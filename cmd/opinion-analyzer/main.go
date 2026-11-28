package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/gomlx/compute"
	gomlxmodel "github.com/gomlx/gomlx/ml/model"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/cluster"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/data"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/graphkg"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/logging"
	nfcmodel "github.com/Mireu-Lab/gomlx-korean-opinion/internal/model"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/nfc"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/onnxembed"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/pretrain"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/release"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/sorter"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/tokenizer"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(2)
	}
	subcmd := os.Args[1]
	fs := flag.NewFlagSet(subcmd, flag.ExitOnError)
	configPath := fs.String("config", "config/config.yaml", "Path to config YAML")
	backendOverride := fs.String("backend", "", "Override runtime.backend (e.g., go, xla:cpu)")
	stepsOverride := fs.Int("steps", 0, "Override training.train_steps")
	batchOverride := fs.Int("batch-size", 0, "Override training.batch_size")
	sortBy := fs.String("sort-by", "", "Override sort mode (topic_nfc_intensity|nfc_intensity|intensity_desc|sentiment_neg_desc)")
	inputPath := fs.String("input", "", "Optional external file (.jsonl/.csv/.tsv/.txt for analyze/infer, or annotated CSV for eval-human)")
	rawText := fs.String("text", "", "Raw Korean text for infer (repeat via --input for batches)")
	hfStagingDir := fs.String("hf-staging-dir", envOrDefault("NFC_HF_STAGING_DIR", "data/hf_model_staging"), "Destination for export-model Hugging Face staging")
	outputDirOverride := fs.String("output-dir", "", "Override paths.output_dir")
	checkpointDirOverride := fs.String("checkpoint-dir", "", "Override paths.checkpoint_dir")
	providerOverride := fs.String("provider", "", "Override embedding.provider (scratch_relu|ko_hub_graph)")
	rootsConfigOverride := fs.String("roots-config", "", "Override embedding.roots_config_file")
	pretrainDirOverride := fs.String("pretrain-dir", "", "Override embedding.pretrain_checkpoint_dir")
	freezeEncoderOverride := fs.String("freeze-encoder", "", "Override embedding.freeze_encoder (true|false)")
	pcaDimOverride := fs.Int("pca-dim", -1, "Override embedding.pca_dim (>=0)")
	limit := fs.Int("limit", 0, "Optional maximum number of articles to analyze (0 = all)")
	resume := fs.Bool("resume", false, "Resume training from existing checkpoint instead of resetting")
	if err := fs.Parse(os.Args[2:]); err != nil {
		os.Exit(2)
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "config load error: %v\n", err)
		os.Exit(1)
	}
	if strings.TrimSpace(*backendOverride) != "" {
		cfg.Runtime.Backend = strings.TrimSpace(*backendOverride)
	}
	if *stepsOverride > 0 {
		cfg.Training.TrainSteps = *stepsOverride
	}
	if *batchOverride > 0 {
		cfg.Training.BatchSize = *batchOverride
	}
	if strings.TrimSpace(*sortBy) != "" {
		cfg.InferenceAndSorting.DefaultSortMode = strings.TrimSpace(*sortBy)
	}
	if strings.TrimSpace(*outputDirOverride) != "" {
		cfg.Paths.OutputDir = strings.TrimSpace(*outputDirOverride)
	}
	if strings.TrimSpace(*checkpointDirOverride) != "" {
		cfg.Paths.CheckpointDir = strings.TrimSpace(*checkpointDirOverride)
	}
	if strings.TrimSpace(*providerOverride) != "" {
		cfg.Embedding.Provider = strings.TrimSpace(*providerOverride)
	}
	if strings.TrimSpace(*rootsConfigOverride) != "" {
		cfg.Embedding.RootsConfigFile = strings.TrimSpace(*rootsConfigOverride)
	}
	if strings.TrimSpace(*pretrainDirOverride) != "" {
		cfg.Embedding.PretrainCheckpointDir = strings.TrimSpace(*pretrainDirOverride)
	}
	if strings.TrimSpace(*freezeEncoderOverride) != "" {
		b, parseErr := strconv.ParseBool(strings.TrimSpace(*freezeEncoderOverride))
		if parseErr != nil {
			fmt.Fprintf(os.Stderr, "invalid --freeze-encoder=%q: %v\n", *freezeEncoderOverride, parseErr)
			os.Exit(1)
		}
		cfg.Embedding.FreezeEncoder = b
	}
	if *pcaDimOverride >= 0 {
		cfg.Embedding.PCADim = *pcaDimOverride
	}
	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "config validation error: %v\n", err)
		os.Exit(1)
	}

	log, err := logging.NewDualLogger(
		cfg.Paths.LogDir,
		cfg.Logging.StatusFilename,
		cfg.Logging.ProcessingFilename,
		cfg.Logging.Level,
		true,
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "logger init error: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = log.Close() }()

	if err := runCommand(subcmd, cfg, *inputPath, *rawText, *hfStagingDir, *limit, *resume, *stepsOverride, *batchOverride, log); err != nil {
		log.Status(strings.ToUpper(subcmd), "FAILED", "%v", err)
		log.Error("cli", "command %q failed: %v", subcmd, err)
		os.Exit(1)
	}
}

func envOrDefault(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func runCommand(
	subcmd string,
	cfg *config.Config,
	inputPath string,
	rawText string,
	hfStagingDir string,
	limit int,
	resume bool,
	stepsOverride int,
	batchOverride int,
	log *logging.DualLogger,
) error {
	tax, err := nfc.LoadTaxonomy(cfg.Paths.TopicsFile, cfg.Paths.NFCTaxonomyFile)
	if err != nil {
		return err
	}
	log.Info("taxonomy", "loaded %d KLUE-YNAT topics and %d Nuclear Feeling Checklist categories",
		tax.NumTopics(), tax.NumFeelings())

	switch subcmd {
	case "download":
		return stageDownload(cfg, log)
	case "build-graph":
		_, err := stageBuildGraph(cfg, tax, log)
		return err
	case "pretrain":
		return stagePretrain(cfg, tax, stepsOverride, batchOverride, log)
	case "export-dataset":
		return stageExportDataset(cfg, tax, log)
	case "export-model":
		return stageExportModel(cfg, tax, hfStagingDir, log)
	case "eval-human":
		return stageEvalHuman(cfg, inputPath, log)
	}
	if cfg.Embedding.Provider == "onnx_backbone_relu" {
		return runONNXCommand(subcmd, cfg, tax, inputPath, rawText, limit, resume, log)
	}

	switch subcmd {
	case "train":
		backend, _, _, _, err := stageTrain(cfg, tax, resume, log)
		if backend != nil {
			backend.Finalize()
		}
		return err
	case "eval":
		return stageEval(cfg, tax, log)
	case "analyze":
		return stageAnalyze(cfg, tax, inputPath, limit, nil, nil, nil, log)
	case "infer":
		return stageInfer(cfg, tax, inputPath, rawText, log)
	case "cluster":
		return stageCluster(cfg, tax, nil, nil, nil, log)
	case "pipeline":
		if err := stageDownload(cfg, log); err != nil {
			return err
		}
		if nfcmodel.IsKoHubProvider(cfg.Embedding.Provider) {
			metaPath := filepath.Join(cfg.Embedding.PretrainCheckpointDir, "pretrain_meta.json")
			if _, statErr := os.Stat(metaPath); statErr != nil {
				if err := stagePretrain(cfg, tax, 0, 0, log); err != nil {
					return err
				}
			}
		}
		backend, store, tok, ynatVal, err := stageTrain(cfg, tax, resume, log)
		if err != nil {
			return err
		}
		defer backend.Finalize()

		nsmcTestRecs, skipped, err := data.LoadNSMC(cfg.NSMCTestPath())
		if err != nil {
			return err
		}
		log.Info("loader", "loaded %d NSMC test rows (skipped %d)", len(nsmcTestRecs), skipped)
		rng := rand.New(rand.NewSource(cfg.Runtime.Seed + 1))
		nsmcTestEx, _ := data.PrepareExamples(nsmcTestRecs, tok, tax, cfg.Training.NFCNegativeCalibrationProb, rng)

		log.Status("EVAL", "START", "evaluating GoMLX ReLU model on KLUE-YNAT val (%d) and NSMC test (%d)",
			len(ynatVal), len(nsmcTestEx))
		evalReport, ynatPreds, err := nfcmodel.Evaluate(backend, store, cfg, tax, tok.VocabLen(), ynatVal, nsmcTestEx)
		if err != nil {
			return err
		}
		logEvalReport(log, evalReport)
		log.Status("EVAL", "SUCCESS", "YNAT_acc=%.4f YNAT_macro_f1=%.4f NSMC_acc=%.4f NFC_recall@3=%.4f",
			evalReport.YNATAccuracy, evalReport.YNATMacroF1, evalReport.NSMCAccuracy, evalReport.NFCRecallAt3)

		pre := &precomputedAnalyze{
			tok:        tok,
			ynatVal:    ynatVal,
			ynatPreds:  ynatPreds,
			nsmcTest:   nsmcTestEx,
			evalReport: evalReport,
		}
		if err := stageAnalyze(cfg, tax, inputPath, limit, backend, store, pre, log); err != nil {
			return err
		}
		if cfg.Clustering.Enabled {
			return stageCluster(cfg, tax, backend, store, pre, log)
		}
		return nil
	default:
		printUsage()
		return fmt.Errorf("unknown subcommand %q", subcmd)
	}
}

type precomputedAnalyze struct {
	tok        *tokenizer.Tokenizer
	ynatVal    []data.EncodedExample
	ynatPreds  []sorter.ArticlePrediction
	nsmcTest   []data.EncodedExample
	evalReport *nfcmodel.EvalReport
}

func stageDownload(cfg *config.Config, log *logging.DualLogger) error {
	log.Status("DOWNLOAD", "START", "ensuring NSMC and KLUE-YNAT datasets in %s", cfg.Paths.DataDir)
	if err := data.EnsureDatasets(cfg, log); err != nil {
		return err
	}
	log.Status("DOWNLOAD", "SUCCESS", "datasets ready in %s", cfg.Paths.DataDir)
	return nil
}

func loadTrainingCorpusTexts(cfg *config.Config, log *logging.DualLogger) ([]string, error) {
	if err := data.EnsureDatasets(cfg, log); err != nil {
		return nil, err
	}
	ynatRecs, err := data.LoadYNAT(cfg.YNATTrainPath())
	if err != nil {
		return nil, err
	}
	nsmcRecs, _, err := data.LoadNSMC(cfg.NSMCTrainPath())
	if err != nil {
		return nil, err
	}
	corpus := make([]string, 0, len(ynatRecs)+len(nsmcRecs))
	for _, r := range ynatRecs {
		corpus = append(corpus, r.Text)
	}
	for _, r := range nsmcRecs {
		corpus = append(corpus, r.Text)
	}
	return corpus, nil
}

func stageBuildGraph(cfg *config.Config, tax *nfc.Taxonomy, log *logging.DualLogger) (*graphkg.Graph, error) {
	rootsCfg, err := graphkg.LoadRootsConfig(cfg.Embedding.RootsConfigFile)
	if err != nil {
		return nil, err
	}
	log.Status("BUILD_GRAPH", "START", "downloading/verifying multilingual roots and building KO-Hub CSR graph")
	if err := graphkg.EnsureSources(rootsCfg, log); err != nil {
		return nil, err
	}
	corpus, err := loadTrainingCorpusTexts(cfg, log)
	if err != nil {
		return nil, err
	}
	g, err := graphkg.BuildGraph(rootsCfg, corpus, tax, log)
	if err != nil {
		return nil, err
	}
	if err := g.Save(rootsCfg.GraphFile); err != nil {
		return nil, err
	}
	log.Info("graphkg", "saved KO-Hub heterogeneous CSR graph to %s", rootsCfg.GraphFile)
	return g, nil
}

func stagePretrain(cfg *config.Config, tax *nfc.Taxonomy, stepsOverride, batchOverride int, log *logging.DualLogger) error {
	rootsCfg, err := graphkg.LoadRootsConfig(cfg.Embedding.RootsConfigFile)
	if err != nil {
		return err
	}
	if strings.TrimSpace(cfg.Embedding.PretrainCheckpointDir) != "" {
		rootsCfg.Pretrain.CheckpointDir = cfg.Embedding.PretrainCheckpointDir
	}
	if stepsOverride > 0 {
		rootsCfg.Pretrain.PretrainSteps = stepsOverride
	}
	if batchOverride > 0 {
		rootsCfg.Pretrain.BatchSize = batchOverride
	}
	rootsCfg.Pretrain.EmbedDim = cfg.Model.EmbedDim
	rootsCfg.Pretrain.OutDim = cfg.Embedding.StudentOutDim

	log.Status("PRETRAIN", "START", "preparing KO-Hub graph & corpus for contrastive pretraining (steps=%d, out_dim=%d)",
		rootsCfg.Pretrain.PretrainSteps, rootsCfg.Pretrain.OutDim)

	var g *graphkg.Graph
	if _, statErr := os.Stat(rootsCfg.GraphFile); statErr == nil {
		g, err = graphkg.Load(rootsCfg.GraphFile)
		if err != nil {
			return err
		}
		log.Info("pretrain", "loaded cached KO-Hub graph from %s (nodes=%d edges=%d)",
			rootsCfg.GraphFile, g.Stats.NumNodes, g.Stats.NumEdges)
	} else {
		g, err = stageBuildGraph(cfg, tax, log)
		if err != nil {
			return err
		}
	}

	corpus, err := loadTrainingCorpusTexts(cfg, log)
	if err != nil {
		return err
	}
	tok := tokenizer.Fit(corpus, cfg.Tokenizer)
	log.Info("pretrain", "fitted shared KO-Hub tokenizer vocab=%d", tok.VocabLen())

	backend, err := compute.NewWithConfig(cfg.Runtime.Backend)
	if err != nil {
		return fmt.Errorf("init GoMLX backend %q: %w", cfg.Runtime.Backend, err)
	}
	defer backend.Finalize()

	store, _, err := pretrain.TrainPretrain(backend, &rootsCfg.Pretrain, tok, g, corpus, cfg.Runtime.Seed, cfg.Tokenizer.VocabFilename, log)
	if store != nil {
		store.Finalize()
	}
	return err
}

func stageTrain(
	cfg *config.Config,
	tax *nfc.Taxonomy,
	resume bool,
	log *logging.DualLogger,
) (compute.Backend, *gomlxmodel.Store, *tokenizer.Tokenizer, []data.EncodedExample, error) {
	log.Status("LOAD", "START", "loading NSMC train and KLUE-YNAT train/val datasets")
	nsmcTrainRecs, skippedTrain, err := data.LoadNSMC(cfg.NSMCTrainPath())
	if err != nil {
		return nil, nil, nil, nil, err
	}
	ynatTrainRecs, err := data.LoadYNAT(cfg.YNATTrainPath())
	if err != nil {
		return nil, nil, nil, nil, err
	}
	ynatValRecs, err := data.LoadYNAT(cfg.YNATValPath())
	if err != nil {
		return nil, nil, nil, nil, err
	}
	log.Info("loader", "loaded NSMC_train=%d (skipped %d), YNAT_train=%d, YNAT_val=%d",
		len(nsmcTrainRecs), skippedTrain, len(ynatTrainRecs), len(ynatValRecs))
	log.Status("LOAD", "SUCCESS", "NSMC_train=%d YNAT_train=%d YNAT_val=%d",
		len(nsmcTrainRecs), len(ynatTrainRecs), len(ynatValRecs))

	log.Status("TOKENIZE", "START", "fitting/loading Korean word + Hangul n-gram tokenizer")
	var tok *tokenizer.Tokenizer
	if resume {
		if loadedTok, err := tokenizer.Load(cfg.VocabPath()); err == nil {
			tok = loadedTok
			log.Info("tokenizer", "loaded existing vocab (%d tokens) from %s", tok.VocabLen(), cfg.VocabPath())
		}
	} else if nfcmodel.IsKoHubProvider(cfg.Embedding.Provider) {
		pretrainTokPath := filepath.Join(cfg.Embedding.PretrainCheckpointDir, cfg.Tokenizer.VocabFilename)
		loadedTok, err := tokenizer.Load(pretrainTokPath)
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("load KO-Hub pretrain vocab %q (run 'pretrain' first): %w", pretrainTokPath, err)
		}
		meta, err := pretrain.LoadPretrainMeta(cfg.Embedding.PretrainCheckpointDir)
		if err != nil {
			return nil, nil, nil, nil, err
		}
		if meta.EmbedDim != cfg.Model.EmbedDim || meta.OutDim != cfg.Embedding.StudentOutDim {
			return nil, nil, nil, nil, fmt.Errorf("pretrain meta dims (embed=%d, out=%d) do not match config (embed=%d, student_out_dim=%d)",
				meta.EmbedDim, meta.OutDim, cfg.Model.EmbedDim, cfg.Embedding.StudentOutDim)
		}
		tok = loadedTok
		log.Info("tokenizer", "reusing KO-Hub pretrain vocab (%d tokens) from %s", tok.VocabLen(), pretrainTokPath)
	}
	if tok == nil {
		corpus := make([]string, 0, len(ynatTrainRecs)+len(nsmcTrainRecs))
		for _, r := range ynatTrainRecs {
			corpus = append(corpus, r.Text)
		}
		for _, r := range nsmcTrainRecs {
			corpus = append(corpus, r.Text)
		}
		tok = tokenizer.Fit(corpus, cfg.Tokenizer)
		log.Info("tokenizer", "fitted vocabulary size=%d (max=%d, min_freq=%d)",
			tok.VocabLen(), cfg.Tokenizer.VocabSize, cfg.Tokenizer.MinFreq)
	}

	rng := rand.New(rand.NewSource(cfg.Runtime.Seed))
	ynatTrainEx, yStats := data.PrepareExamples(ynatTrainRecs, tok, tax, cfg.Training.NFCNegativeCalibrationProb, rng)
	nsmcTrainEx, nStats := data.PrepareExamples(nsmcTrainRecs, tok, tax, cfg.Training.NFCNegativeCalibrationProb, rng)
	ynatValEx, yvStats := data.PrepareExamples(ynatValRecs, tok, tax, cfg.Training.NFCNegativeCalibrationProb, rng)

	log.Info("nfc_weak_label", "YNAT_train NFC active=%d/%d supervised=%d | NSMC_train NFC active=%d/%d supervised=%d | YNAT_val active=%d/%d",
		yStats.NFCActiveExamples, yStats.Total, yStats.NFCSupervised,
		nStats.NFCActiveExamples, nStats.Total, nStats.NFCSupervised,
		yvStats.NFCActiveExamples, yvStats.Total)
	for i, f := range tax.Feelings {
		log.Debug("nfc_coverage", "feeling[%02d] %s: ynat_train=%d nsmc_train=%d ynat_val=%d",
			i, f.KoName, yStats.PerFeelingHits[i], nStats.PerFeelingHits[i], yvStats.PerFeelingHits[i])
	}
	log.Status("TOKENIZE", "SUCCESS", "vocab=%d ynat_nfc_active=%d nsmc_nfc_active=%d",
		tok.VocabLen(), yStats.NFCActiveExamples, nStats.NFCActiveExamples)

	log.Status("TRAIN", "START", "initializing GoMLX backend %q", cfg.Runtime.Backend)
	backend, err := compute.NewWithConfig(cfg.Runtime.Backend)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("init GoMLX backend %q: %w", cfg.Runtime.Backend, err)
	}

	store, err := nfcmodel.Train(backend, cfg, tax, tok.VocabLen(), ynatTrainEx, nsmcTrainEx, resume, log)
	if err != nil {
		backend.Finalize()
		return nil, nil, nil, nil, err
	}
	if err := tok.Save(cfg.VocabPath()); err != nil {
		backend.Finalize()
		return nil, nil, nil, nil, err
	}
	log.Info("tokenizer", "saved vocabulary to %s", cfg.VocabPath())
	log.Status("TRAIN", "SUCCESS", "checkpoint and vocab saved to %s", cfg.Paths.CheckpointDir)
	return backend, store, tok, ynatValEx, nil
}

func stageEval(cfg *config.Config, tax *nfc.Taxonomy, log *logging.DualLogger) error {
	tok, err := tokenizer.Load(cfg.VocabPath())
	if err != nil {
		return err
	}
	ynatValRecs, err := data.LoadYNAT(cfg.YNATValPath())
	if err != nil {
		return err
	}
	nsmcTestRecs, skipped, err := data.LoadNSMC(cfg.NSMCTestPath())
	if err != nil {
		return err
	}
	log.Info("loader", "loaded YNAT_val=%d and NSMC_test=%d (skipped %d)", len(ynatValRecs), len(nsmcTestRecs), skipped)

	rng := rand.New(rand.NewSource(cfg.Runtime.Seed + 1))
	ynatValEx, _ := data.PrepareExamples(ynatValRecs, tok, tax, cfg.Training.NFCNegativeCalibrationProb, rng)
	nsmcTestEx, _ := data.PrepareExamples(nsmcTestRecs, tok, tax, cfg.Training.NFCNegativeCalibrationProb, rng)

	backend, err := compute.NewWithConfig(cfg.Runtime.Backend)
	if err != nil {
		return fmt.Errorf("init GoMLX backend %q: %w", cfg.Runtime.Backend, err)
	}
	defer backend.Finalize()

	graphCfg := nfcmodel.BuildGraphConfig(cfg, tok.VocabLen(), tax.NumTopics(), tax.NumFeelings())
	pred, store, err := nfcmodel.LoadPredictor(backend, cfg.Paths.CheckpointDir, graphCfg, cfg.Training.EvalBatchSize, cfg.Tokenizer.MaxSeqLen, cfg.Tokenizer.WordSeqLen)
	if err != nil {
		return err
	}
	pred.Close()

	log.Status("EVAL", "START", "evaluating checkpoint in %s", cfg.Paths.CheckpointDir)
	report, _, err := nfcmodel.Evaluate(backend, store, cfg, tax, tok.VocabLen(), ynatValEx, nsmcTestEx)
	if err != nil {
		return err
	}
	logEvalReport(log, report)
	log.Status("EVAL", "SUCCESS", "YNAT_acc=%.4f YNAT_macro_f1=%.4f NSMC_acc=%.4f NFC_recall@3=%.4f",
		report.YNATAccuracy, report.YNATMacroF1, report.NSMCAccuracy, report.NFCRecallAt3)
	return nil
}

func stageAnalyze(
	cfg *config.Config,
	tax *nfc.Taxonomy,
	inputPath string,
	limit int,
	existingBackend compute.Backend,
	existingStore *gomlxmodel.Store,
	pre *precomputedAnalyze,
	log *logging.DualLogger,
) error {
	log.Status("ANALYZE", "START", "classifying and sorting articles (mode=%s)", cfg.InferenceAndSorting.DefaultSortMode)

	var tok *tokenizer.Tokenizer
	var examples []data.EncodedExample
	var preds []sorter.ArticlePrediction
	var evalReport *nfcmodel.EvalReport

	if inputPath == "" && pre != nil {
		tok = pre.tok
		examples = pre.ynatVal
		preds = pre.ynatPreds
		evalReport = pre.evalReport
		if limit > 0 && limit < len(examples) {
			examples = examples[:limit]
			preds = preds[:limit]
		}
	} else {
		var err error
		if pre != nil && pre.tok != nil {
			tok = pre.tok
		} else {
			tok, err = tokenizer.Load(cfg.VocabPath())
			if err != nil {
				return err
			}
		}

		var records []data.Record
		if inputPath != "" {
			records, err = data.LoadCustomArticles(inputPath)
			if err != nil {
				return err
			}
			log.Info("analyze", "loaded %d custom articles from %s", len(records), inputPath)
		} else {
			records, err = data.LoadYNAT(cfg.YNATValPath())
			if err != nil {
				return err
			}
			log.Info("analyze", "loaded %d KLUE-YNAT validation articles", len(records))
		}
		if limit > 0 && limit < len(records) {
			records = records[:limit]
		}
		examples, _ = data.PrepareExamples(records, tok, tax, 0, nil)

		backend := existingBackend
		if backend == nil {
			backend, err = compute.NewWithConfig(cfg.Runtime.Backend)
			if err != nil {
				return fmt.Errorf("init GoMLX backend %q: %w", cfg.Runtime.Backend, err)
			}
			defer backend.Finalize()
		}

		graphCfg := nfcmodel.BuildGraphConfig(cfg, tok.VocabLen(), tax.NumTopics(), tax.NumFeelings())
		var pred *nfcmodel.Predictor
		if existingStore != nil {
			pred, err = nfcmodel.NewPredictor(backend, existingStore, graphCfg, cfg.Training.EvalBatchSize, cfg.Tokenizer.MaxSeqLen, cfg.Tokenizer.WordSeqLen)
		} else {
			pred, _, err = nfcmodel.LoadPredictor(backend, cfg.Paths.CheckpointDir, graphCfg, cfg.Training.EvalBatchSize, cfg.Tokenizer.MaxSeqLen, cfg.Tokenizer.WordSeqLen)
		}
		if err != nil {
			return err
		}
		defer pred.Close()

		preds, err = pred.PredictExamples(examples)
		if err != nil {
			return err
		}
		if pre != nil {
			evalReport = pre.evalReport
		}
	}

	records := make([]data.Record, len(examples))
	for i, ex := range examples {
		records[i] = ex.Record
	}

	scored, err := sorter.ScoreArticles(records, preds, tax, cfg.InferenceAndSorting)
	if err != nil {
		return err
	}
	if err := sorter.SortArticles(scored, cfg.InferenceAndSorting.DefaultSortMode, tax); err != nil {
		return err
	}

	report := sorter.BuildReport(scored, tax, cfg.InferenceAndSorting.DefaultSortMode, cfg.InferenceAndSorting.NFCActiveThreshold, evalReport)
	if err := sorter.WriteOutputs(report, cfg.Paths.OutputDir, cfg.InferenceAndSorting, tax); err != nil {
		return err
	}
	log.Status("ANALYZE", "SUCCESS", "wrote %d sorted articles to %s (%s, %s, %s)",
		len(scored), cfg.Paths.OutputDir,
		filepath.Join(cfg.Paths.OutputDir, "articles_sorted.json"),
		filepath.Join(cfg.Paths.OutputDir, "articles_sorted.csv"),
		filepath.Join(cfg.Paths.OutputDir, "social_opinion_report.md"))
	return nil
}

func stageExportModel(cfg *config.Config, tax *nfc.Taxonomy, stagingDir string, log *logging.DualLogger) error {
	if strings.TrimSpace(stagingDir) == "" {
		stagingDir = filepath.Join("data", "hf_model_staging")
	}
	log.Status("EXPORT_MODEL", "START", "staging trained weights for Hugging Face at %s (provider=%s)", stagingDir, cfg.Embedding.Provider)
	man, err := release.StageModel(cfg, stagingDir)
	if err != nil {
		return err
	}
	tokLen := 0
	if tok, err := tokenizer.Load(cfg.VocabPath()); err == nil {
		tokLen = tok.VocabLen()
	}
	spec, err := nfcmodel.ResolveRunSpec(cfg.Paths.CheckpointDir, cfg, tokLen, tax.NumTopics(), tax.NumFeelings())
	if err == nil {
		ckptRel := filepath.Join(stagingDir, "checkpoints", filepath.Base(cfg.Paths.CheckpointDir))
		if err := nfcmodel.SaveRunSpec(ckptRel, spec); err != nil {
			log.Warn("export_model", "could not write graph_config.json: %v", err)
		} else {
			log.Info("export_model", "wrote graph_config.json provider=%s vocab=%d", spec.Provider, spec.Graph.VocabSize)
		}
	}
	log.Status("EXPORT_MODEL", "SUCCESS", "copied %d files to %s; next: ./scripts/publish_hf.sh model", len(man.CopiedFiles), man.StagingDir)
	return nil
}

func stageInfer(cfg *config.Config, tax *nfc.Taxonomy, inputPath, rawText string, log *logging.DualLogger) error {
	records, err := release.CollectInferRecords(rawText, inputPath)
	if err != nil {
		return err
	}
	log.Status("INFER", "START", "running checkpoint inference on %d text(s) provider=%s ckpt=%s",
		len(records), cfg.Embedding.Provider, cfg.Paths.CheckpointDir)

	tok, err := tokenizer.Load(cfg.VocabPath())
	if err != nil {
		return fmt.Errorf("load vocab for infer (train or export-model first): %w", err)
	}
	spec, err := nfcmodel.ResolveRunSpec(cfg.Paths.CheckpointDir, cfg, tok.VocabLen(), tax.NumTopics(), tax.NumFeelings())
	if err != nil {
		return err
	}
	log.Info("infer", "using provider=%s vocab=%d max_seq=%d (graph_config or checkpoint detect)",
		spec.Provider, spec.Graph.VocabSize, spec.MaxSeqLen)
	examples, _ := data.PrepareExamples(records, tok, tax, 0, nil)

	backend, err := compute.NewWithConfig(cfg.Runtime.Backend)
	if err != nil {
		return fmt.Errorf("init GoMLX backend %q: %w", cfg.Runtime.Backend, err)
	}
	defer backend.Finalize()

	graphCfg := spec.Graph
	pred, _, err := nfcmodel.LoadPredictor(backend, cfg.Paths.CheckpointDir, graphCfg, spec.EvalBatchSize, spec.MaxSeqLen, spec.WordSeqLen)
	if err != nil {
		return err
	}
	defer pred.Close()

	preds, err := pred.PredictExamples(examples)
	if err != nil {
		return err
	}
	scored, err := sorter.ScoreArticles(records, preds, tax, cfg.InferenceAndSorting)
	if err != nil {
		return err
	}
	return writeInferJSON(cfg, scored, log)
}

func writeInferJSON(cfg *config.Config, scored []sorter.ScoredArticle, log *logging.DualLogger) error {
	if err := os.MkdirAll(cfg.Paths.OutputDir, 0o755); err != nil {
		return err
	}
	outPath := filepath.Join(cfg.Paths.OutputDir, "infer.json")
	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(scored); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	stdout := json.NewEncoder(os.Stdout)
	stdout.SetEscapeHTML(false)
	stdout.SetIndent("", "  ")
	if err := stdout.Encode(scored); err != nil {
		return err
	}
	log.Status("INFER", "SUCCESS", "wrote %d predictions to %s", len(scored), outPath)
	return nil
}

func stageCluster(
	cfg *config.Config,
	tax *nfc.Taxonomy,
	existingBackend compute.Backend,
	existingStore *gomlxmodel.Store,
	pre *precomputedAnalyze,
	log *logging.DualLogger,
) error {
	log.Status("CLUSTER", "START", "running 3-method clustering (HDBSCAN, Entity Anchors, Hierarchical Topics) & 3-stage validation (ynat=%d, nsmc=%d)",
		cfg.Clustering.YNATSampleSize, cfg.Clustering.NSMCSampleSize)

	var tok *tokenizer.Tokenizer
	var err error
	if pre != nil && pre.tok != nil {
		tok = pre.tok
	} else {
		tok, err = tokenizer.Load(cfg.VocabPath())
		if err != nil {
			return err
		}
	}

	var ynatSampleEx []data.EncodedExample
	var nsmcSampleEx []data.EncodedExample
	var ynatSamplePreds []sorter.ArticlePrediction

	rng := rand.New(rand.NewSource(cfg.Runtime.Seed + 99))
	if pre != nil && len(pre.ynatVal) > 0 && len(pre.ynatPreds) == len(pre.ynatVal) {
		idxs := sampleIndices(len(pre.ynatVal), cfg.Clustering.YNATSampleSize, rng)
		for _, idx := range idxs {
			ynatSampleEx = append(ynatSampleEx, pre.ynatVal[idx])
			ynatSamplePreds = append(ynatSamplePreds, pre.ynatPreds[idx])
		}
	} else {
		ynatRecs, err := data.LoadYNAT(cfg.YNATValPath())
		if err != nil {
			return err
		}
		idxs := sampleIndices(len(ynatRecs), cfg.Clustering.YNATSampleSize, rng)
		sub := make([]data.Record, len(idxs))
		for i, idx := range idxs {
			sub[i] = ynatRecs[idx]
		}
		ynatSampleEx, _ = data.PrepareExamples(sub, tok, tax, 0, nil)
	}

	if pre != nil && len(pre.nsmcTest) > 0 {
		idxs := sampleIndices(len(pre.nsmcTest), cfg.Clustering.NSMCSampleSize, rng)
		for _, idx := range idxs {
			nsmcSampleEx = append(nsmcSampleEx, pre.nsmcTest[idx])
		}
	} else {
		nsmcRecs, _, err := data.LoadNSMC(cfg.NSMCTestPath())
		if err != nil {
			return err
		}
		idxs := sampleIndices(len(nsmcRecs), cfg.Clustering.NSMCSampleSize, rng)
		sub := make([]data.Record, len(idxs))
		for i, idx := range idxs {
			sub[i] = nsmcRecs[idx]
		}
		nsmcSampleEx, _ = data.PrepareExamples(sub, tok, tax, 0, nil)
	}

	backend := existingBackend
	if backend == nil {
		backend, err = compute.NewWithConfig(cfg.Runtime.Backend)
		if err != nil {
			return fmt.Errorf("init GoMLX backend %q: %w", cfg.Runtime.Backend, err)
		}
		defer backend.Finalize()
	}

	graphCfg := nfcmodel.BuildGraphConfig(cfg, tok.VocabLen(), tax.NumTopics(), tax.NumFeelings())
	var pred *nfcmodel.Predictor
	if existingStore != nil {
		pred, err = nfcmodel.NewPredictor(backend, existingStore, graphCfg, cfg.Training.EvalBatchSize, cfg.Tokenizer.MaxSeqLen, cfg.Tokenizer.WordSeqLen)
	} else {
		pred, _, err = nfcmodel.LoadPredictor(backend, cfg.Paths.CheckpointDir, graphCfg, cfg.Training.EvalBatchSize, cfg.Tokenizer.MaxSeqLen, cfg.Tokenizer.WordSeqLen)
	}
	if err != nil {
		return err
	}
	defer pred.Close()

	if len(ynatSamplePreds) != len(ynatSampleEx) {
		ynatSamplePreds, err = pred.PredictExamples(ynatSampleEx)
		if err != nil {
			return err
		}
	}
	nsmcSamplePreds, err := pred.PredictExamples(nsmcSampleEx)
	if err != nil {
		return err
	}

	total := len(ynatSampleEx) + len(nsmcSampleEx)
	mixedRecords := make([]data.Record, 0, total)
	mixedPreds := make([]sorter.ArticlePrediction, 0, total)
	for i, ex := range ynatSampleEx {
		mixedRecords = append(mixedRecords, ex.Record)
		mixedPreds = append(mixedPreds, ynatSamplePreds[i])
	}
	for i, ex := range nsmcSampleEx {
		mixedRecords = append(mixedRecords, ex.Record)
		mixedPreds = append(mixedPreds, nsmcSamplePreds[i])
	}

	scored, err := sorter.ScoreArticles(mixedRecords, mixedPreds, tax, cfg.InferenceAndSorting)
	if err != nil {
		return err
	}

	report, err := cluster.RunClusteringPipeline(context.Background(), mixedRecords, mixedPreds, scored, tax, cfg, log)
	if err != nil {
		return err
	}

	v := report.Validation
	log.Info("cluster", "HDBSCAN clusters=%d (cross_domain=%d), clustered=%d/%d, noise=%d (%.1f%%)",
		v.NumClusters, v.CrossDomainClusters, v.ClusteredDocs, v.TotalDocs, v.NoiseDocs, v.NoiseRatio*100)
	log.Info("cluster_val", "Silhouette=%.4f (target=%.2f), DunnCentroid=%.4f, DunnStrict=%.4f, MeanNPMI=%.4f, MeanUMass=%.4f",
		v.SilhouetteScore, v.SilhouetteTarget, v.DunnIndexCentroid, v.DunnIndexStrict, v.MeanNPMICoherence, v.MeanUMassCoherence)
	log.Info("cluster_ner", "EntityGroups=%d, CrossDomainBridges=%d, HumanEvalSample=%d",
		v.EntityGroupCount, v.CrossDomainEntityBridges, v.HumanEvalSampleCount)
	log.Status("CLUSTER", "SUCCESS", "clusters=%d silhouette=%.4f npmi=%.4f dunn_c=%.4f bridges=%d reports_dir=%s",
		v.NumClusters, v.SilhouetteScore, v.MeanNPMICoherence, v.DunnIndexCentroid, v.CrossDomainEntityBridges, cfg.Paths.OutputDir)
	return nil
}

func stageEvalHuman(cfg *config.Config, inputPath string, log *logging.DualLogger) error {
	if strings.TrimSpace(inputPath) == "" {
		inputPath = filepath.Join(cfg.Paths.OutputDir, "human_eval_sample.csv")
	}
	log.Status("EVAL_HUMAN", "START", "scoring Gold Standard human evaluation file %s", inputPath)
	sc, err := cluster.ScoreHumanEvalCSV(inputPath, cfg.Clustering.HumanEvalTargetAgreement)
	if err != nil {
		return err
	}
	log.Info("eval_human", "rated=%d agree=%d agreement=%.4f (target=%.2f) cohens_kappa=%.4f passes=%t",
		sc.TotalRated, sc.AgreeCount, sc.AgreementRate, sc.TargetAgreement, sc.CohensKappa, sc.PassesTarget)
	log.Status("EVAL_HUMAN", "SUCCESS", "agreement=%.4f kappa=%.4f passes_target=%t",
		sc.AgreementRate, sc.CohensKappa, sc.PassesTarget)
	return nil
}

func stageExportDataset(cfg *config.Config, tax *nfc.Taxonomy, log *logging.DualLogger) error {
	outDir := cfg.Paths.OutputDir
	log.Status("EXPORT_DATASET", "START", "preparing and exporting preprocessed Parquet dataset to %s", outDir)
	if err := stageDownload(cfg, log); err != nil {
		return err
	}

	ynatTrainRecs, err := data.LoadYNAT(cfg.YNATTrainPath())
	if err != nil {
		return err
	}
	ynatValRecs, err := data.LoadYNAT(cfg.YNATValPath())
	if err != nil {
		return err
	}
	nsmcTrainRecs, skippedTrain, err := data.LoadNSMC(cfg.NSMCTrainPath())
	if err != nil {
		return err
	}
	nsmcTestRecs, skippedTest, err := data.LoadNSMC(cfg.NSMCTestPath())
	if err != nil {
		return err
	}
	log.Info("export_dataset", "loaded splits ynat_train=%d ynat_val=%d nsmc_train=%d (skipped %d) nsmc_test=%d (skipped %d)",
		len(ynatTrainRecs), len(ynatValRecs), len(nsmcTrainRecs), skippedTrain, len(nsmcTestRecs), skippedTest)

	pretrainTokPath := filepath.Join(cfg.Embedding.PretrainCheckpointDir, cfg.Tokenizer.VocabFilename)
	var tok *tokenizer.Tokenizer
	if loaded, err := tokenizer.Load(pretrainTokPath); err == nil {
		tok = loaded
		log.Info("export_dataset", "loaded KO-Hub particle-split vocab (%d tokens) from %s", tok.VocabLen(), pretrainTokPath)
	} else {
		corpus := make([]string, 0, len(ynatTrainRecs)+len(nsmcTrainRecs))
		for _, r := range ynatTrainRecs {
			corpus = append(corpus, r.Text)
		}
		for _, r := range nsmcTrainRecs {
			corpus = append(corpus, r.Text)
		}
		tok = tokenizer.Fit(corpus, cfg.Tokenizer)
		log.Info("export_dataset", "fitted fallback vocab (%d tokens) from training corpus", tok.VocabLen())
	}

	trainRng := rand.New(rand.NewSource(cfg.Runtime.Seed))
	evalRng := rand.New(rand.NewSource(cfg.Runtime.Seed + 1))

	type splitSpec struct {
		name         string
		records      []data.Record
		negCalibProb float64
		rng          *rand.Rand
	}
	specs := []splitSpec{
		{name: "ynat_train", records: ynatTrainRecs, negCalibProb: cfg.Training.NFCNegativeCalibrationProb, rng: trainRng},
		{name: "ynat_validation", records: ynatValRecs, negCalibProb: 0, rng: evalRng},
		{name: "nsmc_train", records: nsmcTrainRecs, negCalibProb: cfg.Training.NFCNegativeCalibrationProb, rng: trainRng},
		{name: "nsmc_test", records: nsmcTestRecs, negCalibProb: 0, rng: evalRng},
	}

	splitStats := make(map[string]data.ExportSplitStats, len(specs))
	for _, sp := range specs {
		examples, prepStats := data.PrepareExamples(sp.records, tok, tax, sp.negCalibProb, sp.rng)
		parquetPath := filepath.Join(outDir, "data", fmt.Sprintf("%s-00000-of-00001.parquet", sp.name))
		st, err := data.ExportPreprocessedSplit(parquetPath, sp.name, examples, tok, tax)
		if err != nil {
			return err
		}
		splitStats[sp.name] = st
		log.Info("export_dataset", "wrote %s rows=%d nfc_active=%d guard_hits=%d supervised=%d prep_guard=%d",
			parquetPath, st.Rows, st.NFCActive, st.HomonymGuardHits, st.NFCSupervised, prepStats.GuardHits)
	}

	vocabOutDir := filepath.Join(outDir, "vocab")
	if err := os.MkdirAll(vocabOutDir, 0755); err != nil {
		return fmt.Errorf("create vocab staging dir: %w", err)
	}
	if err := tok.Save(filepath.Join(vocabOutDir, cfg.Tokenizer.VocabFilename)); err != nil {
		return err
	}

	configOutDir := filepath.Join(outDir, "config")
	if err := os.MkdirAll(configOutDir, 0755); err != nil {
		return fmt.Errorf("create config staging dir: %w", err)
	}
	for _, src := range []string{
		cfg.Paths.TopicsFile,
		cfg.Paths.NFCTaxonomyFile,
		cfg.Paths.EntityRulesFile,
		cfg.Embedding.RootsConfigFile,
	} {
		if strings.TrimSpace(src) == "" {
			continue
		}
		if err := copyBinaryFile(src, filepath.Join(configOutDir, filepath.Base(src))); err != nil {
			return err
		}
	}

	var graphOutRel string
	var graphSize int64
	if strings.TrimSpace(cfg.Embedding.RootsConfigFile) != "" {
		if rootsCfg, err := graphkg.LoadRootsConfig(cfg.Embedding.RootsConfigFile); err == nil && rootsCfg.GraphFile != "" {
			if st, statErr := os.Stat(rootsCfg.GraphFile); statErr == nil && !st.IsDir() {
				dest := filepath.Join(outDir, "graph", filepath.Base(rootsCfg.GraphFile))
				if err := copyBinaryFile(rootsCfg.GraphFile, dest); err != nil {
					return err
				}
				graphOutRel = filepath.ToSlash(filepath.Join("graph", filepath.Base(rootsCfg.GraphFile)))
				graphSize = st.Size()
			}
		}
	}

	summary := data.ExportDatasetSummary{
		CreatedAtUTC:   time.Now().UTC().Format(time.RFC3339),
		VocabSize:      tok.VocabLen(),
		MaxSeqLen:      cfg.Tokenizer.MaxSeqLen,
		WordSeqLen:     cfg.Tokenizer.WordSeqLen,
		SplitParticles: cfg.Tokenizer.SplitParticles,
		ParticleCount:  len(cfg.Tokenizer.KoreanParticles),
		GraphFile:      graphOutRel,
		GraphSizeBytes: graphSize,
		Splits:         splitStats,
	}
	if err := data.WriteDatasetSummary(filepath.Join(outDir, "dataset_stats.json"), summary); err != nil {
		return err
	}
	log.Status("EXPORT_DATASET", "SUCCESS", "exported 4 Parquet splits, vocab, configs, and stats to %s", outDir)
	return nil
}

func copyBinaryFile(src, dst string) error {
	raw, err := os.ReadFile(src)
	if err != nil {
		return fmt.Errorf("read %q: %w", src, err)
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return fmt.Errorf("create dir for %q: %w", dst, err)
	}
	if err := os.WriteFile(dst, raw, 0644); err != nil {
		return fmt.Errorf("write %q: %w", dst, err)
	}
	return nil
}

func sampleIndices(total, limit int, rng *rand.Rand) []int {
	if limit <= 0 || limit >= total {
		out := make([]int, total)
		for i := range out {
			out[i] = i
		}
		return out
	}
	perm := rng.Perm(total)
	return perm[:limit]
}

func logEvalReport(log *logging.DualLogger, r *nfcmodel.EvalReport) {
	log.Info("eval", "KLUE-YNAT (n=%d): accuracy=%.4f macro_f1=%.4f", r.YNATCount, r.YNATAccuracy, r.YNATMacroF1)
	for _, tm := range r.YNATPerTopic {
		log.Info("eval_topic", "topic[%d:%s] support=%d prec=%.4f rec=%.4f f1=%.4f",
			tm.TopicID, tm.KoName, tm.Support, tm.Precision, tm.Recall, tm.F1)
	}
	log.Info("eval", "NSMC Test (n=%d): accuracy=%.4f prec=%.4f rec=%.4f f1=%.4f",
		r.NSMCCount, r.NSMCAccuracy, r.NSMCPrecision, r.NSMCRecall, r.NSMCF1)
	log.Info("eval", "Nuclear Feeling Checklist (supervised=%d active=%d): micro_f1=%.4f macro_f1=%.4f recall@3=%.4f covered_classes=%d",
		r.NFCSupervisedEvalCount, r.NFCActiveEvalCount, r.NFCMicroF1, r.NFCMacroF1, r.NFCRecallAt3, r.NFCCoveredClasses)
}

func printUsage() {
	fmt.Fprintf(os.Stderr, `Usage: opinion-analyzer <command> [flags]

Commands:
  download        Download NSMC and KLUE-YNAT datasets configured in config.yaml
  build-graph     Download Unihan/OMW sources and build KO-Hub multilingual roots CSR graph
  pretrain        Run GoMLX Context-Aware Soft Edge contrastive pretraining for KO-Hub Student
  export-dataset  Export preprocessed Parquet dataset splits, vocab, and KO-Hub graph metadata
  export-model    Stage checkpoints + vocab + taxonomy YAML for Hugging Face (not GitHub)
  train           Fit Korean tokenizer and train GoMLX ReLU multi-task model
  eval            Evaluate saved checkpoint on KLUE-YNAT val, NSMC test, and NFC
  infer           Run trained model on --text and/or --input (prints JSON)
  analyze         Classify and sort news articles by topic, Nuclear Feeling, and intensity
  cluster         Run 3-method clustering (HDBSCAN, Entity Anchors, Hierarchical Topics) & 3-stage validation
  eval-human      Score completed human_eval_sample.csv (Agreement Rate & Cohen's Kappa)
  pipeline        Run end-to-end: download -> (pretrain if ko_hub_graph) -> train -> eval -> analyze -> cluster

Flags:
  --config <path>         Path to config YAML (default: config/config.yaml)
  --backend <name>        Override runtime.backend (go, xla:cpu, etc.)
  --provider <name>       scratch_relu | ko_hub_graph | ko_hub_cnn_relu | ko_hub_transformer_relu | ko_hub_lstm_relu | onnx_backbone_relu
  --roots-config <path>   Override embedding.roots_config_file
  --pretrain-dir <path>   Override embedding.pretrain_checkpoint_dir
  --freeze-encoder <bool> Override embedding.freeze_encoder (true|false)
  --pca-dim <int>         Override embedding.pca_dim (0 disables PCA)
  --checkpoint-dir <path> Override paths.checkpoint_dir
  --steps <int>           Override training.train_steps (or pretrain steps)
  --batch-size <int>      Override training.batch_size (or pretrain batch_size)
  --sort-by <mode>        topic_nfc_intensity | nfc_intensity | intensity_desc | sentiment_neg_desc
  --text <string>         Raw Korean sentence for infer
  --input <path>          Custom file (.jsonl/.csv/.tsv/.txt for analyze/infer, or annotated CSV for eval-human)
  --hf-staging-dir <path> Destination for export-model (default: data/hf_model_staging or NFC_HF_STAGING_DIR)
  --output-dir <path>     Override paths.output_dir
  --limit <int>           Max articles to analyze (0 = all)
  --resume                Resume training from existing checkpoint
`)
}

func runONNXCommand(
	subcmd string,
	cfg *config.Config,
	tax *nfc.Taxonomy,
	inputPath string,
	rawText string,
	limit int,
	resume bool,
	log *logging.DualLogger,
) error {
	switch subcmd {
	case "pipeline", "train", "eval", "analyze", "cluster", "infer":
	default:
		printUsage()
		return fmt.Errorf("unknown subcommand %q", subcmd)
	}
	if err := stageDownload(cfg, log); err != nil {
		return err
	}

	ob := cfg.ONNXBackbone
	backend, err := compute.NewWithConfig(cfg.Runtime.Backend)
	if err != nil {
		return fmt.Errorf("init GoMLX backend %q: %w", cfg.Runtime.Backend, err)
	}
	defer backend.Finalize()

	needEncoder := false
	checkCache := func(path string, count int) bool {
		if embeds, err := onnxembed.LoadEmbedCache(path); err == nil && len(embeds) == count {
			return false
		}
		return true
	}

	subSampleRecords := func(recs []data.Record, capN int, seedOffset int64) []data.Record {
		if capN <= 0 || capN >= len(recs) {
			return recs
		}
		idxs := sampleIndices(len(recs), capN, rand.New(rand.NewSource(cfg.Runtime.Seed+seedOffset)))
		out := make([]data.Record, len(idxs))
		for i, idx := range idxs {
			out[i] = recs[idx]
		}
		return out
	}
	extractTexts := func(recs []data.Record) []string {
		out := make([]string, len(recs))
		for i, r := range recs {
			out[i] = r.Text
		}
		return out
	}

	var ynatTrainRecs, nsmcTrainRecs, ynatValRecs, nsmcTestRecs []data.Record
	if subcmd == "pipeline" || subcmd == "train" {
		ynatTrainRecs, err = data.LoadYNAT(cfg.YNATTrainPath())
		if err != nil {
			return err
		}
		nsmcTrainRecs, _, err = data.LoadNSMC(cfg.NSMCTrainPath())
		if err != nil {
			return err
		}
		ynatTrainRecs = subSampleRecords(ynatTrainRecs, ob.MaxTrainPerDomain, 101)
		nsmcTrainRecs = subSampleRecords(nsmcTrainRecs, ob.MaxTrainPerDomain, 102)
		if checkCache(filepath.Join(ob.EmbedCacheDir, fmt.Sprintf("ynat_train_%d.bin", len(ynatTrainRecs))), len(ynatTrainRecs)) ||
			checkCache(filepath.Join(ob.EmbedCacheDir, fmt.Sprintf("nsmc_train_%d.bin", len(nsmcTrainRecs))), len(nsmcTrainRecs)) {
			needEncoder = true
		}
	}

	if subcmd == "pipeline" || subcmd == "eval" || subcmd == "cluster" || (subcmd == "analyze" && inputPath == "") {
		ynatValRecs, err = data.LoadYNAT(cfg.YNATValPath())
		if err != nil {
			return err
		}
		ynatValRecs = subSampleRecords(ynatValRecs, ob.MaxEvalPerDomain, 103)
		if checkCache(filepath.Join(ob.EmbedCacheDir, fmt.Sprintf("ynat_val_%d.bin", len(ynatValRecs))), len(ynatValRecs)) {
			needEncoder = true
		}
	}
	if subcmd == "pipeline" || subcmd == "eval" || subcmd == "cluster" {
		nsmcTestRecs, _, err = data.LoadNSMC(cfg.NSMCTestPath())
		if err != nil {
			return err
		}
		nsmcTestRecs = subSampleRecords(nsmcTestRecs, ob.MaxEvalPerDomain, 104)
		if checkCache(filepath.Join(ob.EmbedCacheDir, fmt.Sprintf("nsmc_test_%d.bin", len(nsmcTestRecs))), len(nsmcTestRecs)) {
			needEncoder = true
		}
	}
	if subcmd == "analyze" && inputPath != "" {
		needEncoder = true
	}
	if subcmd == "infer" {
		needEncoder = true
	}

	var enc *onnxembed.ONNXEncoder
	if needEncoder {
		modelPath, vocabPath, err := onnxembed.EnsureONNXAssets(ob, cfg.Runtime.HTTPTimeoutSec, cfg.Runtime.HTTPRetries, log)
		if err != nil {
			return err
		}
		enc, err = onnxembed.NewONNXEncoder(backend, modelPath, vocabPath, ob.MaxSeqLen, ob.EncodeBatchSize, ob.EmbedDim)
		if err != nil {
			return err
		}
	}

	loadSplit := func(name string, recs []data.Record) ([][]float32, error) {
		cachePath := filepath.Join(ob.EmbedCacheDir, fmt.Sprintf("%s_%d.bin", name, len(recs)))
		return onnxembed.EncodeOrLoadCache(cachePath, extractTexts(recs), enc, log)
	}

	var ynatTrainEmb, nsmcTrainEmb, ynatValEmb, nsmcTestEmb [][]float32
	if len(ynatTrainRecs) > 0 {
		if ynatTrainEmb, err = loadSplit("ynat_train", ynatTrainRecs); err != nil {
			if enc != nil {
				enc.Close()
			}
			return err
		}
	}
	if len(nsmcTrainRecs) > 0 {
		if nsmcTrainEmb, err = loadSplit("nsmc_train", nsmcTrainRecs); err != nil {
			if enc != nil {
				enc.Close()
			}
			return err
		}
	}
	if len(ynatValRecs) > 0 {
		if ynatValEmb, err = loadSplit("ynat_val", ynatValRecs); err != nil {
			if enc != nil {
				enc.Close()
			}
			return err
		}
	}
	if len(nsmcTestRecs) > 0 {
		if nsmcTestEmb, err = loadSplit("nsmc_test", nsmcTestRecs); err != nil {
			if enc != nil {
				enc.Close()
			}
			return err
		}
	}
	if enc != nil && !(subcmd == "analyze" && inputPath != "") && subcmd != "infer" {
		enc.Close()
		enc = nil
	}

	actualDim := ob.EmbedDim
	for _, set := range [][][]float32{ynatTrainEmb, ynatValEmb, nsmcTrainEmb, nsmcTestEmb} {
		if len(set) > 0 && len(set[0]) > 0 {
			actualDim = len(set[0])
			break
		}
	}

	rng := rand.New(rand.NewSource(cfg.Runtime.Seed))
	var store *gomlxmodel.Store
	if subcmd == "pipeline" || subcmd == "train" {
		ynatTrainEx, _ := nfcmodel.PrepareEmbedExamples(ynatTrainRecs, ynatTrainEmb, tax, cfg.Training.NFCNegativeCalibrationProb, rng)
		nsmcTrainEx, _ := nfcmodel.PrepareEmbedExamples(nsmcTrainRecs, nsmcTrainEmb, tax, cfg.Training.NFCNegativeCalibrationProb, rng)
		log.Status("TRAIN", "START", "training GoMLX ReLU heads on ONNX backbone embeddings (dim=%d, ynat=%d, nsmc=%d)",
			actualDim, len(ynatTrainEx), len(nsmcTrainEx))
		store, err = nfcmodel.TrainOnEmbeddings(backend, cfg, tax, actualDim, ynatTrainEx, nsmcTrainEx, resume, log)
		if err != nil {
			return err
		}
		defer store.Finalize()
		log.Status("TRAIN", "SUCCESS", "ONNX head checkpoint saved to %s", cfg.Paths.CheckpointDir)
		if subcmd == "train" {
			return nil
		}
	} else {
		predTmp, loadedStore, err := nfcmodel.LoadEmbedPredictor(backend, cfg.Paths.CheckpointDir, cfg, actualDim, tax.NumTopics(), tax.NumFeelings(), cfg.Training.EvalBatchSize)
		if err != nil {
			return err
		}
		predTmp.Close()
		store = loadedStore
		defer store.Finalize()
	}

	var evalReport *nfcmodel.EvalReport
	var ynatValEx []nfcmodel.EmbedExample
	var ynatValPreds []sorter.ArticlePrediction

	if subcmd == "pipeline" || subcmd == "eval" {
		evalRng := rand.New(rand.NewSource(cfg.Runtime.Seed + 1))
		ynatValEx, _ = nfcmodel.PrepareEmbedExamples(ynatValRecs, ynatValEmb, tax, cfg.Training.NFCNegativeCalibrationProb, evalRng)
		nsmcTestEx, _ := nfcmodel.PrepareEmbedExamples(nsmcTestRecs, nsmcTestEmb, tax, cfg.Training.NFCNegativeCalibrationProb, evalRng)
		log.Status("EVAL", "START", "evaluating onnx_backbone_relu on YNAT val (%d) and NSMC test (%d)", len(ynatValEx), len(nsmcTestEx))
		evalReport, ynatValPreds, err = nfcmodel.EvaluateOnEmbeddings(backend, store, cfg, tax, actualDim, ynatValEx, nsmcTestEx)
		if err != nil {
			return err
		}
		logEvalReport(log, evalReport)
		log.Status("EVAL", "SUCCESS", "YNAT_acc=%.4f YNAT_macro_f1=%.4f NSMC_acc=%.4f NFC_recall@3=%.4f",
			evalReport.YNATAccuracy, evalReport.YNATMacroF1, evalReport.NSMCAccuracy, evalReport.NFCRecallAt3)
		if subcmd == "eval" {
			return nil
		}
	}

	if subcmd == "infer" {
		records, err := release.CollectInferRecords(rawText, inputPath)
		if err != nil {
			return err
		}
		if enc == nil {
			return fmt.Errorf("onnx encoder required for infer")
		}
		emb, err := enc.EncodeTexts(extractTexts(records), log)
		enc.Close()
		if err != nil {
			return err
		}
		ex, _ := nfcmodel.PrepareEmbedExamples(records, emb, tax, 0, nil)
		pred, err := nfcmodel.NewEmbedPredictor(backend, store, cfg, actualDim, tax.NumTopics(), tax.NumFeelings(), cfg.Training.EvalBatchSize)
		if err != nil {
			return err
		}
		preds, err := pred.PredictEmbedExamples(ex)
		pred.Close()
		if err != nil {
			return err
		}
		scored, err := sorter.ScoreArticles(records, preds, tax, cfg.InferenceAndSorting)
		if err != nil {
			return err
		}
		return writeInferJSON(cfg, scored, log)
	}

	if subcmd == "pipeline" || subcmd == "analyze" {
		log.Status("ANALYZE", "START", "classifying and sorting articles with onnx_backbone_relu")
		var analyzeRecs []data.Record
		var analyzePreds []sorter.ArticlePrediction
		if inputPath == "" && len(ynatValPreds) > 0 {
			analyzeRecs = ynatValRecs
			analyzePreds = ynatValPreds
		} else {
			var analyzeEmb [][]float32
			if inputPath != "" {
				analyzeRecs, err = data.LoadCustomArticles(inputPath)
				if err != nil {
					return err
				}
				analyzeEmb, err = enc.EncodeTexts(extractTexts(analyzeRecs), log)
				enc.Close()
				if err != nil {
					return err
				}
			} else {
				analyzeRecs = ynatValRecs
				analyzeEmb = ynatValEmb
			}
			analyzeEx, _ := nfcmodel.PrepareEmbedExamples(analyzeRecs, analyzeEmb, tax, 0, nil)
			pred, err := nfcmodel.NewEmbedPredictor(backend, store, cfg, actualDim, tax.NumTopics(), tax.NumFeelings(), cfg.Training.EvalBatchSize)
			if err != nil {
				return err
			}
			analyzePreds, err = pred.PredictEmbedExamples(analyzeEx)
			pred.Close()
			if err != nil {
				return err
			}
		}
		if limit > 0 && limit < len(analyzeRecs) {
			analyzeRecs = analyzeRecs[:limit]
			analyzePreds = analyzePreds[:limit]
		}
		scored, err := sorter.ScoreArticles(analyzeRecs, analyzePreds, tax, cfg.InferenceAndSorting)
		if err != nil {
			return err
		}
		if err := sorter.SortArticles(scored, cfg.InferenceAndSorting.DefaultSortMode, tax); err != nil {
			return err
		}
		report := sorter.BuildReport(scored, tax, cfg.InferenceAndSorting.DefaultSortMode, cfg.InferenceAndSorting.NFCActiveThreshold, evalReport)
		if err := sorter.WriteOutputs(report, cfg.Paths.OutputDir, cfg.InferenceAndSorting, tax); err != nil {
			return err
		}
		log.Status("ANALYZE", "SUCCESS", "wrote %d sorted articles to %s", len(scored), cfg.Paths.OutputDir)
		if subcmd == "analyze" {
			return nil
		}
	}

	if (subcmd == "pipeline" && cfg.Clustering.Enabled) || subcmd == "cluster" {
		log.Status("CLUSTER", "START", "running 3-method clustering on onnx_backbone_relu embeddings")
		cRng := rand.New(rand.NewSource(cfg.Runtime.Seed + 99))
		yIdxs := sampleIndices(len(ynatValRecs), cfg.Clustering.YNATSampleSize, cRng)
		nIdxs := sampleIndices(len(nsmcTestRecs), cfg.Clustering.NSMCSampleSize, cRng)

		mixedRecs := make([]data.Record, 0, len(yIdxs)+len(nIdxs))
		mixedEmb := make([][]float32, 0, len(yIdxs)+len(nIdxs))
		for _, idx := range yIdxs {
			mixedRecs = append(mixedRecs, ynatValRecs[idx])
			mixedEmb = append(mixedEmb, ynatValEmb[idx])
		}
		for _, idx := range nIdxs {
			mixedRecs = append(mixedRecs, nsmcTestRecs[idx])
			mixedEmb = append(mixedEmb, nsmcTestEmb[idx])
		}
		mixedEx, _ := nfcmodel.PrepareEmbedExamples(mixedRecs, mixedEmb, tax, 0, nil)
		pred, err := nfcmodel.NewEmbedPredictor(backend, store, cfg, actualDim, tax.NumTopics(), tax.NumFeelings(), cfg.Training.EvalBatchSize)
		if err != nil {
			return err
		}
		mixedPreds, err := pred.PredictEmbedExamples(mixedEx)
		pred.Close()
		if err != nil {
			return err
		}
		scored, err := sorter.ScoreArticles(mixedRecs, mixedPreds, tax, cfg.InferenceAndSorting)
		if err != nil {
			return err
		}
		report, err := cluster.RunClusteringPipeline(context.Background(), mixedRecs, mixedPreds, scored, tax, cfg, log)
		if err != nil {
			return err
		}
		v := report.Validation
		log.Info("cluster", "HDBSCAN clusters=%d (cross_domain=%d), clustered=%d/%d, noise=%d (%.1f%%)",
			v.NumClusters, v.CrossDomainClusters, v.ClusteredDocs, v.TotalDocs, v.NoiseDocs, v.NoiseRatio*100)
		log.Info("cluster_val", "Silhouette=%.4f (target=%.2f), DunnCentroid=%.4f, DunnStrict=%.4f, MeanNPMI=%.4f, MeanUMass=%.4f",
			v.SilhouetteScore, v.SilhouetteTarget, v.DunnIndexCentroid, v.DunnIndexStrict, v.MeanNPMICoherence, v.MeanUMassCoherence)
		log.Status("CLUSTER", "SUCCESS", "clusters=%d silhouette=%.4f npmi=%.4f dunn_c=%.4f bridges=%d reports_dir=%s",
			v.NumClusters, v.SilhouetteScore, v.MeanNPMICoherence, v.DunnIndexCentroid, v.CrossDomainEntityBridges, cfg.Paths.OutputDir)
	}
	return nil
}
