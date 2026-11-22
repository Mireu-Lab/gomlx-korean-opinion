package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gomlx/compute"
	gomlxmodel "github.com/gomlx/gomlx/ml/model"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/cluster"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/data"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/graphkg"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/logging"
	nfcmodel "github.com/Mireu-Lab/gomlx-korean-opinion/internal/model"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/nfc"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/pretrain"
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
	inputPath := fs.String("input", "", "Optional external file (.jsonl/.csv/.tsv for analyze, or annotated CSV for eval-human)")
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

	if err := runCommand(subcmd, cfg, *inputPath, *limit, *resume, *stepsOverride, *batchOverride, log); err != nil {
		log.Status(strings.ToUpper(subcmd), "FAILED", "%v", err)
		log.Error("cli", "command %q failed: %v", subcmd, err)
		os.Exit(1)
	}
}

func runCommand(
	subcmd string,
	cfg *config.Config,
	inputPath string,
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
	case "cluster":
		return stageCluster(cfg, tax, nil, nil, nil, log)
	case "eval-human":
		return stageEvalHuman(cfg, inputPath, log)
	case "pipeline":
		if err := stageDownload(cfg, log); err != nil {
			return err
		}
		if cfg.Embedding.Provider == "ko_hub_graph" {
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
	} else if cfg.Embedding.Provider == "ko_hub_graph" {
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
	pred, store, err := nfcmodel.LoadPredictor(backend, cfg.Paths.CheckpointDir, graphCfg, cfg.Training.EvalBatchSize, cfg.Tokenizer.MaxSeqLen)
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
			pred, err = nfcmodel.NewPredictor(backend, existingStore, graphCfg, cfg.Training.EvalBatchSize, cfg.Tokenizer.MaxSeqLen)
		} else {
			pred, _, err = nfcmodel.LoadPredictor(backend, cfg.Paths.CheckpointDir, graphCfg, cfg.Training.EvalBatchSize, cfg.Tokenizer.MaxSeqLen)
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
		pred, err = nfcmodel.NewPredictor(backend, existingStore, graphCfg, cfg.Training.EvalBatchSize, cfg.Tokenizer.MaxSeqLen)
	} else {
		pred, _, err = nfcmodel.LoadPredictor(backend, cfg.Paths.CheckpointDir, graphCfg, cfg.Training.EvalBatchSize, cfg.Tokenizer.MaxSeqLen)
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
  download     Download NSMC and KLUE-YNAT datasets configured in config.yaml
  build-graph  Download Unihan/OMW sources and build KO-Hub multilingual roots CSR graph
  pretrain     Run GoMLX Context-Aware Soft Edge contrastive pretraining for KO-Hub Student
  train        Fit Korean tokenizer and train GoMLX ReLU multi-task model
  eval         Evaluate saved checkpoint on KLUE-YNAT val, NSMC test, and NFC
  analyze      Classify and sort news articles by topic, Nuclear Feeling, and intensity
  cluster      Run 3-method clustering (HDBSCAN, Entity Anchors, Hierarchical Topics) & 3-stage validation
  eval-human   Score completed human_eval_sample.csv (Agreement Rate & Cohen's Kappa)
  pipeline     Run end-to-end: download -> (pretrain if ko_hub_graph) -> train -> eval -> analyze -> cluster

Flags:
  --config <path>         Path to config YAML (default: config/config.yaml)
  --backend <name>        Override runtime.backend (go, xla:cpu, etc.)
  --provider <name>       Override embedding.provider (scratch_relu|ko_hub_graph)
  --roots-config <path>   Override embedding.roots_config_file
  --pretrain-dir <path>   Override embedding.pretrain_checkpoint_dir
  --freeze-encoder <bool> Override embedding.freeze_encoder (true|false)
  --pca-dim <int>         Override embedding.pca_dim (0 disables PCA)
  --checkpoint-dir <path> Override paths.checkpoint_dir
  --steps <int>           Override training.train_steps (or pretrain steps)
  --batch-size <int>      Override training.batch_size (or pretrain batch_size)
  --sort-by <mode>        topic_nfc_intensity | nfc_intensity | intensity_desc | sentiment_neg_desc
  --input <path>          Custom file (.jsonl/.csv/.tsv for analyze, or annotated CSV for eval-human)
  --output-dir <path>     Override paths.output_dir
  --limit <int>           Max articles to analyze (0 = all)
  --resume                Resume training from existing checkpoint
`)
}
