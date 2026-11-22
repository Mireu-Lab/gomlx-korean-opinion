package pretrain

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/ml/train/optimizer/cosineschedule"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/graphkg"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/logging"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/tokenizer"
)

// PretrainMeta stores checkpoint metadata used to verify alignment when loading into downstream models.
type PretrainMeta struct {
	VocabSize   int     `json:"vocab_size"`
	EmbedDim    int     `json:"embed_dim"`
	OutDim      int     `json:"out_dim"`
	DropoutRate float64 `json:"dropout_rate"`
	Steps       int     `json:"steps"`
	FirstLoss   float32 `json:"first_loss"`
	FinalLoss   float32 `json:"final_loss"`
}

// PretrainReport summarizes completed KO-Hub contrastive pretraining.
type PretrainReport struct {
	Meta         PretrainMeta
	DatasetStats PretrainDatasetStats
}

// TrainPretrain runs GoMLX contrastive pretraining over the KO-Hub graph and saves the Student checkpoint + vocab.
func TrainPretrain(
	backend compute.Backend,
	cfg *graphkg.PretrainConfig,
	tok *tokenizer.Tokenizer,
	g *graphkg.Graph,
	texts []string,
	seed int64,
	vocabFilename string,
	log *logging.DualLogger,
) (*model.Store, *PretrainReport, error) {
	if err := os.RemoveAll(cfg.CheckpointDir); err != nil {
		return nil, nil, fmt.Errorf("clean pretrain checkpoint dir %q: %w", cfg.CheckpointDir, err)
	}
	if err := os.MkdirAll(cfg.CheckpointDir, 0755); err != nil {
		return nil, nil, fmt.Errorf("create pretrain checkpoint dir %q: %w", cfg.CheckpointDir, err)
	}

	rv := BuildRootVocab(g)
	examples, dsStats := BuildPretrainExamples(texts, tok, g, rv, cfg.MaxRootsPerView)
	log.Info("pretrain", "built examples=%d (corpus=%d disambig=%d) root_vocab=%d en=%d hanja=%d jazh=%d",
		dsStats.TotalExamples, dsStats.CorpusExamples, dsStats.DisambigExamples,
		dsStats.RootVocabSize, dsStats.WithENRoots, dsStats.WithHanjaRoots, dsStats.WithJaZhRoots)

	ds, err := NewPretrainDataset("ko_hub_pretrain", examples, cfg.BatchSize, tok.Cfg.MaxSeqLen, cfg.MaxRootsPerView, seed)
	if err != nil {
		return nil, nil, err
	}

	store := model.NewStore()
	store.SetParam(optimizer.ParamOptimizer, cfg.Optimizer)
	store.SetParam(optimizer.ParamLearningRate, cfg.LearningRate)
	store.SetParam(optimizer.ParamAdamWeightDecay, cfg.WeightDecay)
	store.SetParam(cosineschedule.ParamPeriodSteps, cfg.PretrainSteps)
	store.SetParam(cosineschedule.ParamMinLearningRate, cfg.MinLearningRate)
	if err := store.SetRNGStateFromSeed(seed); err != nil {
		return nil, nil, fmt.Errorf("set RNG seed: %w", err)
	}

	ckptHandler, err := checkpoint.Build(store).
		Dir(cfg.CheckpointDir).
		Keep(cfg.CheckpointKeep).
		Done()
	if err != nil {
		return nil, nil, fmt.Errorf("init pretrain checkpoint handler: %w", err)
	}

	graphCfg := PretrainGraphConfig{
		VocabSize:       tok.VocabLen(),
		NumRootVocab:    rv.Size,
		EmbedDim:        cfg.EmbedDim,
		OutDim:          cfg.OutDim,
		RootEmbedDim:    cfg.RootEmbedDim,
		AttnDim:         cfg.AttnDim,
		MaxSeqLen:       tok.Cfg.MaxSeqLen,
		MaxRootsPerView: cfg.MaxRootsPerView,
		DropoutRate:     cfg.DropoutRate,
	}
	modelFn := BuildPretrainModelFn(graphCfg, true)
	lossFn := BuildPretrainLoss(*cfg)
	opt := optimizer.FromStore(store)
	trainer := train.NewTrainer(backend, store, modelFn, lossFn, opt, nil, nil)
	loop := train.NewLoop(trainer)

	var firstLoss float32
	hasFirst := false
	start := time.Now()

	loop.OnStep("track-first-loss", 0, func(l *train.Loop, metrics []*tensors.Tensor) error {
		v := tensorScalarFloat32(metrics[0])
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("non-finite pretrain loss at step %d: %v", l.LoopStep, v)
		}
		if !hasFirst {
			firstLoss = v
			hasFirst = true
		}
		return nil
	})
	train.EveryNSteps(loop, cfg.LogEverySteps, "log-pretrain", 0, func(l *train.Loop, metrics []*tensors.Tensor) error {
		batchLoss := tensorScalarFloat32(metrics[0])
		maLoss := tensorScalarFloat32(metrics[1])
		log.Info("pretrain", "step=%d/%d loss=%.4f ma_loss=%.4f median_step=%s elapsed=%s",
			l.LoopStep, l.EndStep, batchLoss, maLoss, l.MedianTrainStepDuration().Round(time.Millisecond), time.Since(start).Round(time.Second))
		return nil
	})

	metrics, err := loop.RunSteps(ds, cfg.PretrainSteps)
	if err != nil {
		return nil, nil, fmt.Errorf("pretraining loop failed: %w", err)
	}
	finalLoss := tensorScalarFloat32(metrics[0])
	if math.IsNaN(float64(finalLoss)) || math.IsInf(float64(finalLoss), 0) {
		return nil, nil, fmt.Errorf("pretraining produced non-finite final loss %v", finalLoss)
	}

	if err := ckptHandler.Save(); err != nil {
		return nil, nil, fmt.Errorf("save pretrain checkpoint: %w", err)
	}
	if vocabFilename == "" {
		vocabFilename = "vocab.json"
	}
	if err := tok.Save(filepath.Join(cfg.CheckpointDir, vocabFilename)); err != nil {
		return nil, nil, err
	}

	meta := PretrainMeta{
		VocabSize:   tok.VocabLen(),
		EmbedDim:    cfg.EmbedDim,
		OutDim:      cfg.OutDim,
		DropoutRate: cfg.DropoutRate,
		Steps:       cfg.PretrainSteps,
		FirstLoss:   firstLoss,
		FinalLoss:   finalLoss,
	}
	metaBytes, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return nil, nil, err
	}
	if err := os.WriteFile(filepath.Join(cfg.CheckpointDir, "pretrain_meta.json"), metaBytes, 0644); err != nil {
		return nil, nil, fmt.Errorf("write pretrain_meta.json: %w", err)
	}

	log.Status("PRETRAIN", "SUCCESS", "steps=%d first_loss=%.4f final_loss=%.4f dir=%s",
		cfg.PretrainSteps, firstLoss, finalLoss, cfg.CheckpointDir)
	return store, &PretrainReport{Meta: meta, DatasetStats: dsStats}, nil
}

// LoadPretrainMeta reads pretrain_meta.json from a saved KO-Hub pretraining checkpoint directory.
func LoadPretrainMeta(pretrainCkptDir string) (*PretrainMeta, error) {
	raw, err := os.ReadFile(filepath.Join(pretrainCkptDir, "pretrain_meta.json"))
	if err != nil {
		return nil, fmt.Errorf("read pretrain_meta.json in %q: %w", pretrainCkptDir, err)
	}
	var meta PretrainMeta
	if err := json.Unmarshal(raw, &meta); err != nil {
		return nil, fmt.Errorf("parse pretrain_meta.json: %w", err)
	}
	return &meta, nil
}

// LoadPretrainedStudentWeights loads ONLY `/ko_hub_student/...` tensor variables from a pretraining
// checkpoint into targetStore (excluding teacher/optimizer state) and sets Trainable=!freeze.
func LoadPretrainedStudentWeights(pretrainCkptDir string, targetStore *model.Store, freeze bool) (int, error) {
	srcStore := model.NewStore()
	defer srcStore.Finalize()
	handler, err := checkpoint.Build(srcStore).
		Dir(pretrainCkptDir).
		Immediate().
		ExcludeAllParams().
		Done()
	if err != nil {
		return 0, fmt.Errorf("load pretrain checkpoint from %q: %w", pretrainCkptDir, err)
	}
	has, err := handler.HasCheckpoints()
	if err != nil {
		return 0, fmt.Errorf("check pretrain checkpoints in %q: %w", pretrainCkptDir, err)
	}
	if !has {
		return 0, fmt.Errorf("no checkpoints found in %q", pretrainCkptDir)
	}

	prefix := "/" + StudentScopeName + "/"
	transferred := 0
	for v := range srcStore.IterVariables() {
		if !strings.HasPrefix(v.Path(), prefix) {
			continue
		}
		cloned, err := v.CloneToStore(targetStore)
		if err != nil {
			return transferred, fmt.Errorf("clone transferred variable %q: %w", v.Path(), err)
		}
		cloned.SetTrainable(!freeze)
		transferred++
	}
	if transferred == 0 {
		return 0, fmt.Errorf("no %q variables found in checkpoint %q", prefix, pretrainCkptDir)
	}
	return transferred, nil
}

// EncodeStudentTexts runs deterministic inference using ONLY the Korean-only Student encoder.
func EncodeStudentTexts(
	backend compute.Backend,
	store *model.Store,
	tok *tokenizer.Tokenizer,
	texts []string,
	embedDim, outDim int,
) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	batchSize := len(texts)
	seqLen := tok.Cfg.MaxSeqLen

	exec, err := model.NewExec(backend, store, func(scope *model.Scope, tokens *Node) *Node {
		g := tokens.Graph()
		scope.Store().SetTraining(g, false)
		return BuildStudentEncoder(scope, tokens, tok.VocabLen(), embedDim, outDim, 0.0, true)
	})
	if err != nil {
		return nil, fmt.Errorf("create student encoder exec: %w", err)
	}
	defer exec.Finalize()

	flat := make([]int32, batchSize*seqLen)
	for i, txt := range texts {
		copy(flat[i*seqLen:(i+1)*seqLen], tok.Encode(txt))
	}
	inTensor := tensors.FromFlatDataAndDimensions(flat, batchSize, seqLen)
	outs, err := exec.Call(inTensor)
	_ = inTensor.FinalizeAll()
	if err != nil {
		return nil, fmt.Errorf("encode student texts: %w", err)
	}
	if len(outs) != 1 {
		for _, o := range outs {
			_ = o.FinalizeAll()
		}
		return nil, fmt.Errorf("expected 1 output tensor, got %d", len(outs))
	}
	outFlat := tensors.MustCopyFlatData[float32](outs[0])
	_ = outs[0].FinalizeAll()

	res := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		vec := make([]float32, outDim)
		copy(vec, outFlat[i*outDim:(i+1)*outDim])
		res[i] = vec
	}
	return res, nil
}

func tensorScalarFloat32(t *tensors.Tensor) float32 {
	if t == nil {
		return 0
	}
	vals := tensors.MustCopyFlatData[float32](t)
	if len(vals) == 0 {
		return 0
	}
	return vals[0]
}
