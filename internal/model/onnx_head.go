package model

import (
	"fmt"
	"iter"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/ml/train/optimizer/cosineschedule"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/data"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/logging"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/nfc"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/pretrain"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/sorter"
)

// EmbedExample pairs a Record and weak NFC labels with a precomputed dense backbone embedding vector.
type EmbedExample struct {
	Record     data.Record
	Embedding  []float32
	NFCTargets []float32
	NFCScores  []float32
	NFCMask    float32
	NFCMatched int
}

// PrepareEmbedExamples attaches weak Nuclear Feeling Checklist annotations to records paired with ONNX embeddings.
func PrepareEmbedExamples(
	records []data.Record,
	embeds [][]float32,
	tax *nfc.Taxonomy,
	negCalibProb float64,
	rng *rand.Rand,
) ([]EmbedExample, data.PrepStats) {
	numNFC := tax.NumFeelings()
	stats := data.PrepStats{
		Total:          len(records),
		PerFeelingHits: make([]int, numNFC),
	}
	out := make([]EmbedExample, len(records))
	for i, rec := range records {
		targets, scores, matched, guardTriggered := tax.WeakAnnotateDetailed(rec.Text)
		var nfcMask float32
		if matched > 0 {
			nfcMask = 1.0
			stats.NFCActiveExamples++
			stats.NFCSupervised++
			for c, v := range targets {
				if v > 0 {
					stats.PerFeelingHits[c]++
				}
			}
		} else if guardTriggered {
			nfcMask = 1.0
			stats.NFCSupervised++
		} else if negCalibProb > 0 && rng != nil {
			if rec.HasSentiment && rec.SentimentLabel >= 0.5 {
				if rng.Float64() < negCalibProb {
					nfcMask = 1.0
					stats.NFCSupervised++
				}
			} else if !rec.HasSentiment && rng.Float64() < negCalibProb*0.15 {
				nfcMask = 1.0
				stats.NFCSupervised++
			}
		}
		out[i] = EmbedExample{
			Record:     rec,
			Embedding:  embeds[i],
			NFCTargets: targets,
			NFCScores:  scores,
			NFCMask:    nfcMask,
			NFCMatched: matched,
		}
	}
	return out, stats
}

// MultiTaskEmbedDataset implements train.Dataset over dense ONNX backbone embeddings.
type MultiTaskEmbedDataset struct {
	name        string
	batchSize   int
	embedDim    int
	numNFC      int
	ynatRatio   float64
	nfcPosRatio float64
	rng         *rand.Rand
	ynatAll     []EmbedExample
	nsmcAll     []EmbedExample
	ynatNFCPos  []EmbedExample
	nsmcNFCPos  []EmbedExample
}

// NewMultiTaskEmbedDataset wraps YNAT and NSMC embedding examples for balanced multi-task ReLU head training.
func NewMultiTaskEmbedDataset(
	name string,
	ynatExamples, nsmcExamples []EmbedExample,
	cfg config.TrainingConfig,
	embedDim, numNFC int,
	seed int64,
) (*MultiTaskEmbedDataset, error) {
	if len(ynatExamples) == 0 || len(nsmcExamples) == 0 {
		return nil, fmt.Errorf("ynatExamples and nsmcExamples must not be empty")
	}
	var ynatPos, nsmcPos []EmbedExample
	for _, ex := range ynatExamples {
		if ex.NFCMatched > 0 {
			ynatPos = append(ynatPos, ex)
		}
	}
	for _, ex := range nsmcExamples {
		if ex.NFCMatched > 0 {
			nsmcPos = append(nsmcPos, ex)
		}
	}
	return &MultiTaskEmbedDataset{
		name:        name,
		batchSize:   cfg.BatchSize,
		embedDim:    embedDim,
		numNFC:      numNFC,
		ynatRatio:   cfg.YNATBatchRatio,
		nfcPosRatio: cfg.NFCPositiveSampleRatio,
		rng:         rand.New(rand.NewSource(seed)),
		ynatAll:     ynatExamples,
		nsmcAll:     nsmcExamples,
		ynatNFCPos:  ynatPos,
		nsmcNFCPos:  nsmcPos,
	}, nil
}

func (d *MultiTaskEmbedDataset) Name() string { return d.name }

func (d *MultiTaskEmbedDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		nYNAT := int(math.Round(float64(d.batchSize) * d.ynatRatio))
		if nYNAT < 1 {
			nYNAT = 1
		}
		if nYNAT >= d.batchSize && d.batchSize > 1 {
			nYNAT = d.batchSize - 1
		}
		nNSMC := d.batchSize - nYNAT
		nYNATPos := int(math.Round(float64(nYNAT) * d.nfcPosRatio))
		if len(d.ynatNFCPos) == 0 {
			nYNATPos = 0
		}
		nNSMCPos := int(math.Round(float64(nNSMC) * d.nfcPosRatio))
		if len(d.nsmcNFCPos) == 0 {
			nNSMCPos = 0
		}
		picked := make([]EmbedExample, d.batchSize)
		for {
			idx := 0
			for i := 0; i < nYNATPos; i++ {
				picked[idx] = d.ynatNFCPos[d.rng.Intn(len(d.ynatNFCPos))]
				idx++
			}
			for i := nYNATPos; i < nYNAT; i++ {
				picked[idx] = d.ynatAll[d.rng.Intn(len(d.ynatAll))]
				idx++
			}
			for i := 0; i < nNSMCPos; i++ {
				picked[idx] = d.nsmcNFCPos[d.rng.Intn(len(d.nsmcNFCPos))]
				idx++
			}
			for i := nNSMCPos; i < nNSMC; i++ {
				picked[idx] = d.nsmcAll[d.rng.Intn(len(d.nsmcAll))]
				idx++
			}
			d.rng.Shuffle(len(picked), func(i, j int) {
				picked[i], picked[j] = picked[j], picked[i]
			})

			bs := len(picked)
			embFlat := make([]float32, bs*d.embedDim)
			topicTarget := make([]int32, bs)
			topicMask := make([]float32, bs)
			sentTarget := make([]float32, bs)
			sentMask := make([]float32, bs)
			nfcTarget := make([]float32, bs*d.numNFC)
			nfcMask := make([]float32, bs)
			for i, ex := range picked {
				copy(embFlat[i*d.embedDim:(i+1)*d.embedDim], ex.Embedding)
				if ex.Record.HasTopic {
					topicTarget[i] = int32(ex.Record.TopicID)
					topicMask[i] = 1.0
				}
				if ex.Record.HasSentiment {
					sentTarget[i] = ex.Record.SentimentLabel
					sentMask[i] = 1.0
				}
				copy(nfcTarget[i*d.numNFC:(i+1)*d.numNFC], ex.NFCTargets)
				nfcMask[i] = ex.NFCMask
			}
			batch := train.Batch{
				Inputs: []*tensors.Tensor{
					tensors.FromFlatDataAndDimensions(embFlat, bs, d.embedDim),
				},
				Labels: []*tensors.Tensor{
					tensors.FromFlatDataAndDimensions(topicTarget, bs, 1),
					tensors.FromFlatDataAndDimensions(topicMask, bs),
					tensors.FromFlatDataAndDimensions(sentTarget, bs, 1),
					tensors.FromFlatDataAndDimensions(sentMask, bs),
					tensors.FromFlatDataAndDimensions(nfcTarget, bs, d.numNFC),
					tensors.FromFlatDataAndDimensions(nfcMask, bs),
				},
				Spec: d.name,
			}
			if !yield(batch, nil) {
				return
			}
		}
	}
}

// BuildEmbedHeadGraph builds GoMLX ReLU multi-task heads over precomputed ONNX backbone embeddings.
func BuildEmbedHeadGraph(
	scope *model.Scope,
	embedIn *Node,
	cfg *config.Config,
	numTopics, numNFC int,
	useCosine bool,
) (topicLogits, sentLogits, nfcLogits, semanticEmbed *Node) {
	g := embedIn.Graph()
	if useCosine && scope.Store().IsTraining(g) {
		cosineschedule.New(scope, g, dtypes.Float32).FromScope().Done()
	}

	scope = scope.In("onnx_relu_heads")
	outDim := cfg.Embedding.StudentOutDim
	h := layers.NewDense(scope.In("proj_0"), embedIn, 256).
		WithBias(true).
		WithActivation(activation.TypeRelu).
		Done()
	h = layers.DropoutStatic(scope.In("drop_0"), h, cfg.Model.DropoutRate)
	h = layers.NewDense(scope.In("proj_1"), h, outDim).
		WithBias(true).
		WithActivation(activation.TypeRelu).
		Done()
	h = layers.DropoutStatic(scope.In("drop_1"), h, cfg.Model.DropoutRate)
	semanticEmbed = pretrain.SafeL2Normalize(h)

	headDim := cfg.Model.HeadDim
	topicH := layers.NewDense(scope.In("topic_hidden"), h, headDim).
		WithBias(true).
		WithActivation(activation.TypeRelu).
		Done()
	topicLogits = layers.NewDense(scope.In("topic_out"), topicH, numTopics).
		WithBias(true).
		Done()

	sentH := layers.NewDense(scope.In("sentiment_hidden"), h, headDim).
		WithBias(true).
		WithActivation(activation.TypeRelu).
		Done()
	sentLogits = layers.NewDense(scope.In("sentiment_out"), sentH, 1).
		WithBias(true).
		Done()

	nfcIn := Concatenate([]*Node{h, topicH, sentH}, -1)
	nfcH := layers.NewDense(scope.In("nfc_hidden"), nfcIn, headDim).
		WithBias(true).
		WithActivation(activation.TypeRelu).
		Done()
	nfcH = layers.DropoutStatic(scope.In("nfc_drop"), nfcH, cfg.Model.DropoutRate)
	nfcLogits = layers.NewDense(scope.In("nfc_out"), nfcH, numNFC).
		WithBias(true).
		Done()

	return topicLogits, sentLogits, nfcLogits, semanticEmbed
}

// TrainOnEmbeddings trains GoMLX ReLU multi-task heads on precomputed ONNX embeddings and saves a checkpoint.
func TrainOnEmbeddings(
	backend compute.Backend,
	cfg *config.Config,
	tax *nfc.Taxonomy,
	embedDim int,
	ynatTrain, nsmcTrain []EmbedExample,
	resume bool,
	log *logging.DualLogger,
) (*model.Store, error) {
	if !resume {
		if err := os.RemoveAll(cfg.Paths.CheckpointDir); err != nil {
			return nil, fmt.Errorf("clean checkpoint dir %q: %w", cfg.Paths.CheckpointDir, err)
		}
	}
	if err := os.MkdirAll(cfg.Paths.CheckpointDir, 0755); err != nil {
		return nil, fmt.Errorf("create checkpoint dir %q: %w", cfg.Paths.CheckpointDir, err)
	}

	store := model.NewStore()
	store.SetParam(optimizer.ParamOptimizer, cfg.Training.Optimizer)
	store.SetParam(optimizer.ParamLearningRate, cfg.Training.LearningRate)
	store.SetParam(optimizer.ParamAdamWeightDecay, cfg.Training.WeightDecay)
	store.SetParam(cosineschedule.ParamPeriodSteps, cfg.Training.TrainSteps)
	store.SetParam(cosineschedule.ParamMinLearningRate, cfg.Training.MinLearningRate)
	if err := store.SetRNGStateFromSeed(cfg.Runtime.Seed); err != nil {
		return nil, err
	}

	ckptHandler, err := checkpoint.Build(store).
		Dir(cfg.Paths.CheckpointDir).
		Keep(cfg.Training.CheckpointKeep).
		Done()
	if err != nil {
		return nil, err
	}

	trainDS, err := NewMultiTaskEmbedDataset(
		"onnx_backbone_relu_train",
		ynatTrain,
		nsmcTrain,
		cfg.Training,
		embedDim,
		tax.NumFeelings(),
		cfg.Runtime.Seed,
	)
	if err != nil {
		return nil, err
	}

	numTopics := tax.NumTopics()
	numNFC := tax.NumFeelings()
	modelFn := func(scope *model.Scope, _ any, inputs []*Node) []*Node {
		tL, sL, nL, se := BuildEmbedHeadGraph(scope, inputs[0], cfg, numTopics, numNFC, true)
		return []*Node{tL, sL, nL, se}
	}
	lossFn := BuildMultiTaskLoss(cfg.Training.LossWeights, numTopics)
	opt := optimizer.FromStore(store)
	trainer := train.NewTrainer(backend, store, modelFn, lossFn, opt, nil, nil)
	loop := train.NewLoop(trainer)

	startTime := time.Now()
	train.EveryNSteps(loop, cfg.Training.LogEverySteps, "log-metrics", 0, func(l *train.Loop, metrics []*tensors.Tensor) error {
		log.Info("trainer.onnx", "step=%d/%d loss=%.4f moving_avg_loss=%.4f median_step=%s elapsed=%s",
			l.LoopStep, l.EndStep, tensorScalarFloat32(metrics[0]), tensorScalarFloat32(metrics[1]),
			l.MedianTrainStepDuration().Round(time.Millisecond), time.Since(startTime).Round(time.Second))
		return nil
	})

	metrics, err := loop.RunSteps(trainDS, cfg.Training.TrainSteps)
	if err != nil {
		return nil, fmt.Errorf("onnx head training failed: %w", err)
	}
	finalLoss := tensorScalarFloat32(metrics[0])
	if math.IsNaN(float64(finalLoss)) || math.IsInf(float64(finalLoss), 0) {
		return nil, fmt.Errorf("onnx head training produced non-finite loss %v", finalLoss)
	}
	if err := ckptHandler.Save(); err != nil {
		return nil, err
	}
	log.Info("trainer.onnx", "training complete: steps=%d final_loss=%.4f total_time=%s",
		loop.LoopStep, finalLoss, time.Since(startTime).Round(time.Second))
	return store, nil
}

// EmbedPredictor runs batched static inference on GoMLX ReLU heads over ONNX backbone embeddings.
type EmbedPredictor struct {
	exec        *model.Exec
	store       *model.Store
	batchSize   int
	embedDim    int
	numTopics   int
	numNFC      int
	semanticDim int
}

// NewEmbedPredictor compiles deterministic inference for onnx_backbone_relu heads.
func NewEmbedPredictor(backend compute.Backend, store *model.Store, cfg *config.Config, embedDim, numTopics, numNFC, batchSize int) (*EmbedPredictor, error) {
	exec, err := model.NewExec(backend, store, func(scope *model.Scope, embedIn *Node) []*Node {
		g := embedIn.Graph()
		scope.Store().SetTraining(g, false)
		tL, sL, nL, se := BuildEmbedHeadGraph(scope, embedIn, cfg, numTopics, numNFC, false)
		return []*Node{Softmax(tL, -1), Sigmoid(sL), Sigmoid(nL), se}
	})
	if err != nil {
		return nil, err
	}
	return &EmbedPredictor{
		exec:        exec,
		store:       store,
		batchSize:   batchSize,
		embedDim:    embedDim,
		numTopics:   numTopics,
		numNFC:      numNFC,
		semanticDim: cfg.Embedding.StudentOutDim,
	}, nil
}

// LoadEmbedPredictor loads onnx_backbone_relu head weights from checkpointDir.
func LoadEmbedPredictor(backend compute.Backend, checkpointDir string, cfg *config.Config, embedDim, numTopics, numNFC, batchSize int) (*EmbedPredictor, *model.Store, error) {
	store := model.NewStore()
	handler, err := checkpoint.Build(store).Dir(checkpointDir).Immediate().Done()
	if err != nil {
		return nil, nil, err
	}
	has, err := handler.HasCheckpoints()
	if err != nil || !has {
		return nil, nil, fmt.Errorf("no checkpoints in %q (err=%v)", checkpointDir, err)
	}
	pred, err := NewEmbedPredictor(backend, store, cfg, embedDim, numTopics, numNFC, batchSize)
	if err != nil {
		return nil, nil, err
	}
	return pred, store, nil
}

// Close releases executor resources.
func (p *EmbedPredictor) Close() {
	if p != nil && p.exec != nil {
		p.exec.Finalize()
	}
}

// PredictEmbedExamples runs inference over EmbedExample slices using static batch padding.
func (p *EmbedPredictor) PredictEmbedExamples(examples []EmbedExample) ([]sorter.ArticlePrediction, error) {
	preds := make([]sorter.ArticlePrediction, len(examples))
	if len(examples) == 0 {
		return preds, nil
	}
	buf := make([]float32, p.batchSize*p.embedDim)
	for start := 0; start < len(examples); start += p.batchSize {
		end := start + p.batchSize
		if end > len(examples) {
			end = len(examples)
		}
		cnt := end - start
		clear(buf)
		for i := 0; i < cnt; i++ {
			copy(buf[i*p.embedDim:(i+1)*p.embedDim], examples[start+i].Embedding)
		}
		inT := tensors.FromFlatDataAndDimensions(buf, p.batchSize, p.embedDim)
		outs, err := p.exec.Call(inT)
		_ = inT.FinalizeAll()
		if err != nil {
			return nil, err
		}
		tFlat := tensors.MustCopyFlatData[float32](outs[0])
		sFlat := tensors.MustCopyFlatData[float32](outs[1])
		nFlat := tensors.MustCopyFlatData[float32](outs[2])
		seFlat := tensors.MustCopyFlatData[float32](outs[3])
		for _, o := range outs {
			_ = o.FinalizeAll()
		}
		for i := 0; i < cnt; i++ {
			tp := make([]float32, p.numTopics)
			copy(tp, tFlat[i*p.numTopics:(i+1)*p.numTopics])
			np := make([]float32, p.numNFC)
			copy(np, nFlat[i*p.numNFC:(i+1)*p.numNFC])
			se := make([]float32, p.semanticDim)
			copy(se, seFlat[i*p.semanticDim:(i+1)*p.semanticDim])
			preds[start+i] = sorter.ArticlePrediction{
				TopicProbs:    tp,
				SentimentProb: sFlat[i],
				NFCProbs:      np,
				SemanticEmbed: se,
			}
		}
	}
	return preds, nil
}

// EvaluateOnEmbeddings computes EvalReport and YNAT predictions for onnx_backbone_relu.
func EvaluateOnEmbeddings(
	backend compute.Backend,
	store *model.Store,
	cfg *config.Config,
	tax *nfc.Taxonomy,
	embedDim int,
	ynatVal, nsmcTest []EmbedExample,
) (*EvalReport, []sorter.ArticlePrediction, error) {
	pred, err := NewEmbedPredictor(backend, store, cfg, embedDim, tax.NumTopics(), tax.NumFeelings(), cfg.Training.EvalBatchSize)
	if err != nil {
		return nil, nil, err
	}
	defer pred.Close()

	ynatPreds, err := pred.PredictEmbedExamples(ynatVal)
	if err != nil {
		return nil, nil, err
	}
	nsmcPreds, err := pred.PredictEmbedExamples(nsmcTest)
	if err != nil {
		return nil, nil, err
	}

	ynatEnc := make([]data.EncodedExample, len(ynatVal))
	for i, ex := range ynatVal {
		ynatEnc[i] = data.EncodedExample{
			Record: ex.Record, NFCTargets: ex.NFCTargets, NFCScores: ex.NFCScores, NFCMask: ex.NFCMask, NFCMatched: ex.NFCMatched,
		}
	}
	nsmcEnc := make([]data.EncodedExample, len(nsmcTest))
	for i, ex := range nsmcTest {
		nsmcEnc[i] = data.EncodedExample{
			Record: ex.Record, NFCTargets: ex.NFCTargets, NFCScores: ex.NFCScores, NFCMask: ex.NFCMask, NFCMatched: ex.NFCMatched,
		}
	}
	return computeEvalMetrics(cfg, tax, ynatEnc, ynatPreds, nsmcEnc, nsmcPreds), ynatPreds, nil
}
