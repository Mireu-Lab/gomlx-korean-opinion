package model

import (
	"fmt"

	"github.com/gomlx/compute"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"

	"github.com/Mireu-Lab/gomlx-nfc-opinion/internal/data"
	"github.com/Mireu-Lab/gomlx-nfc-opinion/internal/sorter"
)

// Predictor runs batched static-shape inference on a trained GoMLX ReLU multi-task model.
type Predictor struct {
	exec        *model.Exec
	store       *model.Store
	batchSize   int
	seqLen      int
	numTopics   int
	numNFC      int
	semanticDim int
}

// NewPredictor compiles a deterministic inference executor (training=false, dropout disabled).
func NewPredictor(backend compute.Backend, store *model.Store, graphCfg GraphConfig, batchSize, seqLen int) (*Predictor, error) {
	if batchSize < 1 || seqLen < 1 {
		return nil, fmt.Errorf("invalid predictor dimensions batchSize=%d seqLen=%d", batchSize, seqLen)
	}
	if len(graphCfg.HiddenDims) == 0 {
		return nil, fmt.Errorf("graphCfg.HiddenDims must not be empty")
	}
	semanticDim := graphCfg.HiddenDims[len(graphCfg.HiddenDims)-1]
	exec, err := model.NewExec(backend, store, func(scope *model.Scope, tokens *Node) []*Node {
		g := tokens.Graph()
		scope.Store().SetTraining(g, false)
		topicLogits, sentLogits, nfcLogits, semanticEmbed := BuildGraphWithEmbedding(scope, tokens, graphCfg, false)
		return []*Node{Softmax(topicLogits, -1), Sigmoid(sentLogits), Sigmoid(nfcLogits), semanticEmbed}
	})
	if err != nil {
		return nil, fmt.Errorf("create predictor exec: %w", err)
	}
	return &Predictor{
		exec:        exec,
		store:       store,
		batchSize:   batchSize,
		seqLen:      seqLen,
		numTopics:   graphCfg.NumTopics,
		numNFC:      graphCfg.NumNFC,
		semanticDim: semanticDim,
	}, nil
}

// LoadPredictor loads model weights from checkpointDir into a fresh Store and creates a Predictor.
func LoadPredictor(backend compute.Backend, checkpointDir string, graphCfg GraphConfig, batchSize, seqLen int) (*Predictor, *model.Store, error) {
	store := model.NewStore()
	handler, err := checkpoint.Build(store).Dir(checkpointDir).Immediate().Done()
	if err != nil {
		return nil, nil, fmt.Errorf("load checkpoint from %q: %w", checkpointDir, err)
	}
	has, err := handler.HasCheckpoints()
	if err != nil {
		return nil, nil, fmt.Errorf("check checkpoints in %q: %w", checkpointDir, err)
	}
	if !has {
		return nil, nil, fmt.Errorf("no checkpoints found in %q", checkpointDir)
	}
	pred, err := NewPredictor(backend, store, graphCfg, batchSize, seqLen)
	if err != nil {
		return nil, nil, err
	}
	return pred, store, nil
}

// Close frees the underlying compiled GoMLX executor resources.
func (p *Predictor) Close() {
	if p != nil && p.exec != nil {
		p.exec.Finalize()
	}
}

// PredictExamples evaluates a slice of EncodedExample using static batch padding to avoid graph recompilation.
func (p *Predictor) PredictExamples(examples []data.EncodedExample) ([]sorter.ArticlePrediction, error) {
	preds := make([]sorter.ArticlePrediction, len(examples))
	if len(examples) == 0 {
		return preds, nil
	}

	tokensBuf := make([]int32, p.batchSize*p.seqLen)
	for start := 0; start < len(examples); start += p.batchSize {
		end := start + p.batchSize
		if end > len(examples) {
			end = len(examples)
		}
		count := end - start

		clear(tokensBuf)
		for i := 0; i < count; i++ {
			copy(tokensBuf[i*p.seqLen:(i+1)*p.seqLen], examples[start+i].Tokens)
		}

		inTensor := tensors.FromFlatDataAndDimensions(tokensBuf, p.batchSize, p.seqLen)
		outputs, err := p.exec.Call(inTensor)
		_ = inTensor.FinalizeAll()
		if err != nil {
			return nil, fmt.Errorf("inference batch [%d:%d]: %w", start, end, err)
		}
		if len(outputs) != 4 {
			for _, out := range outputs {
				_ = out.FinalizeAll()
			}
			return nil, fmt.Errorf("expected 4 output tensors, got %d", len(outputs))
		}

		topicFlat := tensors.MustCopyFlatData[float32](outputs[0])
		sentFlat := tensors.MustCopyFlatData[float32](outputs[1])
		nfcFlat := tensors.MustCopyFlatData[float32](outputs[2])
		semanticFlat := tensors.MustCopyFlatData[float32](outputs[3])
		for _, out := range outputs {
			_ = out.FinalizeAll()
		}

		for i := 0; i < count; i++ {
			tp := make([]float32, p.numTopics)
			copy(tp, topicFlat[i*p.numTopics:(i+1)*p.numTopics])
			np := make([]float32, p.numNFC)
			copy(np, nfcFlat[i*p.numNFC:(i+1)*p.numNFC])
			se := make([]float32, p.semanticDim)
			copy(se, semanticFlat[i*p.semanticDim:(i+1)*p.semanticDim])
			preds[start+i] = sorter.ArticlePrediction{
				TopicProbs:    tp,
				SentimentProb: sentFlat[i],
				NFCProbs:      np,
				SemanticEmbed: se,
			}
		}
	}
	return preds, nil
}
