package model

import (
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/optimizer/cosineschedule"
)

// GraphConfig parameterizes the GoMLX ReLU multi-task NLP graph.
type GraphConfig struct {
	VocabSize   int
	EmbedDim    int
	HiddenDims  []int
	HeadDim     int
	DropoutRate float64
	NumTopics   int
	NumNFC      int
}

// BuildLogits constructs the GoMLX computation graph and returns logits for the 3 multi-task heads.
func BuildLogits(scope *model.Scope, tokens *Node, cfg GraphConfig, useCosine bool) (topicLogits, sentLogits, nfcLogits *Node) {
	topicLogits, sentLogits, nfcLogits, _ = BuildGraphWithEmbedding(scope, tokens, cfg, useCosine)
	return topicLogits, sentLogits, nfcLogits
}

// BuildGraphWithEmbedding constructs the GoMLX computation graph:
// tokens [B, T] -> Embedding -> Masked (MeanPool || MaxPool) -> Shared ReLU MLP ->
// 3 task heads (Topic [B, 7], Sentiment [B, 1], NFC [B, 16]) plus L2-normalized shared semantic embedding [B, D].
func BuildGraphWithEmbedding(scope *model.Scope, tokens *Node, cfg GraphConfig, useCosine bool) (topicLogits, sentLogits, nfcLogits, semanticEmbed *Node) {
	g := tokens.Graph()
	if useCosine && scope.Store().IsTraining(g) {
		cosineschedule.New(scope, g, dtypes.Float32).FromScope().Done()
	}

	// Pin a deterministic scope name so Trainer and inference Exec share exact variable paths.
	scope = scope.In("relu_nlp")

	padMask := NotEqual(tokens, ZerosLike(tokens))
	padMaskFloat := ConvertDType(padMask, dtypes.Float32)
	lengths := ReduceSum(padMaskFloat, 1)
	safeLengths := Max(lengths, OnesLike(lengths))
	safeLengthsCol := InsertAxes(safeLengths, -1)
	hasTokens := GreaterThan(lengths, ZerosLike(lengths))
	hasTokensCol := InsertAxes(hasTokens, -1)

	maxID := Scalar(g, dtypes.Int32, float64(cfg.VocabSize-1))
	maxBroadcast := BroadcastToDims(maxID, tokens.Shape().Dimensions...)
	safeTokens := Where(GreaterOrEqual(tokens, Scalar(g, dtypes.Int32, float64(cfg.VocabSize))), maxBroadcast, tokens)
	safeTokens = Where(LessThan(safeTokens, ZerosLike(safeTokens)), ZerosLike(safeTokens), safeTokens)

	embed := layers.Embedding(scope.In("token_embed"), safeTokens, dtypes.Float32, cfg.VocabSize, cfg.EmbedDim, false)
	embedMask := BroadcastToShape(InsertAxes(padMask, -1), embed.Shape())
	maskedZeroEmbed := Where(embedMask, embed, ZerosLike(embed))

	batchSize := embed.Shape().Dimensions[0]
	meanPool := Div(ReduceSum(maskedZeroEmbed, 1), BroadcastToDims(safeLengthsCol, batchSize, cfg.EmbedDim))

	negInf := MulScalar(OnesLike(embed), -1e9)
	maskedNegEmbed := Where(embedMask, embed, negInf)
	rawMaxPool := ReduceMax(maskedNegEmbed, 1)
	maxPool := Where(BroadcastToShape(hasTokensCol, rawMaxPool.Shape()), rawMaxPool, ZerosLike(rawMaxPool))

	h := Concatenate([]*Node{meanPool, maxPool}, -1)

	for i, dim := range cfg.HiddenDims {
		h = layers.NewDense(scope.In("trunk_%d", i), h, dim).
			WithBias(true).
			WithActivation(activation.TypeRelu).
			Done()
		h = layers.DropoutStatic(scope.In("trunk_drop_%d", i), h, cfg.DropoutRate)
	}

	normSq := ReduceSum(Mul(h, h), -1)
	norm := Sqrt(Add(normSq, Scalar(g, dtypes.Float32, 1e-8)))
	semanticEmbed = Div(h, BroadcastToShape(InsertAxes(norm, -1), h.Shape()))

	topicH := layers.NewDense(scope.In("topic_hidden"), h, cfg.HeadDim).
		WithBias(true).
		WithActivation(activation.TypeRelu).
		Done()
	topicLogits = layers.NewDense(scope.In("topic_out"), topicH, cfg.NumTopics).
		WithBias(true).
		Done()

	sentH := layers.NewDense(scope.In("sentiment_hidden"), h, cfg.HeadDim).
		WithBias(true).
		WithActivation(activation.TypeRelu).
		Done()
	sentLogits = layers.NewDense(scope.In("sentiment_out"), sentH, 1).
		WithBias(true).
		Done()

	nfcIn := Concatenate([]*Node{h, topicH, sentH}, -1)
	nfcH := layers.NewDense(scope.In("nfc_hidden"), nfcIn, cfg.HeadDim).
		WithBias(true).
		WithActivation(activation.TypeRelu).
		Done()
	nfcH = layers.DropoutStatic(scope.In("nfc_drop"), nfcH, cfg.DropoutRate)
	nfcLogits = layers.NewDense(scope.In("nfc_out"), nfcH, cfg.NumNFC).
		WithBias(true).
		Done()

	return topicLogits, sentLogits, nfcLogits, semanticEmbed
}

// NewTrainModelFn returns a GoMLX train.ModelFn outputting [topicLogits, sentLogits, nfcLogits].
func NewTrainModelFn(cfg GraphConfig, useCosine bool) train.ModelFn {
	return func(scope *model.Scope, _ any, inputs []*Node) []*Node {
		topicLogits, sentLogits, nfcLogits := BuildLogits(scope, inputs[0], cfg, useCosine)
		return []*Node{topicLogits, sentLogits, nfcLogits}
	}
}
