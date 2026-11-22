package model

import (
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/optimizer/cosineschedule"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/pretrain"
)

// GraphConfig parameterizes the GoMLX ReLU multi-task NLP graph across all supported encoder providers.
type GraphConfig struct {
	VocabSize         int
	EmbedDim          int
	HiddenDims        []int
	HeadDim           int
	DropoutRate       float64
	NumTopics         int
	NumNFC            int
	EmbeddingProvider string
	StudentOutDim     int
	FreezeEncoder     bool
	CNNChannels       int
	CNNKernels        []int
	TransformerLayers int
	TransformerHeads  int
	TransformerDim    int
	LSTMHidden        int
	AttnPoolDim       int
}

// IsKoHubProvider reports whether provider uses the pretrained KO-Hub Student encoder.
func IsKoHubProvider(provider string) bool {
	switch provider {
	case "ko_hub_graph", "ko_hub_cnn_relu", "ko_hub_transformer_relu", "ko_hub_lstm_relu":
		return true
	default:
		return false
	}
}

// UsesSeqBranch reports whether provider adds a word-order sequence encoder alongside KO-Hub.
func UsesSeqBranch(provider string) bool {
	switch provider {
	case "ko_hub_cnn_relu", "ko_hub_transformer_relu", "ko_hub_lstm_relu":
		return true
	default:
		return false
	}
}

// BuildLogits constructs the GoMLX computation graph and returns logits for the 3 multi-task heads.
func BuildLogits(scope *model.Scope, tokens, wordTokens *Node, cfg GraphConfig, useCosine bool) (topicLogits, sentLogits, nfcLogits *Node) {
	topicLogits, sentLogits, nfcLogits, _ = BuildGraphWithEmbedding(scope, tokens, wordTokens, cfg, useCosine)
	return topicLogits, sentLogits, nfcLogits
}

// BuildGraphWithEmbedding constructs the GoMLX computation graph:
//   - "scratch_relu":           tokens -> Embedding -> Masked (MeanPool || MaxPool) -> Shared ReLU MLP
//   - "ko_hub_graph":           tokens -> Pretrained KO-Hub Student -> Residual ReLU Adapter
//   - "ko_hub_cnn_relu":        KO-Hub Student + Word-Order Multi-Scale Conv1D + AttentionPool + ReLU
//   - "ko_hub_transformer_relu": KO-Hub Student + Word-Order RoPE Self-Attention Transformer + ReLU
//   - "ko_hub_lstm_relu":       KO-Hub Student + Word-Order Ragged BiLSTM + AttentionPool + ReLU
//
// Outputs 3 task heads (Topic [B, 7], Sentiment [B, 1], NFC [B, 16]) plus L2-normalized shared semantic embedding [B, D].
func BuildGraphWithEmbedding(scope *model.Scope, tokens, wordTokens *Node, cfg GraphConfig, useCosine bool) (topicLogits, sentLogits, nfcLogits, semanticEmbed *Node) {
	g := tokens.Graph()
	if useCosine && scope.Store().IsTraining(g) {
		cosineschedule.New(scope, g, dtypes.Float32).FromScope().Done()
	}

	if wordTokens == nil {
		wordTokens = tokens
	}

	var h *Node
	if IsKoHubProvider(cfg.EmbeddingProvider) {
		studentH := pretrain.BuildStudentEncoder(scope, tokens, cfg.VocabSize, cfg.EmbedDim, cfg.StudentOutDim, cfg.DropoutRate, cfg.FreezeEncoder)
		// Pin a deterministic scope name so Trainer and inference Exec share exact variable paths.
		scope = scope.In("relu_nlp")
		switch cfg.EmbeddingProvider {
		case "ko_hub_cnn_relu":
			h = buildKoHubCNNReLU(scope, studentH, wordTokens, cfg)
		case "ko_hub_transformer_relu":
			h = buildKoHubTransformerReLU(scope, studentH, wordTokens, cfg)
		case "ko_hub_lstm_relu":
			h = buildKoHubLSTMReLU(scope, studentH, wordTokens, cfg)
		default:
			adapter := layers.NewDense(scope.In("ko_hub_adapter"), studentH, cfg.StudentOutDim).
				WithBias(true).
				WithActivation(activation.TypeRelu).
				Done()
			adapter = layers.DropoutStatic(scope.In("ko_hub_drop"), adapter, cfg.DropoutRate)
			h = Add(studentH, adapter)
		}
	} else {
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

		h = Concatenate([]*Node{meanPool, maxPool}, -1)

		for i, dim := range cfg.HiddenDims {
			h = layers.NewDense(scope.In("trunk_%d", i), h, dim).
				WithBias(true).
				WithActivation(activation.TypeRelu).
				Done()
			h = layers.DropoutStatic(scope.In("trunk_drop_%d", i), h, cfg.DropoutRate)
		}
	}

	semanticEmbed = pretrain.SafeL2Normalize(h)

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

// NewTrainModelFn returns a GoMLX train.ModelFn outputting [topicLogits, sentLogits, nfcLogits, semanticEmbed].
func NewTrainModelFn(cfg GraphConfig, useCosine bool) train.ModelFn {
	return func(scope *model.Scope, _ any, inputs []*Node) []*Node {
		var wordTokens *Node
		if len(inputs) > 1 {
			wordTokens = inputs[1]
		}
		topicLogits, sentLogits, nfcLogits, semanticEmbed := BuildGraphWithEmbedding(scope, inputs[0], wordTokens, cfg, useCosine)
		return []*Node{topicLogits, sentLogits, nfcLogits, semanticEmbed}
	}
}
