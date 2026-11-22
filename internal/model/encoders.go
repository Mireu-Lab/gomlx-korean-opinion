package model

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/layers/attention/pos"
	"github.com/gomlx/gomlx/ml/layers/lstm"
	"github.com/gomlx/gomlx/ml/layers/norm"
	"github.com/gomlx/gomlx/ml/model"
)

// AttentionPool computes learned additive attention pooling over sequence states seqH [B, T, D]
// masked by padMask [B, T] (true for valid tokens), returning pooled vectors [B, D].
func AttentionPool(scope *model.Scope, seqH, padMask *Node, attnDim int) *Node {
	dims := seqH.Shape().Dimensions
	batchSize := dims[0]
	seqLen := dims[1]
	featDim := dims[2]

	flat := Reshape(seqH, batchSize*seqLen, featDim)
	proj := layers.NewDense(scope.In("attn_proj"), flat, attnDim).
		WithBias(true).
		WithActivation(activation.TypeTanh).
		Done()
	rawScores := layers.NewDense(scope.In("attn_score"), proj, 1).
		WithBias(false).
		Done()
	logits := Reshape(rawScores, batchSize, seqLen)

	negInf := MulScalar(OnesLike(logits), -1e9)
	maskedLogits := Where(padMask, logits, negInf)
	rawWeights := Softmax(maskedLogits, -1)

	padFloat := ConvertDType(padMask, dtypes.Float32)
	rowCounts := ReduceSum(padFloat, -1)
	hasAny := GreaterThan(rowCounts, ZerosLike(rowCounts))
	effectiveMask := LogicalAnd(padMask, BroadcastToDims(InsertAxes(hasAny, -1), batchSize, seqLen))
	weights := Where(effectiveMask, rawWeights, ZerosLike(rawWeights))

	weightsCol := BroadcastToShape(InsertAxes(weights, -1), seqH.Shape())
	return ReduceSum(Mul(weightsCol, seqH), 1)
}

func buildSeqWordEmbedding(
	scope *model.Scope,
	wordTokens *Node,
	vocabSize, embedDim int,
	freeze bool,
) (emb, padMask *Node) {
	g := wordTokens.Graph()
	padMask = NotEqual(wordTokens, ZerosLike(wordTokens))

	maxID := Scalar(g, dtypes.Int32, float64(vocabSize-1))
	maxBroadcast := BroadcastToDims(maxID, wordTokens.Shape().Dimensions...)
	safeTokens := Where(GreaterOrEqual(wordTokens, Scalar(g, dtypes.Int32, float64(vocabSize))), maxBroadcast, wordTokens)
	safeTokens = Where(LessThan(safeTokens, ZerosLike(safeTokens)), ZerosLike(safeTokens), safeTokens)

	emb = layers.Embedding(scope.In("seq_embed"), safeTokens, dtypes.Float32, vocabSize, embedDim, false)
	if freeze {
		emb = StopGradient(emb)
	}
	embMask := BroadcastToShape(InsertAxes(padMask, -1), emb.Shape())
	emb = Where(embMask, emb, ZerosLike(emb))
	return emb, padMask
}

func fuseStudentAndSequence(scope *model.Scope, studentH, seqBranchH *Node, outDim int, dropoutRate float64) *Node {
	fused := layers.NewDense(scope.In("ko_hub_seq_fuse"), Concatenate([]*Node{studentH, seqBranchH}, -1), outDim).
		WithBias(true).
		WithActivation(activation.TypeRelu).
		Done()
	fused = layers.DropoutStatic(scope.In("ko_hub_seq_drop"), fused, dropoutRate)
	return Add(studentH, fused)
}

// buildKoHubCNNReLU combines the KO-Hub Student encoder with Multi-Scale 1D-CNN (kernels 2, 3, 4) + ReLU + AttentionPool.
func buildKoHubCNNReLU(scope *model.Scope, studentH, wordTokens *Node, cfg GraphConfig) *Node {
	g := wordTokens.Graph()
	wordEmb, wordPadMask := buildSeqWordEmbedding(scope, wordTokens, cfg.VocabSize, cfg.EmbedDim, cfg.FreezeEncoder)

	kernels := cfg.CNNKernels
	if len(kernels) == 0 {
		kernels = []int{2, 3, 4}
	}
	channels := cfg.CNNChannels
	if channels < 8 {
		channels = 64
	}

	convOuts := make([]*Node, 0, len(kernels))
	for _, k := range kernels {
		c := layers.Convolution(scope.In("conv_k%d", k), wordEmb).
			Channels(channels).
			KernelSize(k).
			PadSame().
			UseBias(true).
			Done()
		c = activation.Relu(c)
		cMask := BroadcastToShape(InsertAxes(wordPadMask, -1), c.Shape())
		c = Where(cMask, c, ZerosLike(c))
		convOuts = append(convOuts, c)
	}

	seqH := convOuts[0]
	if len(convOuts) > 1 {
		seqH = Concatenate(convOuts, -1)
	}

	attnPoolDim := cfg.AttnPoolDim
	if attnPoolDim < 8 {
		attnPoolDim = 64
	}
	attnH := AttentionPool(scope.In("cnn_attn_pool"), seqH, wordPadMask, attnPoolDim)

	seqMask := BroadcastToShape(InsertAxes(wordPadMask, -1), seqH.Shape())
	negInf := MulScalar(OnesLike(seqH), -1e9)
	rawMax := ReduceMax(Where(seqMask, seqH, negInf), 1)
	rowCounts := ReduceSum(ConvertDType(wordPadMask, dtypes.Float32), -1)
	hasAny := GreaterThan(rowCounts, Scalar(g, dtypes.Float32, 0))
	maxH := Where(BroadcastToShape(InsertAxes(hasAny, -1), rawMax.Shape()), rawMax, ZerosLike(rawMax))

	cnnH := layers.NewDense(scope.In("cnn_proj"), Concatenate([]*Node{attnH, maxH}, -1), cfg.StudentOutDim).
		WithBias(true).
		WithActivation(activation.TypeRelu).
		Done()
	cnnH = layers.DropoutStatic(scope.In("cnn_drop"), cnnH, cfg.DropoutRate)

	return fuseStudentAndSequence(scope, studentH, cnnH, cfg.StudentOutDim, cfg.DropoutRate)
}

// buildKoHubTransformerReLU combines the KO-Hub Student encoder with lightweight RoPE Self-Attention + ReLU FFN layers.
func buildKoHubTransformerReLU(scope *model.Scope, studentH, wordTokens *Node, cfg GraphConfig) *Node {
	g := wordTokens.Graph()
	wordEmb, wordPadMask := buildSeqWordEmbedding(scope, wordTokens, cfg.VocabSize, cfg.EmbedDim, cfg.FreezeEncoder)

	dims := wordEmb.Shape().Dimensions
	batchSize := dims[0]
	wordSeqLen := dims[1]
	inDim := dims[2]

	tDim := cfg.TransformerDim
	if tDim < 8 {
		tDim = 128
	}
	nHeads := cfg.TransformerHeads
	if nHeads < 1 {
		nHeads = 4
	}
	nLayers := cfg.TransformerLayers
	if nLayers < 1 {
		nLayers = 2
	}

	flatIn := Reshape(wordEmb, batchSize*wordSeqLen, inDim)
	xFlat := layers.NewDense(scope.In("trans_in_proj"), flatIn, tDim).WithBias(false).Done()
	x := Reshape(xFlat, batchSize, wordSeqLen, tDim)

	posIdx := Iota(g, shapes.Make(dtypes.Int32, batchSize, wordSeqLen), 1)
	pos0Mask := Equal(posIdx, ZerosLike(posIdx))
	safeMask := LogicalOr(wordPadMask, pos0Mask)

	headDim := tDim / nHeads
	rope := pos.NewRoPE(10000.0)

	for layer := 0; layer < nLayers; layer++ {
		lScope := scope.In("trans_layer_%d", layer)
		n1 := norm.RMSNorm(lScope.In("norm1"), x).Done()
		attnOut := attention.MultiHeadAttention(lScope.In("mha"), n1, n1, n1, nHeads, headDim).
			WithPositionalEncoder(rope).
			WithMask(safeMask).
			WithOutputDim(tDim).
			Done()
		x = Add(x, layers.DropoutStatic(lScope.In("attn_drop"), attnOut, cfg.DropoutRate))

		n2 := norm.RMSNorm(lScope.In("norm2"), x).Done()
		flatN2 := Reshape(n2, batchSize*wordSeqLen, tDim)
		ffn1 := layers.NewDense(lScope.In("ffn1"), flatN2, tDim*2).
			WithBias(true).
			WithActivation(activation.TypeRelu).
			Done()
		ffn1 = layers.DropoutStatic(lScope.In("ffn_drop1"), ffn1, cfg.DropoutRate)
		ffn2Flat := layers.NewDense(lScope.In("ffn2"), ffn1, tDim).WithBias(true).Done()
		ffn2 := Reshape(ffn2Flat, batchSize, wordSeqLen, tDim)
		x = Add(x, layers.DropoutStatic(lScope.In("ffn_drop2"), ffn2, cfg.DropoutRate))
	}

	x = norm.RMSNorm(scope.In("trans_norm_out"), x).Done()
	attnPoolDim := cfg.AttnPoolDim
	if attnPoolDim < 8 {
		attnPoolDim = 64
	}
	attnH := AttentionPool(scope.In("trans_pool"), x, wordPadMask, attnPoolDim)
	transH := layers.NewDense(scope.In("trans_proj"), attnH, cfg.StudentOutDim).
		WithBias(true).
		WithActivation(activation.TypeRelu).
		Done()
	transH = layers.DropoutStatic(scope.In("trans_drop"), transH, cfg.DropoutRate)

	return fuseStudentAndSequence(scope, studentH, transH, cfg.StudentOutDim, cfg.DropoutRate)
}

// buildKoHubLSTMReLU combines the KO-Hub Student encoder with a word-order Ragged Bidirectional LSTM + AttentionPool + ReLU.
func buildKoHubLSTMReLU(scope *model.Scope, studentH, wordTokens *Node, cfg GraphConfig) *Node {
	g := wordTokens.Graph()
	wordEmb, wordPadMask := buildSeqWordEmbedding(scope, wordTokens, cfg.VocabSize, cfg.EmbedDim, cfg.FreezeEncoder)

	dims := wordEmb.Shape().Dimensions
	batchSize := dims[0]
	wordSeqLen := dims[1]

	hidden := cfg.LSTMHidden
	if hidden < 8 {
		hidden = 48
	}

	lengthsF := Max(ReduceSum(ConvertDType(wordPadMask, dtypes.Float32), -1), Scalar(g, dtypes.Float32, 1.0))
	wordLengths := ConvertDType(lengthsF, dtypes.Int32)

	allH, lastH, _ := lstm.New(scope.In("bilstm"), wordEmb, hidden).
		Direction(lstm.DirBidirectional).
		Ragged(wordLengths).
		Done()

	// allH is [wordSeqLen, 2, batchSize, hidden] -> [batchSize, wordSeqLen, 2*hidden]
	seqH := Reshape(TransposeAllDims(allH, 2, 0, 1, 3), batchSize, wordSeqLen, 2*hidden)
	attnPoolDim := cfg.AttnPoolDim
	if attnPoolDim < 8 {
		attnPoolDim = 64
	}
	attnH := AttentionPool(scope.In("lstm_pool"), seqH, wordPadMask, attnPoolDim)

	// lastH is [2, batchSize, hidden] -> [batchSize, 2*hidden]
	endH := Reshape(TransposeAllDims(lastH, 1, 0, 2), batchSize, 2*hidden)

	lstmH := layers.NewDense(scope.In("lstm_proj"), Concatenate([]*Node{attnH, endH}, -1), cfg.StudentOutDim).
		WithBias(true).
		WithActivation(activation.TypeRelu).
		Done()
	lstmH = layers.DropoutStatic(scope.In("lstm_drop"), lstmH, cfg.DropoutRate)

	return fuseStudentAndSequence(scope, studentH, lstmH, cfg.StudentOutDim, cfg.DropoutRate)
}
