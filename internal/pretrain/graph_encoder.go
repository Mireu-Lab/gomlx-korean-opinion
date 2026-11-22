package pretrain

import (
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/optimizer/cosineschedule"
)

// StudentScopeName pins the variable scope shared between pretraining and downstream ko_hub_graph models.
const StudentScopeName = "ko_hub_student"

// PretrainGraphConfig parameterizes the KO-Hub Teacher + Korean-only Student GoMLX computation graph.
type PretrainGraphConfig struct {
	VocabSize       int
	NumRootVocab    int
	EmbedDim        int
	OutDim          int
	RootEmbedDim    int
	AttnDim         int
	MaxSeqLen       int
	MaxRootsPerView int
	DropoutRate     float64
}

// BuildStudentEncoder builds the Korean-only Student text encoder under the root scope `/ko_hub_student/...`.
// Both pretraining and downstream inference call this function at the root scope so variable paths match identically.
func BuildStudentEncoder(
	scope *model.Scope,
	tokens *Node,
	vocabSize, embedDim, outDim int,
	dropoutRate float64,
	freeze bool,
) *Node {
	g := tokens.Graph()
	scope = scope.In(StudentScopeName)

	padMask := NotEqual(tokens, ZerosLike(tokens))
	padMaskFloat := ConvertDType(padMask, dtypes.Float32)
	lengths := ReduceSum(padMaskFloat, 1)
	safeLengths := Max(lengths, OnesLike(lengths))
	safeLengthsCol := InsertAxes(safeLengths, -1)
	hasTokens := GreaterThan(lengths, ZerosLike(lengths))
	hasTokensCol := InsertAxes(hasTokens, -1)

	maxID := Scalar(g, dtypes.Int32, float64(vocabSize-1))
	maxBroadcast := BroadcastToDims(maxID, tokens.Shape().Dimensions...)
	safeTokens := Where(GreaterOrEqual(tokens, Scalar(g, dtypes.Int32, float64(vocabSize))), maxBroadcast, tokens)
	safeTokens = Where(LessThan(safeTokens, ZerosLike(safeTokens)), ZerosLike(safeTokens), safeTokens)

	embed := layers.Embedding(scope.In("token_embed"), safeTokens, dtypes.Float32, vocabSize, embedDim, false)
	embedMask := BroadcastToShape(InsertAxes(padMask, -1), embed.Shape())
	maskedZeroEmbed := Where(embedMask, embed, ZerosLike(embed))

	batchSize := embed.Shape().Dimensions[0]
	meanPool := Div(ReduceSum(maskedZeroEmbed, 1), BroadcastToDims(safeLengthsCol, batchSize, embedDim))

	negInf := MulScalar(OnesLike(embed), -1e9)
	maskedNegEmbed := Where(embedMask, embed, negInf)
	rawMaxPool := ReduceMax(maskedNegEmbed, 1)
	maxPool := Where(BroadcastToShape(hasTokensCol, rawMaxPool.Shape()), rawMaxPool, ZerosLike(rawMaxPool))

	pooled := Concatenate([]*Node{meanPool, maxPool}, -1) // [B, 2*embedDim]

	proj1 := layers.NewDense(scope.In("student_proj1"), pooled, outDim).
		WithBias(true).
		WithActivation(activation.TypeRelu).
		Done()
	proj1 = layers.DropoutStatic(scope.In("student_drop1"), proj1, dropoutRate)
	proj2 := layers.NewDense(scope.In("student_proj2"), proj1, outDim).
		WithBias(true).
		Done()
	skip := layers.NewDense(scope.In("student_skip"), pooled, outDim).
		WithBias(false).
		Done()
	h := activation.Relu(Add(proj2, skip))
	if freeze {
		h = StopGradient(h)
	}
	return SafeL2Normalize(h)
}

// BuildPretrainModelFn builds the GoMLX train.ModelFn for KO-Hub multilingual roots contrastive pretraining.
// Inputs:  [tokens, hanjaIDs, hanjaMask, enIDs, enMask, jazhIDs, jazhMask, hardNegTokens]
// Outputs: [studentEmbed, teacherEmbed, enView, hanjaView, jazhView, hardNegEmbed]
func BuildPretrainModelFn(cfg PretrainGraphConfig, useCosine bool) train.ModelFn {
	return func(scope *model.Scope, _ any, inputs []*Node) []*Node {
		tokens := inputs[0]
		hanjaIDs := inputs[1]
		hanjaMask := inputs[2]
		enIDs := inputs[3]
		enMask := inputs[4]
		jazhIDs := inputs[5]
		jazhMask := inputs[6]
		hardNegTokens := inputs[7]

		g := tokens.Graph()
		if useCosine && scope.Store().IsTraining(g) {
			cosineschedule.New(scope, g, dtypes.Float32).FromScope().Done()
		}

		batchSize := tokens.Shape().Dimensions[0]
		bothTokens := Concatenate([]*Node{tokens, hardNegTokens}, 0)
		bothEmbed := BuildStudentEncoder(scope, bothTokens, cfg.VocabSize, cfg.EmbedDim, cfg.OutDim, cfg.DropoutRate, false)
		studentEmbed := Slice(bothEmbed, AxisRange(0, batchSize))
		hardNegEmbed := Slice(bothEmbed, AxisRange(batchSize, 2*batchSize))

		tScope := scope.In("ko_hub_teacher")
		k := hanjaIDs.Shape().Dimensions[1]
		allRootIDs := Concatenate([]*Node{hanjaIDs, enIDs, jazhIDs}, 1)
		allRootEmb := layers.Embedding(tScope.In("root_embed"), allRootIDs, dtypes.Float32, cfg.NumRootVocab, cfg.RootEmbedDim, false)
		hanjaEmb := Slice(allRootEmb, AxisRange(), AxisRange(0, k))
		enEmb := Slice(allRootEmb, AxisRange(), AxisRange(k, 2*k))
		jazhEmb := Slice(allRootEmb, AxisRange(), AxisRange(2*k, 3*k))

		hanjaAgg, _ := SoftEdgeAttention(tScope.In("attn_hanja"), studentEmbed, hanjaEmb, hanjaMask, cfg.AttnDim)
		enAgg, _ := SoftEdgeAttention(tScope.In("attn_en"), studentEmbed, enEmb, enMask, cfg.AttnDim)
		jazhAgg, _ := SoftEdgeAttention(tScope.In("attn_jazh"), studentEmbed, jazhEmb, jazhMask, cfg.AttnDim)

		hanjaView := hanjaAgg
		enView := enAgg
		jazhView := jazhAgg
		if cfg.RootEmbedDim != cfg.OutDim {
			hanjaView = layers.NewDense(tScope.In("proj_hanja"), hanjaAgg, cfg.OutDim).WithBias(false).Done()
			enView = layers.NewDense(tScope.In("proj_en"), enAgg, cfg.OutDim).WithBias(false).Done()
			jazhView = layers.NewDense(tScope.In("proj_jazh"), jazhAgg, cfg.OutDim).WithBias(false).Done()
		}

		fusedIn := Concatenate([]*Node{studentEmbed, hanjaView, enView, jazhView}, -1)
		fuse1 := layers.NewDense(tScope.In("fuse_mlp1"), fusedIn, cfg.OutDim).
			WithBias(true).
			WithActivation(activation.TypeRelu).
			Done()
		fuse1 = layers.DropoutStatic(tScope.In("fuse_drop"), fuse1, cfg.DropoutRate)
		fuse2 := layers.NewDense(tScope.In("fuse_mlp2"), fuse1, cfg.OutDim).
			WithBias(true).
			Done()
		teacherEmbed := SafeL2Normalize(Add(studentEmbed, fuse2))

		return []*Node{
			studentEmbed,
			teacherEmbed,
			SafeL2Normalize(enView),
			SafeL2Normalize(hanjaView),
			SafeL2Normalize(jazhView),
			hardNegEmbed,
		}
	}
}
