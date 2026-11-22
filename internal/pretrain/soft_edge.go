package pretrain

import (
	"math"

	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/model"
)

// SoftEdgeAttention computes Context-Aware Soft Edge Attention over candidate multilingual root nodes:
// context [B, D], candRoots [B, K, D], candMask [B, K] -> agg [B, D], weights [B, K].
// Candidate roots incompatible with surrounding Korean sentence context receive near-zero weight.
func SoftEdgeAttention(scope *model.Scope, context, candRoots, candMask *Node, attnDim int) (agg, weights *Node) {
	dims := candRoots.Shape().Dimensions
	batchSize := dims[0]
	numCands := dims[1]
	rootDim := dims[2]

	q := layers.NewDense(scope.In("w_q"), context, attnDim).WithBias(false).Done() // [B, Da]
	flatRoots := Reshape(candRoots, batchSize*numCands, rootDim)
	flatK := layers.NewDense(scope.In("w_k"), flatRoots, attnDim).WithBias(false).Done() // [B*K, Da]
	k := Reshape(flatK, batchSize, numCands, attnDim)                                    // [B, K, Da]

	qExpanded := BroadcastToDims(InsertAxes(q, 1), batchSize, numCands, attnDim)
	scale := 1.0 / math.Sqrt(float64(attnDim))
	scores := MulScalar(ReduceSum(Mul(qExpanded, k), -1), scale) // [B, K]

	validMask := GreaterThan(candMask, ZerosLike(candMask))
	negInf := MulScalar(OnesLike(scores), -1e9)
	maskedScores := Where(validMask, scores, negInf)
	rawWeights := Softmax(maskedScores, -1) // [B, K]

	rowValidSum := ReduceSum(candMask, -1) // [B]
	hasAny := GreaterThan(rowValidSum, ZerosLike(rowValidSum))
	hasAnyMat := BroadcastToDims(InsertAxes(hasAny, -1), batchSize, numCands)
	effectiveMask := LogicalAnd(validMask, hasAnyMat)
	weights = Where(effectiveMask, rawWeights, ZerosLike(rawWeights)) // [B, K]

	weightsExpanded := BroadcastToShape(InsertAxes(weights, -1), candRoots.Shape())
	agg = ReduceSum(Mul(weightsExpanded, candRoots), 1) // [B, D]
	return agg, weights
}

// SafeL2Normalize normalizes along the last axis with epsilon guard against zero vectors.
func SafeL2Normalize(x *Node) *Node {
	g := x.Graph()
	normSq := ReduceSum(Mul(x, x), -1)
	norm := Sqrt(Add(normSq, Scalar(g, dtypes.Float32, 1e-8)))
	return Div(x, BroadcastToShape(InsertAxes(norm, -1), x.Shape()))
}
