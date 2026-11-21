package model

import (
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/train"

	"github.com/Mireu-Lab/gomlx-nfc-opinion/internal/config"
)

func maskedMean(perExample, mask *Node, maskEpsilon float64) *Node {
	g := perExample.Graph()
	eps := Scalar(g, dtypes.Float32, maskEpsilon)
	denom := Max(ReduceAllSum(mask), eps)
	return Div(ReduceAllSum(Mul(perExample, mask)), denom)
}

// BuildMultiTaskLoss creates a NaN-safe masked multi-task loss combining:
// 1. Masked Sparse Categorical Cross-Entropy for KLUE-YNAT 7 topics
// 2. Masked Binary Cross-Entropy with Logits for NSMC sentiment
// 3. Masked Positive-Weighted Multi-Label BCE with Logits for 16 Nuclear Feeling Checklist targets
func BuildMultiTaskLoss(cfg config.LossWeightsConfig, numTopics int) train.LossFn {
	return func(labels, predictions []*Node) *Node {
		topicLogits := predictions[0] // [B, numTopics]
		sentLogits := predictions[1]  // [B, 1]
		nfcLogits := predictions[2]   // [B, numNFC]

		topicTarget := labels[0] // (int32)[B, 1]
		topicMask := labels[1]   // (float32)[B]
		sentTarget := labels[2]  // (float32)[B, 1]
		sentMask := labels[3]    // (float32)[B]
		nfcTarget := labels[4]   // (float32)[B, numNFC]
		nfcMask := labels[5]     // (float32)[B]

		batchSize := topicLogits.Shape().Dimensions[0]
		g := topicLogits.Graph()

		// 1. KLUE-YNAT Topic Sparse Cross-Entropy.
		topicIdx := Reshape(topicTarget, batchSize)
		topicOneHot := OneHot(topicIdx, numTopics, dtypes.Float32)
		topicPerEx := ReduceSum(Neg(Mul(topicOneHot, LogSoftmax(topicLogits))), -1)
		topicLoss := maskedMean(topicPerEx, topicMask, cfg.MaskEpsilon)

		// 2. NSMC Binary Sentiment BCE with Logits.
		sentLogitsFlat := Reshape(sentLogits, batchSize)
		sentTargetFlat := Reshape(sentTarget, batchSize)
		sentLogPart := Log1P(Exp(Neg(Abs(sentLogitsFlat))))
		sentMaxPart := Max(sentLogitsFlat, ZerosLike(sentLogitsFlat))
		sentPerEx := Add(Sub(sentMaxPart, Mul(sentLogitsFlat, sentTargetFlat)), sentLogPart)
		sentLoss := maskedMean(sentPerEx, sentMask, cfg.MaskEpsilon)

		// 3. Nuclear Feeling Checklist Positive-Weighted Multi-Label BCE with Logits.
		posWeight := Scalar(g, dtypes.Float32, cfg.NFCPosWeight)
		nfcLogPart := Log1P(Exp(Neg(Abs(nfcLogits))))
		posTerm := Mul(Mul(nfcTarget, posWeight), Add(Max(Neg(nfcLogits), ZerosLike(nfcLogits)), nfcLogPart))
		negTerm := Mul(OneMinus(nfcTarget), Add(Max(nfcLogits, ZerosLike(nfcLogits)), nfcLogPart))
		nfcPerEx := ReduceMean(Add(posTerm, negTerm), -1)
		nfcLoss := maskedMean(nfcPerEx, nfcMask, cfg.MaskEpsilon)

		total := Add(
			Add(MulScalar(topicLoss, cfg.Topic), MulScalar(sentLoss, cfg.Sentiment)),
			MulScalar(nfcLoss, cfg.NFC),
		)
		return total
	}
}
