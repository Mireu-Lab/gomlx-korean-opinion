package model

import (
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/train"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
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
// 4. Optional CORAL mean+variance domain alignment between YNAT (news) and NSMC (reviews) semantic embeddings
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

		// 4. Optional CORAL mean+variance alignment between YNAT and NSMC semantic embeddings.
		if cfg.DomainAlign > 0 && len(predictions) >= 4 {
			emb := predictions[3] // [B, D]
			eps := Scalar(g, dtypes.Float32, cfg.MaskEpsilon)
			ySum := ReduceAllSum(topicMask)
			nSum := ReduceAllSum(sentMask)
			yDenom := Max(ySum, eps)
			nDenom := Max(nSum, eps)

			yCol := BroadcastToShape(InsertAxes(topicMask, -1), emb.Shape())
			nCol := BroadcastToShape(InsertAxes(sentMask, -1), emb.Shape())

			muY := Div(ReduceSum(Mul(emb, yCol), 0), yDenom)
			muN := Div(ReduceSum(Mul(emb, nCol), 0), nDenom)
			dMu := Sub(muY, muN)
			meanAlign := ReduceAllMean(Mul(dMu, dMu))

			centeredY := Mul(Sub(emb, BroadcastToShape(InsertAxes(muY, 0), emb.Shape())), yCol)
			centeredN := Mul(Sub(emb, BroadcastToShape(InsertAxes(muN, 0), emb.Shape())), nCol)
			varY := Div(ReduceSum(Mul(centeredY, centeredY), 0), yDenom)
			varN := Div(ReduceSum(Mul(centeredN, centeredN), 0), nDenom)
			dVar := Sub(varY, varN)
			varAlign := ReduceAllMean(Mul(dVar, dVar))

			zeroS := ZerosLike(ySum)
			bothDomains := ConvertDType(LogicalAnd(GreaterThan(ySum, zeroS), GreaterThan(nSum, zeroS)), dtypes.Float32)
			coral := Mul(Add(meanAlign, varAlign), bothDomains)
			total = Add(total, MulScalar(coral, cfg.DomainAlign))
		}
		return total
	}
}
