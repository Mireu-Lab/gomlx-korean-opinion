package pretrain

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/train"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/graphkg"
)

func maskedInfoNCE(anchor, view, validMask *Node, temperature, maskEpsilon float64) *Node {
	g := anchor.Graph()
	batchSize := anchor.Shape().Dimensions[0]
	sim := MulScalar(Einsum("bd,cd->bc", anchor, view), 1.0/temperature)
	diagIdx := Iota(g, shapes.Make(dtypes.Int32, batchSize), 0)
	targets := OneHot(diagIdx, batchSize, dtypes.Float32)
	perEx := ReduceSum(Neg(Mul(targets, LogSoftmax(sim, -1))), -1)
	eps := Scalar(g, dtypes.Float32, maskEpsilon)
	denom := Max(ReduceAllSum(validMask), eps)
	return Div(ReduceAllSum(Mul(perEx, validMask)), denom)
}

// BuildPretrainLoss constructs the Multi-View Contrastive + Homonym Disambiguation + Distillation loss:
//   - L_{ko<->en}:    InfoNCE between Korean student/teacher and English concept roots
//   - L_{ko<->hanja}: InfoNCE between Korean student/teacher and Hanja morpheme roots
//   - L_{ko<->ja,zh}: InfoNCE between Korean student/teacher and Japanese/Chinese roots
//   - L_disambig:     Cosine hinge margin separating contrasting homonym sense contexts
//   - L_distill:      Cosine alignment from Korean-only Student toward Teacher (StopGradient)
func BuildPretrainLoss(cfg graphkg.PretrainConfig) train.LossFn {
	return func(labels, predictions []*Node) *Node {
		student := predictions[0]
		teacher := predictions[1]
		enView := predictions[2]
		hanjaView := predictions[3]
		jazhView := predictions[4]
		hardNeg := predictions[5]

		enValid := labels[0]
		hanjaValid := labels[1]
		jazhValid := labels[2]
		disambigValid := labels[3]

		g := student.Graph()
		eps := Scalar(g, dtypes.Float32, cfg.MaskEpsilon)

		lossEN := MulScalar(
			Add(
				maskedInfoNCE(student, enView, enValid, cfg.Temperature, cfg.MaskEpsilon),
				maskedInfoNCE(teacher, enView, enValid, cfg.Temperature, cfg.MaskEpsilon),
			),
			0.5,
		)
		lossHanja := MulScalar(
			Add(
				maskedInfoNCE(student, hanjaView, hanjaValid, cfg.Temperature, cfg.MaskEpsilon),
				maskedInfoNCE(teacher, hanjaView, hanjaValid, cfg.Temperature, cfg.MaskEpsilon),
			),
			0.5,
		)
		lossJaZh := MulScalar(
			Add(
				maskedInfoNCE(student, jazhView, jazhValid, cfg.Temperature, cfg.MaskEpsilon),
				maskedInfoNCE(teacher, jazhView, jazhValid, cfg.Temperature, cfg.MaskEpsilon),
			),
			0.5,
		)

		// Homonym disambiguation hinge: penalize cos(student, hardNeg) > -disambigMargin (or cos + margin > 0).
		negCos := ReduceSum(Mul(student, hardNeg), -1)
		margin := Scalar(g, dtypes.Float32, cfg.DisambigMargin)
		hinge := Max(Add(negCos, margin), ZerosLike(negCos))
		disambigDenom := Max(ReduceAllSum(disambigValid), eps)
		lossDisambig := Div(ReduceAllSum(Mul(hinge, disambigValid)), disambigDenom)

		// Asymmetric distillation: pull Korean-only Student toward Teacher without collapsing Teacher.
		simST := ReduceSum(Mul(student, StopGradient(teacher)), -1)
		lossDistill := ReduceAllMean(Sub(OnesLike(simST), simST))

		w := cfg.LossWeights
		total := Add(
			Add(
				MulScalar(lossEN, w.KoEN),
				MulScalar(lossHanja, w.KoHanja),
			),
			Add(
				MulScalar(lossJaZh, w.KoJaZh),
				Add(
					MulScalar(lossDisambig, w.Disambig),
					MulScalar(lossDistill, w.Distill),
				),
			),
		)
		return total
	}
}
