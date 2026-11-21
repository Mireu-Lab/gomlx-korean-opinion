package model

import (
	"fmt"
	"math"
	"os"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/ml/train/optimizer/cosineschedule"

	"github.com/Mireu-Lab/gomlx-nfc-opinion/internal/config"
	"github.com/Mireu-Lab/gomlx-nfc-opinion/internal/data"
	"github.com/Mireu-Lab/gomlx-nfc-opinion/internal/logging"
	"github.com/Mireu-Lab/gomlx-nfc-opinion/internal/nfc"
	"github.com/Mireu-Lab/gomlx-nfc-opinion/internal/sorter"
)

// TopicMetric stores per-topic evaluation precision, recall, F1, and support.
type TopicMetric struct {
	TopicID   int     `json:"topic_id"`
	Code      string  `json:"code"`
	KoName    string  `json:"ko_name"`
	Support   int     `json:"support"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
}

// EvalReport summarizes evaluation across KLUE-YNAT, NSMC, and Nuclear Feeling Checklist weak labels.
type EvalReport struct {
	YNATCount              int           `json:"ynat_count"`
	YNATAccuracy           float64       `json:"ynat_accuracy"`
	YNATMacroF1            float64       `json:"ynat_macro_f1"`
	YNATPerTopic           []TopicMetric `json:"ynat_per_topic"`
	NSMCCount              int           `json:"nsmc_count"`
	NSMCAccuracy           float64       `json:"nsmc_accuracy"`
	NSMCPrecision          float64       `json:"nsmc_precision"`
	NSMCRecall             float64       `json:"nsmc_recall"`
	NSMCF1                 float64       `json:"nsmc_f1"`
	NFCActiveEvalCount     int           `json:"nfc_active_eval_count"`
	NFCSupervisedEvalCount int           `json:"nfc_supervised_eval_count"`
	NFCMicroF1             float64       `json:"nfc_micro_f1"`
	NFCMacroF1             float64       `json:"nfc_macro_f1"`
	NFCRecallAt3           float64       `json:"nfc_recall_at_3"`
	NFCCoveredClasses      int           `json:"nfc_covered_classes"`
}

// BuildGraphConfig maps external config + runtime dimensions into GraphConfig.
func BuildGraphConfig(cfg *config.Config, vocabLen, numTopics, numNFC int) GraphConfig {
	return GraphConfig{
		VocabSize:   vocabLen,
		EmbedDim:    cfg.Model.EmbedDim,
		HiddenDims:  cfg.Model.HiddenDims,
		HeadDim:     cfg.Model.HeadDim,
		DropoutRate: cfg.Model.DropoutRate,
		NumTopics:   numTopics,
		NumNFC:      numNFC,
	}
}

// Train trains the GoMLX ReLU multi-task model on NSMC + KLUE-YNAT + NFC weak labels and saves a checkpoint.
func Train(
	backend compute.Backend,
	cfg *config.Config,
	tax *nfc.Taxonomy,
	vocabLen int,
	ynatTrain, nsmcTrain []data.EncodedExample,
	resume bool,
	log *logging.DualLogger,
) (*model.Store, error) {
	if !resume {
		// Remove prior checkpoints when starting a fresh run so vocab and weights stay aligned.
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
		return nil, fmt.Errorf("set RNG seed: %w", err)
	}

	ckptHandler, err := checkpoint.Build(store).
		Dir(cfg.Paths.CheckpointDir).
		Keep(cfg.Training.CheckpointKeep).
		Done()
	if err != nil {
		return nil, fmt.Errorf("init checkpoint handler: %w", err)
	}

	trainDS, err := data.NewMultiTaskTrainDataset(
		"nsmc_ynat_nfc_train",
		ynatTrain,
		nsmcTrain,
		cfg.Training,
		cfg.Tokenizer.MaxSeqLen,
		tax.NumFeelings(),
		cfg.Runtime.Seed,
	)
	if err != nil {
		return nil, fmt.Errorf("create training dataset: %w", err)
	}

	graphCfg := BuildGraphConfig(cfg, vocabLen, tax.NumTopics(), tax.NumFeelings())
	modelFn := NewTrainModelFn(graphCfg, true)
	lossFn := BuildMultiTaskLoss(cfg.Training.LossWeights, tax.NumTopics())
	opt := optimizer.FromStore(store)
	trainer := train.NewTrainer(backend, store, modelFn, lossFn, opt, nil, nil)
	loop := train.NewLoop(trainer)

	startTime := time.Now()
	train.EveryNSteps(loop, cfg.Training.LogEverySteps, "log-metrics", 0, func(l *train.Loop, metrics []*tensors.Tensor) error {
		batchLoss := tensorScalarFloat32(metrics[0])
		maLoss := tensorScalarFloat32(metrics[1])
		medianDur := l.MedianTrainStepDuration()
		log.Info("trainer", "step=%d/%d loss=%.4f moving_avg_loss=%.4f median_step=%s elapsed=%s",
			l.LoopStep, l.EndStep, batchLoss, maLoss, medianDur.Round(time.Millisecond), time.Since(startTime).Round(time.Second))
		return nil
	})

	currentStep := int(optimizer.GetGlobalStep(store))
	targetSteps := cfg.Training.TrainSteps
	if resume && currentStep >= targetSteps {
		log.Info("trainer", "checkpoint already at global_step=%d >= target=%d, skipping training", currentStep, targetSteps)
		return store, nil
	}
	stepsToRun := targetSteps - currentStep
	log.Info("trainer", "starting GoMLX training on backend=%s for %d steps (batch_size=%d, vocab=%d, embed=%d, hidden=%v)",
		backend.Name(), stepsToRun, cfg.Training.BatchSize, vocabLen, cfg.Model.EmbedDim, cfg.Model.HiddenDims)

	metrics, err := loop.RunSteps(trainDS, stepsToRun)
	if err != nil {
		return nil, fmt.Errorf("training loop failed: %w", err)
	}
	finalLoss := tensorScalarFloat32(metrics[0])
	finalMA := tensorScalarFloat32(metrics[1])
	if math.IsNaN(float64(finalLoss)) || math.IsInf(float64(finalLoss), 0) {
		return nil, fmt.Errorf("training produced non-finite final loss %v", finalLoss)
	}

	if err := ckptHandler.Save(); err != nil {
		return nil, fmt.Errorf("save final checkpoint: %w", err)
	}
	log.Info("trainer", "training complete: steps=%d final_loss=%.4f final_ma_loss=%.4f median_step=%s total_time=%s",
		loop.LoopStep, finalLoss, finalMA, loop.MedianTrainStepDuration().Round(time.Millisecond), time.Since(startTime).Round(time.Second))
	return store, nil
}

// Evaluate computes metrics on KLUE-YNAT validation, NSMC test, and weak NFC annotations,
// and returns YNAT validation predictions for downstream article sorting.
func Evaluate(
	backend compute.Backend,
	store *model.Store,
	cfg *config.Config,
	tax *nfc.Taxonomy,
	vocabLen int,
	ynatVal, nsmcTest []data.EncodedExample,
) (*EvalReport, []sorter.ArticlePrediction, error) {
	graphCfg := BuildGraphConfig(cfg, vocabLen, tax.NumTopics(), tax.NumFeelings())
	pred, err := NewPredictor(backend, store, graphCfg, cfg.Training.EvalBatchSize, cfg.Tokenizer.MaxSeqLen)
	if err != nil {
		return nil, nil, err
	}
	defer pred.Close()

	ynatPreds, err := pred.PredictExamples(ynatVal)
	if err != nil {
		return nil, nil, fmt.Errorf("predict YNAT validation: %w", err)
	}
	nsmcPreds, err := pred.PredictExamples(nsmcTest)
	if err != nil {
		return nil, nil, fmt.Errorf("predict NSMC test: %w", err)
	}

	numTopics := tax.NumTopics()
	topicTP := make([]int, numTopics)
	topicFP := make([]int, numTopics)
	topicFN := make([]int, numTopics)
	topicSupport := make([]int, numTopics)
	ynatCorrect := 0
	for i, ex := range ynatVal {
		predID := argMaxFloat32(ynatPreds[i].TopicProbs)
		trueID := ex.Record.TopicID
		if trueID >= 0 && trueID < numTopics {
			topicSupport[trueID]++
			if predID == trueID {
				ynatCorrect++
				topicTP[trueID]++
			} else {
				topicFN[trueID]++
				if predID >= 0 && predID < numTopics {
					topicFP[predID]++
				}
			}
		}
	}

	perTopic := make([]TopicMetric, 0, numTopics)
	var sumTopicF1 float64
	activeTopics := 0
	for _, t := range tax.Topics {
		tp, fp, fn := float64(topicTP[t.ID]), float64(topicFP[t.ID]), float64(topicFN[t.ID])
		prec := safeDiv(tp, tp+fp)
		rec := safeDiv(tp, tp+fn)
		f1 := safeDiv(2*prec*rec, prec+rec)
		if topicSupport[t.ID] > 0 {
			sumTopicF1 += f1
			activeTopics++
		}
		perTopic = append(perTopic, TopicMetric{
			TopicID:   t.ID,
			Code:      t.Code,
			KoName:    t.KoName,
			Support:   topicSupport[t.ID],
			Precision: prec,
			Recall:    rec,
			F1:        f1,
		})
	}

	var nsmcTP, nsmcFP, nsmcTN, nsmcFN int
	for i, ex := range nsmcTest {
		predPos := nsmcPreds[i].SentimentProb >= 0.5
		truePos := ex.Record.SentimentLabel >= 0.5
		switch {
		case predPos && truePos:
			nsmcTP++
		case predPos && !truePos:
			nsmcFP++
		case !predPos && !truePos:
			nsmcTN++
		case !predPos && truePos:
			nsmcFN++
		}
	}
	nsmcAcc := safeDiv(float64(nsmcTP+nsmcTN), float64(len(nsmcTest)))
	nsmcPrec := safeDiv(float64(nsmcTP), float64(nsmcTP+nsmcFP))
	nsmcRec := safeDiv(float64(nsmcTP), float64(nsmcTP+nsmcFN))
	nsmcF1 := safeDiv(2*nsmcPrec*nsmcRec, nsmcPrec+nsmcRec)

	// Evaluate NFC head across YNAT validation + NSMC test weak annotations.
	numNFC := tax.NumFeelings()
	nfcTP := make([]int, numNFC)
	nfcFP := make([]int, numNFC)
	nfcFN := make([]int, numNFC)
	nfcSupport := make([]int, numNFC)
	nfcThreshold := float32(cfg.InferenceAndSorting.NFCActiveThreshold)
	var totalTP, totalFP, totalFN int
	var recallAt3Hits, activeEvalCount, supervisedEvalCount int

	evalNFCSet := func(examples []data.EncodedExample, preds []sorter.ArticlePrediction) {
		for i, ex := range examples {
			if ex.NFCMask <= 0 && ex.NFCMatched == 0 {
				continue
			}
			supervisedEvalCount++
			probs := preds[i].NFCProbs
			for c := 0; c < numNFC; c++ {
				predOn := probs[c] >= nfcThreshold
				trueOn := ex.NFCTargets[c] >= 0.5
				if trueOn {
					nfcSupport[c]++
				}
				switch {
				case predOn && trueOn:
					nfcTP[c]++
					totalTP++
				case predOn && !trueOn:
					nfcFP[c]++
					totalFP++
				case !predOn && trueOn:
					nfcFN[c]++
					totalFN++
				}
			}
			if ex.NFCMatched > 0 {
				activeEvalCount++
				top3 := topKIndices(probs, 3)
				hit := false
				for _, idx := range top3 {
					if ex.NFCTargets[idx] >= 0.5 {
						hit = true
						break
					}
				}
				if hit {
					recallAt3Hits++
				}
			}
		}
	}
	evalNFCSet(ynatVal, ynatPreds)
	evalNFCSet(nsmcTest, nsmcPreds)

	var nfcSumF1 float64
	nfcClassesWithSupport := 0
	nfcCovered := 0
	for c := 0; c < numNFC; c++ {
		if nfcTP[c] > 0 {
			nfcCovered++
		}
		if nfcSupport[c] > 0 {
			tp, fp, fn := float64(nfcTP[c]), float64(nfcFP[c]), float64(nfcFN[c])
			p := safeDiv(tp, tp+fp)
			r := safeDiv(tp, tp+fn)
			nfcSumF1 += safeDiv(2*p*r, p+r)
			nfcClassesWithSupport++
		}
	}
	microPrec := safeDiv(float64(totalTP), float64(totalTP+totalFP))
	microRec := safeDiv(float64(totalTP), float64(totalTP+totalFN))
	microF1 := safeDiv(2*microPrec*microRec, microPrec+microRec)

	report := &EvalReport{
		YNATCount:              len(ynatVal),
		YNATAccuracy:           safeDiv(float64(ynatCorrect), float64(len(ynatVal))),
		YNATMacroF1:            safeDiv(sumTopicF1, float64(activeTopics)),
		YNATPerTopic:           perTopic,
		NSMCCount:              len(nsmcTest),
		NSMCAccuracy:           nsmcAcc,
		NSMCPrecision:          nsmcPrec,
		NSMCRecall:             nsmcRec,
		NSMCF1:                 nsmcF1,
		NFCActiveEvalCount:     activeEvalCount,
		NFCSupervisedEvalCount: supervisedEvalCount,
		NFCMicroF1:             microF1,
		NFCMacroF1:             safeDiv(nfcSumF1, float64(nfcClassesWithSupport)),
		NFCRecallAt3:           safeDiv(float64(recallAt3Hits), float64(activeEvalCount)),
		NFCCoveredClasses:      nfcCovered,
	}
	return report, ynatPreds, nil
}

func tensorScalarFloat32(t *tensors.Tensor) float32 {
	if t == nil {
		return 0
	}
	vals := tensors.MustCopyFlatData[float32](t)
	if len(vals) == 0 {
		return 0
	}
	return vals[0]
}

func argMaxFloat32(vals []float32) int {
	if len(vals) == 0 {
		return -1
	}
	bestIdx := 0
	bestVal := vals[0]
	for i := 1; i < len(vals); i++ {
		if vals[i] > bestVal {
			bestVal = vals[i]
			bestIdx = i
		}
	}
	return bestIdx
}

func topKIndices(vals []float32, k int) []int {
	if k > len(vals) {
		k = len(vals)
	}
	used := make([]bool, len(vals))
	out := make([]int, 0, k)
	for step := 0; step < k; step++ {
		bestIdx := -1
		bestVal := float32(-1e30)
		for i, v := range vals {
			if !used[i] && v > bestVal {
				bestVal = v
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		used[bestIdx] = true
		out = append(out, bestIdx)
	}
	return out
}

func safeDiv(num, den float64) float64 {
	if den == 0 {
		return 0
	}
	return num / den
}
