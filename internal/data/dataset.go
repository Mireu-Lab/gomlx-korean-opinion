package data

import (
	"fmt"
	"iter"
	"math"
	"math/rand"

	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/train"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/nfc"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/tokenizer"
)

// EncodedExample holds tokenized IDs and multi-task supervision targets/masks for one text item.
type EncodedExample struct {
	Record     Record
	Tokens     []int32
	NFCTargets []float32
	NFCScores  []float32
	NFCMask    float32
	NFCMatched int
}

// PrepStats summarizes weak-supervision NFC coverage on a prepared split.
type PrepStats struct {
	Total             int
	NFCActiveExamples int
	NFCSupervised     int
	PerFeelingHits    []int
}

// PrepareExamples tokenizes records and attaches weak Nuclear Feeling Checklist annotations.
func PrepareExamples(records []Record, tok *tokenizer.Tokenizer, tax *nfc.Taxonomy, negCalibProb float64, rng *rand.Rand) ([]EncodedExample, PrepStats) {
	numNFC := tax.NumFeelings()
	stats := PrepStats{
		Total:          len(records),
		PerFeelingHits: make([]int, numNFC),
	}
	out := make([]EncodedExample, len(records))
	for i, rec := range records {
		tokens := tok.Encode(rec.Text)
		targets, scores, matched, guardTriggered := tax.WeakAnnotateDetailed(rec.Text)
		var nfcMask float32
		if matched > 0 {
			nfcMask = 1.0
			stats.NFCActiveExamples++
			stats.NFCSupervised++
			for c, v := range targets {
				if v > 0 {
					stats.PerFeelingHits[c]++
				}
			}
		} else if guardTriggered {
			// Homonym guard hits provide high-value negative calibration for subword/bigram collisions (e.g., 화가/문화가).
			nfcMask = 1.0
			stats.NFCSupervised++
		} else if negCalibProb > 0 && rng != nil {
			// Calibrate negative NFC examples primarily from positive reviews or a small fraction of neutral headlines.
			if rec.HasSentiment && rec.SentimentLabel >= 0.5 {
				if rng.Float64() < negCalibProb {
					nfcMask = 1.0
					stats.NFCSupervised++
				}
			} else if !rec.HasSentiment && rng.Float64() < negCalibProb*0.15 {
				nfcMask = 1.0
				stats.NFCSupervised++
			}
		}
		out[i] = EncodedExample{
			Record:     rec,
			Tokens:     tokens,
			NFCTargets: targets,
			NFCScores:  scores,
			NFCMask:    nfcMask,
			NFCMatched: matched,
		}
	}
	return out, stats
}

// MultiTaskTrainDataset implements GoMLX train.Dataset with balanced YNAT + NSMC + NFC batching.
type MultiTaskTrainDataset struct {
	name        string
	batchSize   int
	seqLen      int
	numNFC      int
	ynatRatio   float64
	nfcPosRatio float64
	rng         *rand.Rand
	ynatAll     []EncodedExample
	nsmcAll     []EncodedExample
	ynatNFCPos  []EncodedExample
	nsmcNFCPos  []EncodedExample
}

// NewMultiTaskTrainDataset builds a balanced training dataset mixing KLUE-YNAT and NSMC examples.
func NewMultiTaskTrainDataset(
	name string,
	ynatExamples, nsmcExamples []EncodedExample,
	cfg config.TrainingConfig,
	seqLen, numNFC int,
	seed int64,
) (*MultiTaskTrainDataset, error) {
	if len(ynatExamples) == 0 {
		return nil, fmt.Errorf("ynatExamples must not be empty")
	}
	if len(nsmcExamples) == 0 {
		return nil, fmt.Errorf("nsmcExamples must not be empty")
	}
	var ynatPos, nsmcPos []EncodedExample
	for _, ex := range ynatExamples {
		if ex.NFCMatched > 0 {
			ynatPos = append(ynatPos, ex)
		}
	}
	for _, ex := range nsmcExamples {
		if ex.NFCMatched > 0 {
			nsmcPos = append(nsmcPos, ex)
		}
	}
	return &MultiTaskTrainDataset{
		name:        name,
		batchSize:   cfg.BatchSize,
		seqLen:      seqLen,
		numNFC:      numNFC,
		ynatRatio:   cfg.YNATBatchRatio,
		nfcPosRatio: cfg.NFCPositiveSampleRatio,
		rng:         rand.New(rand.NewSource(seed)),
		ynatAll:     ynatExamples,
		nsmcAll:     nsmcExamples,
		ynatNFCPos:  ynatPos,
		nsmcNFCPos:  nsmcPos,
	}, nil
}

// Name returns the dataset identifier for GoMLX Loop reporting.
func (d *MultiTaskTrainDataset) Name() string {
	return d.name
}

// Iter yields multi-task training batches indefinitely until the loop completes its target steps.
func (d *MultiTaskTrainDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		nYNAT := int(math.Round(float64(d.batchSize) * d.ynatRatio))
		if nYNAT < 1 {
			nYNAT = 1
		}
		if nYNAT >= d.batchSize && d.batchSize > 1 {
			nYNAT = d.batchSize - 1
		}
		nNSMC := d.batchSize - nYNAT

		nYNATPos := int(math.Round(float64(nYNAT) * d.nfcPosRatio))
		if len(d.ynatNFCPos) == 0 {
			nYNATPos = 0
		}
		nNSMCPos := int(math.Round(float64(nNSMC) * d.nfcPosRatio))
		if len(d.nsmcNFCPos) == 0 {
			nNSMCPos = 0
		}

		picked := make([]EncodedExample, d.batchSize)
		for {
			idx := 0
			for i := 0; i < nYNATPos; i++ {
				picked[idx] = d.ynatNFCPos[d.rng.Intn(len(d.ynatNFCPos))]
				idx++
			}
			for i := nYNATPos; i < nYNAT; i++ {
				picked[idx] = d.ynatAll[d.rng.Intn(len(d.ynatAll))]
				idx++
			}
			for i := 0; i < nNSMCPos; i++ {
				picked[idx] = d.nsmcNFCPos[d.rng.Intn(len(d.nsmcNFCPos))]
				idx++
			}
			for i := nNSMCPos; i < nNSMC; i++ {
				picked[idx] = d.nsmcAll[d.rng.Intn(len(d.nsmcAll))]
				idx++
			}
			d.rng.Shuffle(len(picked), func(i, j int) {
				picked[i], picked[j] = picked[j], picked[i]
			})

			batch := buildBatch(picked, d.seqLen, d.numNFC, d.name)
			if !yield(batch, nil) {
				return
			}
		}
	}
}

func buildBatch(items []EncodedExample, seqLen, numNFC int, spec any) train.Batch {
	bs := len(items)
	tokensFlat := make([]int32, bs*seqLen)
	topicTarget := make([]int32, bs)
	topicMask := make([]float32, bs)
	sentTarget := make([]float32, bs)
	sentMask := make([]float32, bs)
	nfcTarget := make([]float32, bs*numNFC)
	nfcMask := make([]float32, bs)

	for i, ex := range items {
		copy(tokensFlat[i*seqLen:(i+1)*seqLen], ex.Tokens)
		if ex.Record.HasTopic {
			topicTarget[i] = int32(ex.Record.TopicID)
			topicMask[i] = 1.0
		}
		if ex.Record.HasSentiment {
			sentTarget[i] = ex.Record.SentimentLabel
			sentMask[i] = 1.0
		}
		copy(nfcTarget[i*numNFC:(i+1)*numNFC], ex.NFCTargets)
		nfcMask[i] = ex.NFCMask
	}

	return train.Batch{
		Inputs: []*tensors.Tensor{
			tensors.FromFlatDataAndDimensions(tokensFlat, bs, seqLen),
		},
		Labels: []*tensors.Tensor{
			tensors.FromFlatDataAndDimensions(topicTarget, bs, 1),
			tensors.FromFlatDataAndDimensions(topicMask, bs),
			tensors.FromFlatDataAndDimensions(sentTarget, bs, 1),
			tensors.FromFlatDataAndDimensions(sentMask, bs),
			tensors.FromFlatDataAndDimensions(nfcTarget, bs, numNFC),
			tensors.FromFlatDataAndDimensions(nfcMask, bs),
		},
		Spec: spec,
	}
}
