package sorter

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/data"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/nfc"
)

// ArticlePrediction holds raw neural model probabilities and semantic representation for one article.
type ArticlePrediction struct {
	TopicProbs    []float32
	SentimentProb float32
	NFCProbs      []float32
	SemanticEmbed []float32
}

// ScoredFeeling stores one Nuclear Feeling Checklist item score for an article.
type ScoredFeeling struct {
	ID              int     `json:"id"`
	Code            string  `json:"code"`
	KoName          string  `json:"ko_name"`
	EnName          string  `json:"en_name"`
	FactorGroup     string  `json:"factor_group"`
	NeuralProb      float64 `json:"neural_prob"`
	LexicalScore    float64 `json:"lexical_score"`
	CalibratedScore float64 `json:"calibrated_score"`
}

// ScoredArticle represents an analyzed and classified news article / opinion text.
type ScoredArticle struct {
	ID                     string          `json:"id"`
	Title                  string          `json:"title"`
	Source                 string          `json:"source"`
	Date                   string          `json:"date,omitempty"`
	URL                    string          `json:"url,omitempty"`
	TrueTopicID            int             `json:"true_topic_id"`
	PredTopicID            int             `json:"pred_topic_id"`
	TopicCode              string          `json:"topic_code"`
	TopicKoName            string          `json:"topic_ko_name"`
	TopicEnName            string          `json:"topic_en_name"`
	TopicConfidence        float64         `json:"topic_confidence"`
	TopicProbs             []float64       `json:"topic_probs"`
	SentimentPositiveProb  float64         `json:"sentiment_positive_prob"`
	SentimentNegativeProb  float64         `json:"sentiment_negative_prob"`
	SentimentLabel         string          `json:"sentiment_label"`
	PrimaryNFC             ScoredFeeling   `json:"primary_nfc"`
	TopNFC                 []ScoredFeeling `json:"top_nfc"`
	AllNFCScores           []float64       `json:"all_nfc_scores"`
	NFCActivated           bool            `json:"nfc_activated"`
	SocialOpinionIntensity float64         `json:"social_opinion_intensity"`
}

// TopicStats aggregates distribution and sentiment per KLUE-YNAT topic.
type TopicStats struct {
	TopicID          int     `json:"topic_id"`
	TopicCode        string  `json:"topic_code"`
	TopicKoName      string  `json:"topic_ko_name"`
	Count            int     `json:"count"`
	Share            float64 `json:"share"`
	NegativeRatio    float64 `json:"negative_ratio"`
	MeanIntensity    float64 `json:"mean_intensity"`
	TopFeelingKoName string  `json:"top_feeling_ko_name"`
}

// NFCStats aggregates distribution across each of the 16 Nuclear Feeling Checklist items.
type NFCStats struct {
	FeelingID     int     `json:"feeling_id"`
	Code          string  `json:"code"`
	KoName        string  `json:"ko_name"`
	FactorGroup   string  `json:"factor_group"`
	PrimaryCount  int     `json:"primary_count"`
	ActiveCount   int     `json:"active_count"`
	Share         float64 `json:"share"`
	MeanScore     float64 `json:"mean_score"`
	MeanIntensity float64 `json:"mean_intensity"`
}

// FactorSummary aggregates articles across the 8 validated NFC factor groups.
type FactorSummary struct {
	FactorGroup   string  `json:"factor_group"`
	Count         int     `json:"count"`
	Share         float64 `json:"share"`
	MeanIntensity float64 `json:"mean_intensity"`
}

// MatrixCell holds one Topic x Nuclear Feeling cross-tabulation cell.
type MatrixCell struct {
	TopicID       int     `json:"topic_id"`
	TopicKoName   string  `json:"topic_ko_name"`
	FeelingID     int     `json:"feeling_id"`
	FeelingKoName string  `json:"feeling_ko_name"`
	Count         int     `json:"count"`
	MeanIntensity float64 `json:"mean_intensity"`
	NegativeRatio float64 `json:"negative_ratio"`
}

// AnalysisOutput is the complete structured output written to JSON and Markdown.
type AnalysisOutput struct {
	GeneratedAt   string          `json:"generated_at"`
	SortMode      string          `json:"sort_mode"`
	TotalArticles int             `json:"total_articles"`
	EvalMetrics   any             `json:"eval_metrics,omitempty"`
	TopicSummary  []TopicStats    `json:"topic_summary"`
	NFCSummary    []NFCStats      `json:"nfc_summary"`
	FactorSummary []FactorSummary `json:"factor_summary"`
	Matrix        []MatrixCell    `json:"matrix"`
	Articles      []ScoredArticle `json:"articles"`
}

// ScoreArticles combines GoMLX neural predictions with Nuclear Feeling Checklist calibration and intensity scoring.
func ScoreArticles(
	records []data.Record,
	preds []ArticlePrediction,
	tax *nfc.Taxonomy,
	cfg config.InferenceAndSortingConfig,
) ([]ScoredArticle, error) {
	if len(records) != len(preds) {
		return nil, fmt.Errorf("records length (%d) != predictions length (%d)", len(records), len(preds))
	}
	numTopics := tax.NumTopics()
	numNFC := tax.NumFeelings()
	wSum := cfg.NeuralNFCWeight + cfg.LexicalNFCWeight
	iwSum := cfg.IntensityWeights.MaxNFC + cfg.IntensityWeights.MeanTop3NFC + cfg.IntensityWeights.NegativeSentiment

	out := make([]ScoredArticle, len(records))
	for idx, rec := range records {
		pred := preds[idx]
		if len(pred.TopicProbs) != numTopics {
			return nil, fmt.Errorf("record %d topic probs len %d != %d", idx, len(pred.TopicProbs), numTopics)
		}
		if len(pred.NFCProbs) != numNFC {
			return nil, fmt.Errorf("record %d nfc probs len %d != %d", idx, len(pred.NFCProbs), numNFC)
		}

		bestTopic := 0
		bestTopicProb := float64(-1)
		topicProbs := make([]float64, numTopics)
		for t, p := range pred.TopicProbs {
			fp := float64(p)
			topicProbs[t] = fp
			if fp > bestTopicProb {
				bestTopicProb = fp
				bestTopic = t
			}
		}
		topicDef, _ := tax.TopicByID(bestTopic)

		posProb := float64(pred.SentimentProb)
		if posProb < 0 {
			posProb = 0
		} else if posProb > 1 {
			posProb = 1
		}
		negProb := 1.0 - posProb
		sentLabel := "중립"
		if posProb >= cfg.SentimentPosThreshold {
			sentLabel = "긍정"
		} else if posProb <= cfg.SentimentNegThreshold {
			sentLabel = "부정"
		}

		_, lexScores, _ := tax.WeakAnnotate(rec.Text)
		scoredFeelings := make([]ScoredFeeling, numNFC)
		allNFCScores := make([]float64, numNFC)
		weightedScores := make([]float64, numNFC)

		for fID := 0; fID < numNFC; fID++ {
			fDef := tax.Feelings[fID]
			np := float64(pred.NFCProbs[fID])
			ls := float64(lexScores[fID])
			cal := (cfg.NeuralNFCWeight*np + cfg.LexicalNFCWeight*ls) / wSum
			allNFCScores[fID] = cal
			weightedScores[fID] = cal * fDef.SocialWeight
			scoredFeelings[fID] = ScoredFeeling{
				ID:              fDef.ID,
				Code:            fDef.Code,
				KoName:          fDef.KoName,
				EnName:          fDef.EnName,
				FactorGroup:     fDef.FactorGroup,
				NeuralProb:      np,
				LexicalScore:    ls,
				CalibratedScore: cal,
			}
		}

		sort.Slice(scoredFeelings, func(i, j int) bool {
			if scoredFeelings[i].CalibratedScore != scoredFeelings[j].CalibratedScore {
				return scoredFeelings[i].CalibratedScore > scoredFeelings[j].CalibratedScore
			}
			return scoredFeelings[i].ID < scoredFeelings[j].ID
		})
		sort.Slice(weightedScores, func(i, j int) bool { return weightedScores[i] > weightedScores[j] })

		maxWS := weightedScores[0]
		topN := 3
		if len(weightedScores) < topN {
			topN = len(weightedScores)
		}
		var sumTop3 float64
		for i := 0; i < topN; i++ {
			sumTop3 += weightedScores[i]
		}
		meanTop3WS := sumTop3 / float64(topN)

		intensity := (cfg.IntensityWeights.MaxNFC*maxWS +
			cfg.IntensityWeights.MeanTop3NFC*meanTop3WS +
			cfg.IntensityWeights.NegativeSentiment*negProb) / iwSum

		topK := cfg.TopKNFC
		if topK > len(scoredFeelings) {
			topK = len(scoredFeelings)
		}
		topSlice := make([]ScoredFeeling, topK)
		copy(topSlice, scoredFeelings[:topK])

		trueTopic := -1
		if rec.HasTopic {
			trueTopic = rec.TopicID
		}

		out[idx] = ScoredArticle{
			ID:                     rec.ID,
			Title:                  rec.Text,
			Source:                 rec.Source,
			Date:                   rec.Date,
			URL:                    rec.URL,
			TrueTopicID:            trueTopic,
			PredTopicID:            bestTopic,
			TopicCode:              topicDef.Code,
			TopicKoName:            topicDef.KoName,
			TopicEnName:            topicDef.EnName,
			TopicConfidence:        bestTopicProb,
			TopicProbs:             topicProbs,
			SentimentPositiveProb:  posProb,
			SentimentNegativeProb:  negProb,
			SentimentLabel:         sentLabel,
			PrimaryNFC:             scoredFeelings[0],
			TopNFC:                 topSlice,
			AllNFCScores:           allNFCScores,
			NFCActivated:           scoredFeelings[0].CalibratedScore >= cfg.NFCActiveThreshold,
			SocialOpinionIntensity: intensity,
		}
	}
	return out, nil
}

// SortArticles sorts scored articles deterministically according to the selected sort mode.
func SortArticles(articles []ScoredArticle, mode string, tax *nfc.Taxonomy) error {
	topicOrder := make(map[int]int, tax.NumTopics())
	for _, t := range tax.Topics {
		topicOrder[t.ID] = t.DisplayOrder
	}

	switch mode {
	case "topic_nfc_intensity":
		sort.SliceStable(articles, func(i, j int) bool {
			oi, oj := topicOrder[articles[i].PredTopicID], topicOrder[articles[j].PredTopicID]
			if oi != oj {
				return oi < oj
			}
			if articles[i].PrimaryNFC.ID != articles[j].PrimaryNFC.ID {
				return articles[i].PrimaryNFC.ID < articles[j].PrimaryNFC.ID
			}
			if articles[i].SocialOpinionIntensity != articles[j].SocialOpinionIntensity {
				return articles[i].SocialOpinionIntensity > articles[j].SocialOpinionIntensity
			}
			return articles[i].ID < articles[j].ID
		})
	case "nfc_intensity":
		sort.SliceStable(articles, func(i, j int) bool {
			if articles[i].PrimaryNFC.ID != articles[j].PrimaryNFC.ID {
				return articles[i].PrimaryNFC.ID < articles[j].PrimaryNFC.ID
			}
			if articles[i].SocialOpinionIntensity != articles[j].SocialOpinionIntensity {
				return articles[i].SocialOpinionIntensity > articles[j].SocialOpinionIntensity
			}
			oi, oj := topicOrder[articles[i].PredTopicID], topicOrder[articles[j].PredTopicID]
			if oi != oj {
				return oi < oj
			}
			return articles[i].ID < articles[j].ID
		})
	case "intensity_desc":
		sort.SliceStable(articles, func(i, j int) bool {
			if articles[i].SocialOpinionIntensity != articles[j].SocialOpinionIntensity {
				return articles[i].SocialOpinionIntensity > articles[j].SocialOpinionIntensity
			}
			return articles[i].ID < articles[j].ID
		})
	case "sentiment_neg_desc":
		sort.SliceStable(articles, func(i, j int) bool {
			if articles[i].SentimentNegativeProb != articles[j].SentimentNegativeProb {
				return articles[i].SentimentNegativeProb > articles[j].SentimentNegativeProb
			}
			if articles[i].SocialOpinionIntensity != articles[j].SocialOpinionIntensity {
				return articles[i].SocialOpinionIntensity > articles[j].SocialOpinionIntensity
			}
			return articles[i].ID < articles[j].ID
		})
	default:
		return fmt.Errorf("unsupported sort mode %q", mode)
	}
	return nil
}

// BuildReport constructs Topic, 16-Feeling, 8-Factor, and Topic x Feeling cross-tabulation summaries.
func BuildReport(articles []ScoredArticle, tax *nfc.Taxonomy, sortMode string, nfcActiveThreshold float64, evalSummary any) AnalysisOutput {
	total := len(articles)
	numTopics := tax.NumTopics()
	numNFC := tax.NumFeelings()

	type topicAccum struct {
		count     int
		negCount  int
		sumInt    float64
		nfcCounts []int
	}
	tAcc := make([]topicAccum, numTopics)
	for i := range tAcc {
		tAcc[i].nfcCounts = make([]int, numNFC)
	}

	type nfcAccum struct {
		primaryCount int
		activeCount  int
		sumScore     float64
		sumInt       float64
	}
	nAcc := make([]nfcAccum, numNFC)

	type cellAccum struct {
		count    int
		negCount int
		sumInt   float64
	}
	matrixAcc := make([][]cellAccum, numTopics)
	for t := 0; t < numTopics; t++ {
		matrixAcc[t] = make([]cellAccum, numNFC)
	}

	type factorAccum struct {
		count  int
		sumInt float64
	}
	factorMap := make(map[string]*factorAccum)

	for _, a := range articles {
		tID := a.PredTopicID
		fID := a.PrimaryNFC.ID
		if tID >= 0 && tID < numTopics {
			ta := &tAcc[tID]
			ta.count++
			ta.sumInt += a.SocialOpinionIntensity
			if a.SentimentLabel == "부정" {
				ta.negCount++
			}
			if fID >= 0 && fID < numNFC {
				ta.nfcCounts[fID]++
			}
		}
		if fID >= 0 && fID < numNFC {
			nAcc[fID].primaryCount++
			nAcc[fID].sumInt += a.SocialOpinionIntensity
		}
		for c, sc := range a.AllNFCScores {
			nAcc[c].sumScore += sc
			if sc >= nfcActiveThreshold {
				nAcc[c].activeCount++
			}
		}
		if tID >= 0 && tID < numTopics && fID >= 0 && fID < numNFC {
			cell := &matrixAcc[tID][fID]
			cell.count++
			cell.sumInt += a.SocialOpinionIntensity
			if a.SentimentLabel == "부정" {
				cell.negCount++
			}
		}
		fg := a.PrimaryNFC.FactorGroup
		if _, ok := factorMap[fg]; !ok {
			factorMap[fg] = &factorAccum{}
		}
		factorMap[fg].count++
		factorMap[fg].sumInt += a.SocialOpinionIntensity
	}

	orderedTopics := tax.SortedTopicsByDisplayOrder()
	topicSummary := make([]TopicStats, 0, numTopics)
	for _, t := range orderedTopics {
		acc := tAcc[t.ID]
		topFID := 0
		topFCount := -1
		for fID, cnt := range acc.nfcCounts {
			if cnt > topFCount {
				topFCount = cnt
				topFID = fID
			}
		}
		topFName := "-"
		if acc.count > 0 {
			if fDef, ok := tax.FeelingByID(topFID); ok {
				topFName = fDef.KoName
			}
		}
		var share, negRatio, meanInt float64
		if total > 0 {
			share = float64(acc.count) / float64(total)
		}
		if acc.count > 0 {
			negRatio = float64(acc.negCount) / float64(acc.count)
			meanInt = acc.sumInt / float64(acc.count)
		}
		topicSummary = append(topicSummary, TopicStats{
			TopicID:          t.ID,
			TopicCode:        t.Code,
			TopicKoName:      t.KoName,
			Count:            acc.count,
			Share:            share,
			NegativeRatio:    negRatio,
			MeanIntensity:    meanInt,
			TopFeelingKoName: topFName,
		})
	}

	nfcSummary := make([]NFCStats, 0, numNFC)
	for _, f := range tax.Feelings {
		acc := nAcc[f.ID]
		var share, meanScore, meanInt float64
		if total > 0 {
			share = float64(acc.primaryCount) / float64(total)
			meanScore = acc.sumScore / float64(total)
		}
		if acc.primaryCount > 0 {
			meanInt = acc.sumInt / float64(acc.primaryCount)
		}
		nfcSummary = append(nfcSummary, NFCStats{
			FeelingID:     f.ID,
			Code:          f.Code,
			KoName:        f.KoName,
			FactorGroup:   f.FactorGroup,
			PrimaryCount:  acc.primaryCount,
			ActiveCount:   acc.activeCount,
			Share:         share,
			MeanScore:     meanScore,
			MeanIntensity: meanInt,
		})
	}

	factorSummary := make([]FactorSummary, 0, len(factorMap))
	for fg, acc := range factorMap {
		var share, meanInt float64
		if total > 0 {
			share = float64(acc.count) / float64(total)
		}
		if acc.count > 0 {
			meanInt = acc.sumInt / float64(acc.count)
		}
		factorSummary = append(factorSummary, FactorSummary{
			FactorGroup:   fg,
			Count:         acc.count,
			Share:         share,
			MeanIntensity: meanInt,
		})
	}
	sort.Slice(factorSummary, func(i, j int) bool {
		if factorSummary[i].Count != factorSummary[j].Count {
			return factorSummary[i].Count > factorSummary[j].Count
		}
		return factorSummary[i].FactorGroup < factorSummary[j].FactorGroup
	})

	matrix := make([]MatrixCell, 0, numTopics*numNFC)
	for _, t := range orderedTopics {
		for _, f := range tax.Feelings {
			c := matrixAcc[t.ID][f.ID]
			var meanInt, negRatio float64
			if c.count > 0 {
				meanInt = c.sumInt / float64(c.count)
				negRatio = float64(c.negCount) / float64(c.count)
			}
			matrix = append(matrix, MatrixCell{
				TopicID:       t.ID,
				TopicKoName:   t.KoName,
				FeelingID:     f.ID,
				FeelingKoName: f.KoName,
				Count:         c.count,
				MeanIntensity: meanInt,
				NegativeRatio: negRatio,
			})
		}
	}

	return AnalysisOutput{
		GeneratedAt:   time.Now().UTC().Format(time.RFC3339),
		SortMode:      sortMode,
		TotalArticles: total,
		EvalMetrics:   evalSummary,
		TopicSummary:  topicSummary,
		NFCSummary:    nfcSummary,
		FactorSummary: factorSummary,
		Matrix:        matrix,
		Articles:      articles,
	}
}

// WriteOutputs serializes JSON, CSV, and Korean Markdown report files to outputDir.
func WriteOutputs(out AnalysisOutput, outputDir string, cfg config.InferenceAndSortingConfig, tax *nfc.Taxonomy) error {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("create output dir %q: %w", outputDir, err)
	}
	if err := writeJSON(filepath.Join(outputDir, "articles_sorted.json"), out); err != nil {
		return err
	}
	if err := writeCSV(filepath.Join(outputDir, "articles_sorted.csv"), out.Articles); err != nil {
		return err
	}
	if err := writeMarkdown(filepath.Join(outputDir, "social_opinion_report.md"), out, cfg, tax); err != nil {
		return err
	}
	return nil
}

func writeJSON(path string, out AnalysisOutput) error {
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal output json: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write %q: %w", path, err)
	}
	return nil
}

func writeCSV(path string, articles []ScoredArticle) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create csv %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	w := csv.NewWriter(f)
	header := []string{
		"rank", "id", "topic_id", "topic_ko", "topic_confidence",
		"sentiment_label", "sentiment_pos_prob", "sentiment_neg_prob",
		"primary_nfc_id", "primary_nfc_ko", "primary_nfc_factor", "primary_nfc_score",
		"top_nfc_summary", "nfc_activated", "social_opinion_intensity",
		"date", "url", "title",
	}
	if err := w.Write(header); err != nil {
		return err
	}
	for i, a := range articles {
		topParts := make([]string, 0, len(a.TopNFC))
		for _, nf := range a.TopNFC {
			topParts = append(topParts, fmt.Sprintf("%s:%.3f", nf.KoName, nf.CalibratedScore))
		}
		row := []string{
			fmt.Sprintf("%d", i+1),
			a.ID,
			fmt.Sprintf("%d", a.PredTopicID),
			a.TopicKoName,
			fmt.Sprintf("%.4f", a.TopicConfidence),
			a.SentimentLabel,
			fmt.Sprintf("%.4f", a.SentimentPositiveProb),
			fmt.Sprintf("%.4f", a.SentimentNegativeProb),
			fmt.Sprintf("%d", a.PrimaryNFC.ID),
			a.PrimaryNFC.KoName,
			a.PrimaryNFC.FactorGroup,
			fmt.Sprintf("%.4f", a.PrimaryNFC.CalibratedScore),
			strings.Join(topParts, " | "),
			fmt.Sprintf("%t", a.NFCActivated),
			fmt.Sprintf("%.4f", a.SocialOpinionIntensity),
			a.Date,
			a.URL,
			a.Title,
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

func writeMarkdown(path string, out AnalysisOutput, cfg config.InferenceAndSortingConfig, tax *nfc.Taxonomy) error {
	var b strings.Builder
	b.WriteString("# 사회적 여론 감정 & 핵심감정 체크리스트(Nuclear Feeling Checklist) 기사 분석 보고서\n\n")
	fmt.Fprintf(&b, "- **생성 시각 (UTC)**: %s\n", out.GeneratedAt)
	fmt.Fprintf(&b, "- **정렬 기준 (Sort Mode)**: `%s`\n", out.SortMode)
	fmt.Fprintf(&b, "- **분석 기사 총수**: %d 건\n\n", out.TotalArticles)

	if out.EvalMetrics != nil {
		raw, _ := json.MarshalIndent(out.EvalMetrics, "", "  ")
		b.WriteString("## 1. GoMLX ReLU 멀티태스크 모델 평가 지표\n\n```json\n")
		b.Write(raw)
		b.WriteString("\n```\n\n")
	}

	b.WriteString("## 2. 기사 주제별 사회적 여론 분포 (KLUE-YNAT 7개 주제)\n\n")
	b.WriteString("| 주제 (Topic) | 기사 수 | 비중 | 부정 여론 비율 | 평균 여론 강도 | 최다 핵심감정 |\n")
	b.WriteString("|---|---:|---:|---:|---:|---|\n")
	for _, ts := range out.TopicSummary {
		fmt.Fprintf(&b, "| %s (`%s`) | %d | %.1f%% | %.1f%% | %.3f | %s |\n",
			ts.TopicKoName, ts.TopicCode, ts.Count, ts.Share*100, ts.NegativeRatio*100, ts.MeanIntensity, ts.TopFeelingKoName)
	}
	b.WriteString("\n")

	b.WriteString("## 3. Nuclear Feeling Checklist 16개 핵심감정 분포\n\n")
	b.WriteString("| ID | 핵심감정 | 요인군 (8-Factor) | 주감정 기사 수 | 활성 기사 수 | 비중 | 평균 점수 | 평균 여론 강도 |\n")
	b.WriteString("|---:|---|---|---:|---:|---:|---:|---:|\n")
	for _, ns := range out.NFCSummary {
		fmt.Fprintf(&b, "| %d | **%s** (`%s`) | %s | %d | %d | %.1f%% | %.3f | %.3f |\n",
			ns.FeelingID, ns.KoName, ns.Code, ns.FactorGroup, ns.PrimaryCount, ns.ActiveCount, ns.Share*100, ns.MeanScore, ns.MeanIntensity)
	}
	b.WriteString("\n")

	b.WriteString("### 8개 학술 요인군 요약\n\n")
	b.WriteString("| 요인군 | 기사 수 | 비중 | 평균 여론 강도 |\n")
	b.WriteString("|---|---:|---:|---:|\n")
	for _, fs := range out.FactorSummary {
		fmt.Fprintf(&b, "| %s | %d | %.1f%% | %.3f |\n", fs.FactorGroup, fs.Count, fs.Share*100, fs.MeanIntensity)
	}
	b.WriteString("\n")

	b.WriteString("## 4. 주제(Topic) × 핵심감정(16 NFC) 교차 분포 매트릭스\n\n")
	b.WriteString("| 주제 |")
	for _, f := range tax.Feelings {
		fmt.Fprintf(&b, " %s |", f.KoName)
	}
	b.WriteString("\n|---|")
	for range tax.Feelings {
		b.WriteString("---:|")
	}
	b.WriteString("\n")
	cellMap := make(map[string]MatrixCell, len(out.Matrix))
	for _, c := range out.Matrix {
		cellMap[fmt.Sprintf("%d:%d", c.TopicID, c.FeelingID)] = c
	}
	for _, t := range tax.SortedTopicsByDisplayOrder() {
		fmt.Fprintf(&b, "| **%s** |", t.KoName)
		for _, f := range tax.Feelings {
			c := cellMap[fmt.Sprintf("%d:%d", t.ID, f.ID)]
			fmt.Fprintf(&b, " %d |", c.Count)
		}
		b.WriteString("\n")
	}
	b.WriteString("\n")

	b.WriteString("## 5. 주제별 주요 사회적 여론 핵심감정 기사 (상위 정렬 샘플)\n\n")
	byTopic := make(map[int][]ScoredArticle)
	for _, a := range out.Articles {
		byTopic[a.PredTopicID] = append(byTopic[a.PredTopicID], a)
	}
	for _, t := range tax.SortedTopicsByDisplayOrder() {
		items := byTopic[t.ID]
		if len(items) == 0 {
			continue
		}
		// Highlight highest intensity articles within each topic.
		ranked := make([]ScoredArticle, len(items))
		copy(ranked, items)
		sort.SliceStable(ranked, func(i, j int) bool {
			if ranked[i].SocialOpinionIntensity != ranked[j].SocialOpinionIntensity {
				return ranked[i].SocialOpinionIntensity > ranked[j].SocialOpinionIntensity
			}
			return ranked[i].ID < ranked[j].ID
		})
		limit := cfg.ReportTopArticlesPerGroup
		if limit > len(ranked) {
			limit = len(ranked)
		}
		fmt.Fprintf(&b, "### %s (%d건)\n\n", t.KoName, len(items))
		b.WriteString("| 순위 | 핵심감정 (점수) | 감성 (부정확률) | 여론 강도 | 기사 제목 |\n")
		b.WriteString("|---:|---|---|---:|---|\n")
		for i := 0; i < limit; i++ {
			a := ranked[i]
			fmt.Fprintf(&b, "| %d | %s (%.3f) | %s (%.3f) | %.3f | %s |\n",
				i+1, a.PrimaryNFC.KoName, a.PrimaryNFC.CalibratedScore,
				a.SentimentLabel, a.SentimentNegativeProb, a.SocialOpinionIntensity,
				escapeMD(a.Title))
		}
		b.WriteString("\n")
	}

	b.WriteString("## 6. 전체 사회적 여론 강도 상위 기사\n\n")
	globalCopy := make([]ScoredArticle, len(out.Articles))
	copy(globalCopy, out.Articles)
	sort.SliceStable(globalCopy, func(i, j int) bool {
		if globalCopy[i].SocialOpinionIntensity != globalCopy[j].SocialOpinionIntensity {
			return globalCopy[i].SocialOpinionIntensity > globalCopy[j].SocialOpinionIntensity
		}
		return globalCopy[i].ID < globalCopy[j].ID
	})
	gLimit := cfg.ReportTopGlobalArticles
	if gLimit > len(globalCopy) {
		gLimit = len(globalCopy)
	}
	b.WriteString("| 순위 | 주제 | 주 핵심감정 | 상위 3 핵심감정 | 감성 | 여론 강도 | 기사 제목 |\n")
	b.WriteString("|---:|---|---|---|---|---:|---|\n")
	for i := 0; i < gLimit; i++ {
		a := globalCopy[i]
		topNames := make([]string, 0, len(a.TopNFC))
		for _, nf := range a.TopNFC {
			topNames = append(topNames, fmt.Sprintf("%s(%.2f)", nf.KoName, nf.CalibratedScore))
		}
		fmt.Fprintf(&b, "| %d | %s | %s | %s | %s | %.3f | %s |\n",
			i+1, a.TopicKoName, a.PrimaryNFC.KoName, strings.Join(topNames, ", "),
			a.SentimentLabel, a.SocialOpinionIntensity, escapeMD(a.Title))
	}

	if err := os.WriteFile(path, []byte(b.String()), 0644); err != nil {
		return fmt.Errorf("write markdown report %q: %w", path, err)
	}
	return nil
}

func escapeMD(s string) string {
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "|", "\\|")
	return s
}
