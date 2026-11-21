package sorter

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/data"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/nfc"
)

func TestScoreSortAndWriteOutputs(t *testing.T) {
	tax, err := nfc.LoadTaxonomy(
		filepath.Join("..", "..", "config", "topics.yaml"),
		filepath.Join("..", "..", "config", "nfc_taxonomy.yaml"),
	)
	if err != nil {
		t.Fatalf("LoadTaxonomy: %v", err)
	}
	cfg := config.InferenceAndSortingConfig{
		DefaultSortMode:           "topic_nfc_intensity",
		NeuralNFCWeight:           0.85,
		LexicalNFCWeight:          0.15,
		NFCActiveThreshold:        0.35,
		SentimentPosThreshold:     0.60,
		SentimentNegThreshold:     0.40,
		IntensityWeights:          config.IntensityWeightsConfig{MaxNFC: 0.55, MeanTop3NFC: 0.25, NegativeSentiment: 0.20},
		TopKNFC:                   3,
		ReportTopArticlesPerGroup: 3,
		ReportTopGlobalArticles:   5,
	}

	mkPred := func(topicID int, sentPos float32, nfcID int, nfcProb float32) ArticlePrediction {
		tp := make([]float32, 7)
		tp[topicID] = 0.9
		np := make([]float32, 16)
		np[nfcID] = nfcProb
		return ArticlePrediction{TopicProbs: tp, SentimentProb: sentPos, NFCProbs: np}
	}

	records := []data.Record{
		{ID: "doc-1", Text: "정치 갈등과 적대적 공방 격화", Source: "ynat"},
		{ID: "doc-2", Text: "고물가 부담과 가계 빚 압박 심화", Source: "ynat"},
		{ID: "doc-3", Text: "경제 불확실성 속 급락 불안", Source: "ynat"},
	}
	preds := []ArticlePrediction{
		mkPred(6, 0.15, 15, 0.88), // politics, hostility
		mkPred(1, 0.20, 0, 0.80),  // economy, burden
		mkPred(1, 0.10, 12, 0.92), // economy, anxiety
	}

	scored, err := ScoreArticles(records, preds, tax, cfg)
	if err != nil {
		t.Fatalf("ScoreArticles: %v", err)
	}
	if scored[0].PrimaryNFC.Code != "hostility" || scored[1].PrimaryNFC.Code != "burden" || scored[2].PrimaryNFC.Code != "anxiety" {
		t.Fatalf("unexpected primary NFC assignments: %+v", scored)
	}

	if err := SortArticles(scored, "topic_nfc_intensity", tax); err != nil {
		t.Fatalf("SortArticles topic_nfc_intensity: %v", err)
	}
	// Economy (order 1) before Politics (order 6), and burden (0) before anxiety (12).
	if scored[0].ID != "doc-2" || scored[1].ID != "doc-3" || scored[2].ID != "doc-1" {
		t.Fatalf("unexpected topic_nfc_intensity order: %s, %s, %s", scored[0].ID, scored[1].ID, scored[2].ID)
	}

	if err := SortArticles(scored, "intensity_desc", tax); err != nil {
		t.Fatalf("SortArticles intensity_desc: %v", err)
	}
	if scored[0].SocialOpinionIntensity < scored[1].SocialOpinionIntensity {
		t.Fatalf("expected descending intensity")
	}

	report := BuildReport(scored, tax, "intensity_desc", cfg.NFCActiveThreshold, map[string]any{"ok": true})
	outDir := t.TempDir()
	if err := WriteOutputs(report, outDir, cfg, tax); err != nil {
		t.Fatalf("WriteOutputs: %v", err)
	}
	for _, fname := range []string{"articles_sorted.json", "articles_sorted.csv", "social_opinion_report.md"} {
		st, err := os.Stat(filepath.Join(outDir, fname))
		if err != nil || st.Size() == 0 {
			t.Fatalf("expected non-empty output %s (err=%v)", fname, err)
		}
	}
}
