package data

import (
	"math/rand"
	"path/filepath"
	"strings"
	"testing"

	"github.com/parquet-go/parquet-go"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/nfc"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/tokenizer"
)

func TestExportPreprocessedSplitRoundTrip(t *testing.T) {
	tax, err := nfc.LoadTaxonomy(
		filepath.Join("..", "..", "config", "topics.yaml"),
		filepath.Join("..", "..", "config", "nfc_taxonomy.yaml"),
	)
	if err != nil {
		t.Fatalf("LoadTaxonomy: %v", err)
	}

	tokCfg := config.TokenizerConfig{
		VocabSize:            256,
		MaxSeqLen:            16,
		WordSeqLen:           10,
		MinFreq:              1,
		UseCharBigrams:       true,
		UseCharTrigrams:      true,
		SplitParticles:       true,
		ParticleMinStemRunes: 2,
		KoreanParticles:      []string{"가", "을", "에서"},
		PadToken:             "<PAD>",
		UnkToken:             "<UNK>",
	}
	recs := []Record{
		{
			ID:       "ynat_1",
			Text:     "남춘모 부조적 단색화가 미술계에서 주목받는다",
			Source:   "klue_ynat",
			Date:     "2026-11-22",
			TopicID:  3,
			HasTopic: true,
		},
		{
			ID:             "nsmc_1",
			Text:           "혼자 남겨진 외로움과 슬픔에 너무 화가 난다",
			Source:         "nsmc",
			TopicID:        -1,
			SentimentLabel: 0,
			HasSentiment:   true,
		},
	}
	corpus := []string{recs[0].Text, recs[1].Text}
	tok := tokenizer.Fit(corpus, tokCfg)
	examples, prepStats := PrepareExamples(recs, tok, tax, 0.35, rand.New(rand.NewSource(42)))

	if prepStats.GuardHits != 1 {
		t.Fatalf("expected 1 homonym guard hit, got %d", prepStats.GuardHits)
	}
	if !examples[0].GuardTriggered {
		t.Fatalf("expected examples[0].GuardTriggered=true")
	}

	outPath := filepath.Join(t.TempDir(), "data", "sample-00000-of-00001.parquet")
	exportStats, err := ExportPreprocessedSplit(outPath, "validation", examples, tok, tax)
	if err != nil {
		t.Fatalf("ExportPreprocessedSplit: %v", err)
	}
	if exportStats.Rows != 2 || exportStats.HomonymGuardHits != 1 {
		t.Fatalf("unexpected exportStats: %+v", exportStats)
	}

	rows, err := parquet.ReadFile[PreprocessedRow](outPath)
	if err != nil {
		t.Fatalf("parquet.ReadFile: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("expected 2 parquet rows, got %d", len(rows))
	}

	row0 := rows[0]
	if !row0.NFCHomonymGuardHit {
		t.Fatalf("expected row0 NFCHomonymGuardHit=true")
	}
	if row0.NFCSupervisedMask != 1.0 {
		t.Fatalf("expected guard-calibrated row0 NFCSupervisedMask=1.0, got %v", row0.NFCSupervisedMask)
	}
	if strings.Contains(row0.NFCActiveLabels, "화") {
		t.Fatalf("expected '화' suppressed for 단색화가, got %q", row0.NFCActiveLabels)
	}
	if !strings.Contains(row0.WordTokens, "w:단색화 p:가") {
		t.Fatalf("expected particle-split token 'w:단색화 p:가' in WordTokens, got %q", row0.WordTokens)
	}
	if row0.TopicLabel != "생활문화" {
		t.Fatalf("expected TopicLabel=생활문화, got %q", row0.TopicLabel)
	}

	row1 := rows[1]
	if !strings.Contains(row1.NFCActiveLabels, "외로움") || !strings.Contains(row1.NFCActiveLabels, "화") {
		t.Fatalf("expected '외로움' and '화' active in row1, got %q", row1.NFCActiveLabels)
	}
}
