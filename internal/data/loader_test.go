package data

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/nfc"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/tokenizer"
)

func TestLoadNSMCAndCustomArticlesAndBatching(t *testing.T) {
	tmp := t.TempDir()
	nsmcPath := filepath.Join(tmp, "nsmc.txt")
	nsmcContent := "id\tdocument\tlabel\n1\t정말 화가 나고 짜증나네요\t0\n2\t최고의 감동 명작입니다\t1\n3\t\t0\nbadrow\n"
	if err := os.WriteFile(nsmcPath, []byte(nsmcContent), 0644); err != nil {
		t.Fatalf("write nsmc: %v", err)
	}
	nsmcRecs, skipped, err := LoadNSMC(nsmcPath)
	if err != nil {
		t.Fatalf("LoadNSMC: %v", err)
	}
	if len(nsmcRecs) != 2 || skipped != 2 {
		t.Fatalf("expected 2 records and 2 skipped, got %d records and %d skipped", len(nsmcRecs), skipped)
	}

	jsonlPath := filepath.Join(tmp, "articles.jsonl")
	jsonlContent := "{\"id\":\"a1\",\"title\":\"고물가 부담 확산\",\"text\":\"서민 우려\",\"date\":\"2026-11-21\"}\n"
	if err := os.WriteFile(jsonlPath, []byte(jsonlContent), 0644); err != nil {
		t.Fatalf("write jsonl: %v", err)
	}
	customRecs, err := LoadCustomArticles(jsonlPath)
	if err != nil || len(customRecs) != 1 {
		t.Fatalf("LoadCustomArticles jsonl: err=%v len=%d", err, len(customRecs))
	}

	tax, err := nfc.LoadTaxonomy(
		filepath.Join("..", "..", "config", "topics.yaml"),
		filepath.Join("..", "..", "config", "nfc_taxonomy.yaml"),
	)
	if err != nil {
		t.Fatalf("LoadTaxonomy: %v", err)
	}
	tokCfg := config.TokenizerConfig{
		VocabSize: 100, MaxSeqLen: 12, MinFreq: 1, UseCharBigrams: true, PadToken: "<PAD>", UnkToken: "<UNK>",
	}
	tok := tokenizer.Fit([]string{nsmcRecs[0].Text, nsmcRecs[1].Text, customRecs[0].Text}, tokCfg)
	rng := rand.New(rand.NewSource(42))
	ynatRecs := []Record{
		{ID: "y1", Text: "경제 위기 불안 우려", Source: "ynat", TopicID: 1, HasTopic: true},
		{ID: "y2", Text: "신형 AI 기술 공개", Source: "ynat", TopicID: 0, HasTopic: true},
	}
	ynatEx, _ := PrepareExamples(ynatRecs, tok, tax, 0.3, rng)
	nsmcEx, _ := PrepareExamples(nsmcRecs, tok, tax, 0.3, rng)

	ds, err := NewMultiTaskTrainDataset("train", ynatEx, nsmcEx, config.TrainingConfig{
		BatchSize:              4,
		YNATBatchRatio:         0.5,
		NSMCBatchRatio:         0.5,
		NFCPositiveSampleRatio: 0.5,
	}, tokCfg.MaxSeqLen, tax.NumFeelings(), 42)
	if err != nil {
		t.Fatalf("NewMultiTaskTrainDataset: %v", err)
	}

	for batch, err := range ds.Iter() {
		if err != nil {
			t.Fatalf("Iter error: %v", err)
		}
		if len(batch.Inputs) != 1 || len(batch.Labels) != 6 {
			t.Fatalf("unexpected batch tensor counts: inputs=%d labels=%d", len(batch.Inputs), len(batch.Labels))
		}
		if err := batch.Finalize(); err != nil {
			t.Fatalf("Finalize failed: %v", err)
		}
		break
	}
}
