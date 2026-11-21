package tokenizer

import (
	"path/filepath"
	"testing"

	"github.com/Mireu-Lab/gomlx-nfc-opinion/internal/config"
)

func TestFitEncodeSaveLoad(t *testing.T) {
	cfg := config.TokenizerConfig{
		VocabSize:       100,
		MaxSeqLen:       12,
		MinFreq:         1,
		UseCharBigrams:  true,
		UseCharTrigrams: true,
		PadToken:        "<PAD>",
		UnkToken:        "<UNK>",
		VocabFilename:   "vocab.json",
	}
	corpus := []string{
		"고물가 부담과 금리 인상 불안 확산!",
		"경제 위기 우려 속 서민 분노",
		"AI 반도체 혁신 기술 발표",
	}
	tok := Fit(corpus, cfg)
	if tok.VocabLen() <= 2 {
		t.Fatalf("expected non-trivial vocab size, got %d", tok.VocabLen())
	}

	ids := tok.Encode("고물가 부담 불안")
	if len(ids) != cfg.MaxSeqLen {
		t.Fatalf("expected encoded len %d, got %d", cfg.MaxSeqLen, len(ids))
	}
	if ids[0] <= 1 {
		t.Fatalf("expected known token ID > 1 at index 0, got %d", ids[0])
	}

	unkIDs := tok.Encode("존재하지않는희귀어")
	if unkIDs[0] != 1 {
		t.Fatalf("expected UNK token ID 1, got %d", unkIDs[0])
	}

	savePath := filepath.Join(t.TempDir(), "vocab.json")
	if err := tok.Save(savePath); err != nil {
		t.Fatalf("Save failed: %v", err)
	}
	loaded, err := Load(savePath)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	ids2 := loaded.Encode("고물가 부담 불안")
	for i := range ids {
		if ids[i] != ids2[i] {
			t.Fatalf("token mismatch at %d: %d vs %d", i, ids[i], ids2[i])
		}
	}
}
