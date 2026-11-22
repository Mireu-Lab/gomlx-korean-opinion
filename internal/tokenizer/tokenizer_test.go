package tokenizer

import (
	"path/filepath"
	"testing"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
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

func TestParticleSplitPreventsCrossBoundaryBigramsAndPreservesOrder(t *testing.T) {
	cfg := config.TokenizerConfig{
		VocabSize:            500,
		MaxSeqLen:            32,
		WordSeqLen:           12,
		MinFreq:              1,
		UseCharBigrams:       true,
		UseCharTrigrams:      true,
		SplitParticles:       true,
		ParticleMinStemRunes: 2,
		KoreanParticles:      []string{"에서", "으로", "와", "가", "는", "을"},
		PadToken:             "<PAD>",
		UnkToken:             "<UNK>",
	}

	artTokens := ExtractTokens("남춘모 부조적 단색화가 펼쳐 보이는 자연", cfg)
	for _, tok := range artTokens {
		if tok == "2:화가" {
			t.Fatalf("particle boundary split failed: cross-boundary bigram 2:화가 should not be generated from 단색화가")
		}
	}
	hasStem := false
	hasPart := false
	for _, tok := range artTokens {
		if tok == "w:단색화" {
			hasStem = true
		}
		if tok == "p:가" {
			hasPart = true
		}
	}
	if !hasStem || !hasPart {
		t.Fatalf("expected w:단색화 and p:가 in tokens %v", artTokens)
	}

	angerTokens := ExtractTokens("시민들 화가 났다", cfg)
	hasAngerWord := false
	hasAngerBigram := false
	for _, tok := range angerTokens {
		if tok == "w:화가" {
			hasAngerWord = true
		}
		if tok == "2:화가" {
			hasAngerBigram = true
		}
	}
	if !hasAngerWord || !hasAngerBigram {
		t.Fatalf("expected bare 2-rune 화가 to keep w:화가 and 2:화가, got %v", angerTokens)
	}

	wordSeq := ExtractWordTokens("단색화가 펼쳐 자연", cfg)
	want := []string{"w:단색화", "p:가", "w:펼쳐", "w:자연"}
	if len(wordSeq) != len(want) {
		t.Fatalf("word seq mismatch: got %v want %v", wordSeq, want)
	}
	for i := range want {
		if wordSeq[i] != want[i] {
			t.Fatalf("word seq[%d]: got %q want %q", i, wordSeq[i], want[i])
		}
	}
}
