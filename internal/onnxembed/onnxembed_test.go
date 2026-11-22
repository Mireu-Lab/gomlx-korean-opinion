package onnxembed

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWordPieceAndCacheRoundTrip(t *testing.T) {
	tmpDir := t.TempDir()
	vocabPath := filepath.Join(tmpDir, "vocab.txt")
	vocabContent := "[PAD]\n[UNK]\n[CLS]\n[SEP]\n남\n##춘\n##모\n단색\n##화\n##가\n자연\n!\n"
	if err := os.WriteFile(vocabPath, []byte(vocabContent), 0644); err != nil {
		t.Fatalf("write vocab: %v", err)
	}
	wp, err := LoadWordPiece(vocabPath)
	if err != nil {
		t.Fatalf("LoadWordPiece: %v", err)
	}
	if wp.VocabSize() != 12 {
		t.Fatalf("expected 12 vocab entries, got %d", wp.VocabSize())
	}

	ids, mask, tids := wp.Encode("단색화가 자연!", 12)
	if len(ids) != 12 || len(mask) != 12 || len(tids) != 12 {
		t.Fatalf("unexpected encoded slice lengths")
	}
	// Expect [CLS]=2, 단색=7, ##화=8, ##가=9, 자연=10, !=11, [SEP]=3, then PAD=0...
	wantIDs := []int64{2, 7, 8, 9, 10, 11, 3, 0, 0, 0, 0, 0}
	wantMask := []int64{1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0}
	for i := range wantIDs {
		if ids[i] != wantIDs[i] {
			t.Fatalf("id[%d]=%d want=%d (all=%v)", i, ids[i], wantIDs[i], ids)
		}
		if mask[i] != wantMask[i] {
			t.Fatalf("mask[%d]=%d want=%d", i, mask[i], wantMask[i])
		}
	}

	cachePath := filepath.Join(tmpDir, "embeds.bin")
	sample := [][]float32{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}
	if err := SaveEmbedCache(cachePath, sample); err != nil {
		t.Fatalf("SaveEmbedCache: %v", err)
	}
	loaded, err := LoadEmbedCache(cachePath)
	if err != nil {
		t.Fatalf("LoadEmbedCache: %v", err)
	}
	if len(loaded) != 2 || loaded[1][2] != 0.6 {
		t.Fatalf("cache round-trip mismatch: %+v", loaded)
	}
}
