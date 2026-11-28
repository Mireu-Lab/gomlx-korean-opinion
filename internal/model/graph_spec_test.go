package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDetectProviderFromCheckpointJSON(t *testing.T) {
	dir := t.TempDir()
	raw := []byte(`{"Variables":[{"ParameterName":"var:/relu_nlp/cnn_proj/dense/weights"},{"ParameterName":"var:/ko_hub_student/token_embed/embeddings"}]}`)
	if err := os.WriteFile(filepath.Join(dir, "checkpoint-n1.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	got, ok := DetectProviderFromCheckpointDir(dir)
	if !ok || got != "ko_hub_cnn_relu" {
		t.Fatalf("got %q ok=%t, want ko_hub_cnn_relu", got, ok)
	}
}

func TestSaveLoadRunSpec(t *testing.T) {
	dir := t.TempDir()
	spec := RunSpec{
		Graph:         GraphConfig{VocabSize: 100, EmbedDim: 32, HiddenDims: []int{64}, HeadDim: 16, EmbeddingProvider: "scratch_relu"},
		MaxSeqLen:     16,
		WordSeqLen:    8,
		VocabFilename: "vocab.json",
		EvalBatchSize: 4,
		Provider:      "scratch_relu",
	}
	if err := SaveRunSpec(dir, spec); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadRunSpec(dir)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Provider != "scratch_relu" || loaded.Graph.VocabSize != 100 {
		t.Fatalf("unexpected spec: %+v", loaded)
	}
}
