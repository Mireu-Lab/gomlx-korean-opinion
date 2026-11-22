package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadAndOverrides(t *testing.T) {
	cfgPath := filepath.Join("..", "..", "config", "config.yaml")
	cfg, err := Load(cfgPath)
	if err != nil {
		t.Fatalf("Load(%q) failed: %v", cfgPath, err)
	}
	if cfg.Model.Activation != "relu" {
		t.Fatalf("expected activation relu, got %q", cfg.Model.Activation)
	}
	if cfg.Runtime.Backend != "go" {
		t.Fatalf("expected default backend go, got %q", cfg.Runtime.Backend)
	}
	if cfg.Embedding.Provider != "scratch_relu" {
		t.Fatalf("expected default embedding.provider scratch_relu, got %q", cfg.Embedding.Provider)
	}

	t.Setenv("NFC_BACKEND", "xla:cpu")
	t.Setenv("NFC_TRAIN_STEPS", "42")
	t.Setenv("NFC_SORT_MODE", "intensity_desc")
	t.Setenv("NFC_CLUSTER_MIN_SIZE", "12")
	t.Setenv("NFC_EMBEDDING_PROVIDER", "ko_hub_graph")
	t.Setenv("NFC_FREEZE_ENCODER", "true")
	t.Setenv("NFC_PCA_DIM", "16")
	overridden, err := Load(cfgPath)
	if err != nil {
		t.Fatalf("Load with env overrides failed: %v", err)
	}
	if overridden.Runtime.Backend != "xla:cpu" {
		t.Fatalf("expected NFC_BACKEND override xla:cpu, got %q", overridden.Runtime.Backend)
	}
	if overridden.Training.TrainSteps != 42 {
		t.Fatalf("expected NFC_TRAIN_STEPS override 42, got %d", overridden.Training.TrainSteps)
	}
	if overridden.InferenceAndSorting.DefaultSortMode != "intensity_desc" {
		t.Fatalf("expected NFC_SORT_MODE override intensity_desc, got %q", overridden.InferenceAndSorting.DefaultSortMode)
	}
	if overridden.Clustering.MinClusterSize != 12 {
		t.Fatalf("expected NFC_CLUSTER_MIN_SIZE override 12, got %d", overridden.Clustering.MinClusterSize)
	}
	if overridden.Embedding.Provider != "ko_hub_graph" || !overridden.Embedding.FreezeEncoder || overridden.Embedding.PCADim != 16 {
		t.Fatalf("unexpected embedding overrides: %+v", overridden.Embedding)
	}
	overridden.Embedding.Provider = "invalid_provider"
	if err := overridden.Validate(); err == nil {
		t.Fatalf("expected validation error for invalid embedding.provider")
	}
}

func TestValidateRejectsInvalidActivation(t *testing.T) {
	tmpDir := t.TempDir()
	raw, err := os.ReadFile(filepath.Join("..", "..", "config", "config.yaml"))
	if err != nil {
		t.Fatalf("read base config: %v", err)
	}
	badPath := filepath.Join(tmpDir, "bad.yaml")
	cfg, err := Load(filepath.Join("..", "..", "config", "config.yaml"))
	if err != nil {
		t.Fatalf("load base: %v", err)
	}
	_ = raw
	cfg.Model.Activation = "gelu"
	if err := cfg.Validate(); err == nil {
		t.Fatalf("expected validation error for non-relu activation")
	}
	if err := os.WriteFile(badPath, []byte("runtime:\n  backend: \"\"\n"), 0644); err != nil {
		t.Fatalf("write bad yaml: %v", err)
	}
	if _, err := Load(badPath); err == nil {
		t.Fatalf("expected error on incomplete yaml")
	}
}
