package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
)

const GraphSpecFilename = "graph_config.json"

// RunSpec records the exact graph + tokenizer sizes used when a checkpoint was written.
type RunSpec struct {
	Graph         GraphConfig `json:"graph"`
	MaxSeqLen     int         `json:"max_seq_len"`
	WordSeqLen    int         `json:"word_seq_len"`
	VocabFilename string      `json:"vocab_filename"`
	EvalBatchSize int         `json:"eval_batch_size"`
	Provider      string      `json:"provider"`
}

// SaveRunSpec writes graph_config.json next to checkpoint weights.
func SaveRunSpec(dir string, spec RunSpec) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, GraphSpecFilename), append(raw, '\n'), 0o644)
}

func RunSpecFromConfig(cfg *config.Config, vocabLen, numTopics, numNFC int) RunSpec {
	g := BuildGraphConfig(cfg, vocabLen, numTopics, numNFC)
	return RunSpec{
		Graph:         g,
		MaxSeqLen:     cfg.Tokenizer.MaxSeqLen,
		WordSeqLen:    cfg.Tokenizer.WordSeqLen,
		VocabFilename: cfg.Tokenizer.VocabFilename,
		EvalBatchSize: cfg.Training.EvalBatchSize,
		Provider:      cfg.Embedding.Provider,
	}
}

// LoadRunSpec reads graph_config.json if present.
func LoadRunSpec(dir string) (*RunSpec, error) {
	raw, err := os.ReadFile(filepath.Join(dir, GraphSpecFilename))
	if err != nil {
		return nil, err
	}
	var spec RunSpec
	if err := json.Unmarshal(raw, &spec); err != nil {
		return nil, fmt.Errorf("parse %s: %w", GraphSpecFilename, err)
	}
	if spec.Provider == "" {
		spec.Provider = spec.Graph.EmbeddingProvider
	}
	if spec.Graph.EmbeddingProvider == "" {
		spec.Graph.EmbeddingProvider = spec.Provider
	}
	return &spec, nil
}

// ResolveRunSpec prefers graph_config.json, otherwise uses YAML and infers provider from checkpoint variable names.
func ResolveRunSpec(dir string, cfg *config.Config, vocabLen, numTopics, numNFC int) (RunSpec, error) {
	if spec, err := LoadRunSpec(dir); err == nil {
		if spec.Graph.VocabSize == 0 {
			spec.Graph.VocabSize = vocabLen
		}
		if spec.Graph.NumTopics == 0 {
			spec.Graph.NumTopics = numTopics
		}
		if spec.Graph.NumNFC == 0 {
			spec.Graph.NumNFC = numNFC
		}
		return *spec, nil
	}
	spec := RunSpecFromConfig(cfg, vocabLen, numTopics, numNFC)
	if p, ok := DetectProviderFromCheckpointDir(dir); ok {
		spec.Provider = p
		spec.Graph.EmbeddingProvider = p
	}
	return spec, nil
}

type checkpointIndex struct {
	Variables []struct {
		ParameterName string `json:"ParameterName"`
	} `json:"Variables"`
}

// DetectProviderFromCheckpointDir inspects the newest checkpoint-*.json for encoder variable paths.
func DetectProviderFromCheckpointDir(dir string) (string, bool) {
	matches, err := filepath.Glob(filepath.Join(dir, "checkpoint-*.json"))
	if err != nil || len(matches) == 0 {
		return "", false
	}
	newest := matches[0]
	for _, m := range matches[1:] {
		if m > newest {
			newest = m
		}
	}
	raw, err := os.ReadFile(newest)
	if err != nil {
		return "", false
	}
	var idx checkpointIndex
	if err := json.Unmarshal(raw, &idx); err != nil {
		return "", false
	}
	joined := ""
	for _, v := range idx.Variables {
		joined += v.ParameterName + "\n"
	}
	switch {
	case strings.Contains(joined, "/relu_nlp/cnn_"):
		return "ko_hub_cnn_relu", true
	case strings.Contains(joined, "/relu_nlp/transformer") || strings.Contains(joined, "/relu_nlp/mhsa"):
		return "ko_hub_transformer_relu", true
	case strings.Contains(joined, "/relu_nlp/lstm") || strings.Contains(joined, "/relu_nlp/bilstm"):
		return "ko_hub_lstm_relu", true
	case strings.Contains(joined, "/ko_hub_student/"):
		return "ko_hub_graph", true
	case strings.Contains(joined, "/relu_nlp/"):
		return "scratch_relu", true
	default:
		return "", false
	}
}
