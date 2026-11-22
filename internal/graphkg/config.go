package graphkg

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// RootsConfig externalizes all URLs, cache paths, limits, edge weights, pretraining settings,
// and curated homonym/NFC seeds for the KO-Hub multilingual roots graph.
type RootsConfig struct {
	CacheDir       string            `yaml:"cache_dir"`
	GraphFile      string            `yaml:"graph_file"`
	HTTPTimeoutSec int               `yaml:"http_timeout_sec"`
	HTTPRetries    int               `yaml:"http_retries"`
	Sources        SourcesConfig     `yaml:"sources"`
	Limits         LimitsConfig      `yaml:"limits"`
	EdgeWeights    EdgeWeightsConfig `yaml:"edge_weights"`
	Pretrain       PretrainConfig    `yaml:"pretrain"`
	PolysemySeeds  []PolysemySeed    `yaml:"polysemy_seeds"`
}

// SourcesConfig stores remote URLs and local filenames for Unihan and OMW multilingual resources.
type SourcesConfig struct {
	UnihanZipURL   string `yaml:"unihan_zip_url"`
	UnihanZipFile  string `yaml:"unihan_zip_file"`
	OMWKorWiktURL  string `yaml:"omw_kor_wikt_url"`
	OMWKorWiktFile string `yaml:"omw_kor_wikt_file"`
	OMWKorCLDRURL  string `yaml:"omw_kor_cldr_url"`
	OMWKorCLDRFile string `yaml:"omw_kor_cldr_file"`
	OMWEngURL      string `yaml:"omw_eng_url"`
	OMWEngFile     string `yaml:"omw_eng_file"`
	OMWJpnURL      string `yaml:"omw_jpn_url"`
	OMWJpnFile     string `yaml:"omw_jpn_file"`
	OMWCmnURL      string `yaml:"omw_cmn_url"`
	OMWCmnFile     string `yaml:"omw_cmn_file"`
}

// LimitsConfig bounds graph size and memory consumption for local CPU runs.
type LimitsConfig struct {
	MinKoFreq           int     `yaml:"min_ko_freq"`
	MaxKoSurfaceNodes   int     `yaml:"max_ko_surface_nodes"`
	MaxHanjaPerSyllable int     `yaml:"max_hanja_per_syllable"`
	MaxNeighborsPerNode int     `yaml:"max_neighbors_per_node"`
	MaxCandidateSenses  int     `yaml:"max_candidate_senses"`
	MinEdgeConfidence   float64 `yaml:"min_edge_confidence"`
	CooccurWindow       int     `yaml:"cooccur_window"`
	MinCooccurCount     int     `yaml:"min_cooccur_count"`
	MaxCorpusDocs       int     `yaml:"max_corpus_docs"`
}

// EdgeWeightsConfig assigns weights to each heterogeneous relation type.
type EdgeWeightsConfig struct {
	HasCandidateSense float32 `yaml:"has_candidate_sense"`
	ComposedOfHanja   float32 `yaml:"composed_of_hanja"`
	AlignedEN         float32 `yaml:"aligned_en"`
	AlignedJA         float32 `yaml:"aligned_ja"`
	AlignedZH         float32 `yaml:"aligned_zh"`
	CooccursKO        float32 `yaml:"cooccurs_ko"`
	NFCAnchor         float32 `yaml:"nfc_anchor"`
}

// PretrainConfig configures GoMLX contrastive pretraining over the KO-Hub graph.
type PretrainConfig struct {
	CheckpointDir   string              `yaml:"checkpoint_dir"`
	EmbedDim        int                 `yaml:"embed_dim"`
	OutDim          int                 `yaml:"out_dim"`
	RootEmbedDim    int                 `yaml:"root_embed_dim"`
	AttnDim         int                 `yaml:"attn_dim"`
	MaxRootsPerView int                 `yaml:"max_roots_per_view"`
	BatchSize       int                 `yaml:"batch_size"`
	PretrainSteps   int                 `yaml:"pretrain_steps"`
	LogEverySteps   int                 `yaml:"log_every_steps"`
	CheckpointKeep  int                 `yaml:"checkpoint_keep"`
	Optimizer       string              `yaml:"optimizer"`
	LearningRate    float64             `yaml:"learning_rate"`
	MinLearningRate float64             `yaml:"min_learning_rate"`
	WeightDecay     float64             `yaml:"weight_decay"`
	DropoutRate     float64             `yaml:"dropout_rate"`
	Temperature     float64             `yaml:"temperature"`
	DisambigMargin  float64             `yaml:"disambig_margin"`
	MaskEpsilon     float64             `yaml:"mask_epsilon"`
	LossWeights     PretrainLossWeights `yaml:"loss_weights"`
}

// PretrainLossWeights weights each loss term in the multi-view contrastive objective.
type PretrainLossWeights struct {
	KoEN     float64 `yaml:"ko_en"`
	KoHanja  float64 `yaml:"ko_hanja"`
	KoJaZh   float64 `yaml:"ko_jazh"`
	Disambig float64 `yaml:"disambig"`
	Distill  float64 `yaml:"distill"`
}

// PolysemySeed declares curated Korean surface words with disambiguated multilingual senses.
type PolysemySeed struct {
	Surface string      `yaml:"surface"`
	Senses  []SenseSeed `yaml:"senses"`
}

// SenseSeed declares one sense of a Korean word with multilingual root links and context cues.
type SenseSeed struct {
	SenseKey    string   `yaml:"sense_key"`
	Gloss       string   `yaml:"gloss"`
	Hanja       []string `yaml:"hanja"`
	EN          []string `yaml:"en"`
	JA          []string `yaml:"ja"`
	ZH          []string `yaml:"zh"`
	ContextCues []string `yaml:"context_cues"`
	NFCAnchor   string   `yaml:"nfc_anchor,omitempty"`
}

// LoadRootsConfig reads, applies environment overrides, and validates multilingual_roots.yaml.
func LoadRootsConfig(path string) (*RootsConfig, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read roots config %q: %w", path, err)
	}
	var cfg RootsConfig
	if err := yaml.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("parse roots config %q: %w", path, err)
	}
	if err := applyRootsEnvOverrides(&cfg); err != nil {
		return nil, err
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func applyRootsEnvOverrides(cfg *RootsConfig) error {
	if v := strings.TrimSpace(os.Getenv("KO_HUB_CACHE_DIR")); v != "" {
		cfg.CacheDir = v
	}
	if v := strings.TrimSpace(os.Getenv("KO_HUB_GRAPH_FILE")); v != "" {
		cfg.GraphFile = v
	}
	if v := strings.TrimSpace(os.Getenv("KO_HUB_PRETRAIN_DIR")); v != "" {
		cfg.Pretrain.CheckpointDir = v
	}
	if v := strings.TrimSpace(os.Getenv("KO_HUB_PRETRAIN_STEPS")); v != "" {
		steps, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("invalid KO_HUB_PRETRAIN_STEPS=%q: %w", v, err)
		}
		cfg.Pretrain.PretrainSteps = steps
	}
	if v := strings.TrimSpace(os.Getenv("KO_HUB_MAX_DOCS")); v != "" {
		maxDocs, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("invalid KO_HUB_MAX_DOCS=%q: %w", v, err)
		}
		cfg.Limits.MaxCorpusDocs = maxDocs
	}
	return nil
}

// Validate checks all fields of RootsConfig to prevent invalid or partial runs.
func (c *RootsConfig) Validate() error {
	if strings.TrimSpace(c.CacheDir) == "" || strings.TrimSpace(c.GraphFile) == "" {
		return fmt.Errorf("cache_dir and graph_file must not be empty")
	}
	if c.HTTPTimeoutSec <= 0 || c.HTTPRetries < 1 {
		return fmt.Errorf("http_timeout_sec must be > 0 and http_retries >= 1")
	}
	requiredSources := map[string]string{
		"sources.unihan_zip_url":    c.Sources.UnihanZipURL,
		"sources.unihan_zip_file":   c.Sources.UnihanZipFile,
		"sources.omw_kor_wikt_url":  c.Sources.OMWKorWiktURL,
		"sources.omw_kor_wikt_file": c.Sources.OMWKorWiktFile,
		"sources.omw_kor_cldr_url":  c.Sources.OMWKorCLDRURL,
		"sources.omw_kor_cldr_file": c.Sources.OMWKorCLDRFile,
		"sources.omw_eng_url":       c.Sources.OMWEngURL,
		"sources.omw_eng_file":      c.Sources.OMWEngFile,
		"sources.omw_jpn_url":       c.Sources.OMWJpnURL,
		"sources.omw_jpn_file":      c.Sources.OMWJpnFile,
		"sources.omw_cmn_url":       c.Sources.OMWCmnURL,
		"sources.omw_cmn_file":      c.Sources.OMWCmnFile,
	}
	for k, v := range requiredSources {
		if strings.TrimSpace(v) == "" {
			return fmt.Errorf("%s must not be empty", k)
		}
	}
	if c.Limits.MinKoFreq < 1 || c.Limits.MaxKoSurfaceNodes < 10 || c.Limits.MaxHanjaPerSyllable < 1 ||
		c.Limits.MaxNeighborsPerNode < 1 || c.Limits.MaxCandidateSenses < 1 || c.Limits.CooccurWindow < 1 ||
		c.Limits.MinCooccurCount < 1 || c.Limits.MaxCorpusDocs < 1 {
		return fmt.Errorf("limits fields must be positive")
	}
	if c.Limits.MinEdgeConfidence < 0 || c.Limits.MinEdgeConfidence > 1 {
		return fmt.Errorf("limits.min_edge_confidence must be in [0, 1]")
	}
	if c.EdgeWeights.HasCandidateSense <= 0 || c.EdgeWeights.ComposedOfHanja <= 0 ||
		c.EdgeWeights.AlignedEN <= 0 || c.EdgeWeights.AlignedJA <= 0 ||
		c.EdgeWeights.AlignedZH <= 0 || c.EdgeWeights.CooccursKO <= 0 || c.EdgeWeights.NFCAnchor <= 0 {
		return fmt.Errorf("all edge_weights must be > 0")
	}
	p := c.Pretrain
	if strings.TrimSpace(p.CheckpointDir) == "" {
		return fmt.Errorf("pretrain.checkpoint_dir must not be empty")
	}
	if p.EmbedDim < 4 || p.OutDim < 4 || p.RootEmbedDim < 4 || p.AttnDim < 4 || p.MaxRootsPerView < 1 {
		return fmt.Errorf("pretrain dimensions and max_roots_per_view must be valid positive integers")
	}
	if p.BatchSize < 2 || p.PretrainSteps < 1 || p.LogEverySteps < 1 || p.CheckpointKeep < 1 {
		return fmt.Errorf("pretrain batch_size >= 2 and step/keep settings >= 1 required")
	}
	if p.LearningRate <= 0 || p.MinLearningRate < 0 || p.WeightDecay < 0 || p.Temperature <= 0 || p.MaskEpsilon <= 0 {
		return fmt.Errorf("pretrain optimizer/loss parameters invalid")
	}
	if p.DropoutRate < 0 || p.DropoutRate >= 1 {
		return fmt.Errorf("pretrain.dropout_rate must be in [0, 1)")
	}
	if p.LossWeights.KoEN < 0 || p.LossWeights.KoHanja < 0 || p.LossWeights.KoJaZh < 0 ||
		p.LossWeights.Disambig < 0 || p.LossWeights.Distill < 0 {
		return fmt.Errorf("pretrain.loss_weights must be non-negative")
	}
	return nil
}
