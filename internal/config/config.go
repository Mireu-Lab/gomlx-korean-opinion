package config

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// ValidSortModes lists the supported article sorting modes.
var ValidSortModes = []string{
	"topic_nfc_intensity",
	"nfc_intensity",
	"intensity_desc",
	"sentiment_neg_desc",
}

// Config holds the full externalized configuration for the GoMLX NFC pipeline.
type Config struct {
	Runtime             RuntimeConfig             `yaml:"runtime"`
	Paths               PathsConfig               `yaml:"paths"`
	Datasets            DatasetsConfig            `yaml:"datasets"`
	Tokenizer           TokenizerConfig           `yaml:"tokenizer"`
	Model               ModelConfig               `yaml:"model"`
	Embedding           EmbeddingConfig           `yaml:"embedding"`
	Training            TrainingConfig            `yaml:"training"`
	InferenceAndSorting InferenceAndSortingConfig `yaml:"inference_and_sorting"`
	Logging             LoggingConfig             `yaml:"logging"`
	Clustering          ClusteringConfig          `yaml:"clustering"`
}

// RuntimeConfig configures execution backend, RNG seed, and HTTP behavior.
type RuntimeConfig struct {
	Backend        string `yaml:"backend"`
	Seed           int64  `yaml:"seed"`
	HTTPTimeoutSec int    `yaml:"http_timeout_sec"`
	HTTPRetries    int    `yaml:"http_retries"`
}

// PathsConfig holds directories and taxonomy file paths.
type PathsConfig struct {
	DataDir         string `yaml:"data_dir"`
	CheckpointDir   string `yaml:"checkpoint_dir"`
	OutputDir       string `yaml:"output_dir"`
	LogDir          string `yaml:"log_dir"`
	TopicsFile      string `yaml:"topics_file"`
	NFCTaxonomyFile string `yaml:"nfc_taxonomy_file"`
	EntityRulesFile string `yaml:"entity_rules_file"`
}

// DatasetsConfig holds external URLs and local filenames for NSMC and KLUE-YNAT.
type DatasetsConfig struct {
	NSMCTrainURL  string `yaml:"nsmc_train_url"`
	NSMCTestURL   string `yaml:"nsmc_test_url"`
	YNATTrainURL  string `yaml:"ynat_train_url"`
	YNATValURL    string `yaml:"ynat_val_url"`
	NSMCTrainFile string `yaml:"nsmc_train_file"`
	NSMCTestFile  string `yaml:"nsmc_test_file"`
	YNATTrainFile string `yaml:"ynat_train_file"`
	YNATValFile   string `yaml:"ynat_val_file"`
}

// TokenizerConfig configures Korean tokenization and vocabulary bounds.
type TokenizerConfig struct {
	VocabSize       int    `yaml:"vocab_size"`
	MaxSeqLen       int    `yaml:"max_seq_len"`
	MinFreq         int    `yaml:"min_freq"`
	UseCharBigrams  bool   `yaml:"use_char_bigrams"`
	UseCharTrigrams bool   `yaml:"use_char_trigrams"`
	PadToken        string `yaml:"pad_token"`
	UnkToken        string `yaml:"unk_token"`
	VocabFilename   string `yaml:"vocab_filename"`
}

// ValidEmbeddingProviders lists the supported Korean text encoder providers.
var ValidEmbeddingProviders = []string{
	"scratch_relu",
	"ko_hub_graph",
}

// EmbeddingConfig selects between scratch ReLU embedding and pretrained KO-Hub multilingual student encoder.
type EmbeddingConfig struct {
	Provider              string `yaml:"provider"`
	RootsConfigFile       string `yaml:"roots_config_file"`
	PretrainCheckpointDir string `yaml:"pretrain_checkpoint_dir"`
	FreezeEncoder         bool   `yaml:"freeze_encoder"`
	StudentOutDim         int    `yaml:"student_out_dim"`
	PCADim                int    `yaml:"pca_dim"`
}

// ModelConfig defines the GoMLX ReLU multi-task MLP architecture.
type ModelConfig struct {
	EmbedDim    int     `yaml:"embed_dim"`
	HiddenDims  []int   `yaml:"hidden_dims"`
	HeadDim     int     `yaml:"head_dim"`
	DropoutRate float64 `yaml:"dropout_rate"`
	Activation  string  `yaml:"activation"`
}

// TrainingConfig holds optimization, batching, and loss weighting parameters.
type TrainingConfig struct {
	BatchSize                  int               `yaml:"batch_size"`
	EvalBatchSize              int               `yaml:"eval_batch_size"`
	TrainSteps                 int               `yaml:"train_steps"`
	LogEverySteps              int               `yaml:"log_every_steps"`
	CheckpointKeep             int               `yaml:"checkpoint_keep"`
	Optimizer                  string            `yaml:"optimizer"`
	LearningRate               float64           `yaml:"learning_rate"`
	MinLearningRate            float64           `yaml:"min_learning_rate"`
	WeightDecay                float64           `yaml:"weight_decay"`
	YNATBatchRatio             float64           `yaml:"ynat_batch_ratio"`
	NSMCBatchRatio             float64           `yaml:"nsmc_batch_ratio"`
	NFCPositiveSampleRatio     float64           `yaml:"nfc_positive_sample_ratio"`
	NFCNegativeCalibrationProb float64           `yaml:"nfc_negative_calibration_prob"`
	LossWeights                LossWeightsConfig `yaml:"loss_weights"`
}

// LossWeightsConfig configures multi-task loss weighting and numerical stability.
type LossWeightsConfig struct {
	Topic        float64 `yaml:"topic"`
	Sentiment    float64 `yaml:"sentiment"`
	NFC          float64 `yaml:"nfc"`
	NFCPosWeight float64 `yaml:"nfc_pos_weight"`
	MaskEpsilon  float64 `yaml:"mask_epsilon"`
}

// InferenceAndSortingConfig configures calibrated NFC scoring, intensity, and article sorting.
type InferenceAndSortingConfig struct {
	DefaultSortMode           string                 `yaml:"default_sort_mode"`
	NeuralNFCWeight           float64                `yaml:"neural_nfc_weight"`
	LexicalNFCWeight          float64                `yaml:"lexical_nfc_weight"`
	NFCActiveThreshold        float64                `yaml:"nfc_active_threshold"`
	SentimentPosThreshold     float64                `yaml:"sentiment_pos_threshold"`
	SentimentNegThreshold     float64                `yaml:"sentiment_neg_threshold"`
	IntensityWeights          IntensityWeightsConfig `yaml:"intensity_weights"`
	TopKNFC                   int                    `yaml:"top_k_nfc"`
	ReportTopArticlesPerGroup int                    `yaml:"report_top_articles_per_group"`
	ReportTopGlobalArticles   int                    `yaml:"report_top_global_articles"`
}

// IntensityWeightsConfig configures the composite social opinion intensity score.
type IntensityWeightsConfig struct {
	MaxNFC            float64 `yaml:"max_nfc"`
	MeanTop3NFC       float64 `yaml:"mean_top3_nfc"`
	NegativeSentiment float64 `yaml:"negative_sentiment"`
}

// LoggingConfig defines dual-log filenames and verbosity level.
type LoggingConfig struct {
	StatusFilename     string `yaml:"status_filename"`
	ProcessingFilename string `yaml:"processing_filename"`
	Level              string `yaml:"level"`
}

// ClusteringConfig configures hybrid HDBSCAN clustering, NER anchors, hierarchical naming, and 3-stage validation.
type ClusteringConfig struct {
	Enabled                      bool                  `yaml:"enabled"`
	YNATSampleSize               int                   `yaml:"ynat_sample_size"`
	NSMCSampleSize               int                   `yaml:"nsmc_sample_size"`
	MinClusterSize               int                   `yaml:"min_cluster_size"`
	MinSamples                   int                   `yaml:"min_samples"`
	ClusterSelectionEpsilon      float64               `yaml:"cluster_selection_epsilon"`
	FeatureWeights               ClusterFeatureWeights `yaml:"feature_weights"`
	TopKeywordsPerCluster        int                   `yaml:"top_keywords_per_cluster"`
	RepresentativeDocsPerCluster int                   `yaml:"representative_docs_per_cluster"`
	CoherenceEpsilon             float64               `yaml:"coherence_epsilon"`
	HumanEvalSampleSize          int                   `yaml:"human_eval_sample_size"`
	HumanEvalTargetAgreement     float64               `yaml:"human_eval_target_agreement"`
	SilhouetteTarget             float64               `yaml:"silhouette_target"`
	LLMLabeling                  LLMLabelingConfig     `yaml:"llm_labeling"`
}

// ClusterFeatureWeights configures how shared GoMLX latent, topic, NFC, and sentiment vectors combine.
type ClusterFeatureWeights struct {
	Latent     float64 `yaml:"latent"`
	TopicProbs float64 `yaml:"topic_probs"`
	NFCProbs   float64 `yaml:"nfc_probs"`
	Sentiment  float64 `yaml:"sentiment"`
}

// LLMLabelingConfig configures optional OpenAI-compatible cluster label refinement using env vars only.
type LLMLabelingConfig struct {
	Enabled     bool   `yaml:"enabled"`
	EndpointEnv string `yaml:"endpoint_env"`
	APIKeyEnv   string `yaml:"api_key_env"`
	ModelEnv    string `yaml:"model_env"`
	TimeoutSec  int    `yaml:"timeout_sec"`
}

// Load reads a YAML configuration file, applies environment overrides, and validates all values.
func Load(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config %q: %w", path, err)
	}
	var cfg Config
	if err := yaml.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("parse yaml config %q: %w", path, err)
	}
	if err := applyEnvOverrides(&cfg); err != nil {
		return nil, err
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func applyEnvOverrides(cfg *Config) error {
	if v := strings.TrimSpace(os.Getenv("GOMLX_BACKEND")); v != "" {
		cfg.Runtime.Backend = v
	}
	if v := strings.TrimSpace(os.Getenv("NFC_BACKEND")); v != "" {
		cfg.Runtime.Backend = v
	}
	if v := strings.TrimSpace(os.Getenv("NFC_DATA_DIR")); v != "" {
		cfg.Paths.DataDir = v
	}
	if v := strings.TrimSpace(os.Getenv("NFC_CHECKPOINT_DIR")); v != "" {
		cfg.Paths.CheckpointDir = v
	}
	if v := strings.TrimSpace(os.Getenv("NFC_OUTPUT_DIR")); v != "" {
		cfg.Paths.OutputDir = v
	}
	if v := strings.TrimSpace(os.Getenv("NFC_LOG_DIR")); v != "" {
		cfg.Paths.LogDir = v
	}
	if v := strings.TrimSpace(os.Getenv("NFC_TOPICS_FILE")); v != "" {
		cfg.Paths.TopicsFile = v
	}
	if v := strings.TrimSpace(os.Getenv("NFC_TAXONOMY_FILE")); v != "" {
		cfg.Paths.NFCTaxonomyFile = v
	}
	if v := strings.TrimSpace(os.Getenv("NFC_ENTITY_RULES_FILE")); v != "" {
		cfg.Paths.EntityRulesFile = v
	}
	if v := strings.TrimSpace(os.Getenv("NFC_SORT_MODE")); v != "" {
		cfg.InferenceAndSorting.DefaultSortMode = v
	}
	if v := strings.TrimSpace(os.Getenv("NFC_TRAIN_STEPS")); v != "" {
		steps, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("invalid NFC_TRAIN_STEPS=%q: %w", v, err)
		}
		cfg.Training.TrainSteps = steps
	}
	if v := strings.TrimSpace(os.Getenv("NFC_BATCH_SIZE")); v != "" {
		bs, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("invalid NFC_BATCH_SIZE=%q: %w", v, err)
		}
		cfg.Training.BatchSize = bs
	}
	if v := strings.TrimSpace(os.Getenv("NFC_CLUSTER_MIN_SIZE")); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("invalid NFC_CLUSTER_MIN_SIZE=%q: %w", v, err)
		}
		cfg.Clustering.MinClusterSize = n
	}
	if v := strings.TrimSpace(os.Getenv("NFC_CLUSTER_MIN_SAMPLES")); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("invalid NFC_CLUSTER_MIN_SAMPLES=%q: %w", v, err)
		}
		cfg.Clustering.MinSamples = n
	}
	if v := strings.TrimSpace(os.Getenv("NFC_CLUSTER_YNAT_SAMPLES")); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("invalid NFC_CLUSTER_YNAT_SAMPLES=%q: %w", v, err)
		}
		cfg.Clustering.YNATSampleSize = n
	}
	if v := strings.TrimSpace(os.Getenv("NFC_CLUSTER_NSMC_SAMPLES")); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("invalid NFC_CLUSTER_NSMC_SAMPLES=%q: %w", v, err)
		}
		cfg.Clustering.NSMCSampleSize = n
	}
	if v := strings.TrimSpace(os.Getenv("NFC_HUMAN_EVAL_SAMPLES")); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("invalid NFC_HUMAN_EVAL_SAMPLES=%q: %w", v, err)
		}
		cfg.Clustering.HumanEvalSampleSize = n
	}
	if v := strings.TrimSpace(os.Getenv("NFC_EMBEDDING_PROVIDER")); v != "" {
		cfg.Embedding.Provider = v
	}
	if v := strings.TrimSpace(os.Getenv("NFC_ROOTS_CONFIG_FILE")); v != "" {
		cfg.Embedding.RootsConfigFile = v
	}
	if v := strings.TrimSpace(os.Getenv("NFC_PRETRAIN_CHECKPOINT_DIR")); v != "" {
		cfg.Embedding.PretrainCheckpointDir = v
	}
	if v := strings.TrimSpace(os.Getenv("NFC_FREEZE_ENCODER")); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return fmt.Errorf("invalid NFC_FREEZE_ENCODER=%q: %w", v, err)
		}
		cfg.Embedding.FreezeEncoder = b
	}
	if v := strings.TrimSpace(os.Getenv("NFC_STUDENT_OUT_DIM")); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("invalid NFC_STUDENT_OUT_DIM=%q: %w", v, err)
		}
		cfg.Embedding.StudentOutDim = n
	}
	if v := strings.TrimSpace(os.Getenv("NFC_PCA_DIM")); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("invalid NFC_PCA_DIM=%q: %w", v, err)
		}
		cfg.Embedding.PCADim = n
	}
	return nil
}

// Validate checks every configuration section to prevent missing or invalid settings.
func (c *Config) Validate() error {
	if strings.TrimSpace(c.Runtime.Backend) == "" {
		return fmt.Errorf("runtime.backend must not be empty")
	}
	if c.Runtime.HTTPTimeoutSec <= 0 {
		return fmt.Errorf("runtime.http_timeout_sec must be > 0, got %d", c.Runtime.HTTPTimeoutSec)
	}
	if c.Runtime.HTTPRetries < 1 {
		return fmt.Errorf("runtime.http_retries must be >= 1, got %d", c.Runtime.HTTPRetries)
	}
	requiredPaths := map[string]string{
		"paths.data_dir":          c.Paths.DataDir,
		"paths.checkpoint_dir":    c.Paths.CheckpointDir,
		"paths.output_dir":        c.Paths.OutputDir,
		"paths.log_dir":           c.Paths.LogDir,
		"paths.topics_file":       c.Paths.TopicsFile,
		"paths.nfc_taxonomy_file": c.Paths.NFCTaxonomyFile,
		"paths.entity_rules_file": c.Paths.EntityRulesFile,
	}
	for k, v := range requiredPaths {
		if strings.TrimSpace(v) == "" {
			return fmt.Errorf("%s must not be empty", k)
		}
	}
	requiredDS := map[string]string{
		"datasets.nsmc_train_url":  c.Datasets.NSMCTrainURL,
		"datasets.nsmc_test_url":   c.Datasets.NSMCTestURL,
		"datasets.ynat_train_url":  c.Datasets.YNATTrainURL,
		"datasets.ynat_val_url":    c.Datasets.YNATValURL,
		"datasets.nsmc_train_file": c.Datasets.NSMCTrainFile,
		"datasets.nsmc_test_file":  c.Datasets.NSMCTestFile,
		"datasets.ynat_train_file": c.Datasets.YNATTrainFile,
		"datasets.ynat_val_file":   c.Datasets.YNATValFile,
	}
	for k, v := range requiredDS {
		if strings.TrimSpace(v) == "" {
			return fmt.Errorf("%s must not be empty", k)
		}
	}
	if c.Tokenizer.VocabSize < 10 {
		return fmt.Errorf("tokenizer.vocab_size must be >= 10, got %d", c.Tokenizer.VocabSize)
	}
	if c.Tokenizer.MaxSeqLen < 4 {
		return fmt.Errorf("tokenizer.max_seq_len must be >= 4, got %d", c.Tokenizer.MaxSeqLen)
	}
	if c.Tokenizer.MinFreq < 1 {
		return fmt.Errorf("tokenizer.min_freq must be >= 1, got %d", c.Tokenizer.MinFreq)
	}
	if c.Tokenizer.PadToken == "" || c.Tokenizer.UnkToken == "" || c.Tokenizer.VocabFilename == "" {
		return fmt.Errorf("tokenizer pad_token, unk_token, and vocab_filename must not be empty")
	}
	if c.Model.EmbedDim < 4 {
		return fmt.Errorf("model.embed_dim must be >= 4, got %d", c.Model.EmbedDim)
	}
	if len(c.Model.HiddenDims) == 0 {
		return fmt.Errorf("model.hidden_dims must contain at least one layer")
	}
	for i, dim := range c.Model.HiddenDims {
		if dim < 4 {
			return fmt.Errorf("model.hidden_dims[%d] must be >= 4, got %d", i, dim)
		}
	}
	if c.Model.HeadDim < 4 {
		return fmt.Errorf("model.head_dim must be >= 4, got %d", c.Model.HeadDim)
	}
	if c.Model.DropoutRate < 0 || c.Model.DropoutRate >= 1 {
		return fmt.Errorf("model.dropout_rate must be in [0, 1), got %v", c.Model.DropoutRate)
	}
	if strings.ToLower(strings.TrimSpace(c.Model.Activation)) != "relu" {
		return fmt.Errorf("model.activation must be \"relu\", got %q", c.Model.Activation)
	}
	if !slices.Contains(ValidEmbeddingProviders, c.Embedding.Provider) {
		return fmt.Errorf("embedding.provider %q not in %v", c.Embedding.Provider, ValidEmbeddingProviders)
	}
	if c.Embedding.StudentOutDim < 8 {
		return fmt.Errorf("embedding.student_out_dim must be >= 8, got %d", c.Embedding.StudentOutDim)
	}
	if c.Embedding.PCADim < 0 {
		return fmt.Errorf("embedding.pca_dim must be >= 0, got %d", c.Embedding.PCADim)
	}
	if c.Embedding.Provider == "ko_hub_graph" {
		if strings.TrimSpace(c.Embedding.RootsConfigFile) == "" || strings.TrimSpace(c.Embedding.PretrainCheckpointDir) == "" {
			return fmt.Errorf("embedding.roots_config_file and pretrain_checkpoint_dir must not be empty when provider=ko_hub_graph")
		}
	}
	if c.Training.BatchSize < 1 || c.Training.EvalBatchSize < 1 {
		return fmt.Errorf("training batch_size and eval_batch_size must be >= 1")
	}
	if c.Training.TrainSteps < 1 || c.Training.LogEverySteps < 1 || c.Training.CheckpointKeep < 1 {
		return fmt.Errorf("training train_steps, log_every_steps, and checkpoint_keep must be >= 1")
	}
	if c.Training.LearningRate <= 0 || c.Training.MinLearningRate < 0 || c.Training.WeightDecay < 0 {
		return fmt.Errorf("training learning_rate must be > 0 and min_learning_rate/weight_decay >= 0")
	}
	if c.Training.YNATBatchRatio <= 0 || c.Training.NSMCBatchRatio <= 0 || math.Abs((c.Training.YNATBatchRatio+c.Training.NSMCBatchRatio)-1.0) > 1e-6 {
		return fmt.Errorf("training.ynat_batch_ratio + training.nsmc_batch_ratio must equal 1.0, got %v + %v",
			c.Training.YNATBatchRatio, c.Training.NSMCBatchRatio)
	}
	if c.Training.NFCPositiveSampleRatio < 0 || c.Training.NFCPositiveSampleRatio > 1 {
		return fmt.Errorf("training.nfc_positive_sample_ratio must be in [0, 1]")
	}
	if c.Training.NFCNegativeCalibrationProb < 0 || c.Training.NFCNegativeCalibrationProb > 1 {
		return fmt.Errorf("training.nfc_negative_calibration_prob must be in [0, 1]")
	}
	if c.Training.LossWeights.Topic < 0 || c.Training.LossWeights.Sentiment < 0 || c.Training.LossWeights.NFC < 0 ||
		c.Training.LossWeights.NFCPosWeight <= 0 || c.Training.LossWeights.MaskEpsilon <= 0 {
		return fmt.Errorf("training.loss_weights must be valid non-negative weights with positive nfc_pos_weight and mask_epsilon")
	}
	if !slices.Contains(ValidSortModes, c.InferenceAndSorting.DefaultSortMode) {
		return fmt.Errorf("inference_and_sorting.default_sort_mode %q not in %v",
			c.InferenceAndSorting.DefaultSortMode, ValidSortModes)
	}
	if c.InferenceAndSorting.NeuralNFCWeight < 0 || c.InferenceAndSorting.LexicalNFCWeight < 0 ||
		(c.InferenceAndSorting.NeuralNFCWeight+c.InferenceAndSorting.LexicalNFCWeight) <= 0 {
		return fmt.Errorf("inference_and_sorting neural/lexical weights must be non-negative and sum > 0")
	}
	if c.InferenceAndSorting.TopKNFC < 1 || c.InferenceAndSorting.ReportTopArticlesPerGroup < 1 || c.InferenceAndSorting.ReportTopGlobalArticles < 1 {
		return fmt.Errorf("inference_and_sorting top_k_nfc and report limits must be >= 1")
	}
	if c.InferenceAndSorting.IntensityWeights.MaxNFC+c.InferenceAndSorting.IntensityWeights.MeanTop3NFC+c.InferenceAndSorting.IntensityWeights.NegativeSentiment <= 0 {
		return fmt.Errorf("inference_and_sorting.intensity_weights must sum > 0")
	}
	if strings.TrimSpace(c.Logging.StatusFilename) == "" || strings.TrimSpace(c.Logging.ProcessingFilename) == "" {
		return fmt.Errorf("logging filenames must not be empty")
	}
	switch strings.ToUpper(strings.TrimSpace(c.Logging.Level)) {
	case "DEBUG", "INFO", "WARN", "ERROR":
	default:
		return fmt.Errorf("logging.level must be one of DEBUG, INFO, WARN, ERROR, got %q", c.Logging.Level)
	}
	if c.Clustering.YNATSampleSize < 1 || c.Clustering.NSMCSampleSize < 1 {
		return fmt.Errorf("clustering ynat_sample_size and nsmc_sample_size must be >= 1")
	}
	if c.Clustering.MinClusterSize < 2 || c.Clustering.MinSamples < 1 {
		return fmt.Errorf("clustering min_cluster_size must be >= 2 and min_samples >= 1")
	}
	if c.Clustering.ClusterSelectionEpsilon < 0 {
		return fmt.Errorf("clustering.cluster_selection_epsilon must be >= 0")
	}
	fw := c.Clustering.FeatureWeights
	if fw.Latent < 0 || fw.TopicProbs < 0 || fw.NFCProbs < 0 || fw.Sentiment < 0 ||
		(fw.Latent+fw.TopicProbs+fw.NFCProbs+fw.Sentiment) <= 0 {
		return fmt.Errorf("clustering.feature_weights must be non-negative and sum > 0")
	}
	if c.Clustering.TopKeywordsPerCluster < 2 || c.Clustering.RepresentativeDocsPerCluster < 1 {
		return fmt.Errorf("clustering top_keywords_per_cluster >= 2 and representative_docs_per_cluster >= 1 required")
	}
	if c.Clustering.CoherenceEpsilon <= 0 {
		return fmt.Errorf("clustering.coherence_epsilon must be > 0")
	}
	if c.Clustering.HumanEvalSampleSize < 1 || c.Clustering.HumanEvalTargetAgreement <= 0 || c.Clustering.HumanEvalTargetAgreement > 1 {
		return fmt.Errorf("clustering human_eval settings invalid")
	}
	if c.Clustering.LLMLabeling.Enabled {
		if strings.TrimSpace(c.Clustering.LLMLabeling.EndpointEnv) == "" ||
			strings.TrimSpace(c.Clustering.LLMLabeling.APIKeyEnv) == "" ||
			strings.TrimSpace(c.Clustering.LLMLabeling.ModelEnv) == "" ||
			c.Clustering.LLMLabeling.TimeoutSec <= 0 {
			return fmt.Errorf("clustering.llm_labeling env var names and timeout_sec must be set when enabled")
		}
	}
	return nil
}

// VocabPath returns the path where the fitted tokenizer vocabulary is stored.
func (c *Config) VocabPath() string {
	return filepath.Join(c.Paths.CheckpointDir, c.Tokenizer.VocabFilename)
}

// NSMCTrainPath returns the local path for NSMC training TSV.
func (c *Config) NSMCTrainPath() string {
	return filepath.Join(c.Paths.DataDir, c.Datasets.NSMCTrainFile)
}

// NSMCTestPath returns the local path for NSMC test TSV.
func (c *Config) NSMCTestPath() string {
	return filepath.Join(c.Paths.DataDir, c.Datasets.NSMCTestFile)
}

// YNATTrainPath returns the local path for KLUE-YNAT training Parquet.
func (c *Config) YNATTrainPath() string {
	return filepath.Join(c.Paths.DataDir, c.Datasets.YNATTrainFile)
}

// YNATValPath returns the local path for KLUE-YNAT validation Parquet.
func (c *Config) YNATValPath() string {
	return filepath.Join(c.Paths.DataDir, c.Datasets.YNATValFile)
}
