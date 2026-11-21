package cluster

import (
	"github.com/Mireu-Lab/gomlx-nfc-opinion/internal/data"
	"github.com/Mireu-Lab/gomlx-nfc-opinion/internal/sorter"
)

// Entity represents a domain anchor extracted from text using externalized NER rules.
type Entity struct {
	Type      string `json:"type"`
	Value     string `json:"value"`
	Canonical string `json:"canonical"`
}

// RepDoc is a medoid representative document for a cluster.
type RepDoc struct {
	ID             string  `json:"id"`
	Source         string  `json:"source"`
	Title          string  `json:"title"`
	TopicKoName    string  `json:"topic_ko_name"`
	PrimaryNFC     string  `json:"primary_nfc"`
	SentimentLabel string  `json:"sentiment_label"`
	Intensity      float64 `json:"intensity"`
	MembershipProb float64 `json:"membership_prob"`
	DistToCentroid float64 `json:"dist_to_centroid"`
}

// ClusterDoc holds a mixed YNAT/NSMC document along with its GoMLX neural features and cluster assignment.
type ClusterDoc struct {
	Record          data.Record
	Scored          sorter.ScoredArticle
	SemanticEmbed   []float32
	Entities        []Entity
	Keywords        []string
	HybridVector    []float64
	ClusterID       int
	MembershipProb  float64
	PointSilhouette float64
}

// ClusterSummary summarizes one HDBSCAN cluster across both news and review domains.
type ClusterSummary struct {
	ClusterID          int      `json:"cluster_id"`
	Size               int      `json:"size"`
	YNATCount          int      `json:"ynat_count"`
	NSMCCount          int      `json:"nsmc_count"`
	CrossDomainRatio   float64  `json:"cross_domain_ratio"`
	DominantTopic      string   `json:"dominant_topic"`
	DominantNFC        string   `json:"dominant_nfc"`
	DominantFactor     string   `json:"dominant_factor"`
	MeanIntensity      float64  `json:"mean_intensity"`
	MeanNegativeProb   float64  `json:"mean_negative_prob"`
	MeanSilhouette     float64  `json:"mean_silhouette"`
	TopKeywords        []string `json:"top_keywords"`
	NPMICoherence      float64  `json:"npmi_coherence"`
	UMassCoherence     float64  `json:"umass_coherence"`
	TopEntities        []string `json:"top_entities"`
	HierarchicalLabel  string   `json:"hierarchical_label"`
	LLMRefinedLabel    string   `json:"llm_refined_label,omitempty"`
	RepresentativeDocs []RepDoc `json:"representative_docs"`
}

// EntityGroup aggregates documents anchored by the same named entity or public issue.
type EntityGroup struct {
	Type          string   `json:"type"`
	Entity        string   `json:"entity"`
	TotalCount    int      `json:"total_count"`
	YNATCount     int      `json:"ynat_count"`
	NSMCCount     int      `json:"nsmc_count"`
	CrossDomain   bool     `json:"cross_domain"`
	DominantTopic string   `json:"dominant_topic"`
	DominantNFC   string   `json:"dominant_nfc"`
	MeanIntensity float64  `json:"mean_intensity"`
	SampleTitles  []string `json:"sample_titles"`
}

// ValidationMetrics holds the 3-stage clustering validation metrics.
type ValidationMetrics struct {
	SilhouetteScore          float64 `json:"silhouette_score"`
	SilhouetteTarget         float64 `json:"silhouette_target"`
	DunnIndexStrict          float64 `json:"dunn_index_strict"`
	DunnIndexCentroid        float64 `json:"dunn_index_centroid"`
	MeanNPMICoherence        float64 `json:"mean_npmi_coherence"`
	MeanUMassCoherence       float64 `json:"mean_umass_coherence"`
	NumClusters              int     `json:"num_clusters"`
	TotalDocs                int     `json:"total_docs"`
	ClusteredDocs            int     `json:"clustered_docs"`
	NoiseDocs                int     `json:"noise_docs"`
	NoiseRatio               float64 `json:"noise_ratio"`
	CrossDomainClusters      int     `json:"cross_domain_clusters"`
	EntityGroupCount         int     `json:"entity_group_count"`
	CrossDomainEntityBridges int     `json:"cross_domain_entity_bridges"`
	HumanEvalSampleCount     int     `json:"human_eval_sample_count"`
	HumanEvalTargetAgreement float64 `json:"human_eval_target_agreement"`
}

// ClusterDocSummary is a compact JSON-friendly view of one clustered document.
type ClusterDocSummary struct {
	ID              string   `json:"id"`
	Source          string   `json:"source"`
	Title           string   `json:"title"`
	ClusterID       int      `json:"cluster_id"`
	MembershipProb  float64  `json:"membership_prob"`
	PointSilhouette float64  `json:"point_silhouette"`
	PredTopic       string   `json:"pred_topic"`
	PrimaryNFC      string   `json:"primary_nfc"`
	SentimentLabel  string   `json:"sentiment_label"`
	Intensity       float64  `json:"intensity"`
	Entities        []Entity `json:"entities,omitempty"`
}

// ClusteringReport is the full output serialized to clusters_report.json.
type ClusteringReport struct {
	GeneratedAt        string              `json:"generated_at"`
	Validation         ValidationMetrics   `json:"validation"`
	Clusters           []ClusterSummary    `json:"clusters"`
	NoiseSamples       []ClusterDocSummary `json:"noise_samples"`
	EntityGroups       []EntityGroup       `json:"entity_groups"`
	CrossDomainBridges []EntityGroup       `json:"cross_domain_bridges"`
	Documents          []ClusterDocSummary `json:"documents"`
}

// HumanEvalScore summarizes Stage 3 Gold Standard human evaluation results.
type HumanEvalScore struct {
	TotalRated      int     `json:"total_rated"`
	AgreeCount      int     `json:"agree_count"`
	AgreementRate   float64 `json:"agreement_rate"`
	CohensKappa     float64 `json:"cohens_kappa"`
	TargetAgreement float64 `json:"target_agreement"`
	PassesTarget    bool    `json:"passes_target"`
}
