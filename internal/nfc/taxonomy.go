package nfc

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// TopicDef defines one KLUE-YNAT news topic category loaded from YAML.
type TopicDef struct {
	ID           int    `yaml:"id" json:"id"`
	Code         string `yaml:"code" json:"code"`
	KoName       string `yaml:"ko_name" json:"ko_name"`
	EnName       string `yaml:"en_name" json:"en_name"`
	DisplayOrder int    `yaml:"display_order" json:"display_order"`
}

// FeelingDef defines one Nuclear Feeling Checklist item loaded from YAML.
type FeelingDef struct {
	ID               int      `yaml:"id" json:"id"`
	Code             string   `yaml:"code" json:"code"`
	KoName           string   `yaml:"ko_name" json:"ko_name"`
	EnName           string   `yaml:"en_name" json:"en_name"`
	FactorGroup      string   `yaml:"factor_group" json:"factor_group"`
	SocialWeight     float64  `yaml:"social_weight" json:"social_weight"`
	PolarityAffinity string   `yaml:"polarity_affinity" json:"polarity_affinity"`
	Description      string   `yaml:"description" json:"description"`
	Keywords         []string `yaml:"keywords" json:"keywords"`
	ExcludePatterns  []string `yaml:"exclude_patterns,omitempty" json:"exclude_patterns,omitempty"`

	normalizedKeywords        []string
	normalizedExcludePatterns []string
}

type topicsFile struct {
	Topics []TopicDef `yaml:"topics"`
}

type nfcTaxonomyFile struct {
	LexicalSaturation float64      `yaml:"lexical_saturation"`
	Feelings          []FeelingDef `yaml:"feelings"`
}

// Taxonomy bundles externalized KLUE-YNAT topics and the 16 Nuclear Feeling Checklist categories.
type Taxonomy struct {
	Topics            []TopicDef
	Feelings          []FeelingDef
	LexicalSaturation float64

	topicsByID   map[int]TopicDef
	feelingsByID map[int]FeelingDef
}

// LoadTaxonomy reads and validates topics.yaml and nfc_taxonomy.yaml.
func LoadTaxonomy(topicsPath, nfcPath string) (*Taxonomy, error) {
	tRaw, err := os.ReadFile(topicsPath)
	if err != nil {
		return nil, fmt.Errorf("read topics file %q: %w", topicsPath, err)
	}
	var tf topicsFile
	if err := yaml.Unmarshal(tRaw, &tf); err != nil {
		return nil, fmt.Errorf("parse topics file %q: %w", topicsPath, err)
	}

	nRaw, err := os.ReadFile(nfcPath)
	if err != nil {
		return nil, fmt.Errorf("read nfc taxonomy %q: %w", nfcPath, err)
	}
	var nf nfcTaxonomyFile
	if err := yaml.Unmarshal(nRaw, &nf); err != nil {
		return nil, fmt.Errorf("parse nfc taxonomy %q: %w", nfcPath, err)
	}

	if len(tf.Topics) == 0 {
		return nil, fmt.Errorf("topics list in %q is empty", topicsPath)
	}
	if len(nf.Feelings) == 0 {
		return nil, fmt.Errorf("feelings list in %q is empty", nfcPath)
	}
	if nf.LexicalSaturation <= 0 {
		return nil, fmt.Errorf("lexical_saturation in %q must be > 0, got %v", nfcPath, nf.LexicalSaturation)
	}

	sort.Slice(tf.Topics, func(i, j int) bool { return tf.Topics[i].ID < tf.Topics[j].ID })
	topicsByID := make(map[int]TopicDef, len(tf.Topics))
	for i, topic := range tf.Topics {
		if topic.ID != i {
			return nil, fmt.Errorf("topics IDs must be contiguous 0..N-1, expected %d got %d", i, topic.ID)
		}
		if strings.TrimSpace(topic.Code) == "" || strings.TrimSpace(topic.KoName) == "" {
			return nil, fmt.Errorf("topic id %d has empty code or ko_name", topic.ID)
		}
		topicsByID[topic.ID] = topic
	}

	sort.Slice(nf.Feelings, func(i, j int) bool { return nf.Feelings[i].ID < nf.Feelings[j].ID })
	feelingsByID := make(map[int]FeelingDef, len(nf.Feelings))
	for i := range nf.Feelings {
		f := &nf.Feelings[i]
		if f.ID != i {
			return nil, fmt.Errorf("feelings IDs must be contiguous 0..M-1, expected %d got %d", i, f.ID)
		}
		if strings.TrimSpace(f.Code) == "" || strings.TrimSpace(f.KoName) == "" {
			return nil, fmt.Errorf("feeling id %d has empty code or ko_name", f.ID)
		}
		if f.SocialWeight <= 0 {
			return nil, fmt.Errorf("feeling %q social_weight must be > 0, got %v", f.Code, f.SocialWeight)
		}
		if len(f.Keywords) == 0 {
			return nil, fmt.Errorf("feeling %q must define keywords", f.Code)
		}
		seenKW := make(map[string]struct{}, len(f.Keywords))
		f.normalizedKeywords = make([]string, 0, len(f.Keywords))
		for _, kw := range f.Keywords {
			norm := strings.ToLower(strings.TrimSpace(kw))
			if norm == "" {
				continue
			}
			if _, ok := seenKW[norm]; ok {
				continue
			}
			seenKW[norm] = struct{}{}
			f.normalizedKeywords = append(f.normalizedKeywords, norm)
		}
		seenEx := make(map[string]struct{}, len(f.ExcludePatterns))
		f.normalizedExcludePatterns = make([]string, 0, len(f.ExcludePatterns))
		for _, ex := range f.ExcludePatterns {
			norm := strings.ToLower(strings.TrimSpace(ex))
			if norm == "" {
				continue
			}
			if _, ok := seenEx[norm]; ok {
				continue
			}
			seenEx[norm] = struct{}{}
			f.normalizedExcludePatterns = append(f.normalizedExcludePatterns, norm)
		}
		feelingsByID[f.ID] = *f
	}

	return &Taxonomy{
		Topics:            tf.Topics,
		Feelings:          nf.Feelings,
		LexicalSaturation: nf.LexicalSaturation,
		topicsByID:        topicsByID,
		feelingsByID:      feelingsByID,
	}, nil
}

// NumTopics returns the number of topic classes.
func (t *Taxonomy) NumTopics() int {
	return len(t.Topics)
}

// NumFeelings returns the number of Nuclear Feeling Checklist categories.
func (t *Taxonomy) NumFeelings() int {
	return len(t.Feelings)
}

// TopicByID returns the topic definition for an ID.
func (t *Taxonomy) TopicByID(id int) (TopicDef, bool) {
	v, ok := t.topicsByID[id]
	return v, ok
}

// FeelingByID returns the Nuclear Feeling definition for an ID.
func (t *Taxonomy) FeelingByID(id int) (FeelingDef, bool) {
	v, ok := t.feelingsByID[id]
	return v, ok
}

// SortedTopicsByDisplayOrder returns topics ordered by configured display_order, then ID.
func (t *Taxonomy) SortedTopicsByDisplayOrder() []TopicDef {
	out := make([]TopicDef, len(t.Topics))
	copy(out, t.Topics)
	sort.Slice(out, func(i, j int) bool {
		if out[i].DisplayOrder != out[j].DisplayOrder {
			return out[i].DisplayOrder < out[j].DisplayOrder
		}
		return out[i].ID < out[j].ID
	})
	return out
}

// WeakAnnotate scans normalized text against the Nuclear Feeling Checklist keyword stems.
// It returns binary multi-label targets (0/1), calibrated lexical scores in [0, 1], and total matched categories.
func (t *Taxonomy) WeakAnnotate(text string) (targets []float32, scores []float32, matchedCount int) {
	targets, scores, matchedCount, _ = t.WeakAnnotateDetailed(text)
	return targets, scores, matchedCount
}

// WeakAnnotateDetailed additionally reports whether any homonym exclude_patterns matched,
// allowing training preparation to treat disambiguated compounds as explicit negative calibration examples.
func (t *Taxonomy) WeakAnnotateDetailed(text string) (targets []float32, scores []float32, matchedCount int, guardTriggered bool) {
	n := len(t.Feelings)
	targets = make([]float32, n)
	scores = make([]float32, n)
	norm := strings.ToLower(strings.TrimSpace(text))
	if norm == "" {
		return targets, scores, 0, false
	}
	sat := float32(t.LexicalSaturation)
	for i, f := range t.Feelings {
		cleanNorm := norm
		for _, pat := range f.normalizedExcludePatterns {
			if strings.Contains(cleanNorm, pat) {
				guardTriggered = true
				cleanNorm = strings.ReplaceAll(cleanNorm, pat, " ")
			}
		}
		hits := 0
		for _, kw := range f.normalizedKeywords {
			if strings.Contains(cleanNorm, kw) {
				hits++
			}
		}
		if hits > 0 {
			targets[i] = 1.0
			sc := float32(hits) / sat
			if sc > 1.0 {
				sc = 1.0
			}
			scores[i] = sc
			matchedCount++
		}
	}
	return targets, scores, matchedCount, guardTriggered
}
