package cluster

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/Mireu-Lab/gomlx-nfc-opinion/internal/config"
	"github.com/Mireu-Lab/gomlx-nfc-opinion/internal/logging"
	"github.com/Mireu-Lab/gomlx-nfc-opinion/internal/tokenizer"
)

// ExtractDocKeywords returns filtered domain-meaningful keywords for c-TF-IDF and topic coherence.
func ExtractDocKeywords(text string, stopwords map[string]struct{}) []string {
	norm := tokenizer.Normalize(text)
	if norm == "" {
		return nil
	}
	fields := strings.Fields(norm)
	out := make([]string, 0, len(fields))
	for _, w := range fields {
		if len([]rune(w)) < 2 {
			continue
		}
		if _, err := strconv.Atoi(w); err == nil {
			continue
		}
		if _, isStop := stopwords[w]; isStop {
			continue
		}
		out = append(out, w)
	}
	return out
}

// ComputeClusterTopKeywords computes class-based TF-IDF (c-TF-IDF) top keywords per cluster.
func ComputeClusterTopKeywords(docs []ClusterDoc, topK int) map[int][]string {
	clusterTF := make(map[int]map[string]int)
	clusterDF := make(map[int]map[string]int)
	clusterTotalToks := make(map[int]int)
	globalTF := make(map[string]int)

	for _, doc := range docs {
		cid := doc.ClusterID
		if cid < 0 {
			continue
		}
		if _, ok := clusterTF[cid]; !ok {
			clusterTF[cid] = make(map[string]int)
			clusterDF[cid] = make(map[string]int)
		}
		seenInDoc := make(map[string]struct{}, len(doc.Keywords))
		for _, kw := range doc.Keywords {
			clusterTF[cid][kw]++
			clusterTotalToks[cid]++
			globalTF[kw]++
			if _, ok := seenInDoc[kw]; !ok {
				seenInDoc[kw] = struct{}{}
				clusterDF[cid][kw]++
			}
		}
	}

	if len(clusterTF) == 0 {
		return map[int][]string{}
	}

	var sumTokens int
	for _, tot := range clusterTotalToks {
		sumTokens += tot
	}
	avgTokens := float64(sumTokens) / float64(len(clusterTF))
	if avgTokens < 1.0 {
		avgTokens = 1.0
	}

	out := make(map[int][]string, len(clusterTF))
	type scoredKW struct {
		kw    string
		df    int
		score float64
	}
	for cid, tfMap := range clusterTF {
		items := make([]scoredKW, 0, len(tfMap))
		for kw, tf := range tfMap {
			fGlobal := float64(globalTF[kw])
			if fGlobal < 1.0 {
				fGlobal = 1.0
			}
			score := float64(tf) * math.Log(1.0+avgTokens/fGlobal)
			items = append(items, scoredKW{kw: kw, df: clusterDF[cid][kw], score: score})
		}
		sort.Slice(items, func(i, j int) bool {
			// Prefer keywords occurring in >= 2 documents within the cluster to filter single-doc noise.
			iMulti := items[i].df >= 2
			jMulti := items[j].df >= 2
			if iMulti != jMulti {
				return iMulti
			}
			if items[i].score != items[j].score {
				return items[i].score > items[j].score
			}
			return items[i].kw < items[j].kw
		})
		limit := topK
		if limit > len(items) {
			limit = len(items)
		}
		kws := make([]string, limit)
		for i := 0; i < limit; i++ {
			kws[i] = items[i].kw
		}
		out[cid] = kws
	}
	return out
}

// SelectRepresentatives chooses medoid documents closest to the cluster centroid, balancing YNAT and NSMC sources.
func SelectRepresentatives(clusterDocs []*ClusterDoc, limit int) []RepDoc {
	if len(clusterDocs) == 0 || limit <= 0 {
		return nil
	}
	dim := len(clusterDocs[0].HybridVector)
	centroid := make([]float64, dim)
	for _, d := range clusterDocs {
		for k, v := range d.HybridVector {
			centroid[k] += v
		}
	}
	centroid = l2NormalizeFloat64(centroid)

	type candidate struct {
		doc  *ClusterDoc
		dist float64
	}
	var ynatCands, nsmcCands []candidate
	for _, d := range clusterDocs {
		var dot float64
		for k, v := range d.HybridVector {
			dot += v * centroid[k]
		}
		dist := 1.0 - dot
		if dist < 0 {
			dist = 0
		}
		cand := candidate{doc: d, dist: dist}
		if d.Record.Source == "ynat" {
			ynatCands = append(ynatCands, cand)
		} else {
			nsmcCands = append(nsmcCands, cand)
		}
	}
	sortCands := func(s []candidate) {
		sort.Slice(s, func(i, j int) bool {
			if s[i].dist != s[j].dist {
				return s[i].dist < s[j].dist
			}
			return s[i].doc.Record.ID < s[j].doc.Record.ID
		})
	}
	sortCands(ynatCands)
	sortCands(nsmcCands)

	// Interleave YNAT and NSMC medoids so cross-domain clusters show both news and reviews.
	var picked []candidate
	yi, ni := 0, 0
	for len(picked) < limit && (yi < len(ynatCands) || ni < len(nsmcCands)) {
		if yi < len(ynatCands) {
			picked = append(picked, ynatCands[yi])
			yi++
		}
		if len(picked) >= limit {
			break
		}
		if ni < len(nsmcCands) {
			picked = append(picked, nsmcCands[ni])
			ni++
		}
	}

	out := make([]RepDoc, len(picked))
	for i, c := range picked {
		out[i] = RepDoc{
			ID:             c.doc.Record.ID,
			Source:         c.doc.Record.Source,
			Title:          c.doc.Record.Text,
			TopicKoName:    c.doc.Scored.TopicKoName,
			PrimaryNFC:     c.doc.Scored.PrimaryNFC.KoName,
			SentimentLabel: c.doc.Scored.SentimentLabel,
			Intensity:      c.doc.Scored.SocialOpinionIntensity,
			MembershipProb: c.doc.MembershipProb,
			DistToCentroid: c.dist,
		}
	}
	return out
}

// BuildHierarchicalLabel formats a 3-tier interpretable topic label:
// [상위: Domain Topic] - [중위: NFC Factor/Feeling] - [하위: Top Entity or Keywords].
func BuildHierarchicalLabel(domTopic, domFactor, domNFC string, topEntities, topKeywords []string) string {
	sub := "일반 이슈"
	if len(topEntities) > 0 {
		sub = topEntities[0]
		if len(topKeywords) > 0 {
			sub += "·" + topKeywords[0]
		}
	} else if len(topKeywords) >= 2 {
		sub = topKeywords[0] + "·" + topKeywords[1]
	} else if len(topKeywords) == 1 {
		sub = topKeywords[0]
	}
	return fmt.Sprintf("[상위: %s] - [중위: %s/%s] - [하위: %s]", domTopic, domFactor, domNFC, sub)
}

// MaybeRefineLabelsWithLLM optionally refines hierarchical labels via an OpenAI-compatible endpoint
// configured purely via environment variables (no hardcoded URLs, models, or credentials).
func MaybeRefineLabelsWithLLM(
	ctx context.Context,
	summaries []ClusterSummary,
	cfg config.LLMLabelingConfig,
	log *logging.DualLogger,
) {
	if !cfg.Enabled {
		return
	}
	endpoint := strings.TrimSpace(os.Getenv(cfg.EndpointEnv))
	apiKey := strings.TrimSpace(os.Getenv(cfg.APIKeyEnv))
	modelName := strings.TrimSpace(os.Getenv(cfg.ModelEnv))
	if endpoint == "" || apiKey == "" || modelName == "" {
		log.Warn("cluster_llm", "llm_labeling enabled but env vars (%s, %s, %s) not all set; keeping deterministic hierarchical labels",
			cfg.EndpointEnv, cfg.APIKeyEnv, cfg.ModelEnv)
		return
	}

	client := &http.Client{Timeout: time.Duration(cfg.TimeoutSec) * time.Second}
	for i := range summaries {
		s := &summaries[i]
		var repLines []string
		for _, r := range s.RepresentativeDocs {
			repLines = append(repLines, fmt.Sprintf("- (%s) %s", r.Source, r.Title))
		}
		prompt := fmt.Sprintf(
			"다음은 한국어 뉴스/리뷰 군집의 키워드(%s)와 대표 문서입니다:\n%s\n\n"+
				"공통 주제를 '[상위: 대분류] - [중위: 정서/이슈] - [하위: 세부주제]' 한 줄 형식으로만 작성해 주세요.",
			strings.Join(s.TopKeywords, ", "), strings.Join(repLines, "\n"),
		)
		reqBody, _ := json.Marshal(map[string]any{
			"model": modelName,
			"messages": []map[string]string{
				{"role": "user", "content": prompt},
			},
			"temperature": 0.2,
			"max_tokens":  80,
		})
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(reqBody))
		if err != nil {
			log.Warn("cluster_llm", "build request failed for cluster %d: %v", s.ClusterID, err)
			continue
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+apiKey)

		resp, err := client.Do(req)
		if err != nil {
			log.Warn("cluster_llm", "request failed for cluster %d: %v", s.ClusterID, err)
			continue
		}
		var parsed struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		decErr := json.NewDecoder(resp.Body).Decode(&parsed)
		_ = resp.Body.Close()
		if resp.StatusCode < 200 || resp.StatusCode >= 300 || decErr != nil || len(parsed.Choices) == 0 {
			log.Warn("cluster_llm", "unexpected LLM response for cluster %d (status=%s err=%v)", s.ClusterID, resp.Status, decErr)
			continue
		}
		s.LLMRefinedLabel = strings.TrimSpace(parsed.Choices[0].Message.Content)
	}
}
