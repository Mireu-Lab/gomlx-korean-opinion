package cluster

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/data"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/logging"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/nfc"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/sorter"
)

// RunClusteringPipeline executes all 3 clustering methods (Hybrid HDBSCAN, NER Entity Anchors,
// Hierarchical Topic Naming) and 3 validation stages (Silhouette/Dunn, NPMI/UMass Coherence,
// Human Eval Stratified Export) over mixed KLUE-YNAT news and NSMC movie reviews.
func RunClusteringPipeline(
	ctx context.Context,
	records []data.Record,
	preds []sorter.ArticlePrediction,
	scored []sorter.ScoredArticle,
	tax *nfc.Taxonomy,
	cfg *config.Config,
	log *logging.DualLogger,
) (*ClusteringReport, error) {
	if len(records) != len(preds) || len(records) != len(scored) {
		return nil, fmt.Errorf("input slice length mismatch: records=%d preds=%d scored=%d",
			len(records), len(preds), len(scored))
	}
	extractor, err := LoadEntityExtractor(cfg.Paths.EntityRulesFile)
	if err != nil {
		return nil, err
	}

	docs := make([]ClusterDoc, len(records))
	for i := range records {
		docs[i] = ClusterDoc{
			Record:        records[i],
			Scored:        scored[i],
			SemanticEmbed: preds[i].SemanticEmbed,
			Entities:      extractor.Extract(records[i].Text),
			Keywords:      ExtractDocKeywords(records[i].Text, extractor.Stopwords()),
			ClusterID:     -1,
		}
	}

	// Method 1: Hybrid Embedding + Pure-Go HDBSCAN Density Clustering.
	vectors := BuildHybridVectors(docs, cfg.Clustering.FeatureWeights)
	for i := range docs {
		docs[i].HybridVector = vectors[i]
	}
	distMatrix := ComputeCosineDistanceMatrix(vectors)
	hdb := RunHDBSCAN(
		distMatrix,
		len(docs),
		cfg.Clustering.MinClusterSize,
		cfg.Clustering.MinSamples,
		cfg.Clustering.ClusterSelectionEpsilon,
	)
	for i := range docs {
		docs[i].ClusterID = hdb.Labels[i]
		docs[i].MembershipProb = hdb.Probs[i]
	}

	// Stage 1 Validation: Silhouette & Dunn Index.
	silOverall, pointSil, clusterSil := ComputeSilhouette(distMatrix, len(docs), hdb.Labels)
	for i := range docs {
		docs[i].PointSilhouette = pointSil[i]
	}
	strictDunn, centroidDunn := ComputeDunnIndex(distMatrix, vectors, hdb.Labels)

	// Method 2: Entity (NER) Anchor Grouping & Cross-Domain News <-> Review Bridges.
	entityGroups, crossBridges := BuildEntityGroups(docs)

	// Method 3: c-TF-IDF Keywords + Medoid Representatives + Hierarchical Labels (+ optional LLM).
	clusterTopKWs := ComputeClusterTopKeywords(docs, cfg.Clustering.TopKeywordsPerCluster)
	meanNPMI, meanUMass, clusterCoherence := ComputeTopicCoherence(docs, clusterTopKWs, cfg.Clustering.CoherenceEpsilon)

	byCluster := make(map[int][]*ClusterDoc)
	noiseCount := 0
	for i := range docs {
		cid := docs[i].ClusterID
		if cid < 0 {
			noiseCount++
			continue
		}
		byCluster[cid] = append(byCluster[cid], &docs[i])
	}

	cids := make([]int, 0, len(byCluster))
	for cid := range byCluster {
		cids = append(cids, cid)
	}
	sort.Ints(cids)

	summaries := make([]ClusterSummary, 0, len(cids))
	summaryMap := make(map[int]ClusterSummary, len(cids))
	crossDomainClusters := 0

	for _, cid := range cids {
		members := byCluster[cid]
		var ynatCnt, nsmcCnt int
		var sumInt, sumNeg float64
		topicCounts := make(map[string]int)
		nfcCounts := make(map[string]int)
		factorCounts := make(map[string]int)
		entityCounts := make(map[string]int)

		for _, d := range members {
			if d.Record.Source == "ynat" {
				ynatCnt++
			} else {
				nsmcCnt++
			}
			sumInt += d.Scored.SocialOpinionIntensity
			sumNeg += d.Scored.SentimentNegativeProb
			topicCounts[d.Scored.TopicKoName]++
			nfcCounts[d.Scored.PrimaryNFC.KoName]++
			factorCounts[d.Scored.PrimaryNFC.FactorGroup]++
			for _, ent := range d.Entities {
				entityCounts[ent.Canonical+"("+ent.Type+")"]++
			}
		}

		if ynatCnt > 0 && nsmcCnt > 0 {
			crossDomainClusters++
		}
		minSide := ynatCnt
		if nsmcCnt < minSide {
			minSide = nsmcCnt
		}
		crossRatio := 0.0
		if len(members) > 0 {
			crossRatio = float64(2*minSide) / float64(len(members))
		}

		topEntities := topNKeys(entityCounts, 5)
		domTopic := topCountKey(topicCounts)
		domNFC := topCountKey(nfcCounts)
		domFactor := topCountKey(factorCounts)
		topKWs := clusterTopKWs[cid]
		hierLabel := BuildHierarchicalLabel(domTopic, domFactor, domNFC, topEntities, topKWs)
		reps := SelectRepresentatives(members, cfg.Clustering.RepresentativeDocsPerCluster)
		coh := clusterCoherence[cid]

		sum := ClusterSummary{
			ClusterID:          cid,
			Size:               len(members),
			YNATCount:          ynatCnt,
			NSMCCount:          nsmcCnt,
			CrossDomainRatio:   crossRatio,
			DominantTopic:      domTopic,
			DominantNFC:        domNFC,
			DominantFactor:     domFactor,
			MeanIntensity:      sumInt / float64(len(members)),
			MeanNegativeProb:   sumNeg / float64(len(members)),
			MeanSilhouette:     clusterSil[cid],
			TopKeywords:        topKWs,
			NPMICoherence:      coh[0],
			UMassCoherence:     coh[1],
			TopEntities:        topEntities,
			HierarchicalLabel:  hierLabel,
			RepresentativeDocs: reps,
		}
		summaries = append(summaries, sum)
		summaryMap[cid] = sum
	}

	MaybeRefineLabelsWithLLM(ctx, summaries, cfg.Clustering.LLMLabeling, log)
	for _, s := range summaries {
		summaryMap[s.ClusterID] = s
	}

	humanSampleCount, err := ExportHumanEvalSample(
		docs,
		summaryMap,
		cfg.Clustering.HumanEvalSampleSize,
		cfg.Runtime.Seed,
		cfg.Paths.OutputDir,
	)
	if err != nil {
		return nil, err
	}

	docSummaries := make([]ClusterDocSummary, len(docs))
	var noiseSamples []ClusterDocSummary
	for i, d := range docs {
		ds := ClusterDocSummary{
			ID:              d.Record.ID,
			Source:          d.Record.Source,
			Title:           d.Record.Text,
			ClusterID:       d.ClusterID,
			MembershipProb:  d.MembershipProb,
			PointSilhouette: d.PointSilhouette,
			PredTopic:       d.Scored.TopicKoName,
			PrimaryNFC:      d.Scored.PrimaryNFC.KoName,
			SentimentLabel:  d.Scored.SentimentLabel,
			Intensity:       d.Scored.SocialOpinionIntensity,
			Entities:        d.Entities,
		}
		docSummaries[i] = ds
		if d.ClusterID < 0 && len(noiseSamples) < 15 {
			noiseSamples = append(noiseSamples, ds)
		}
	}

	totalDocs := len(docs)
	clusteredDocs := totalDocs - noiseCount
	noiseRatio := 0.0
	if totalDocs > 0 {
		noiseRatio = float64(noiseCount) / float64(totalDocs)
	}

	report := &ClusteringReport{
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Validation: ValidationMetrics{
			SilhouetteScore:          silOverall,
			SilhouetteTarget:         cfg.Clustering.SilhouetteTarget,
			DunnIndexStrict:          strictDunn,
			DunnIndexCentroid:        centroidDunn,
			MeanNPMICoherence:        meanNPMI,
			MeanUMassCoherence:       meanUMass,
			NumClusters:              hdb.NumClusters,
			TotalDocs:                totalDocs,
			ClusteredDocs:            clusteredDocs,
			NoiseDocs:                noiseCount,
			NoiseRatio:               noiseRatio,
			CrossDomainClusters:      crossDomainClusters,
			EntityGroupCount:         len(entityGroups),
			CrossDomainEntityBridges: len(crossBridges),
			HumanEvalSampleCount:     humanSampleCount,
			HumanEvalTargetAgreement: cfg.Clustering.HumanEvalTargetAgreement,
		},
		Clusters:           summaries,
		NoiseSamples:       noiseSamples,
		EntityGroups:       limitEntityGroups(entityGroups, 50),
		CrossDomainBridges: limitEntityGroups(crossBridges, 30),
		Documents:          docSummaries,
	}

	if err := WriteClusteringReports(report, cfg.Paths.OutputDir, tax); err != nil {
		return nil, err
	}
	return report, nil
}

// WriteClusteringReports writes clusters_report.json and clustering_validation_report.md.
func WriteClusteringReports(report *ClusteringReport, outputDir string, _ *nfc.Taxonomy) error {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("create output dir %q: %w", outputDir, err)
	}
	jsonPath := filepath.Join(outputDir, "clusters_report.json")
	raw, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal cluster report: %w", err)
	}
	if err := os.WriteFile(jsonPath, raw, 0644); err != nil {
		return fmt.Errorf("write %q: %w", jsonPath, err)
	}

	mdPath := filepath.Join(outputDir, "clustering_validation_report.md")
	var b strings.Builder
	v := report.Validation
	b.WriteString("# 다차원 의미·엔티티 기반 뉴스+리뷰 클러스터링 및 3단계 검증 보고서\n\n")
	fmt.Fprintf(&b, "- **생성 시각 (UTC)**: %s\n", report.GeneratedAt)
	fmt.Fprintf(&b, "- **총 분석 문서**: %d 건 (군집 배정 %d 건 / HDBSCAN 노이즈 필터링 %d 건 [%.1f%%])\n",
		v.TotalDocs, v.ClusteredDocs, v.NoiseDocs, v.NoiseRatio*100)
	fmt.Fprintf(&b, "- **발견된 HDBSCAN 군집 수**: %d 개 (뉴스+리뷰 혼합 군집 %d 개)\n\n",
		v.NumClusters, v.CrossDomainClusters)

	b.WriteString("## 1. 3단계 검증 스코어카드 (Validation Scorecard)\n\n")
	b.WriteString("| 검증 단계 | 검증 척도 | 실측 결과 | 목표 / 해석 |\n")
	b.WriteString("|---|---|---:|---|\n")
	silStatus := "양호 (군집 분리 명확)"
	if v.SilhouetteScore < v.SilhouetteTarget {
		silStatus = fmt.Sprintf("목표(%.2f) 미만 - 혼합 스타일 반영", v.SilhouetteTarget)
	}
	fmt.Fprintf(&b, "| 1. 수학적 공간 타당성 | Cosine Silhouette Score | **%.4f** | 목표 >= %.2f (%s) |\n",
		v.SilhouetteScore, v.SilhouetteTarget, silStatus)
	fmt.Fprintf(&b, "| 1. 수학적 공간 타당성 | Dunn Index (Centroid / Strict) | **%.4f** / %.4f | 값이 클수록 군집 간 분리도 우수 |\n",
		v.DunnIndexCentroid, v.DunnIndexStrict)
	fmt.Fprintf(&b, "| 2. 키워드 일관성 | Mean NPMI Topic Coherence | **%.4f** | 양수일수록 동일 주제어 응집도 높음 |\n",
		v.MeanNPMICoherence)
	fmt.Fprintf(&b, "| 2. 키워드 일관성 | Mean UMass Topic Coherence | **%.4f** | 0에 가까울수록 공동 출현 일관성 우수 |\n",
		v.MeanUMassCoherence)
	fmt.Fprintf(&b, "| 3. 전문가 인간 검증 | Gold Standard 층화 샘플 | **%d 건 추출** | `human_eval_sample.csv` 기준 정합성 >= %.0f%% |\n\n",
		v.HumanEvalSampleCount, v.HumanEvalTargetAgreement*100)

	b.WriteString("## 2. 방식 ① & ③: 하이브리드 임베딩 HDBSCAN 군집 및 계층적 주제 분류\n\n")
	b.WriteString("| 군집 | 규모 (YNAT/NSMC) | 계층적 주제 라벨 | Silhouette | NPMI | Top-10 c-TF-IDF 키워드 |\n")
	b.WriteString("|---:|---:|---|---:|---:|---|\n")
	for _, c := range report.Clusters {
		lbl := c.HierarchicalLabel
		if c.LLMRefinedLabel != "" {
			lbl = c.LLMRefinedLabel
		}
		fmt.Fprintf(&b, "| C%02d | %d (%d / %d) | %s | %.3f | %.3f | %s |\n",
			c.ClusterID, c.Size, c.YNATCount, c.NSMCCount, lbl,
			c.MeanSilhouette, c.NPMICoherence, strings.Join(c.TopKeywords, ", "))
	}
	b.WriteString("\n")

	b.WriteString("### 군집별 대표 문서 (Medoids: 뉴스 + 리뷰 스타일 브리지)\n\n")
	for _, c := range report.Clusters {
		fmt.Fprintf(&b, "#### 군집 C%02d — %s (여론강도 평균 %.3f, 부정비율 평균 %.1f%%)\n\n",
			c.ClusterID, c.HierarchicalLabel, c.MeanIntensity, c.MeanNegativeProb*100)
		if len(c.TopEntities) > 0 {
			fmt.Fprintf(&b, "- **주요 엔티티 앵커**: %s\n", strings.Join(c.TopEntities, ", "))
		}
		b.WriteString("\n| 출처 | 주제 | 핵심감정 | 감성 | 소속확률 | 제목/본문 |\n")
		b.WriteString("|---|---|---|---|---:|---|\n")
		for _, rd := range c.RepresentativeDocs {
			fmt.Fprintf(&b, "| %s | %s | %s | %s | %.2f | %s |\n",
				rd.Source, rd.TopicKoName, rd.PrimaryNFC, rd.SentimentLabel, rd.MembershipProb, escapeMDCluster(rd.Title))
		}
		b.WriteString("\n")
	}

	b.WriteString("## 3. 방식 ②: 개체명(NER)·이슈 앵커 그룹화 및 교차 도메인 브리지 (뉴스 ↔ 리뷰)\n\n")
	fmt.Fprintf(&b, "- **전체 엔티티 그룹(2건 이상)**: %d 개 / **뉴스-리뷰 교차 브리지**: %d 개\n\n",
		v.EntityGroupCount, v.CrossDomainEntityBridges)
	if len(report.CrossDomainBridges) > 0 {
		b.WriteString("| 유형 | 엔티티/이슈 | 총 건수 | 뉴스(YNAT) | 리뷰(NSMC) | 주 토픽 | 주 핵심감정 | 대표 예시 |\n")
		b.WriteString("|---|---|---:|---:|---:|---|---|---|\n")
		for _, eg := range report.CrossDomainBridges {
			sample := "-"
			if len(eg.SampleTitles) > 0 {
				sample = escapeMDCluster(strings.Join(eg.SampleTitles[:min(2, len(eg.SampleTitles))], " / "))
			}
			fmt.Fprintf(&b, "| %s | **%s** | %d | %d | %d | %s | %s | %s |\n",
				eg.Type, eg.Entity, eg.TotalCount, eg.YNATCount, eg.NSMCCount, eg.DominantTopic, eg.DominantNFC, sample)
		}
		b.WriteString("\n")
	} else {
		b.WriteString("*현재 샘플 내 뉴스와 리뷰에 동시 출현한 교차 엔티티가 없습니다.*\n\n")
	}

	if len(report.NoiseSamples) > 0 {
		b.WriteString("## 4. HDBSCAN 자동 노이즈(-1) 필터링 샘플\n\n")
		b.WriteString("| 출처 | 예측 주제 | 핵심감정 | 감성 | 본문 |\n")
		b.WriteString("|---|---|---|---|---|\n")
		for _, ns := range report.NoiseSamples {
			fmt.Fprintf(&b, "| %s | %s | %s | %s | %s |\n",
				ns.Source, ns.PredTopic, ns.PrimaryNFC, ns.SentimentLabel, escapeMDCluster(ns.Title))
		}
	}

	if err := os.WriteFile(mdPath, []byte(b.String()), 0644); err != nil {
		return fmt.Errorf("write %q: %w", mdPath, err)
	}
	return nil
}

func topNKeys(m map[string]int, limit int) []string {
	type kv struct {
		k string
		v int
	}
	items := make([]kv, 0, len(m))
	for k, v := range m {
		items = append(items, kv{k: k, v: v})
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].v != items[j].v {
			return items[i].v > items[j].v
		}
		return items[i].k < items[j].k
	})
	if limit > len(items) {
		limit = len(items)
	}
	out := make([]string, limit)
	for i := 0; i < limit; i++ {
		out[i] = items[i].k
	}
	return out
}

func limitEntityGroups(in []EntityGroup, limit int) []EntityGroup {
	if len(in) <= limit {
		return in
	}
	return in[:limit]
}

func escapeMDCluster(s string) string {
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "|", "\\|")
	return s
}
