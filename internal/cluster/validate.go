package cluster

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// ComputeSilhouette computes exact cosine-distance Silhouette scores overall, per point, and per cluster.
func ComputeSilhouette(dist []float64, n int, labels []int) (overall float64, pointSil []float64, clusterSil map[int]float64) {
	pointSil = make([]float64, n)
	clusterSil = make(map[int]float64)

	clusters := make(map[int][]int)
	for i, lbl := range labels {
		if lbl >= 0 {
			clusters[lbl] = append(clusters[lbl], i)
		}
	}
	if len(clusters) < 2 {
		return 0, pointSil, clusterSil
	}

	clusterIDs := make([]int, 0, len(clusters))
	for cid := range clusters {
		clusterIDs = append(clusterIDs, cid)
	}
	sort.Ints(clusterIDs)

	var sumAll float64
	var countAll int
	clusterSums := make(map[int]float64, len(clusters))
	clusterCounts := make(map[int]int, len(clusters))

	for i, ci := range labels {
		if ci < 0 {
			continue
		}
		sameMembers := clusters[ci]
		var a float64
		if len(sameMembers) > 1 {
			var sumA float64
			for _, j := range sameMembers {
				if j != i {
					sumA += dist[i*n+j]
				}
			}
			a = sumA / float64(len(sameMembers)-1)
		}

		b := math.Inf(1)
		for _, ck := range clusterIDs {
			if ck == ci {
				continue
			}
			otherMembers := clusters[ck]
			var sumB float64
			for _, j := range otherMembers {
				sumB += dist[i*n+j]
			}
			meanB := sumB / float64(len(otherMembers))
			if meanB < b {
				b = meanB
			}
		}

		var s float64
		if len(sameMembers) > 1 && !math.IsInf(b, 1) {
			denom := math.Max(a, b)
			if denom > 0 {
				s = (b - a) / denom
			}
		}
		pointSil[i] = s
		sumAll += s
		countAll++
		clusterSums[ci] += s
		clusterCounts[ci]++
	}

	if countAll > 0 {
		overall = sumAll / float64(countAll)
	}
	for _, cid := range clusterIDs {
		if clusterCounts[cid] > 0 {
			clusterSil[cid] = clusterSums[cid] / float64(clusterCounts[cid])
		}
	}
	return overall, pointSil, clusterSil
}

// ComputeDunnIndex computes both strict complete-linkage Dunn Index and centroid/mean-intra Dunn Index.
func ComputeDunnIndex(dist []float64, vectors [][]float64, labels []int) (strictDunn float64, centroidDunn float64) {
	n := len(labels)
	clusters := make(map[int][]int)
	for i, lbl := range labels {
		if lbl >= 0 {
			clusters[lbl] = append(clusters[lbl], i)
		}
	}
	if len(clusters) < 2 {
		return 0, 0
	}
	cids := make([]int, 0, len(clusters))
	for cid := range clusters {
		cids = append(cids, cid)
	}
	sort.Ints(cids)

	// Strict Dunn: min inter-cluster distance / max intra-cluster diameter.
	minInter := math.Inf(1)
	maxIntra := 0.0
	for a := 0; a < len(cids); a++ {
		ptsA := clusters[cids[a]]
		for iIdx := 0; iIdx < len(ptsA); iIdx++ {
			i := ptsA[iIdx]
			for jIdx := iIdx + 1; jIdx < len(ptsA); jIdx++ {
				j := ptsA[jIdx]
				if d := dist[i*n+j]; d > maxIntra {
					maxIntra = d
				}
			}
		}
		for b := a + 1; b < len(cids); b++ {
			ptsB := clusters[cids[b]]
			for _, i := range ptsA {
				for _, j := range ptsB {
					if d := dist[i*n+j]; d < minInter {
						minInter = d
					}
				}
			}
		}
	}
	if maxIntra > 0 && !math.IsInf(minInter, 1) {
		strictDunn = minInter / maxIntra
	}

	// Centroid / Mean-Intra Dunn (robust to single outliers).
	dim := len(vectors[0])
	centroids := make(map[int][]float64, len(cids))
	maxMeanIntra := 0.0
	for _, cid := range cids {
		pts := clusters[cid]
		cen := make([]float64, dim)
		for _, idx := range pts {
			for d, v := range vectors[idx] {
				cen[d] += v
			}
		}
		cen = l2NormalizeFloat64(cen)
		centroids[cid] = cen

		var sumIntra float64
		for _, idx := range pts {
			var dot float64
			for d, v := range vectors[idx] {
				dot += v * cen[d]
			}
			cd := 1.0 - dot
			if cd < 0 {
				cd = 0
			}
			sumIntra += cd
		}
		meanIntra := sumIntra / float64(len(pts))
		if meanIntra > maxMeanIntra {
			maxMeanIntra = meanIntra
		}
	}
	minCentroidInter := math.Inf(1)
	for a := 0; a < len(cids); a++ {
		ca := centroids[cids[a]]
		for b := a + 1; b < len(cids); b++ {
			cb := centroids[cids[b]]
			var dot float64
			for d := range ca {
				dot += ca[d] * cb[d]
			}
			cd := 1.0 - dot
			if cd < 0 {
				cd = 0
			}
			if cd < minCentroidInter {
				minCentroidInter = cd
			}
		}
	}
	if maxMeanIntra > 0 && !math.IsInf(minCentroidInter, 1) {
		centroidDunn = minCentroidInter / maxMeanIntra
	}
	return strictDunn, centroidDunn
}

// ComputeTopicCoherence computes information-theoretic NPMI and UMass coherence over cluster Top-K keywords.
func ComputeTopicCoherence(docs []ClusterDoc, clusterKeywords map[int][]string, eps float64) (meanNPMI, meanUMass float64, perCluster map[int][2]float64) {
	perCluster = make(map[int][2]float64, len(clusterKeywords))
	if len(docs) == 0 || len(clusterKeywords) == 0 {
		return 0, 0, perCluster
	}

	// Build inverted index (sorted slice of doc indices per keyword) for fast exact co-occurrence.
	postings := make(map[string][]int)
	for docIdx, d := range docs {
		seen := make(map[string]struct{}, len(d.Keywords))
		for _, kw := range d.Keywords {
			if _, ok := seen[kw]; ok {
				continue
			}
			seen[kw] = struct{}{}
			postings[kw] = append(postings[kw], docIdx)
		}
	}

	countCoOccur := func(a, b []int) int {
		i, j, c := 0, 0, 0
		for i < len(a) && j < len(b) {
			if a[i] == b[j] {
				c++
				i++
				j++
			} else if a[i] < b[j] {
				i++
			} else {
				j++
			}
		}
		return c
	}

	nDocs := float64(len(docs))
	var sumNPMIClusters, sumUMassClusters float64
	validClusters := 0

	for cid, kws := range clusterKeywords {
		if len(kws) < 2 {
			perCluster[cid] = [2]float64{0, 0}
			continue
		}
		var pairNPMISum, pairUMassSum float64
		var pairCount int
		for i := 0; i < len(kws); i++ {
			pa := postings[kws[i]]
			pA := float64(len(pa)) / nDocs
			for j := i + 1; j < len(kws); j++ {
				pb := postings[kws[j]]
				pB := float64(len(pb)) / nDocs
				co := countCoOccur(pa, pb)
				pAB := float64(co) / nDocs

				pmi := math.Log((pAB + eps) / (pA*pB + eps))
				denom := -math.Log(pAB + eps)
				var npmi float64
				if denom > 0 {
					npmi = pmi / denom
				}
				// UMass: log((D(w_i, w_j) + 1) / D(w_i)) where w_i is higher-ranked keyword.
				umass := math.Log((float64(co) + 1.0) / (float64(len(pa)) + 1.0))

				pairNPMISum += npmi
				pairUMassSum += umass
				pairCount++
			}
		}
		if pairCount > 0 {
			cNPMI := pairNPMISum / float64(pairCount)
			cUMass := pairUMassSum / float64(pairCount)
			perCluster[cid] = [2]float64{cNPMI, cUMass}
			sumNPMIClusters += cNPMI
			sumUMassClusters += cUMass
			validClusters++
		}
	}
	if validClusters > 0 {
		meanNPMI = sumNPMIClusters / float64(validClusters)
		meanUMass = sumUMassClusters / float64(validClusters)
	}
	return meanNPMI, meanUMass, perCluster
}

// ExportHumanEvalSample writes a stratified Gold Standard human evaluation CSV + rubric to outputDir.
func ExportHumanEvalSample(
	docs []ClusterDoc,
	summaryMap map[int]ClusterSummary,
	sampleSize int,
	seed int64,
	outputDir string,
) (int, error) {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return 0, fmt.Errorf("create output dir %q: %w", outputDir, err)
	}
	if sampleSize <= 0 || len(docs) == 0 {
		return 0, nil
	}
	if sampleSize > len(docs) {
		sampleSize = len(docs)
	}

	rng := rand.New(rand.NewSource(seed))
	byCluster := make(map[int][]ClusterDoc)
	var clusterIDs []int
	for _, d := range docs {
		if _, ok := byCluster[d.ClusterID]; !ok {
			clusterIDs = append(clusterIDs, d.ClusterID)
		}
		byCluster[d.ClusterID] = append(byCluster[d.ClusterID], d)
	}
	sort.Ints(clusterIDs)
	for _, cid := range clusterIDs {
		s := byCluster[cid]
		rng.Shuffle(len(s), func(i, j int) { s[i], s[j] = s[j], s[i] })
	}

	var sampled []ClusterDoc
	cursors := make(map[int]int, len(clusterIDs))
	// Round-robin stratified draw across clusters (reserve noise for tail if needed).
	for len(sampled) < sampleSize {
		progressed := false
		for _, cid := range clusterIDs {
			if len(sampled) >= sampleSize {
				break
			}
			cur := cursors[cid]
			if cur < len(byCluster[cid]) {
				sampled = append(sampled, byCluster[cid][cur])
				cursors[cid] = cur + 1
				progressed = true
			}
		}
		if !progressed {
			break
		}
	}

	csvPath := filepath.Join(outputDir, "human_eval_sample.csv")
	f, err := os.Create(csvPath)
	if err != nil {
		return 0, fmt.Errorf("create human eval csv %q: %w", csvPath, err)
	}
	defer func() { _ = f.Close() }()

	w := csv.NewWriter(f)
	header := []string{
		"sample_id", "doc_id", "source", "cluster_id", "hierarchical_label",
		"predicted_topic", "predicted_nfc", "sentiment", "top_keywords",
		"title", "human_agree", "human_corrected_topic", "evaluator_note",
	}
	if err := w.Write(header); err != nil {
		return 0, err
	}
	for i, d := range sampled {
		label := "노이즈/비군집(-1)"
		kws := ""
		if sum, ok := summaryMap[d.ClusterID]; ok {
			label = sum.HierarchicalLabel
			kws = strings.Join(sum.TopKeywords, ", ")
		}
		row := []string{
			fmt.Sprintf("S-%03d", i+1),
			d.Record.ID,
			d.Record.Source,
			fmt.Sprintf("%d", d.ClusterID),
			label,
			d.Scored.TopicKoName,
			d.Scored.PrimaryNFC.KoName,
			d.Scored.SentimentLabel,
			kws,
			d.Record.Text,
			"",
			"",
			"",
		}
		if err := w.Write(row); err != nil {
			return 0, err
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		return 0, err
	}

	rubricPath := filepath.Join(outputDir, "human_eval_rubric.md")
	rubric := fmt.Sprintf(`# Gold Standard 전문가 인간 검증 (Human Evaluation) 가이드

- **추출 샘플 수**: %d 건 (군집별 층화 추출)
- **평가 파일**: `+"`human_eval_sample.csv`"+`
- **목표 기준**: AI 군집/주제 분류와 인간 평가자 기준 정합성 **80%% 이상** (`+"`human_agree = 1`"+` 비율)

## 평가 방법
1. `+"`human_eval_sample.csv`"+` 를 스프레드시트 또는 편집기로 엽니다.
2. 각 행의 `+"`title`"+`(원문)을 읽고, AI가 배정한 `+"`hierarchical_label`"+` / `+"`predicted_topic`"+` / `+"`predicted_nfc`"+` 가 문맥상 타당한지 판단합니다.
3. `+"`human_agree`"+` 컬럼에 타당하면 `+"`1`"+`, 부적합하면 `+"`0`"+` 을 입력합니다.
4. 부적합(`+"`0`"+`)인 경우 `+"`human_corrected_topic`"+` 에 올바른 주제/군집명을 기입합니다.
5. 저장 후 아래 명령으로 정합성(Agreement Rate)과 Cohen's Kappa를 자동 산출합니다:

`+"```bash"+`
go run ./cmd/nfc-analyzer eval-human --config config/config.yaml --input %s
`+"```"+`
`, len(sampled), csvPath)
	if err := os.WriteFile(rubricPath, []byte(rubric), 0644); err != nil {
		return 0, fmt.Errorf("write human eval rubric %q: %w", rubricPath, err)
	}
	return len(sampled), nil
}

// ScoreHumanEvalCSV scores a completed human_eval_sample.csv file for Agreement Rate and Cohen's Kappa.
func ScoreHumanEvalCSV(csvPath string, targetAgreement float64) (*HumanEvalScore, error) {
	f, err := os.Open(csvPath)
	if err != nil {
		return nil, fmt.Errorf("open human eval csv %q: %w", csvPath, err)
	}
	defer func() { _ = f.Close() }()

	r := csv.NewReader(f)
	r.FieldsPerRecord = -1
	header, err := r.Read()
	if err != nil {
		return nil, fmt.Errorf("read header %q: %w", csvPath, err)
	}
	colIdx := make(map[string]int, len(header))
	for i, h := range header {
		colIdx[strings.ToLower(strings.TrimSpace(strings.TrimPrefix(h, "\ufeff")))] = i
	}
	agreeIdx, ok := colIdx["human_agree"]
	if !ok {
		return nil, fmt.Errorf("missing required column human_agree in %q", csvPath)
	}
	predIdx, hasPred := colIdx["predicted_topic"]
	corrIdx, hasCorr := colIdx["human_corrected_topic"]

	var rated, agree int
	aiCounts := make(map[string]int)
	humanCounts := make(map[string]int)
	rowNum := 1
	for {
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read row %d in %q: %w", rowNum+1, csvPath, err)
		}
		rowNum++
		if agreeIdx >= len(row) {
			continue
		}
		valStr := strings.TrimSpace(row[agreeIdx])
		if valStr == "" {
			continue
		}
		val, err := strconv.Atoi(valStr)
		if err != nil || (val != 0 && val != 1) {
			return nil, fmt.Errorf("row %d invalid human_agree=%q (expected 0 or 1)", rowNum, valStr)
		}
		rated++
		if val == 1 {
			agree++
		}
		if hasPred && predIdx < len(row) {
			aiLabel := strings.TrimSpace(row[predIdx])
			humanLabel := aiLabel
			if val == 0 && hasCorr && corrIdx < len(row) {
				if corr := strings.TrimSpace(row[corrIdx]); corr != "" {
					humanLabel = corr
				} else {
					humanLabel = "__disagree__"
				}
			}
			aiCounts[aiLabel]++
			humanCounts[humanLabel]++
		}
	}
	if rated == 0 {
		return nil, fmt.Errorf("no rated rows (human_agree=0/1) found in %q", csvPath)
	}

	po := float64(agree) / float64(rated)
	var pe float64
	if len(aiCounts) > 0 {
		n := float64(rated)
		for lbl, cAI := range aiCounts {
			cHuman := humanCounts[lbl]
			pe += (float64(cAI) / n) * (float64(cHuman) / n)
		}
	} else {
		pe = 0.5
	}
	var kappa float64
	if 1.0-pe > 1e-12 {
		kappa = (po - pe) / (1.0 - pe)
	} else if po == 1.0 {
		kappa = 1.0
	}

	return &HumanEvalScore{
		TotalRated:      rated,
		AgreeCount:      agree,
		AgreementRate:   po,
		CohensKappa:     kappa,
		TargetAgreement: targetAgreement,
		PassesTarget:    po >= targetAgreement,
	}, nil
}
