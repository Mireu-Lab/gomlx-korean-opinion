package cluster

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/data"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/sorter"
)

func TestRunHDBSCANAndMathValidation(t *testing.T) {
	// Construct 3 distinct dense clusters (inter-cluster cosine dist ~0.35) + 4 scattered outlier noise points (dist >= 0.9).
	var vectors [][]float64
	centers := [][]float64{
		{0.82, 0.57, 0.00, 0.00, 0.00, 0.00},
		{0.82, -0.57, 0.00, 0.00, 0.00, 0.00},
		{0.82, 0.00, 0.57, 0.00, 0.00, 0.00},
	}
	for _, ctr := range centers {
		for k := 0; k < 10; k++ {
			v := make([]float64, len(ctr))
			copy(v, ctr)
			v[3] = 0.015 * float64(k-5)
			vectors = append(vectors, l2NormalizeFloat64(v))
		}
	}
	// Add 4 distant outlier noise points in orthogonal / antipodal subspaces.
	vectors = append(vectors,
		l2NormalizeFloat64([]float64{0, 0, 0, 1, 0, 0}),
		l2NormalizeFloat64([]float64{0, 0, 0, 0, 1, 0}),
		l2NormalizeFloat64([]float64{0, 0, 0, 0, 0, 1}),
		l2NormalizeFloat64([]float64{-1, 0, 0, 0, 0, 0}),
	)

	dist := ComputeCosineDistanceMatrix(vectors)
	res := RunHDBSCAN(dist, len(vectors), 5, 4, 0.0)
	if res.NumClusters != 3 {
		t.Fatalf("expected 3 HDBSCAN clusters, got %d (labels=%v)", res.NumClusters, res.Labels)
	}
	for i := 30; i < 34; i++ {
		if res.Labels[i] != -1 {
			t.Fatalf("expected outlier point %d to be noise (-1), got cluster %d", i, res.Labels[i])
		}
	}

	sil, _, _ := ComputeSilhouette(dist, len(vectors), res.Labels)
	if sil <= 0.7 {
		t.Fatalf("expected high Silhouette score > 0.7, got %v", sil)
	}
	strictDunn, centroidDunn := ComputeDunnIndex(dist, vectors, res.Labels)
	if strictDunn <= 0 || centroidDunn <= 0 {
		t.Fatalf("expected positive Dunn indices, got strict=%v centroid=%v", strictDunn, centroidDunn)
	}
}

func TestEntityExtractorAndCrossDomainBridges(t *testing.T) {
	ext, err := LoadEntityExtractor(filepath.Join("..", "..", "config", "entity_rules.yaml"))
	if err != nil {
		t.Fatalf("LoadEntityExtractor: %v", err)
	}

	docs := []ClusterDoc{
		{
			Record: data.Record{ID: "y1", Source: "ynat", Text: "삼성전자 반도체 투자 확대… 미국 시장 공략"},
			Scored: sorter.ScoredArticle{TopicKoName: "경제", PrimaryNFC: sorter.ScoredFeeling{KoName: "경쟁심"}, SocialOpinionIntensity: 0.7},
		},
		{
			Record: data.Record{ID: "n1", Source: "nsmc", Text: "영화 《기생충》 봉준호 감독 최고 작품! 빈부격차 현실 반영"},
			Scored: sorter.ScoredArticle{TopicKoName: "생활문화", PrimaryNFC: sorter.ScoredFeeling{KoName: "그리움"}, SocialOpinionIntensity: 0.5},
		},
		{
			Record: data.Record{ID: "y2", Source: "ynat", Text: "봉준호 감독 《기생충》 아카데미 흥행 돌풍… 한국 영화 위상"},
			Scored: sorter.ScoredArticle{TopicKoName: "생활문화", PrimaryNFC: sorter.ScoredFeeling{KoName: "경쟁심"}, SocialOpinionIntensity: 0.6},
		},
	}
	for i := range docs {
		docs[i].Entities = ext.Extract(docs[i].Record.Text)
	}
	if len(docs[0].Entities) == 0 || len(docs[1].Entities) == 0 {
		t.Fatalf("expected entities extracted from Korean news & review")
	}

	allGroups, bridges := BuildEntityGroups(docs)
	if len(allGroups) == 0 || len(bridges) == 0 {
		t.Fatalf("expected cross-domain bridges between ynat and nsmc, got groups=%d bridges=%d", len(allGroups), len(bridges))
	}
}

func TestTopicCoherenceAndHumanEvalScoring(t *testing.T) {
	ext, err := LoadEntityExtractor(filepath.Join("..", "..", "config", "entity_rules.yaml"))
	if err != nil {
		t.Fatalf("LoadEntityExtractor: %v", err)
	}
	rawDocs := []string{
		"금리 인상 물가 부담 가계 경제 위기",
		"고물가 금리 압박 서민 경제 불안",
		"금리 동결 물가 안정 경제 전망",
		"영화 관객 흥행 박스오피스 대작",
		"영화 감동 명작 관객 호평 흥행",
		"박스오피스 1위 영화 흥행 돌풍",
	}
	docs := make([]ClusterDoc, len(rawDocs))
	for i, txt := range rawDocs {
		docs[i] = ClusterDoc{
			Record:    data.Record{ID: fmt.Sprintf("d%d", i), Source: "ynat", Text: txt},
			ClusterID: i / 3,
			Keywords:  ExtractDocKeywords(txt, ext.Stopwords()),
		}
	}
	coherent := map[int][]string{
		0: {"금리", "물가", "경제"},
	}
	incoherent := map[int][]string{
		0: {"금리", "박스오피스", "명작"},
	}
	npmiGood, _, _ := ComputeTopicCoherence(docs, coherent, 1e-9)
	npmiBad, _, _ := ComputeTopicCoherence(docs, incoherent, 1e-9)
	if npmiGood <= npmiBad {
		t.Fatalf("expected coherent keywords NPMI (%v) > incoherent (%v)", npmiGood, npmiBad)
	}

	tmpDir := t.TempDir()
	sumMap := map[int]ClusterSummary{
		0: {ClusterID: 0, HierarchicalLabel: "[상위: 경제] - [중위: 불안/부담감] - [하위: 금리·물가]", TopKeywords: []string{"금리", "물가"}},
		1: {ClusterID: 1, HierarchicalLabel: "[상위: 생활문화] - [중위: 경쟁심/경쟁심] - [하위: 영화·흥행]", TopKeywords: []string{"영화", "흥행"}},
	}
	for i := range docs {
		docs[i].Scored = sorter.ScoredArticle{TopicKoName: "경제", PrimaryNFC: sorter.ScoredFeeling{KoName: "불안"}, SentimentLabel: "부정"}
	}
	cnt, err := ExportHumanEvalSample(docs, sumMap, 4, 42, tmpDir)
	if err != nil || cnt != 4 {
		t.Fatalf("ExportHumanEvalSample: cnt=%d err=%v", cnt, err)
	}

	ratedPath := filepath.Join(tmpDir, "rated.csv")
	csvData := "sample_id,doc_id,source,cluster_id,hierarchical_label,predicted_topic,predicted_nfc,sentiment,top_keywords,title,human_agree,human_corrected_topic,evaluator_note\n" +
		"S-001,d0,ynat,0,lbl,경제,불안,부정,금리,t0,1,,\n" +
		"S-002,d1,ynat,0,lbl,경제,불안,부정,금리,t1,1,,\n" +
		"S-003,d2,ynat,0,lbl,경제,불안,부정,금리,t2,1,,\n" +
		"S-004,d3,ynat,1,lbl,경제,불안,부정,영화,t3,1,,\n" +
		"S-005,d4,ynat,1,lbl,생활문화,불안,긍정,영화,t4,0,사회,\n"
	if err := os.WriteFile(ratedPath, []byte(csvData), 0644); err != nil {
		t.Fatalf("write rated csv: %v", err)
	}
	sc, err := ScoreHumanEvalCSV(ratedPath, 0.80)
	if err != nil {
		t.Fatalf("ScoreHumanEvalCSV: %v", err)
	}
	if sc.TotalRated != 5 || sc.AgreeCount != 4 || math.Abs(sc.AgreementRate-0.80) > 1e-6 || !sc.PassesTarget {
		t.Fatalf("unexpected human eval score: %+v", sc)
	}
}
