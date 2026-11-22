package graphkg

import (
	"archive/zip"
	"os"
	"path/filepath"
	"testing"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/logging"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/nfc"
)

func TestParseUnihanOMWAndPolysemyGraphRoundTrip(t *testing.T) {
	tmpDir := t.TempDir()

	// Create synthetic Unihan.zip with Unihan_Readings.txt.
	zipPath := filepath.Join(tmpDir, "Unihan.zip")
	zf, err := os.Create(zipPath)
	if err != nil {
		t.Fatalf("create temp zip: %v", err)
	}
	zw := zip.NewWriter(zf)
	fw, err := zw.Create("Unihan_Readings.txt")
	if err != nil {
		t.Fatalf("create zip entry: %v", err)
	}
	_, _ = fw.Write([]byte(
		"# Unihan readings fixture\n" +
			"U+756B\tkHangul\t화:0E\n" +
			"U+756B\tkDefinition\ta painting, picture, drawing; to draw\n" +
			"U+5BB6\tkHangul\t가:0E\n" +
			"U+5BB6\tkDefinition\thouse, home; specialist\n" +
			"U+706B\tkHangul\t화:0E\n" +
			"U+706B\tkDefinition\tfire, flame; anger, rage\n" +
			"U+5B64\tkHangul\t고:0E\n" +
			"U+5B64\tkDefinition\torphan; solitary, lonely\n" +
			"U+7368\tkHangul\t독:0E\n" +
			"U+7368\tkDefinition\talone, single, solitary\n",
	))
	if err := zw.Close(); err != nil {
		t.Fatalf("close zip writer: %v", err)
	}
	if err := zf.Close(); err != nil {
		t.Fatalf("close zip file: %v", err)
	}

	unihan, err := ParseUnihanZip(zipPath)
	if err != nil {
		t.Fatalf("ParseUnihanZip failed: %v", err)
	}
	if len(unihan.ByHangul["화"]) < 2 {
		t.Fatalf("expected >= 2 Hanja entries for syllable '화', got %d", len(unihan.ByHangul["화"]))
	}

	// Create synthetic OMW tab fixtures.
	korPath := filepath.Join(tmpDir, "wn-wikt-kor.tab")
	engPath := filepath.Join(tmpDir, "wn-data-eng.tab")
	jpnPath := filepath.Join(tmpDir, "wn-data-jpn.tab")
	cmnPath := filepath.Join(tmpDir, "wn-data-cmn.tab")
	_ = os.WriteFile(korPath, []byte("# Wikt\n10391653-n\tkor:lemma\t화가\n07516354-n\tkor:lemma\t분노\n"), 0644)
	_ = os.WriteFile(engPath, []byte("# Eng\n10391653-n\tlemma\tpainter\n07516354-n\tlemma\tanger\n"), 0644)
	_ = os.WriteFile(jpnPath, []byte("# Jpn\n10391653-n\tjpn:lemma\t画家\n07516354-n\tjpn:lemma\t怒り\n"), 0644)
	_ = os.WriteFile(cmnPath, []byte("# Cmn\n10391653-n\tcmn:lemma\t画家\n07516354-n\tcmn:lemma\t愤怒\n"), 0644)

	korMap, err := ParseOMWTab(korPath)
	if err != nil {
		t.Fatalf("ParseOMWTab(kor): %v", err)
	}
	engMap, _ := ParseOMWTab(engPath)
	jpnMap, _ := ParseOMWTab(jpnPath)
	cmnMap, _ := ParseOMWTab(cmnPath)
	concepts := MergeOMWSynsets(korMap, nil, engMap, jpnMap, cmnMap)
	if len(concepts) != 2 {
		t.Fatalf("expected 2 merged OMW concepts, got %d", len(concepts))
	}

	cfg, err := LoadRootsConfig(filepath.Join("..", "..", "config", "multilingual_roots.yaml"))
	if err != nil {
		t.Fatalf("LoadRootsConfig: %v", err)
	}
	tax, err := nfc.LoadTaxonomy(
		filepath.Join("..", "..", "config", "topics.yaml"),
		filepath.Join("..", "..", "config", "nfc_taxonomy.yaml"),
	)
	if err != nil {
		t.Fatalf("LoadTaxonomy: %v", err)
	}

	corpus := []string{
		"남춘모 부조적 단색화가 펼쳐 보이는 자연 미술 전시회",
		"차병원 한국여류화가협회 자선전시회 개최 미술 작품",
		"고물가 금리 부담에 시민들 화가 났다 거센 분노 항의",
		"사기 피해 경찰 고소 구속 기만 범죄",
		"국가대표 선수 감독 승리 사기 진작 응원",
	}
	log, err := logging.NewDualLogger(tmpDir, "status.log", "processing.log", "INFO", false)
	if err != nil {
		t.Fatalf("NewDualLogger: %v", err)
	}
	defer func() { _ = log.Close() }()

	g, err := BuildGraphFromParsed(cfg, unihan, concepts, corpus, tax, log)
	if err != nil {
		t.Fatalf("BuildGraphFromParsed: %v", err)
	}

	assertTwoSensesBranching := func(surface, senseGlossA, hanjaA, senseGlossB, hanjaB string) {
		t.Helper()
		senses := g.CandidateSenses(surface)
		if len(senses) < 2 {
			t.Fatalf("expected >= 2 candidate senses for %q, got %d", surface, len(senses))
		}
		var foundA, foundB bool
		for _, s := range senses {
			hIDs, enIDs, jazhIDs := g.SenseRoots(s.ID)
			if len(hIDs)+len(enIDs)+len(jazhIDs) == 0 {
				continue
			}
			hasHanja := func(target string) bool {
				for _, hid := range hIDs {
					if g.Nodes[hid].Label == target {
						return true
					}
				}
				return false
			}
			if s.Label == senseGlossA && hasHanja(hanjaA) {
				foundA = true
			}
			if s.Label == senseGlossB && hasHanja(hanjaB) {
				foundB = true
			}
		}
		if !foundA || !foundB {
			t.Fatalf("expected %q senses (%s[%s]=%t, %s[%s]=%t) to branch separately",
				surface, senseGlossA, hanjaA, foundA, senseGlossB, hanjaB, foundB)
		}
	}

	assertTwoSensesBranching("화가", "painter/artist", "畫", "anger/rage", "火")
	assertTwoSensesBranching("사기", "fraud/scam", "詐", "morale/spirit", "士")

	graphPath := filepath.Join(tmpDir, "ko_hub_graph.bin")
	if err := g.Save(graphPath); err != nil {
		t.Fatalf("Graph.Save: %v", err)
	}
	loaded, err := Load(graphPath)
	if err != nil {
		t.Fatalf("Graph.Load: %v", err)
	}
	if loaded.Stats.NumNodes != g.Stats.NumNodes || loaded.Stats.NumEdges != g.Stats.NumEdges {
		t.Fatalf("CSR round-trip mismatch: nodes %d vs %d, edges %d vs %d",
			loaded.Stats.NumNodes, g.Stats.NumNodes, loaded.Stats.NumEdges, g.Stats.NumEdges)
	}
	if len(loaded.CandidateSenses("화가")) != len(g.CandidateSenses("화가")) {
		t.Fatalf("candidate senses mismatch after reload")
	}
}
