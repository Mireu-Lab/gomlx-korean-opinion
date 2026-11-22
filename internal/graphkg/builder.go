package graphkg

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/logging"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/nfc"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/tokenizer"
)

type adjKey struct {
	dst      int32
	edgeType EdgeType
}

type graphBuilder struct {
	nodes     []Node
	keyToID   map[string]int32
	adj       []map[adjKey]float32
	senseCues map[int32][]string
	minConf   float32
}

func newGraphBuilder(minConf float64) *graphBuilder {
	return &graphBuilder{
		keyToID:   make(map[string]int32, 32768),
		senseCues: make(map[int32][]string),
		minConf:   float32(minConf),
	}
}

func (b *graphBuilder) getOrAddNode(typ NodeType, label string, suffix ...string) int32 {
	key := fmt.Sprintf("%s:%s", typ.String(), label)
	if len(suffix) > 0 && suffix[0] != "" {
		key = fmt.Sprintf("%s:%s:%s", typ.String(), label, suffix[0])
	}
	if id, ok := b.keyToID[key]; ok {
		return id
	}
	id := int32(len(b.nodes))
	b.nodes = append(b.nodes, Node{
		ID:    id,
		Type:  typ,
		Key:   key,
		Label: label,
	})
	b.keyToID[key] = id
	b.adj = append(b.adj, make(map[adjKey]float32))
	return id
}

func (b *graphBuilder) addEdge(src, dst int32, typ EdgeType, weight float32) {
	if src == dst || weight < b.minConf {
		return
	}
	k := adjKey{dst: dst, edgeType: typ}
	if prev, ok := b.adj[src][k]; ok && prev >= weight {
		return
	}
	b.adj[src][k] = weight
}

func (b *graphBuilder) addCue(senseID int32, cues []string) {
	if len(cues) == 0 {
		return
	}
	existing := b.senseCues[senseID]
	seen := make(map[string]struct{}, len(existing)+len(cues))
	for _, c := range existing {
		seen[c] = struct{}{}
	}
	for _, c := range cues {
		norm := tokenizer.Normalize(c)
		if norm == "" {
			continue
		}
		for _, tok := range strings.Fields(norm) {
			if _, ok := seen[tok]; !ok {
				seen[tok] = struct{}{}
				existing = append(existing, tok)
			}
		}
	}
	b.senseCues[senseID] = existing
}

func (b *graphBuilder) finalizeCSR(maxNeighborsPerNode int) *Graph {
	n := len(b.nodes)
	rowPtr := make([]int32, n+1)
	colIdx := make([]int32, 0, n*4)
	edgeTypes := make([]EdgeType, 0, n*4)
	edgeWeights := make([]float32, 0, n*4)

	nodesByType := make(map[string]int)
	edgesByType := make(map[string]int)

	type rawEdge struct {
		dst    int32
		typ    EdgeType
		weight float32
	}

	for src := 0; src < n; src++ {
		nodesByType[b.nodes[src].Type.String()]++
		rowPtr[src] = int32(len(colIdx))

		edges := make([]rawEdge, 0, len(b.adj[src]))
		for k, w := range b.adj[src] {
			edges = append(edges, rawEdge{dst: k.dst, typ: k.edgeType, weight: w})
		}

		// Preserve all candidate senses & multilingual root links while capping noisy co-occurrences.
		sort.Slice(edges, func(i, j int) bool {
			if edges[i].typ != edges[j].typ {
				return edges[i].typ < edges[j].typ
			}
			if edges[i].weight != edges[j].weight {
				return edges[i].weight > edges[j].weight
			}
			return edges[i].dst < edges[j].dst
		})

		cooccurCount := 0
		for _, e := range edges {
			if e.typ == EdgeCooccursKO {
				if cooccurCount >= maxNeighborsPerNode {
					continue
				}
				cooccurCount++
			}
			colIdx = append(colIdx, e.dst)
			edgeTypes = append(edgeTypes, e.typ)
			edgeWeights = append(edgeWeights, e.weight)
			edgesByType[e.typ.String()]++
		}
	}
	rowPtr[n] = int32(len(colIdx))

	g := &Graph{
		Nodes:       b.nodes,
		RowPtr:      rowPtr,
		ColIdx:      colIdx,
		EdgeTypes:   edgeTypes,
		EdgeWeights: edgeWeights,
		SenseCues:   b.senseCues,
		Stats: GraphStats{
			NumNodes:    n,
			NumEdges:    len(colIdx),
			NodesByType: nodesByType,
			EdgesByType: edgesByType,
		},
	}
	g.RebuildIndices()
	g.Stats.MultiSenseSurfaces = len(g.MultiSenseSurfaces())
	return g
}

// BuildGraph loads cached Unihan and OMW sources from cfg.CacheDir and constructs the KO-Hub CSR graph.
func BuildGraph(cfg *RootsConfig, corpus []string, tax *nfc.Taxonomy, log *logging.DualLogger) (*Graph, error) {
	unihanPath := filepath.Join(cfg.CacheDir, cfg.Sources.UnihanZipFile)
	unihan, err := ParseUnihanZip(unihanPath)
	if err != nil {
		return nil, err
	}
	log.Info("graphkg", "parsed Unihan readings: %d Hangul syllables, %d Hanja entries",
		len(unihan.ByHangul), len(unihan.ByChar))

	korWikt, err := ParseOMWTab(filepath.Join(cfg.CacheDir, cfg.Sources.OMWKorWiktFile))
	if err != nil {
		return nil, err
	}
	korCLDR, err := ParseOMWTab(filepath.Join(cfg.CacheDir, cfg.Sources.OMWKorCLDRFile))
	if err != nil {
		return nil, err
	}
	eng, err := ParseOMWTab(filepath.Join(cfg.CacheDir, cfg.Sources.OMWEngFile))
	if err != nil {
		return nil, err
	}
	jpn, err := ParseOMWTab(filepath.Join(cfg.CacheDir, cfg.Sources.OMWJpnFile))
	if err != nil {
		return nil, err
	}
	cmn, err := ParseOMWTab(filepath.Join(cfg.CacheDir, cfg.Sources.OMWCmnFile))
	if err != nil {
		return nil, err
	}

	concepts := MergeOMWSynsets(korWikt, korCLDR, eng, jpn, cmn)
	log.Info("graphkg", "merged %d multilingual OMW synset concepts (kor_wikt=%d kor_cldr=%d eng=%d jpn=%d cmn=%d)",
		len(concepts), len(korWikt), len(korCLDR), len(eng), len(jpn), len(cmn))

	return BuildGraphFromParsed(cfg, unihan, concepts, corpus, tax, log)
}

// BuildGraphFromParsed builds the KO-Hub heterogeneous CSR graph from already-parsed lexical resources and corpus texts.
func BuildGraphFromParsed(
	cfg *RootsConfig,
	unihan *UnihanLexicon,
	concepts []OMWConcept,
	corpus []string,
	tax *nfc.Taxonomy,
	log *logging.DualLogger,
) (*Graph, error) {
	if cfg == nil {
		return nil, fmt.Errorf("roots config must not be nil")
	}
	b := newGraphBuilder(cfg.Limits.MinEdgeConfidence)
	ew := cfg.EdgeWeights

	// 1. Add curated polysemy & Sino-Korean/NFC seeds first so disambiguation senses stay guaranteed.
	for _, ps := range cfg.PolysemySeeds {
		surfNorm := tokenizer.Normalize(ps.Surface)
		if surfNorm == "" {
			continue
		}
		surfID := b.getOrAddNode(NodeKOSurface, surfNorm)
		for _, ss := range ps.Senses {
			senseID := b.getOrAddNode(NodeKOSense, ss.Gloss, ss.SenseKey)
			if ss.NFCAnchor != "" {
				b.nodes[senseID].NFCAnchor = ss.NFCAnchor
			}
			b.addEdge(surfID, senseID, EdgeHasCandidateSense, ew.HasCandidateSense)
			b.addCue(senseID, ss.ContextCues)

			for _, h := range ss.Hanja {
				h = strings.TrimSpace(h)
				if h == "" {
					continue
				}
				hID := b.getOrAddNode(NodeHanjaMorpheme, h)
				b.addEdge(senseID, hID, EdgeComposedOfHanja, ew.ComposedOfHanja)
				b.addEdge(surfID, hID, EdgeComposedOfHanja, ew.ComposedOfHanja*0.85)
			}
			for _, w := range ss.EN {
				w = strings.ToLower(strings.TrimSpace(w))
				if w == "" {
					continue
				}
				eID := b.getOrAddNode(NodeENConcept, w)
				b.addEdge(senseID, eID, EdgeAlignedEN, ew.AlignedEN)
				b.addEdge(surfID, eID, EdgeAlignedEN, ew.AlignedEN*0.85)
			}
			for _, w := range ss.JA {
				w = strings.TrimSpace(w)
				if w == "" {
					continue
				}
				jID := b.getOrAddNode(NodeJAConcept, w)
				b.addEdge(senseID, jID, EdgeAlignedJA, ew.AlignedJA)
				b.addEdge(surfID, jID, EdgeAlignedJA, ew.AlignedJA*0.85)
			}
			for _, w := range ss.ZH {
				w = strings.TrimSpace(w)
				if w == "" {
					continue
				}
				zID := b.getOrAddNode(NodeZHConcept, w)
				b.addEdge(senseID, zID, EdgeAlignedZH, ew.AlignedZH)
				b.addEdge(surfID, zID, EdgeAlignedZH, ew.AlignedZH*0.85)
			}
			if ss.NFCAnchor != "" {
				nfcEN := b.getOrAddNode(NodeENConcept, strings.ToLower(strings.TrimSpace(ss.NFCAnchor)))
				b.addEdge(senseID, nfcEN, EdgeNFCAnchor, ew.NFCAnchor)
			}
		}
	}

	// 2. Integrate OMW multilingual synset concepts under Korean hub nodes.
	senseCountPerSurface := make(map[string]int)
	for _, c := range concepts {
		for _, rawKO := range c.KO {
			koNorm := tokenizer.Normalize(rawKO)
			if koNorm == "" || strings.Contains(koNorm, " ") {
				continue
			}
			if senseCountPerSurface[koNorm] >= cfg.Limits.MaxCandidateSenses {
				continue
			}
			senseCountPerSurface[koNorm]++
			surfID := b.getOrAddNode(NodeKOSurface, koNorm)
			glossLabel := c.SynsetID
			if len(c.EN) > 0 {
				glossLabel = c.EN[0]
			}
			senseID := b.getOrAddNode(NodeKOSense, glossLabel, koNorm+"_"+c.SynsetID)
			b.addEdge(surfID, senseID, EdgeHasCandidateSense, ew.HasCandidateSense*0.95)
			b.addCue(senseID, c.KO)

			limitEN := minInt(len(c.EN), cfg.Limits.MaxNeighborsPerNode)
			for i := 0; i < limitEN; i++ {
				eID := b.getOrAddNode(NodeENConcept, c.EN[i])
				b.addEdge(senseID, eID, EdgeAlignedEN, ew.AlignedEN)
				b.addEdge(surfID, eID, EdgeAlignedEN, ew.AlignedEN*0.80)
			}
			limitJA := minInt(len(c.JA), cfg.Limits.MaxNeighborsPerNode)
			for i := 0; i < limitJA; i++ {
				jID := b.getOrAddNode(NodeJAConcept, c.JA[i])
				b.addEdge(senseID, jID, EdgeAlignedJA, ew.AlignedJA)
				b.addEdge(surfID, jID, EdgeAlignedJA, ew.AlignedJA*0.80)
			}
			limitZH := minInt(len(c.ZH), cfg.Limits.MaxNeighborsPerNode)
			for i := 0; i < limitZH; i++ {
				zID := b.getOrAddNode(NodeZHConcept, c.ZH[i])
				b.addEdge(senseID, zID, EdgeAlignedZH, ew.AlignedZH)
				b.addEdge(surfID, zID, EdgeAlignedZH, ew.AlignedZH*0.80)
			}
		}
	}

	// 3. Extract frequent Korean surface words and local co-occurrence edges from corpus.
	wordFreq := make(map[string]int, 32768)
	docWords := make([][]string, 0, minInt(len(corpus), cfg.Limits.MaxCorpusDocs))
	maxDocs := minInt(len(corpus), cfg.Limits.MaxCorpusDocs)
	for i := 0; i < maxDocs; i++ {
		norm := tokenizer.Normalize(corpus[i])
		if norm == "" {
			continue
		}
		fields := strings.Fields(norm)
		docWords = append(docWords, fields)
		for _, w := range fields {
			if len([]rune(w)) >= 2 {
				wordFreq[w]++
			}
		}
	}

	type wf struct {
		word string
		freq int
	}
	rankedWords := make([]wf, 0, len(wordFreq))
	for w, f := range wordFreq {
		if f >= cfg.Limits.MinKoFreq {
			rankedWords = append(rankedWords, wf{word: w, freq: f})
		}
	}
	sort.Slice(rankedWords, func(i, j int) bool {
		if rankedWords[i].freq != rankedWords[j].freq {
			return rankedWords[i].freq > rankedWords[j].freq
		}
		return rankedWords[i].word < rankedWords[j].word
	})
	if len(rankedWords) > cfg.Limits.MaxKoSurfaceNodes {
		rankedWords = rankedWords[:cfg.Limits.MaxKoSurfaceNodes]
	}

	selectedCorpusWords := make(map[string]int32, len(rankedWords))
	for _, item := range rankedWords {
		sid := b.getOrAddNode(NodeKOSurface, item.word)
		selectedCorpusWords[item.word] = sid
	}

	// Build sliding-window co-occurrence counts among selected surface nodes.
	type pair struct {
		a, b int32
	}
	cooccur := make(map[pair]int, len(selectedCorpusWords)*4)
	win := cfg.Limits.CooccurWindow
	for _, words := range docWords {
		for i := 0; i < len(words); i++ {
			idA, okA := selectedCorpusWords[words[i]]
			if !okA {
				continue
			}
			end := minInt(len(words), i+1+win)
			for j := i + 1; j < end; j++ {
				idB, okB := selectedCorpusWords[words[j]]
				if !okB || idA == idB {
					continue
				}
				p := pair{a: idA, b: idB}
				if idB < idA {
					p = pair{a: idB, b: idA}
				}
				cooccur[p]++
			}
		}
	}
	for p, cnt := range cooccur {
		if cnt < cfg.Limits.MinCooccurCount {
			continue
		}
		scale := float32(cnt) / float32(cnt+cfg.Limits.MinCooccurCount)
		w := ew.CooccursKO * scale
		b.addEdge(p.a, p.b, EdgeCooccursKO, w)
		b.addEdge(p.b, p.a, EdgeCooccursKO, w)
	}

	// 4. Decompose Hangul surface words via Unihan Hanja morphemes and gloss anchors.
	if unihan != nil {
		for _, n := range append([]Node(nil), b.nodes...) {
			if n.Type != NodeKOSurface {
				continue
			}
			runes := []rune(n.Label)
			if len(runes) < 2 || len(runes) > 6 {
				continue
			}
			allHangul := true
			for _, r := range runes {
				if r < 0xAC00 || r > 0xD7A3 {
					allHangul = false
					break
				}
			}
			if !allHangul {
				continue
			}
			for _, r := range runes {
				entries := unihan.ByHangul[string(r)]
				limit := minInt(len(entries), cfg.Limits.MaxHanjaPerSyllable)
				for idx := 0; idx < limit; idx++ {
					he := entries[idx]
					hID := b.getOrAddNode(NodeHanjaMorpheme, he.Char)
					weight := ew.ComposedOfHanja * (1.0 / float32(idx+1))
					b.addEdge(n.ID, hID, EdgeComposedOfHanja, weight)
					for _, gTok := range he.Glosses {
						eID := b.getOrAddNode(NodeENConcept, gTok)
						b.addEdge(hID, eID, EdgeAlignedEN, ew.AlignedEN*0.70)
					}
				}
			}
		}
	}

	// 5. Anchor Nuclear Feeling Checklist concepts and keywords to English emotion roots.
	if tax != nil {
		for _, f := range tax.Feelings {
			enCode := strings.ToLower(strings.TrimSpace(f.EnName))
			enID := b.getOrAddNode(NodeENConcept, enCode)
			koID := b.getOrAddNode(NodeKOSurface, tokenizer.Normalize(f.KoName))
			b.addEdge(koID, enID, EdgeNFCAnchor, ew.NFCAnchor)
			b.addEdge(koID, enID, EdgeAlignedEN, ew.AlignedEN)
			for _, kw := range f.Keywords {
				kwNorm := tokenizer.Normalize(kw)
				if kwNorm == "" || strings.Contains(kwNorm, " ") {
					continue
				}
				kwID := b.getOrAddNode(NodeKOSurface, kwNorm)
				b.addEdge(kwID, enID, EdgeNFCAnchor, ew.NFCAnchor*0.9)
			}
		}
	}

	g := b.finalizeCSR(cfg.Limits.MaxNeighborsPerNode)
	log.Status("BUILD_GRAPH", "SUCCESS", "nodes=%d edges=%d multi_sense=%d (by_type=%v)",
		g.Stats.NumNodes, g.Stats.NumEdges, g.Stats.MultiSenseSurfaces, g.Stats.NodesByType)
	log.Info("graphkg", "edges_by_type=%v", g.Stats.EdgesByType)
	return g, nil
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
