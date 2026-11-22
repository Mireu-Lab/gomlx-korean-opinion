package pretrain

import (
	"fmt"
	"iter"
	"math/rand"
	"sort"
	"strings"

	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/train"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/graphkg"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/tokenizer"
)

// RootVocab maps multilingual graph root nodes (Hanja, EN, JA, ZH) into compact embedding table IDs.
type RootVocab struct {
	NodeToID map[int32]int32
	Size     int
}

// BuildRootVocab constructs a deterministic vocabulary across all non-Korean root nodes in g.
func BuildRootVocab(g *graphkg.Graph) *RootVocab {
	var keys []struct {
		key    string
		nodeID int32
	}
	for _, n := range g.Nodes {
		switch n.Type {
		case graphkg.NodeHanjaMorpheme, graphkg.NodeENConcept, graphkg.NodeJAConcept, graphkg.NodeZHConcept:
			keys = append(keys, struct {
				key    string
				nodeID int32
			}{key: n.Key, nodeID: n.ID})
		}
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i].key < keys[j].key })

	nodeToID := make(map[int32]int32, len(keys))
	for idx, item := range keys {
		nodeToID[item.nodeID] = int32(idx + 1) // 0 is reserved for <PAD_ROOT>
	}
	return &RootVocab{
		NodeToID: nodeToID,
		Size:     len(keys) + 1,
	}
}

// PretrainExample holds one Korean text sample paired with candidate multilingual root IDs
// and an optional contrasting-sense hard negative text.
type PretrainExample struct {
	Tokens        []int32
	HanjaIDs      []int32
	HanjaMask     []float32
	ENIDs         []int32
	ENMask        []float32
	JaZhIDs       []int32
	JaZhMask      []float32
	HardNegTokens []int32
	ENValid       float32
	HanjaValid    float32
	JaZhValid     float32
	DisambigValid float32
	IsDisambig    bool
}

// PretrainDatasetStats summarizes constructed corpus and disambiguation examples.
type PretrainDatasetStats struct {
	TotalExamples    int
	CorpusExamples   int
	DisambigExamples int
	WithENRoots      int
	WithHanjaRoots   int
	WithJaZhRoots    int
	RootVocabSize    int
}

// BuildPretrainExamples creates multi-view contrastive and homonym-disambiguation training examples
// from corpus sentences and the KO-Hub heterogeneous graph.
func BuildPretrainExamples(
	texts []string,
	tok *tokenizer.Tokenizer,
	g *graphkg.Graph,
	rv *RootVocab,
	maxRootsPerView int,
) ([]PretrainExample, PretrainDatasetStats) {
	stats := PretrainDatasetStats{RootVocabSize: rv.Size}

	// Build explicit disambiguation anchor examples for every multi-sense surface in the graph.
	multiSenses := g.MultiSenseSurfaces()
	multiSet := make(map[int32]string, len(multiSenses))
	var disambigList []PretrainExample

	for _, sid := range multiSenses {
		surfWord := g.Nodes[sid].Label
		multiSet[sid] = surfWord
		senses := g.CandidateSenses(surfWord)
		if len(senses) < 2 {
			continue
		}
		for i, sPos := range senses {
			cuesPos := g.SenseCues[sPos.ID]
			if len(cuesPos) == 0 {
				continue
			}
			posText := surfWord + " " + strings.Join(cuesPos, " ")
			hanjaNodes, enNodes, jazhNodes := g.SenseRoots(sPos.ID)
			for j, sNeg := range senses {
				if i == j {
					continue
				}
				cuesNeg := g.SenseCues[sNeg.ID]
				if len(cuesNeg) == 0 {
					continue
				}
				negText := surfWord + " " + strings.Join(cuesNeg, " ")
				ex := packPretrainExample(tok, rv, maxRootsPerView, posText, negText, hanjaNodes, enNodes, jazhNodes, true)
				if ex.ENValid > 0 || ex.HanjaValid > 0 || ex.JaZhValid > 0 {
					disambigList = append(disambigList, ex)
				}
			}
		}
	}

	// Extract corpus examples with context-filtered root nodes.
	corpusList := make([]PretrainExample, 0, len(texts)/2)
	for _, txt := range texts {
		norm := tokenizer.Normalize(txt)
		if norm == "" {
			continue
		}
		words := strings.Fields(norm)
		wordSet := make(map[string]struct{}, len(words))
		for _, w := range words {
			wordSet[w] = struct{}{}
		}

		var hanjaNodes, enNodes, jazhNodes []int32
		var bestNegText string

		for _, w := range words {
			sid, ok := g.FindSurface(w)
			if !ok {
				continue
			}
			senses := g.CandidateSenses(w)
			if len(senses) >= 2 {
				bestIdx, altIdx := selectBestAndContrastSense(senses, g.SenseCues, wordSet)
				h, e, jz := g.SenseRoots(senses[bestIdx].ID)
				hanjaNodes = append(hanjaNodes, h...)
				enNodes = append(enNodes, e...)
				jazhNodes = append(jazhNodes, jz...)
				if altIdx >= 0 && bestNegText == "" {
					if negCues := g.SenseCues[senses[altIdx].ID]; len(negCues) > 0 {
						bestNegText = w + " " + strings.Join(negCues, " ")
					}
				}
				continue
			}

			// Single-sense or corpus-derived surface: collect direct roots and single sense roots.
			h, e, jz := g.SenseRoots(sid)
			hanjaNodes = append(hanjaNodes, h...)
			enNodes = append(enNodes, e...)
			jazhNodes = append(jazhNodes, jz...)
			for _, s := range senses {
				sh, se, sjz := g.SenseRoots(s.ID)
				hanjaNodes = append(hanjaNodes, sh...)
				enNodes = append(enNodes, se...)
				jazhNodes = append(jazhNodes, sjz...)
			}
		}

		// Also check if a curated multi-sense surface appears inside a compound token (e.g., 여류화가협회).
		if bestNegText == "" && len(multiSenses) > 0 {
			for _, mWord := range multiSet {
				if strings.Contains(norm, mWord) {
					senses := g.CandidateSenses(mWord)
					if len(senses) >= 2 {
						bestIdx, altIdx := selectBestAndContrastSenseBySubstr(senses, g.SenseCues, norm)
						h, e, jz := g.SenseRoots(senses[bestIdx].ID)
						hanjaNodes = append(hanjaNodes, h...)
						enNodes = append(enNodes, e...)
						jazhNodes = append(jazhNodes, jz...)
						if altIdx >= 0 {
							if negCues := g.SenseCues[senses[altIdx].ID]; len(negCues) > 0 {
								bestNegText = mWord + " " + strings.Join(negCues, " ")
							}
						}
						break
					}
				}
			}
		}

		if len(hanjaNodes) == 0 && len(enNodes) == 0 && len(jazhNodes) == 0 {
			continue
		}
		ex := packPretrainExample(tok, rv, maxRootsPerView, txt, bestNegText, hanjaNodes, enNodes, jazhNodes, bestNegText != "")
		if ex.ENValid > 0 || ex.HanjaValid > 0 || ex.JaZhValid > 0 {
			corpusList = append(corpusList, ex)
		}
	}

	all := make([]PretrainExample, 0, len(corpusList)+len(disambigList))
	all = append(all, corpusList...)
	all = append(all, disambigList...)

	stats.CorpusExamples = len(corpusList)
	stats.DisambigExamples = len(disambigList)
	stats.TotalExamples = len(all)
	for _, ex := range all {
		if ex.ENValid > 0 {
			stats.WithENRoots++
		}
		if ex.HanjaValid > 0 {
			stats.WithHanjaRoots++
		}
		if ex.JaZhValid > 0 {
			stats.WithJaZhRoots++
		}
	}
	return all, stats
}

func selectBestAndContrastSense(senses []graphkg.Node, senseCues map[int32][]string, wordSet map[string]struct{}) (bestIdx, altIdx int) {
	scores := make([]int, len(senses))
	for i, s := range senses {
		for _, cue := range senseCues[s.ID] {
			if _, ok := wordSet[cue]; ok {
				scores[i]++
			}
		}
	}
	bestIdx = 0
	altIdx = 1
	for i := 1; i < len(senses); i++ {
		if scores[i] > scores[bestIdx] {
			bestIdx = i
		}
	}
	minScore := scores[0]
	for i := 0; i < len(senses); i++ {
		if i == bestIdx {
			continue
		}
		if altIdx == bestIdx || scores[i] <= minScore {
			minScore = scores[i]
			altIdx = i
		}
	}
	if altIdx == bestIdx {
		altIdx = -1
	}
	return bestIdx, altIdx
}

func selectBestAndContrastSenseBySubstr(senses []graphkg.Node, senseCues map[int32][]string, normText string) (bestIdx, altIdx int) {
	scores := make([]int, len(senses))
	for i, s := range senses {
		for _, cue := range senseCues[s.ID] {
			if strings.Contains(normText, cue) {
				scores[i]++
			}
		}
	}
	bestIdx = 0
	altIdx = 1
	for i := 1; i < len(senses); i++ {
		if scores[i] > scores[bestIdx] {
			bestIdx = i
		}
	}
	if len(senses) < 2 {
		return 0, -1
	}
	for i := 0; i < len(senses); i++ {
		if i != bestIdx {
			altIdx = i
			break
		}
	}
	return bestIdx, altIdx
}

func packPretrainExample(
	tok *tokenizer.Tokenizer,
	rv *RootVocab,
	maxK int,
	posText, negText string,
	hanjaNodes, enNodes, jazhNodes []int32,
	hasDisambig bool,
) PretrainExample {
	hIDs, hMask, hValid := mapRootIDs(hanjaNodes, rv, maxK)
	eIDs, eMask, eValid := mapRootIDs(enNodes, rv, maxK)
	jzIDs, jzMask, jzValid := mapRootIDs(jazhNodes, rv, maxK)

	var negToks []int32
	var disValid float32
	if hasDisambig && strings.TrimSpace(negText) != "" {
		negToks = tok.Encode(negText)
		disValid = 1.0
	} else {
		negToks = make([]int32, tok.Cfg.MaxSeqLen)
	}

	return PretrainExample{
		Tokens:        tok.Encode(posText),
		HanjaIDs:      hIDs,
		HanjaMask:     hMask,
		ENIDs:         eIDs,
		ENMask:        eMask,
		JaZhIDs:       jzIDs,
		JaZhMask:      jzMask,
		HardNegTokens: negToks,
		ENValid:       eValid,
		HanjaValid:    hValid,
		JaZhValid:     jzValid,
		DisambigValid: disValid,
		IsDisambig:    hasDisambig,
	}
}

func mapRootIDs(nodes []int32, rv *RootVocab, maxK int) (ids []int32, mask []float32, valid float32) {
	ids = make([]int32, maxK)
	mask = make([]float32, maxK)
	seen := make(map[int32]struct{}, maxK)
	count := 0
	for _, nid := range nodes {
		rid, ok := rv.NodeToID[nid]
		if !ok || rid <= 0 {
			continue
		}
		if _, dup := seen[rid]; dup {
			continue
		}
		seen[rid] = struct{}{}
		ids[count] = rid
		mask[count] = 1.0
		count++
		if count >= maxK {
			break
		}
	}
	if count > 0 {
		valid = 1.0
	}
	return ids, mask, valid
}

// PretrainDataset implements train.Dataset yielding balanced multi-view + disambiguation batches.
type PretrainDataset struct {
	name      string
	batchSize int
	seqLen    int
	maxK      int
	rng       *rand.Rand
	all       []PretrainExample
	disambig  []PretrainExample
}

// NewPretrainDataset wraps pretraining examples into a GoMLX train.Dataset.
func NewPretrainDataset(name string, examples []PretrainExample, batchSize, seqLen, maxK int, seed int64) (*PretrainDataset, error) {
	if len(examples) == 0 {
		return nil, fmt.Errorf("pretrain examples must not be empty")
	}
	var dis []PretrainExample
	for _, ex := range examples {
		if ex.DisambigValid > 0 {
			dis = append(dis, ex)
		}
	}
	return &PretrainDataset{
		name:      name,
		batchSize: batchSize,
		seqLen:    seqLen,
		maxK:      maxK,
		rng:       rand.New(rand.NewSource(seed)),
		all:       examples,
		disambig:  dis,
	}, nil
}

// Name returns the dataset name for GoMLX train.Loop.
func (d *PretrainDataset) Name() string {
	return d.name
}

// Iter yields infinite fixed-shape contrastive batches mixing corpus and disambiguation samples.
func (d *PretrainDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		nDis := 0
		if len(d.disambig) > 0 {
			nDis = d.batchSize / 4
			if nDis < 2 {
				nDis = minInt(2, d.batchSize)
			}
		}
		picked := make([]PretrainExample, d.batchSize)
		for {
			idx := 0
			for i := 0; i < nDis; i++ {
				picked[idx] = d.disambig[d.rng.Intn(len(d.disambig))]
				idx++
			}
			for i := nDis; i < d.batchSize; i++ {
				picked[idx] = d.all[d.rng.Intn(len(d.all))]
				idx++
			}
			d.rng.Shuffle(len(picked), func(i, j int) {
				picked[i], picked[j] = picked[j], picked[i]
			})
			batch := buildPretrainBatch(picked, d.seqLen, d.maxK, d.name)
			if !yield(batch, nil) {
				return
			}
		}
	}
}

func buildPretrainBatch(items []PretrainExample, seqLen, maxK int, spec any) train.Batch {
	bs := len(items)
	tokens := make([]int32, bs*seqLen)
	hardNeg := make([]int32, bs*seqLen)
	hanjaIDs := make([]int32, bs*maxK)
	hanjaMask := make([]float32, bs*maxK)
	enIDs := make([]int32, bs*maxK)
	enMask := make([]float32, bs*maxK)
	jazhIDs := make([]int32, bs*maxK)
	jazhMask := make([]float32, bs*maxK)
	enValid := make([]float32, bs)
	hanjaValid := make([]float32, bs)
	jazhValid := make([]float32, bs)
	disambigValid := make([]float32, bs)

	for i, ex := range items {
		copy(tokens[i*seqLen:(i+1)*seqLen], ex.Tokens)
		copy(hardNeg[i*seqLen:(i+1)*seqLen], ex.HardNegTokens)
		copy(hanjaIDs[i*maxK:(i+1)*maxK], ex.HanjaIDs)
		copy(hanjaMask[i*maxK:(i+1)*maxK], ex.HanjaMask)
		copy(enIDs[i*maxK:(i+1)*maxK], ex.ENIDs)
		copy(enMask[i*maxK:(i+1)*maxK], ex.ENMask)
		copy(jazhIDs[i*maxK:(i+1)*maxK], ex.JaZhIDs)
		copy(jazhMask[i*maxK:(i+1)*maxK], ex.JaZhMask)
		enValid[i] = ex.ENValid
		hanjaValid[i] = ex.HanjaValid
		jazhValid[i] = ex.JaZhValid
		disambigValid[i] = ex.DisambigValid
	}

	return train.Batch{
		Inputs: []*tensors.Tensor{
			tensors.FromFlatDataAndDimensions(tokens, bs, seqLen),
			tensors.FromFlatDataAndDimensions(hanjaIDs, bs, maxK),
			tensors.FromFlatDataAndDimensions(hanjaMask, bs, maxK),
			tensors.FromFlatDataAndDimensions(enIDs, bs, maxK),
			tensors.FromFlatDataAndDimensions(enMask, bs, maxK),
			tensors.FromFlatDataAndDimensions(jazhIDs, bs, maxK),
			tensors.FromFlatDataAndDimensions(jazhMask, bs, maxK),
			tensors.FromFlatDataAndDimensions(hardNeg, bs, seqLen),
		},
		Labels: []*tensors.Tensor{
			tensors.FromFlatDataAndDimensions(enValid, bs),
			tensors.FromFlatDataAndDimensions(hanjaValid, bs),
			tensors.FromFlatDataAndDimensions(jazhValid, bs),
			tensors.FromFlatDataAndDimensions(disambigValid, bs),
		},
		Spec: spec,
	}
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
