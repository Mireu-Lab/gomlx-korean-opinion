package graphkg

import (
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// NodeType identifies the 6 heterogeneous node kinds in the KO-Hub graph.
type NodeType uint8

const (
	NodeKOSurface NodeType = iota
	NodeKOSense
	NodeHanjaMorpheme
	NodeENConcept
	NodeJAConcept
	NodeZHConcept
)

func (t NodeType) String() string {
	switch t {
	case NodeKOSurface:
		return "KO_SURFACE"
	case NodeKOSense:
		return "KO_SENSE"
	case NodeHanjaMorpheme:
		return "HANJA_MORPHEME"
	case NodeENConcept:
		return "EN_CONCEPT"
	case NodeJAConcept:
		return "JA_CONCEPT"
	case NodeZHConcept:
		return "ZH_CONCEPT"
	default:
		return "UNKNOWN"
	}
}

// EdgeType identifies the 7 relation types connecting Korean hubs to multilingual roots.
type EdgeType uint8

const (
	EdgeHasCandidateSense EdgeType = iota
	EdgeComposedOfHanja
	EdgeAlignedEN
	EdgeAlignedJA
	EdgeAlignedZH
	EdgeCooccursKO
	EdgeNFCAnchor
)

func (e EdgeType) String() string {
	switch e {
	case EdgeHasCandidateSense:
		return "HAS_CANDIDATE_SENSE"
	case EdgeComposedOfHanja:
		return "COMPOSED_OF_HANJA"
	case EdgeAlignedEN:
		return "ALIGNED_EN"
	case EdgeAlignedJA:
		return "ALIGNED_JA"
	case EdgeAlignedZH:
		return "ALIGNED_ZH"
	case EdgeCooccursKO:
		return "COOCCURS_KO"
	case EdgeNFCAnchor:
		return "NFC_ANCHOR"
	default:
		return "UNKNOWN"
	}
}

// Node represents one vertex in the KO-Hub heterogeneous graph.
type Node struct {
	ID        int32
	Type      NodeType
	Key       string
	Label     string
	NFCAnchor string
}

// Edge represents a typed, weighted outgoing edge returned by Graph.Neighbors.
type Edge struct {
	Dst    int32
	Type   EdgeType
	Weight float32
}

// GraphStats summarizes node and edge counts across all types.
type GraphStats struct {
	NumNodes           int            `json:"num_nodes"`
	NumEdges           int            `json:"num_edges"`
	MultiSenseSurfaces int            `json:"multi_sense_surfaces"`
	NodesByType        map[string]int `json:"nodes_by_type"`
	EdgesByType        map[string]int `json:"edges_by_type"`
}

// Graph stores the heterogeneous KO-Hub graph in compact Compressed Sparse Row (CSR) layout.
type Graph struct {
	Nodes       []Node
	RowPtr      []int32
	ColIdx      []int32
	EdgeTypes   []EdgeType
	EdgeWeights []float32
	SenseCues   map[int32][]string
	Stats       GraphStats

	keyToID     map[string]int32
	surfaceToID map[string]int32
}

// RebuildIndices builds in-memory lookup maps after construction or deserialization.
func (g *Graph) RebuildIndices() {
	g.keyToID = make(map[string]int32, len(g.Nodes))
	g.surfaceToID = make(map[string]int32)
	for i := range g.Nodes {
		n := &g.Nodes[i]
		g.keyToID[n.Key] = n.ID
		if n.Type == NodeKOSurface {
			g.surfaceToID[n.Label] = n.ID
		}
	}
}

// FindSurface looks up a Korean surface word node ID.
func (g *Graph) FindSurface(word string) (int32, bool) {
	if g.surfaceToID == nil {
		g.RebuildIndices()
	}
	id, ok := g.surfaceToID[word]
	return id, ok
}

// Neighbors returns outgoing edges from src, optionally filtered by EdgeType.
func (g *Graph) Neighbors(src int32, filterTypes ...EdgeType) []Edge {
	if int(src) < 0 || int(src)+1 >= len(g.RowPtr) {
		return nil
	}
	start := g.RowPtr[src]
	end := g.RowPtr[src+1]
	if start >= end {
		return nil
	}
	out := make([]Edge, 0, end-start)
	for pos := start; pos < end; pos++ {
		et := g.EdgeTypes[pos]
		if len(filterTypes) > 0 && !containsEdgeType(filterTypes, et) {
			continue
		}
		out = append(out, Edge{
			Dst:    g.ColIdx[pos],
			Type:   et,
			Weight: g.EdgeWeights[pos],
		})
	}
	return out
}

// CandidateSenses returns all KO_SENSE nodes linked from a Korean surface word.
func (g *Graph) CandidateSenses(surface string) []Node {
	sid, ok := g.FindSurface(surface)
	if !ok {
		return nil
	}
	edges := g.Neighbors(sid, EdgeHasCandidateSense)
	out := make([]Node, 0, len(edges))
	for _, e := range edges {
		if int(e.Dst) >= 0 && int(e.Dst) < len(g.Nodes) {
			out = append(out, g.Nodes[e.Dst])
		}
	}
	return out
}

// SenseRoots returns linked Hanja, English, and JA/ZH root node IDs for a KO_SENSE (or KO_SURFACE) node.
func (g *Graph) SenseRoots(nodeID int32) (hanja, en, jazh []int32) {
	for _, e := range g.Neighbors(nodeID) {
		switch e.Type {
		case EdgeComposedOfHanja:
			hanja = append(hanja, e.Dst)
		case EdgeAlignedEN:
			en = append(en, e.Dst)
		case EdgeAlignedJA, EdgeAlignedZH:
			jazh = append(jazh, e.Dst)
		}
	}
	return hanja, en, jazh
}

// MultiSenseSurfaces returns sorted node IDs of KO_SURFACE vertices with >= 2 candidate senses.
func (g *Graph) MultiSenseSurfaces() []int32 {
	var out []int32
	for _, n := range g.Nodes {
		if n.Type != NodeKOSurface {
			continue
		}
		if len(g.Neighbors(n.ID, EdgeHasCandidateSense)) >= 2 {
			out = append(out, n.ID)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// Save writes the CSR graph to disk using gzip-compressed gob encoding.
func (g *Graph) Save(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("create graph dir for %q: %w", path, err)
	}
	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("create graph file %q: %w", path, err)
	}
	gw := gzip.NewWriter(f)
	enc := gob.NewEncoder(gw)
	if err := enc.Encode(g); err != nil {
		_ = gw.Close()
		_ = f.Close()
		_ = os.Remove(tmpPath)
		return fmt.Errorf("encode graph to %q: %w", path, err)
	}
	if err := gw.Close(); err != nil {
		_ = f.Close()
		_ = os.Remove(tmpPath)
		return fmt.Errorf("close gzip stream for %q: %w", path, err)
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("close graph file %q: %w", path, err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("rename graph file %q: %w", path, err)
	}
	return nil
}

// Load reads a CSR graph serialized via Save and rebuilds its lookup indices.
func Load(path string) (*Graph, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open graph %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	gr, err := gzip.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("open gzip graph %q: %w", path, err)
	}
	defer func() { _ = gr.Close() }()

	var g Graph
	if err := gob.NewDecoder(gr).Decode(&g); err != nil {
		return nil, fmt.Errorf("decode graph %q: %w", path, err)
	}
	g.RebuildIndices()
	return &g, nil
}

func containsEdgeType(list []EdgeType, target EdgeType) bool {
	for _, x := range list {
		if x == target {
			return true
		}
	}
	return false
}
