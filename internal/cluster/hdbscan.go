package cluster

import (
	"math"
	"sort"

	"github.com/Mireu-Lab/gomlx-nfc-opinion/internal/config"
)

// HDBSCANResult contains cluster assignments (-1 for noise) and membership strengths.
type HDBSCANResult struct {
	Labels      []int
	Probs       []float64
	NumClusters int
}

// BuildHybridVectors combines L2-normalized GoMLX latent embeddings, topic probs, NFC scores, and sentiment probs.
func BuildHybridVectors(docs []ClusterDoc, w config.ClusterFeatureWeights) [][]float64 {
	out := make([][]float64, len(docs))
	wLat := math.Sqrt(w.Latent)
	wTop := math.Sqrt(w.TopicProbs)
	wNFC := math.Sqrt(w.NFCProbs)
	wSent := math.Sqrt(w.Sentiment)

	for i, doc := range docs {
		lat := l2NormalizeFloat32(doc.SemanticEmbed)
		top := l2NormalizeFloat64(doc.Scored.TopicProbs)
		nfcVec := l2NormalizeFloat64(doc.Scored.AllNFCScores)
		sent := l2NormalizeFloat64([]float64{doc.Scored.SentimentPositiveProb, doc.Scored.SentimentNegativeProb})

		vec := make([]float64, 0, len(lat)+len(top)+len(nfcVec)+len(sent))
		for _, v := range lat {
			vec = append(vec, wLat*v)
		}
		for _, v := range top {
			vec = append(vec, wTop*v)
		}
		for _, v := range nfcVec {
			vec = append(vec, wNFC*v)
		}
		for _, v := range sent {
			vec = append(vec, wSent*v)
		}
		out[i] = l2NormalizeFloat64(vec)
	}
	return out
}

func l2NormalizeFloat32(in []float32) []float64 {
	out := make([]float64, len(in))
	var sumSq float64
	for i, v := range in {
		fv := float64(v)
		out[i] = fv
		sumSq += fv * fv
	}
	if sumSq > 1e-12 {
		inv := 1.0 / math.Sqrt(sumSq)
		for i := range out {
			out[i] *= inv
		}
	}
	return out
}

func l2NormalizeFloat64(in []float64) []float64 {
	out := make([]float64, len(in))
	var sumSq float64
	for i, v := range in {
		out[i] = v
		sumSq += v * v
	}
	if sumSq > 1e-12 {
		inv := 1.0 / math.Sqrt(sumSq)
		for i := range out {
			out[i] *= inv
		}
	}
	return out
}

// ComputeCosineDistanceMatrix builds a symmetric N x N cosine distance matrix for L2-normalized vectors.
func ComputeCosineDistanceMatrix(vectors [][]float64) []float64 {
	n := len(vectors)
	dist := make([]float64, n*n)
	for i := 0; i < n; i++ {
		vi := vectors[i]
		for j := i + 1; j < n; j++ {
			vj := vectors[j]
			var dot float64
			for d := range vi {
				dot += vi[d] * vj[d]
			}
			dVal := 1.0 - dot
			if dVal < 0 {
				dVal = 0
			} else if dVal > 2 {
				dVal = 2
			}
			dist[i*n+j] = dVal
			dist[j*n+i] = dVal
		}
	}
	return dist
}

// RunHDBSCAN runs pure-Go HDBSCAN density clustering over a precomputed symmetric distance matrix.
func RunHDBSCAN(dist []float64, n int, minClusterSize, minSamples int, selectionEpsilon float64) HDBSCANResult {
	labels := make([]int, n)
	probs := make([]float64, n)
	for i := range labels {
		labels[i] = -1
	}
	if n < 2 {
		return HDBSCANResult{Labels: labels, Probs: probs}
	}
	if minClusterSize < 2 {
		minClusterSize = 2
	}
	if minSamples < 1 {
		minSamples = 1
	}
	if minSamples > n {
		minSamples = n
	}
	if n < minClusterSize {
		return HDBSCANResult{Labels: labels, Probs: probs}
	}

	// 1. Core distance per point = distance to (minSamples-1)-th nearest neighbor (including self at 0).
	coreDist := make([]float64, n)
	rowBuf := make([]float64, n)
	kIdx := minSamples - 1
	for i := 0; i < n; i++ {
		copy(rowBuf, dist[i*n:(i+1)*n])
		sort.Float64s(rowBuf)
		coreDist[i] = rowBuf[kIdx]
	}

	// 2. Build exact MST over mutual reachability distance via O(N^2) Prim's algorithm.
	type mstEdge struct {
		u, v   int
		weight float64
	}
	edges := make([]mstEdge, 0, n-1)
	inTree := make([]bool, n)
	minEdgeWeight := make([]float64, n)
	minEdgeParent := make([]int, n)
	for i := 0; i < n; i++ {
		minEdgeWeight[i] = math.Inf(1)
		minEdgeParent[i] = -1
	}
	minEdgeWeight[0] = 0

	for step := 0; step < n; step++ {
		u := -1
		bestW := math.Inf(1)
		for v := 0; v < n; v++ {
			if !inTree[v] && minEdgeWeight[v] < bestW {
				bestW = minEdgeWeight[v]
				u = v
			}
		}
		if u < 0 {
			break
		}
		inTree[u] = true
		if minEdgeParent[u] >= 0 {
			edges = append(edges, mstEdge{u: minEdgeParent[u], v: u, weight: bestW})
		}
		cu := coreDist[u]
		rowU := dist[u*n : (u+1)*n]
		for v := 0; v < n; v++ {
			if inTree[v] {
				continue
			}
			mrd := rowU[v]
			if cu > mrd {
				mrd = cu
			}
			if coreDist[v] > mrd {
				mrd = coreDist[v]
			}
			if mrd < minEdgeWeight[v] {
				minEdgeWeight[v] = mrd
				minEdgeParent[v] = u
			}
		}
	}

	// 3. Sort MST edges ascending and construct single-linkage hierarchy (2*n - 1 nodes).
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].weight != edges[j].weight {
			return edges[i].weight < edges[j].weight
		}
		if edges[i].u != edges[j].u {
			return edges[i].u < edges[j].u
		}
		return edges[i].v < edges[j].v
	})

	totalNodes := 2*n - 1
	ufParent := make([]int, totalNodes)
	nodeSize := make([]int, totalNodes)
	leftChild := make([]int, totalNodes)
	rightChild := make([]int, totalNodes)
	mergeDist := make([]float64, totalNodes)

	for i := 0; i < totalNodes; i++ {
		ufParent[i] = i
		leftChild[i] = -1
		rightChild[i] = -1
		if i < n {
			nodeSize[i] = 1
		}
	}
	var findRoot func(int) int
	findRoot = func(x int) int {
		for ufParent[x] != x {
			ufParent[x] = ufParent[ufParent[x]]
			x = ufParent[x]
		}
		return x
	}

	for step, e := range edges {
		ru := findRoot(e.u)
		rv := findRoot(e.v)
		if ru == rv {
			continue
		}
		m := n + step
		leftChild[m] = ru
		rightChild[m] = rv
		mergeDist[m] = e.weight
		nodeSize[m] = nodeSize[ru] + nodeSize[rv]
		ufParent[ru] = m
		ufParent[rv] = m
	}

	// Helper to collect leaf point indices inside any dendrogram node.
	collectLeaves := func(startNode int) []int {
		var out []int
		stack := []int{startNode}
		for len(stack) > 0 {
			curr := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if curr < n {
				out = append(out, curr)
				continue
			}
			stack = append(stack, leftChild[curr], rightChild[curr])
		}
		return out
	}

	// 4. Condense the hierarchy top-down using minClusterSize.
	type pointFallout struct {
		pointID int
		lambda  float64
	}
	type condensedCluster struct {
		parentCluster int
		children      []int
		birthLambda   float64
		deathLambda   float64
		subtreeSize   int
		fallouts      []pointFallout
	}

	rootNode := totalNodes - 1
	clusters := []condensedCluster{
		{
			parentCluster: -1,
			birthLambda:   0.0,
			subtreeSize:   nodeSize[rootNode],
		},
	}

	const distFloor = 1e-12
	var condense func(treeNode, clusterID int)
	condense = func(treeNode, clusterID int) {
		if treeNode < n {
			return
		}
		d := mergeDist[treeNode]
		if d < distFloor {
			d = distFloor
		}
		lambda := 1.0 / d
		L := leftChild[treeNode]
		R := rightChild[treeNode]
		sL := nodeSize[L]
		sR := nodeSize[R]

		switch {
		case sL >= minClusterSize && sR >= minClusterSize:
			clusters[clusterID].deathLambda = lambda
			cL := len(clusters)
			clusters = append(clusters, condensedCluster{
				parentCluster: clusterID,
				birthLambda:   lambda,
				subtreeSize:   sL,
			})
			cR := len(clusters)
			clusters = append(clusters, condensedCluster{
				parentCluster: clusterID,
				birthLambda:   lambda,
				subtreeSize:   sR,
			})
			clusters[clusterID].children = []int{cL, cR}
			condense(L, cL)
			condense(R, cR)

		case sL >= minClusterSize && sR < minClusterSize:
			for _, pt := range collectLeaves(R) {
				clusters[clusterID].fallouts = append(clusters[clusterID].fallouts, pointFallout{
					pointID: pt,
					lambda:  lambda,
				})
			}
			condense(L, clusterID)

		case sR >= minClusterSize && sL < minClusterSize:
			for _, pt := range collectLeaves(L) {
				clusters[clusterID].fallouts = append(clusters[clusterID].fallouts, pointFallout{
					pointID: pt,
					lambda:  lambda,
				})
			}
			condense(R, clusterID)

		default:
			clusters[clusterID].deathLambda = lambda
			for _, pt := range collectLeaves(L) {
				clusters[clusterID].fallouts = append(clusters[clusterID].fallouts, pointFallout{
					pointID: pt,
					lambda:  lambda,
				})
			}
			for _, pt := range collectLeaves(R) {
				clusters[clusterID].fallouts = append(clusters[clusterID].fallouts, pointFallout{
					pointID: pt,
					lambda:  lambda,
				})
			}
		}
	}
	condense(rootNode, 0)

	numCond := len(clusters)
	stability := make([]float64, numCond)
	for cid := 0; cid < numCond; cid++ {
		c := clusters[cid]
		var st float64
		for _, fo := range c.fallouts {
			diff := fo.lambda - c.birthLambda
			if diff > 0 {
				st += diff
			}
		}
		for _, childID := range c.children {
			diff := clusters[childID].birthLambda - c.birthLambda
			if diff > 0 {
				st += float64(clusters[childID].subtreeSize) * diff
			}
		}
		stability[cid] = st
	}

	// 5. Bottom-up Excess of Mass (EOM) cluster selection.
	selected := make([]bool, numCond)
	bestStab := make([]float64, numCond)

	var unselectDescendants func(cid int)
	unselectDescendants = func(cid int) {
		for _, child := range clusters[cid].children {
			selected[child] = false
			unselectDescendants(child)
		}
	}

	// Post-order traversal via reverse pre-order cluster IDs (parent < child always holds).
	for cid := numCond - 1; cid >= 0; cid-- {
		children := clusters[cid].children
		if len(children) == 0 {
			if cid > 0 {
				selected[cid] = true
			}
			bestStab[cid] = stability[cid]
			continue
		}
		var childSum float64
		for _, ch := range children {
			childSum += bestStab[ch]
		}
		epsMerge := false
		if cid > 0 && selectionEpsilon > 0 && clusters[cid].deathLambda > 0 {
			splitDist := 1.0 / clusters[cid].deathLambda
			if splitDist <= selectionEpsilon {
				epsMerge = true
			}
		}
		if cid > 0 && (stability[cid] > childSum || epsMerge) {
			selected[cid] = true
			unselectDescendants(cid)
			bestStab[cid] = stability[cid]
		} else {
			selected[cid] = false
			bestStab[cid] = childSum
		}
	}

	// 6. Map selected clusters to 0..K-1 and compute per-point membership strengths.
	nextLabel := 0
	for cid := 0; cid < numCond; cid++ {
		if !selected[cid] {
			continue
		}
		flatLabel := nextLabel
		nextLabel++

		var pts []pointFallout
		stack := []int{cid}
		for len(stack) > 0 {
			curr := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			pts = append(pts, clusters[curr].fallouts...)
			stack = append(stack, clusters[curr].children...)
		}

		maxLambda := 0.0
		for _, pf := range pts {
			if pf.lambda > maxLambda {
				maxLambda = pf.lambda
			}
		}
		for _, pf := range pts {
			labels[pf.pointID] = flatLabel
			if maxLambda > 0 {
				p := pf.lambda / maxLambda
				if p > 1 {
					p = 1
				} else if p < 0 {
					p = 0
				}
				probs[pf.pointID] = p
			} else {
				probs[pf.pointID] = 1.0
			}
		}
	}

	return HDBSCANResult{
		Labels:      labels,
		Probs:       probs,
		NumClusters: nextLabel,
	}
}
