package cluster

import (
	"math"
)

// ProjectPCA projects high-dimensional SemanticEmbed vectors (e.g. 128d) down to targetDim (e.g. 24d)
// using pure-Go covariance + Power Iteration with Gram-Schmidt orthogonalization and Hotelling deflation.
// Projected vectors are L2-normalized before returning along with the explained variance ratio.
func ProjectPCA(vectors [][]float32, targetDim int) ([][]float32, float64) {
	n := len(vectors)
	if n == 0 {
		return nil, 1.0
	}
	origDim := len(vectors[0])
	if origDim == 0 || n < 2 || targetDim <= 0 || targetDim >= origDim {
		out := make([][]float32, n)
		for i, v := range vectors {
			cp := make([]float32, len(v))
			copy(cp, v)
			out[i] = cp
		}
		return out, 1.0
	}

	mean := make([]float64, origDim)
	for _, v := range vectors {
		for d := 0; d < origDim && d < len(v); d++ {
			mean[d] += float64(v[d])
		}
	}
	invN := 1.0 / float64(n)
	for d := range mean {
		mean[d] *= invN
	}

	centered := make([]float64, n*origDim)
	for i, v := range vectors {
		row := centered[i*origDim : (i+1)*origDim]
		for d := 0; d < origDim && d < len(v); d++ {
			row[d] = float64(v[d]) - mean[d]
		}
	}

	cov := make([]float64, origDim*origDim)
	for i := 0; i < n; i++ {
		row := centered[i*origDim : (i+1)*origDim]
		for a := 0; a < origDim; a++ {
			va := row[a]
			if va == 0 {
				continue
			}
			cRow := cov[a*origDim:]
			for b := a; b < origDim; b++ {
				cRow[b] += va * row[b]
			}
		}
	}
	for a := 0; a < origDim; a++ {
		for b := a; b < origDim; b++ {
			val := cov[a*origDim+b] * invN
			cov[a*origDim+b] = val
			cov[b*origDim+a] = val
		}
	}

	var totalVar float64
	for d := 0; d < origDim; d++ {
		totalVar += cov[d*origDim+d]
	}
	if totalVar <= 1e-12 {
		out := make([][]float32, n)
		for i := range vectors {
			out[i] = make([]float32, targetDim)
			copy(out[i], vectors[i])
		}
		return out, 1.0
	}

	basis := make([][]float64, 0, targetDim)
	tmp := make([]float64, origDim)
	var capturedVar float64
	const powerIters = 80

	for c := 0; c < targetDim; c++ {
		vec := make([]float64, origDim)
		for d := 0; d < origDim; d++ {
			vec[d] = float64(((d + c*3) % 11) + 1)
		}
		orthogonalizeAndNorm(vec, basis)

		for iter := 0; iter < powerIters; iter++ {
			for a := 0; a < origDim; a++ {
				var sum float64
				cRow := cov[a*origDim : (a+1)*origDim]
				for b := 0; b < origDim; b++ {
					sum += cRow[b] * vec[b]
				}
				tmp[a] = sum
			}
			copy(vec, tmp)
			if !orthogonalizeAndNorm(vec, basis) {
				break
			}
		}

		// Rayleigh quotient eigenvalue: v^T * C * v.
		var lambda float64
		for a := 0; a < origDim; a++ {
			var rowDot float64
			cRow := cov[a*origDim : (a+1)*origDim]
			for b := 0; b < origDim; b++ {
				rowDot += cRow[b] * vec[b]
			}
			lambda += vec[a] * rowDot
		}
		if lambda < 0 {
			lambda = 0
		}
		capturedVar += lambda
		basis = append(basis, vec)

		// Hotelling deflation: C <- C - lambda * v * v^T.
		for a := 0; a < origDim; a++ {
			lva := lambda * vec[a]
			cRow := cov[a*origDim:]
			for b := a; b < origDim; b++ {
				updated := cRow[b] - lva*vec[b]
				cRow[b] = updated
				cov[b*origDim+a] = updated
			}
		}
	}

	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		row := centered[i*origDim : (i+1)*origDim]
		proj := make([]float64, len(basis))
		var sumSq float64
		for c, bVec := range basis {
			var dot float64
			for d := 0; d < origDim; d++ {
				dot += row[d] * bVec[d]
			}
			proj[c] = dot
			sumSq += dot * dot
		}
		outVec := make([]float32, len(basis))
		invNorm := 1.0
		if sumSq > 1e-12 {
			invNorm = 1.0 / math.Sqrt(sumSq)
		}
		for c, val := range proj {
			outVec[c] = float32(val * invNorm)
		}
		out[i] = outVec
	}

	explainedRatio := capturedVar / totalVar
	if explainedRatio > 1.0 {
		explainedRatio = 1.0
	} else if explainedRatio < 0 {
		explainedRatio = 0
	}
	return out, explainedRatio
}

func orthogonalizeAndNorm(vec []float64, basis [][]float64) bool {
	for _, b := range basis {
		var dot float64
		for d := range vec {
			dot += vec[d] * b[d]
		}
		for d := range vec {
			vec[d] -= dot * b[d]
		}
	}
	var normSq float64
	for _, v := range vec {
		normSq += v * v
	}
	if normSq <= 1e-18 {
		return false
	}
	inv := 1.0 / math.Sqrt(normSq)
	for d := range vec {
		vec[d] *= inv
	}
	return true
}
