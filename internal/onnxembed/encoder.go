package onnxembed

import (
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/onnx-gomlx/onnx/parser"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/logging"
)

// ONNXEncoder wraps an imported Hugging Face ONNX Transformer model executed purely inside GoMLX.
type ONNXEncoder struct {
	tok         *WordPieceTokenizer
	store       *model.Store
	exec        *model.Exec
	inputNames  []string
	inputDTypes []dtypes.DType
	maxSeqLen   int
	batchSize   int
	embedDim    int
}

// EnsureONNXAssets downloads the configured ONNX model and WordPiece vocab.txt if not already cached.
func EnsureONNXAssets(cfg config.ONNXBackboneConfig, timeoutSec, retries int, log *logging.DualLogger) (modelPath, vocabPath string, err error) {
	if err := os.MkdirAll(cfg.CacheDir, 0755); err != nil {
		return "", "", fmt.Errorf("create onnx cache dir %q: %w", cfg.CacheDir, err)
	}
	modelPath = filepath.Join(cfg.CacheDir, cfg.ModelFile)
	vocabPath = filepath.Join(cfg.CacheDir, cfg.VocabFile)

	client := &http.Client{Timeout: time.Duration(maxInt(timeoutSec, 120)) * time.Second}
	assets := []struct {
		name string
		url  string
		dest string
	}{
		{name: "ONNX WordPiece Vocab", url: cfg.VocabURL, dest: vocabPath},
		{name: "ONNX Backbone Model", url: cfg.ModelURL, dest: modelPath},
	}
	for _, a := range assets {
		if st, err := os.Stat(a.dest); err == nil && st.Size() > 0 {
			log.Info("onnxembed", "%s already cached at %s (%d bytes)", a.name, a.dest, st.Size())
			continue
		}
		log.Info("onnxembed", "downloading %s from %s -> %s", a.name, a.url, a.dest)
		if err := downloadWithRetry(client, a.url, a.dest, maxInt(retries, 3), log); err != nil {
			return "", "", fmt.Errorf("download %s: %w", a.name, err)
		}
	}
	return modelPath, vocabPath, nil
}

// NewONNXEncoder loads an ONNX model into GoMLX, configures masked mean-pooling + L2 normalization, and compiles an Exec.
func NewONNXEncoder(backend compute.Backend, modelPath, vocabPath string, maxSeqLen, batchSize, expectedDim int) (*ONNXEncoder, error) {
	tok, err := LoadWordPiece(vocabPath)
	if err != nil {
		return nil, err
	}
	onnxModel, err := parser.ParseFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("read ONNX model %q: %w", modelPath, err)
	}
	onnxModel.AllowDTypePromotion()

	store := model.NewStore()
	if err := onnxModel.VariablesToScope(store.RootScope()); err != nil {
		store.Finalize()
		return nil, fmt.Errorf("import ONNX variables: %w", err)
	}

	inNames, inShapes := onnxModel.Inputs()
	if len(inNames) == 0 {
		store.Finalize()
		return nil, fmt.Errorf("ONNX model %q defines no inputs", modelPath)
	}
	inDTypes := make([]dtypes.DType, len(inNames))
	for i, sh := range inShapes {
		dt := sh.DType
		if dt != dtypes.Int32 && dt != dtypes.Int64 {
			dt = dtypes.Int64
		}
		inDTypes[i] = dt
	}

	outNames, _ := onnxModel.Outputs()
	selectedOut := ""
	for _, name := range outNames {
		if name == "last_hidden_state" || name == "sentence_embedding" {
			selectedOut = name
			break
		}
	}
	if selectedOut == "" && len(outNames) > 0 {
		selectedOut = outNames[0]
	}

	exec, err := model.NewExec(backend, store, func(scope *model.Scope, args []*Node) *Node {
		g := args[0].Graph()
		scope.Store().SetTraining(g, false)
		feed := make(map[string]*Node, len(inNames))
		var attnMaskNode *Node
		for i, name := range inNames {
			feed[name] = args[i]
			if strings.Contains(strings.ToLower(name), "mask") {
				attnMaskNode = args[i]
			}
		}

		var outs []*Node
		if selectedOut != "" {
			outs = onnxModel.CallGraph(scope, g, feed, selectedOut)
		} else {
			outs = onnxModel.CallGraph(scope, g, feed)
		}
		hidden := outs[0]
		var pooled *Node
		if hidden.Rank() == 3 {
			if attnMaskNode != nil {
				maskF := ConvertDType(attnMaskNode, hidden.DType())
				maskExp := BroadcastToShape(InsertAxes(maskF, -1), hidden.Shape())
				sumHidden := ReduceSum(Mul(hidden, maskExp), 1)
				denom := Max(ReduceSum(maskF, 1), Scalar(g, hidden.DType(), 1.0))
				pooled = Div(sumHidden, BroadcastToShape(InsertAxes(denom, -1), sumHidden.Shape()))
			} else {
				pooled = ReduceMean(hidden, 1)
			}
		} else {
			pooled = hidden
		}
		if pooled.DType() != dtypes.Float32 {
			pooled = ConvertDType(pooled, dtypes.Float32)
		}
		normSq := ReduceSum(Mul(pooled, pooled), -1)
		norm := Sqrt(Add(normSq, Scalar(g, dtypes.Float32, 1e-8)))
		return Div(pooled, BroadcastToShape(InsertAxes(norm, -1), pooled.Shape()))
	})
	if err != nil {
		store.Finalize()
		return nil, fmt.Errorf("create ONNX GoMLX exec: %w", err)
	}

	return &ONNXEncoder{
		tok:         tok,
		store:       store,
		exec:        exec,
		inputNames:  inNames,
		inputDTypes: inDTypes,
		maxSeqLen:   maxSeqLen,
		batchSize:   batchSize,
		embedDim:    expectedDim,
	}, nil
}

// Close releases GoMLX executor and ONNX variable store resources.
func (e *ONNXEncoder) Close() {
	if e == nil {
		return
	}
	if e.exec != nil {
		e.exec.Finalize()
	}
	if e.store != nil {
		e.store.Finalize()
	}
}

// EncodeTexts runs batched inference over texts using GoMLX and returns L2-normalized float32 embeddings.
func (e *ONNXEncoder) EncodeTexts(texts []string, log *logging.DualLogger) ([][]float32, error) {
	res := make([][]float32, len(texts))
	if len(texts) == 0 {
		return res, nil
	}

	start := time.Now()
	bs := e.batchSize
	seq := e.maxSeqLen

	idsBuf := make([]int64, bs*seq)
	maskBuf := make([]int64, bs*seq)
	typeBuf := make([]int64, bs*seq)

	for offset := 0; offset < len(texts); offset += bs {
		end := offset + bs
		if end > len(texts) {
			end = len(texts)
		}
		cnt := end - offset

		for i := range idsBuf {
			idsBuf[i] = e.tok.padID
		}
		clear(maskBuf)
		clear(typeBuf)
		for i := 0; i < cnt; i++ {
			ids, mask, tids := e.tok.Encode(texts[offset+i], seq)
			copy(idsBuf[i*seq:(i+1)*seq], ids)
			copy(maskBuf[i*seq:(i+1)*seq], mask)
			copy(typeBuf[i*seq:(i+1)*seq], tids)
		}

		args := make([]any, len(e.inputNames))
		inTensors := make([]*tensors.Tensor, len(e.inputNames))
		for i, name := range e.inputNames {
			lname := strings.ToLower(name)
			src := typeBuf
			switch {
			case strings.Contains(lname, "mask"):
				src = maskBuf
			case strings.Contains(lname, "input") || strings.Contains(lname, "id") && !strings.Contains(lname, "type"):
				src = idsBuf
			}
			t := makeIntTensor(src, bs, seq, e.inputDTypes[i])
			inTensors[i] = t
			args[i] = t
		}

		outs, err := e.exec.Call(args...)
		for _, t := range inTensors {
			_ = t.FinalizeAll()
		}
		if err != nil {
			return nil, fmt.Errorf("ONNX encode batch [%d:%d]: %w", offset, end, err)
		}
		if len(outs) == 0 {
			return nil, fmt.Errorf("ONNX encode returned 0 outputs")
		}
		outShape := outs[0].Shape()
		dim := outShape.Dimensions[len(outShape.Dimensions)-1]
		flat := tensors.MustCopyFlatData[float32](outs[0])
		for _, o := range outs {
			_ = o.FinalizeAll()
		}

		for i := 0; i < cnt; i++ {
			vec := make([]float32, dim)
			copy(vec, flat[i*dim:(i+1)*dim])
			res[offset+i] = vec
		}
		if log != nil && ((offset/bs+1)%25 == 0 || end == len(texts)) {
			log.Info("onnxembed", "encoded %d/%d texts (dim=%d, elapsed=%s)",
				end, len(texts), dim, time.Since(start).Round(time.Second))
		}
	}
	return res, nil
}

// EncodeOrLoadCache loads cached embeddings if present and matching len(texts), otherwise runs ONNXEncoder and persists.
func EncodeOrLoadCache(cachePath string, texts []string, enc *ONNXEncoder, log *logging.DualLogger) ([][]float32, error) {
	if cached, err := LoadEmbedCache(cachePath); err == nil && len(cached) == len(texts) {
		if log != nil {
			log.Info("onnxembed", "loaded %d cached ONNX embeddings from %s", len(cached), cachePath)
		}
		return cached, nil
	}
	if enc == nil {
		return nil, fmt.Errorf("embedding cache %q missing/mismatched and encoder is nil", cachePath)
	}
	embeds, err := enc.EncodeTexts(texts, log)
	if err != nil {
		return nil, err
	}
	if err := SaveEmbedCache(cachePath, embeds); err != nil {
		return nil, err
	}
	if log != nil {
		log.Info("onnxembed", "saved %d ONNX embeddings to cache %s", len(embeds), cachePath)
	}
	return embeds, nil
}

// SaveEmbedCache writes precomputed embeddings to disk as gzip-compressed gob.
func SaveEmbedCache(path string, embeds [][]float32) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	gw := gzip.NewWriter(f)
	if err := gob.NewEncoder(gw).Encode(embeds); err != nil {
		_ = gw.Close()
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := gw.Close(); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// LoadEmbedCache reads gzip-compressed gob embeddings from disk.
func LoadEmbedCache(path string) ([][]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	gr, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer func() { _ = gr.Close() }()
	var embeds [][]float32
	if err := gob.NewDecoder(gr).Decode(&embeds); err != nil {
		return nil, err
	}
	return embeds, nil
}

func makeIntTensor(src []int64, bs, seq int, dt dtypes.DType) *tensors.Tensor {
	if dt == dtypes.Int32 {
		buf32 := make([]int32, len(src))
		for i, v := range src {
			buf32[i] = int32(v)
		}
		return tensors.FromFlatDataAndDimensions(buf32, bs, seq)
	}
	return tensors.FromFlatDataAndDimensions(src, bs, seq)
}

func downloadWithRetry(client *http.Client, url, destPath string, maxRetries int, log *logging.DualLogger) error {
	var lastErr error
	for attempt := 1; attempt <= maxRetries; attempt++ {
		if err := downloadOnce(client, url, destPath); err != nil {
			lastErr = err
			if log != nil {
				log.Warn("onnxembed", "attempt %d/%d failed for %s: %v", attempt, maxRetries, url, err)
			}
			if attempt < maxRetries {
				time.Sleep(time.Duration(attempt*attempt) * time.Second)
			}
			continue
		}
		return nil
	}
	return lastErr
}

func downloadOnce(client *http.Client, url, destPath string) error {
	if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "gomlx-korean-opinion/0.2")
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("unexpected HTTP status %s", resp.Status)
	}
	tmp := destPath + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	n, copyErr := io.Copy(f, resp.Body)
	closeErr := f.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	if n == 0 {
		_ = os.Remove(tmp)
		return fmt.Errorf("empty download from %s", url)
	}
	return os.Rename(tmp, destPath)
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
