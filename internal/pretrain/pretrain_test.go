package pretrain

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/ml/model"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/graphkg"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/logging"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/nfc"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/tokenizer"
)

func TestPretrainContrastiveDisambiguationAndTransfer(t *testing.T) {
	backend, err := compute.NewWithConfig("go")
	if err != nil {
		t.Fatalf("compute.NewWithConfig(go): %v", err)
	}
	defer backend.Finalize()

	tmpDir := t.TempDir()
	rootsCfg, err := graphkg.LoadRootsConfig(filepath.Join("..", "..", "config", "multilingual_roots.yaml"))
	if err != nil {
		t.Fatalf("LoadRootsConfig: %v", err)
	}
	rootsCfg.Pretrain.CheckpointDir = filepath.Join(tmpDir, "pretrain_ckpt")
	rootsCfg.Pretrain.EmbedDim = 16
	rootsCfg.Pretrain.OutDim = 32
	rootsCfg.Pretrain.RootEmbedDim = 32
	rootsCfg.Pretrain.AttnDim = 16
	rootsCfg.Pretrain.BatchSize = 8
	rootsCfg.Pretrain.PretrainSteps = 35
	rootsCfg.Pretrain.LogEverySteps = 35
	rootsCfg.Pretrain.LearningRate = 0.02

	tax, err := nfc.LoadTaxonomy(
		filepath.Join("..", "..", "config", "topics.yaml"),
		filepath.Join("..", "..", "config", "nfc_taxonomy.yaml"),
	)
	if err != nil {
		t.Fatalf("LoadTaxonomy: %v", err)
	}

	corpus := []string{
		"화가 전시회 미술 단색화 작품 협회 그림 갤러리",
		"서양화 화가 화백 초대전 전시회 작품",
		"화가 분노 짜증 항의 규탄 열받 치밀 억울 최악",
		"물가 폭등 시민들 화가 났다 분노 항의 규탄",
		"사기 피해 고소 구속 혐의 경찰 검찰 기만",
		"국가대표 경기 감독 선수 승리 사기 진작 응원",
		"고금리 물가 대출 서민 부담 한국은행",
		"독거노인 고독 혼자 외로움 고립 은둔",
	}

	log, err := logging.NewDualLogger(tmpDir, "status.log", "processing.log", "INFO", false)
	if err != nil {
		t.Fatalf("NewDualLogger: %v", err)
	}
	defer func() { _ = log.Close() }()

	g, err := graphkg.BuildGraphFromParsed(rootsCfg, nil, nil, corpus, tax, log)
	if err != nil {
		t.Fatalf("BuildGraphFromParsed: %v", err)
	}

	tok := tokenizer.Fit(corpus, config.TokenizerConfig{
		VocabSize:       300,
		MaxSeqLen:       16,
		MinFreq:         1,
		UseCharBigrams:  true,
		UseCharTrigrams: false,
		PadToken:        "<PAD>",
		UnkToken:        "<UNK>",
		VocabFilename:   "vocab.json",
	})

	store, report, err := TrainPretrain(backend, &rootsCfg.Pretrain, tok, g, corpus, 42, "vocab.json", log)
	if err != nil {
		t.Fatalf("TrainPretrain failed: %v", err)
	}
	if report.Meta.FinalLoss >= report.Meta.FirstLoss {
		t.Fatalf("expected pretrain loss to decrease over %d steps: first=%v final=%v",
			rootsCfg.Pretrain.PretrainSteps, report.Meta.FirstLoss, report.Meta.FinalLoss)
	}

	probes := []string{
		"화가 전시회 미술 작품",
		"화가 단색화 그림 협회",
		"화가 분노 짜증 항의",
	}
	embsOrig, err := EncodeStudentTexts(backend, store, tok, probes, rootsCfg.Pretrain.EmbedDim, rootsCfg.Pretrain.OutDim)
	if err != nil {
		t.Fatalf("EncodeStudentTexts(orig): %v", err)
	}

	simSameSense := cosineSim32(embsOrig[0], embsOrig[1])
	simDiffSense := cosineSim32(embsOrig[0], embsOrig[2])
	if simSameSense <= simDiffSense {
		t.Fatalf("expected same-sense painter cosine (%v) > opposite-sense anger cosine (%v)", simSameSense, simDiffSense)
	}

	// Verify selective Student-only transfer and freeze enforcement.
	transferredStore := model.NewStore()
	nVars, err := LoadPretrainedStudentWeights(rootsCfg.Pretrain.CheckpointDir, transferredStore, true)
	if err != nil {
		t.Fatalf("LoadPretrainedStudentWeights: %v", err)
	}
	if nVars < 5 {
		t.Fatalf("expected >= 5 transferred Student variables, got %d", nVars)
	}
	for v := range transferredStore.IterVariables() {
		if v.Trainable {
			t.Fatalf("expected transferred variable %s to be frozen (Trainable=false)", v.Path())
		}
	}

	embsLoaded, err := EncodeStudentTexts(backend, transferredStore, tok, probes, rootsCfg.Pretrain.EmbedDim, rootsCfg.Pretrain.OutDim)
	if err != nil {
		t.Fatalf("EncodeStudentTexts(loaded): %v", err)
	}
	for i := range embsOrig {
		for d := range embsOrig[i] {
			if math.Abs(float64(embsOrig[i][d]-embsLoaded[i][d])) > 1e-5 {
				t.Fatalf("student weight transfer mismatch at [%d][%d]: %v vs %v", i, d, embsOrig[i][d], embsLoaded[i][d])
			}
		}
	}
}

func cosineSim32(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
