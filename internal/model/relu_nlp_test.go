package model

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/optimizer"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/data"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/graphkg"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/logging"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/nfc"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/pretrain"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/tokenizer"
)

func TestReLUNLPGraphTrainDecreaseAndCheckpointParity(t *testing.T) {
	backend, err := compute.NewWithConfig("go")
	if err != nil {
		t.Fatalf("compute.NewWithConfig(go): %v", err)
	}
	defer backend.Finalize()

	tax, err := nfc.LoadTaxonomy(
		filepath.Join("..", "..", "config", "topics.yaml"),
		filepath.Join("..", "..", "config", "nfc_taxonomy.yaml"),
	)
	if err != nil {
		t.Fatalf("LoadTaxonomy: %v", err)
	}

	tokCfg := config.TokenizerConfig{
		VocabSize: 200, MaxSeqLen: 16, WordSeqLen: 10, MinFreq: 1, UseCharBigrams: true, PadToken: "<PAD>", UnkToken: "<UNK>",
	}
	ynatRecs := []data.Record{
		{ID: "y1", Text: "물가 부담 금리 압박 가중", Source: "ynat", TopicID: 1, HasTopic: true},
		{ID: "y2", Text: "신규 인공지능 반도체 기술 개발", Source: "ynat", TopicID: 0, HasTopic: true},
		{ID: "y3", Text: "정치권 갈등 적대 공방 격화", Source: "ynat", TopicID: 6, HasTopic: true},
		{ID: "y4", Text: "국가대표 결승 치열한 경쟁 승부", Source: "ynat", TopicID: 5, HasTopic: true},
	}
	nsmcRecs := []data.Record{
		{ID: "n1", Text: "정말 최악의 영화 시간낭비 짜증 분노", Source: "nsmc", TopicID: -1, SentimentLabel: 0, HasSentiment: true},
		{ID: "n2", Text: "감동적이고 재미있는 최고의 작품", Source: "nsmc", TopicID: -1, SentimentLabel: 1, HasSentiment: true},
		{ID: "n3", Text: "너무 무섭고 공포스럽고 불안하다", Source: "nsmc", TopicID: -1, SentimentLabel: 0, HasSentiment: true},
		{ID: "n4", Text: "따뜻한 여운과 그리움이 남는다", Source: "nsmc", TopicID: -1, SentimentLabel: 1, HasSentiment: true},
	}

	corpus := make([]string, 0, len(ynatRecs)+len(nsmcRecs))
	for _, r := range ynatRecs {
		corpus = append(corpus, r.Text)
	}
	for _, r := range nsmcRecs {
		corpus = append(corpus, r.Text)
	}
	tok := tokenizer.Fit(corpus, tokCfg)
	rng := rand.New(rand.NewSource(42))
	ynatEx, _ := data.PrepareExamples(ynatRecs, tok, tax, 0.5, rng)
	nsmcEx, _ := data.PrepareExamples(nsmcRecs, tok, tax, 0.5, rng)

	trainCfg := config.TrainingConfig{
		BatchSize:              4,
		YNATBatchRatio:         0.5,
		NSMCBatchRatio:         0.5,
		NFCPositiveSampleRatio: 0.5,
		LossWeights: config.LossWeightsConfig{
			Topic: 1.0, Sentiment: 1.0, NFC: 1.5, NFCPosWeight: 4.0, MaskEpsilon: 1e-6,
		},
	}
	ds, err := data.NewMultiTaskTrainDataset("test_train", ynatEx, nsmcEx, trainCfg, tokCfg.MaxSeqLen, tokCfg.WordSeqLen, tax.NumFeelings(), 42)
	if err != nil {
		t.Fatalf("NewMultiTaskTrainDataset: %v", err)
	}

	graphCfg := GraphConfig{
		VocabSize:   tok.VocabLen(),
		EmbedDim:    16,
		HiddenDims:  []int{32, 16},
		HeadDim:     16,
		DropoutRate: 0.1,
		NumTopics:   tax.NumTopics(),
		NumNFC:      tax.NumFeelings(),
	}

	store := model.NewStore()
	store.SetParam(optimizer.ParamOptimizer, "adamw")
	store.SetParam(optimizer.ParamLearningRate, 0.02)
	if err := store.SetRNGStateFromSeed(42); err != nil {
		t.Fatalf("SetRNGStateFromSeed: %v", err)
	}

	ckptDir := filepath.Join(t.TempDir(), "ckpt")
	ckptHandler, err := checkpoint.Build(store).Dir(ckptDir).Keep(2).Done()
	if err != nil {
		t.Fatalf("checkpoint.Build: %v", err)
	}

	trainer := train.NewTrainer(
		backend,
		store,
		NewTrainModelFn(graphCfg, false),
		BuildMultiTaskLoss(trainCfg.LossWeights, tax.NumTopics()),
		optimizer.FromStore(store),
		nil, nil,
	)
	loop := train.NewLoop(trainer)

	var firstLoss, lastLoss float32
	loop.OnStep("record-loss", 0, func(l *train.Loop, metrics []*tensors.Tensor) error {
		v := tensorScalarFloat32(metrics[0])
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite loss at step %d: %v", l.LoopStep, v)
		}
		if l.LoopStep == 0 {
			firstLoss = v
		}
		lastLoss = v
		return nil
	})

	if _, err := loop.RunSteps(ds, 30); err != nil {
		t.Fatalf("RunSteps: %v", err)
	}
	if lastLoss >= firstLoss {
		t.Fatalf("expected loss to decrease over 30 steps: first=%v last=%v", firstLoss, lastLoss)
	}
	if err := ckptHandler.Save(); err != nil {
		t.Fatalf("ckptHandler.Save: %v", err)
	}

	pred1, err := NewPredictor(backend, store, graphCfg, 4, tokCfg.MaxSeqLen, tokCfg.WordSeqLen)
	if err != nil {
		t.Fatalf("NewPredictor(orig): %v", err)
	}
	defer pred1.Close()

	out1, err := pred1.PredictExamples(ynatEx)
	if err != nil {
		t.Fatalf("PredictExamples(orig): %v", err)
	}

	pred2, _, err := LoadPredictor(backend, ckptDir, graphCfg, 4, tokCfg.MaxSeqLen, tokCfg.WordSeqLen)
	if err != nil {
		t.Fatalf("LoadPredictor: %v", err)
	}
	defer pred2.Close()

	out2, err := pred2.PredictExamples(ynatEx)
	if err != nil {
		t.Fatalf("PredictExamples(loaded): %v", err)
	}

	for i := range out1 {
		if len(out1[i].TopicProbs) != tax.NumTopics() || len(out1[i].NFCProbs) != tax.NumFeelings() {
			t.Fatalf("unexpected prediction vector sizes")
		}
		if len(out1[i].SemanticEmbed) != graphCfg.HiddenDims[len(graphCfg.HiddenDims)-1] {
			t.Fatalf("unexpected semantic embedding len: got %d", len(out1[i].SemanticEmbed))
		}
		var normSq float64
		for _, v := range out1[i].SemanticEmbed {
			normSq += float64(v * v)
		}
		if normSq > 0 && math.Abs(math.Sqrt(normSq)-1.0) > 1e-4 {
			t.Fatalf("expected L2-normalized semantic embedding at %d, got norm=%v", i, math.Sqrt(normSq))
		}
		if math.Abs(float64(out1[i].SentimentProb-out2[i].SentimentProb)) > 1e-5 {
			t.Fatalf("checkpoint reload sentiment mismatch at %d: %v vs %v", i, out1[i].SentimentProb, out2[i].SentimentProb)
		}
		for c := range out1[i].NFCProbs {
			if math.Abs(float64(out1[i].NFCProbs[c]-out2[i].NFCProbs[c])) > 1e-5 {
				t.Fatalf("checkpoint reload NFC mismatch at [%d][%d]: %v vs %v", i, c, out1[i].NFCProbs[c], out2[i].NFCProbs[c])
			}
		}
		for d := range out1[i].SemanticEmbed {
			if math.Abs(float64(out1[i].SemanticEmbed[d]-out2[i].SemanticEmbed[d])) > 1e-5 {
				t.Fatalf("checkpoint reload semantic embed mismatch at [%d][%d]: %v vs %v", i, d, out1[i].SemanticEmbed[d], out2[i].SemanticEmbed[d])
			}
		}
	}
}

func TestKoHubGraphIntegrationAndCheckpointParity(t *testing.T) {
	backend, err := compute.NewWithConfig("go")
	if err != nil {
		t.Fatalf("compute.NewWithConfig(go): %v", err)
	}
	defer backend.Finalize()

	cfg, err := config.Load(filepath.Join("..", "..", "config", "config.yaml"))
	if err != nil {
		t.Fatalf("config.Load: %v", err)
	}
	tax, err := nfc.LoadTaxonomy(
		filepath.Join("..", "..", "config", "topics.yaml"),
		filepath.Join("..", "..", "config", "nfc_taxonomy.yaml"),
	)
	if err != nil {
		t.Fatalf("LoadTaxonomy: %v", err)
	}
	rootsCfg, err := graphkg.LoadRootsConfig(filepath.Join("..", "..", "config", "multilingual_roots.yaml"))
	if err != nil {
		t.Fatalf("LoadRootsConfig: %v", err)
	}

	tmpDir := t.TempDir()
	pretrainDir := filepath.Join(tmpDir, "pretrain")
	downstreamDir := filepath.Join(tmpDir, "downstream")
	log, err := logging.NewDualLogger(tmpDir, "status.log", "processing.log", "INFO", false)
	if err != nil {
		t.Fatalf("NewDualLogger: %v", err)
	}
	defer func() { _ = log.Close() }()

	ynatRecs := []data.Record{
		{ID: "y1", Text: "남춘모 단색화가 미술 전시회 개최", Source: "ynat", TopicID: 3, HasTopic: true},
		{ID: "y2", Text: "고물가 금리 부담 시민들 화가 났다 분노", Source: "ynat", TopicID: 1, HasTopic: true},
		{ID: "y3", Text: "신규 인공지능 반도체 기술 개발", Source: "ynat", TopicID: 0, HasTopic: true},
		{ID: "y4", Text: "국가대표 결승 치열한 경쟁 승부", Source: "ynat", TopicID: 5, HasTopic: true},
	}
	nsmcRecs := []data.Record{
		{ID: "n1", Text: "최악의 영화 시간낭비 짜증 분노", Source: "nsmc", TopicID: -1, SentimentLabel: 0, HasSentiment: true},
		{ID: "n2", Text: "감동적이고 재미있는 최고의 작품", Source: "nsmc", TopicID: -1, SentimentLabel: 1, HasSentiment: true},
		{ID: "n3", Text: "너무 무섭고 공포스럽고 불안하다", Source: "nsmc", TopicID: -1, SentimentLabel: 0, HasSentiment: true},
		{ID: "n4", Text: "따뜻한 여운과 그리움이 남는다", Source: "nsmc", TopicID: -1, SentimentLabel: 1, HasSentiment: true},
	}
	corpus := make([]string, 0, len(ynatRecs)+len(nsmcRecs))
	for _, r := range ynatRecs {
		corpus = append(corpus, r.Text)
	}
	for _, r := range nsmcRecs {
		corpus = append(corpus, r.Text)
	}

	tok := tokenizer.Fit(corpus, config.TokenizerConfig{
		VocabSize: 300, MaxSeqLen: 16, WordSeqLen: 10, MinFreq: 1, UseCharBigrams: true, SplitParticles: true,
		ParticleMinStemRunes: 2, KoreanParticles: []string{"가", "는", "을", "를", "의", "와", "과"},
		PadToken: "<PAD>", UnkToken: "<UNK>", VocabFilename: "vocab.json",
	})
	g, err := graphkg.BuildGraphFromParsed(rootsCfg, nil, nil, corpus, tax, log)
	if err != nil {
		t.Fatalf("BuildGraphFromParsed: %v", err)
	}

	rootsCfg.Pretrain.CheckpointDir = pretrainDir
	rootsCfg.Pretrain.EmbedDim = 16
	rootsCfg.Pretrain.OutDim = 32
	rootsCfg.Pretrain.RootEmbedDim = 32
	rootsCfg.Pretrain.AttnDim = 16
	rootsCfg.Pretrain.BatchSize = 4
	rootsCfg.Pretrain.PretrainSteps = 20
	rootsCfg.Pretrain.LogEverySteps = 20
	rootsCfg.Pretrain.LearningRate = 0.01

	preStore, _, err := pretrain.TrainPretrain(backend, &rootsCfg.Pretrain, tok, g, corpus, 42, "vocab.json", log)
	if err != nil {
		t.Fatalf("TrainPretrain: %v", err)
	}
	preStore.Finalize()

	rng := rand.New(rand.NewSource(42))
	ynatEx, _ := data.PrepareExamples(ynatRecs, tok, tax, 0.5, rng)
	nsmcEx, _ := data.PrepareExamples(nsmcRecs, tok, tax, 0.5, rng)

	providers := []string{
		"ko_hub_graph",
		"ko_hub_cnn_relu",
		"ko_hub_transformer_relu",
		"ko_hub_lstm_relu",
	}
	for _, prov := range providers {
		t.Run(prov, func(t *testing.T) {
			provDir := filepath.Join(downstreamDir, prov)
			cfg.Paths.CheckpointDir = provDir
			cfg.Tokenizer.MaxSeqLen = 16
			cfg.Tokenizer.WordSeqLen = 10
			cfg.Model.EmbedDim = 16
			cfg.Model.HeadDim = 16
			cfg.Model.DropoutRate = 0.1
			cfg.Embedding.Provider = prov
			cfg.Embedding.PretrainCheckpointDir = pretrainDir
			cfg.Embedding.FreezeEncoder = false
			cfg.Embedding.StudentOutDim = 32
			cfg.Embedding.CNNChannels = 16
			cfg.Embedding.CNNKernels = []int{2, 3, 4}
			cfg.Embedding.TransformerLayers = 1
			cfg.Embedding.TransformerHeads = 2
			cfg.Embedding.TransformerDim = 32
			cfg.Embedding.LSTMHidden = 16
			cfg.Embedding.AttnPoolDim = 16
			cfg.Training.BatchSize = 4
			cfg.Training.EvalBatchSize = 4
			cfg.Training.TrainSteps = 15
			cfg.Training.LogEverySteps = 15
			cfg.Training.LearningRate = 0.01

			downStore, err := Train(backend, cfg, tax, tok.VocabLen(), ynatEx, nsmcEx, false, log)
			if err != nil {
				t.Fatalf("Train(%s): %v", prov, err)
			}
			defer downStore.Finalize()

			graphCfg := BuildGraphConfig(cfg, tok.VocabLen(), tax.NumTopics(), tax.NumFeelings())
			pred1, err := NewPredictor(backend, downStore, graphCfg, 4, tok.Cfg.MaxSeqLen, tok.Cfg.WordSeqLen)
			if err != nil {
				t.Fatalf("NewPredictor(%s): %v", prov, err)
			}
			defer pred1.Close()
			out1, err := pred1.PredictExamples(ynatEx)
			if err != nil {
				t.Fatalf("PredictExamples(orig,%s): %v", prov, err)
			}

			pred2, loadedStore, err := LoadPredictor(backend, provDir, graphCfg, 4, tok.Cfg.MaxSeqLen, tok.Cfg.WordSeqLen)
			if err != nil {
				t.Fatalf("LoadPredictor(%s): %v", prov, err)
			}
			defer pred2.Close()
			defer loadedStore.Finalize()
			out2, err := pred2.PredictExamples(ynatEx)
			if err != nil {
				t.Fatalf("PredictExamples(loaded,%s): %v", prov, err)
			}

			for i := range out1 {
				if len(out1[i].SemanticEmbed) != cfg.Embedding.StudentOutDim {
					t.Fatalf("%s expected StudentOutDim=%d, got %d", prov, cfg.Embedding.StudentOutDim, len(out1[i].SemanticEmbed))
				}
				var normSq float64
				for d := range out1[i].SemanticEmbed {
					v1, v2 := out1[i].SemanticEmbed[d], out2[i].SemanticEmbed[d]
					if math.IsNaN(float64(v1)) || math.IsInf(float64(v1), 0) {
						t.Fatalf("%s produced non-finite embedding", prov)
					}
					normSq += float64(v1 * v1)
					if math.Abs(float64(v1-v2)) > 1e-5 {
						t.Fatalf("%s reload embed mismatch at [%d][%d]: %v vs %v", prov, i, d, v1, v2)
					}
				}
				if math.Abs(math.Sqrt(normSq)-1.0) > 1e-4 {
					t.Fatalf("%s expected L2-normalized embedding at %d, got norm=%v", prov, i, math.Sqrt(normSq))
				}
			}
		})
	}
}

func TestEmbedHeadTrainAndCheckpointParity(t *testing.T) {
	backend, err := compute.NewWithConfig("go")
	if err != nil {
		t.Fatalf("compute.NewWithConfig(go): %v", err)
	}
	defer backend.Finalize()

	cfg, err := config.Load(filepath.Join("..", "..", "config", "config.yaml"))
	if err != nil {
		t.Fatalf("config.Load: %v", err)
	}
	tax, err := nfc.LoadTaxonomy(
		filepath.Join("..", "..", "config", "topics.yaml"),
		filepath.Join("..", "..", "config", "nfc_taxonomy.yaml"),
	)
	if err != nil {
		t.Fatalf("LoadTaxonomy: %v", err)
	}

	tmpDir := t.TempDir()
	ckptDir := filepath.Join(tmpDir, "onnx_head_ckpt")
	cfg.Paths.CheckpointDir = ckptDir
	cfg.Embedding.Provider = "onnx_backbone_relu"
	cfg.Embedding.StudentOutDim = 32
	cfg.Model.HeadDim = 16
	cfg.Training.BatchSize = 4
	cfg.Training.EvalBatchSize = 4
	cfg.Training.TrainSteps = 20
	cfg.Training.LogEverySteps = 20
	cfg.Training.LearningRate = 0.01

	log, err := logging.NewDualLogger(tmpDir, "status.log", "processing.log", "INFO", false)
	if err != nil {
		t.Fatalf("NewDualLogger: %v", err)
	}
	defer func() { _ = log.Close() }()

	makeVec := func(seedVal float32) []float32 {
		v := make([]float32, 64)
		for i := range v {
			v[i] = float32(math.Sin(float64(i+1) * float64(seedVal)))
		}
		return v
	}
	ynatRecs := []data.Record{
		{ID: "y1", Text: "물가 부담", Source: "ynat", TopicID: 1, HasTopic: true},
		{ID: "y2", Text: "AI 반도체", Source: "ynat", TopicID: 0, HasTopic: true},
		{ID: "y3", Text: "정치 갈등", Source: "ynat", TopicID: 6, HasTopic: true},
		{ID: "y4", Text: "스포츠 우승", Source: "ynat", TopicID: 5, HasTopic: true},
	}
	nsmcRecs := []data.Record{
		{ID: "n1", Text: "최악 분노", Source: "nsmc", TopicID: -1, SentimentLabel: 0, HasSentiment: true},
		{ID: "n2", Text: "감동 명작", Source: "nsmc", TopicID: -1, SentimentLabel: 1, HasSentiment: true},
		{ID: "n3", Text: "공포 불안", Source: "nsmc", TopicID: -1, SentimentLabel: 0, HasSentiment: true},
		{ID: "n4", Text: "여운 그리움", Source: "nsmc", TopicID: -1, SentimentLabel: 1, HasSentiment: true},
	}
	ynatVecs := [][]float32{makeVec(1), makeVec(2), makeVec(3), makeVec(4)}
	nsmcVecs := [][]float32{makeVec(5), makeVec(6), makeVec(7), makeVec(8)}

	rng := rand.New(rand.NewSource(42))
	ynatEx, _ := PrepareEmbedExamples(ynatRecs, ynatVecs, tax, 0.5, rng)
	nsmcEx, _ := PrepareEmbedExamples(nsmcRecs, nsmcVecs, tax, 0.5, rng)

	store, err := TrainOnEmbeddings(backend, cfg, tax, 64, ynatEx, nsmcEx, false, log)
	if err != nil {
		t.Fatalf("TrainOnEmbeddings: %v", err)
	}
	defer store.Finalize()

	pred1, err := NewEmbedPredictor(backend, store, cfg, 64, tax.NumTopics(), tax.NumFeelings(), 4)
	if err != nil {
		t.Fatalf("NewEmbedPredictor: %v", err)
	}
	defer pred1.Close()
	out1, err := pred1.PredictEmbedExamples(ynatEx)
	if err != nil {
		t.Fatalf("PredictEmbedExamples(orig): %v", err)
	}

	pred2, loadedStore, err := LoadEmbedPredictor(backend, ckptDir, cfg, 64, tax.NumTopics(), tax.NumFeelings(), 4)
	if err != nil {
		t.Fatalf("LoadEmbedPredictor: %v", err)
	}
	defer pred2.Close()
	defer loadedStore.Finalize()
	out2, err := pred2.PredictEmbedExamples(ynatEx)
	if err != nil {
		t.Fatalf("PredictEmbedExamples(loaded): %v", err)
	}

	for i := range out1 {
		for d := range out1[i].SemanticEmbed {
			if math.Abs(float64(out1[i].SemanticEmbed[d]-out2[i].SemanticEmbed[d])) > 1e-5 {
				t.Fatalf("onnx head reload mismatch at [%d][%d]", i, d)
			}
		}
	}
}
