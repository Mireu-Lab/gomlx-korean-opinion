package release

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
)

func TestCollectInferRecordsFromTextAndTxtFile(t *testing.T) {
	dir := t.TempDir()
	txt := filepath.Join(dir, "lines.txt")
	if err := os.WriteFile(txt, []byte("첫 문장\n\n둘째 문장\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	recs, err := CollectInferRecords("직접 입력", txt)
	if err != nil {
		t.Fatal(err)
	}
	if len(recs) != 3 {
		t.Fatalf("got %d records, want 3", len(recs))
	}
	if recs[0].Text != "직접 입력" || recs[1].Text != "첫 문장" || recs[2].Text != "둘째 문장" {
		t.Fatalf("unexpected texts: %+v", recs)
	}
	if _, err := CollectInferRecords("", ""); err == nil {
		t.Fatal("expected error when no text or input is given")
	}
}

func TestStageModelCopiesWeightsNotBinaries(t *testing.T) {
	src := t.TempDir()
	ckpt := filepath.Join(src, "relu_nlp")
	if err := os.MkdirAll(ckpt, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ckpt, "vocab.json"), []byte(`{"cfg":{}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ckpt, "weights.bin"), []byte("fake-weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfgDir := filepath.Join(src, "config")
	if err := os.MkdirAll(cfgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	topics := filepath.Join(cfgDir, "topics.yaml")
	nfc := filepath.Join(cfgDir, "nfc.yaml")
	ent := filepath.Join(cfgDir, "entity.yaml")
	for _, p := range []string{topics, nfc, ent} {
		if err := os.WriteFile(p, []byte("ok: true\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	cfg := &config.Config{}
	cfg.Runtime.Backend = "go"
	cfg.Embedding.Provider = "scratch_relu"
	cfg.Paths.CheckpointDir = ckpt
	cfg.Paths.TopicsFile = topics
	cfg.Paths.NFCTaxonomyFile = nfc
	cfg.Paths.EntityRulesFile = ent
	cfg.Tokenizer.VocabFilename = "vocab.json"

	dest := filepath.Join(src, "hf_model_staging")
	man, err := StageModel(cfg, dest)
	if err != nil {
		t.Fatal(err)
	}
	if man.Provider != "scratch_relu" {
		t.Fatalf("provider=%s", man.Provider)
	}
	mustExist := []string{
		filepath.Join(dest, "README.md"),
		filepath.Join(dest, "inference.json"),
		filepath.Join(dest, "checkpoints", "relu_nlp", "weights.bin"),
		filepath.Join(dest, "checkpoints", "relu_nlp", "vocab.json"),
		filepath.Join(dest, "config", "topics.yaml"),
	}
	for _, p := range mustExist {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("missing %s: %v", p, err)
		}
	}
	card, err := os.ReadFile(filepath.Join(dest, "README.md"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(card), "go run ./cmd/opinion-analyzer infer") {
		t.Fatalf("model card missing infer command:\n%s", card)
	}
}
