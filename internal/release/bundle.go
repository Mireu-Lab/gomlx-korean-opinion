package release

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
)

// InferenceSpec is stored next to Hugging Face model weights so CLI inference can be reproduced.
type InferenceSpec struct {
	Framework        string `json:"framework"`
	Provider         string `json:"provider"`
	BackendDefault   string `json:"backend_default"`
	CheckpointSubdir string `json:"checkpoint_subdir"`
	VocabFile        string `json:"vocab_file"`
	PretrainSubdir   string `json:"pretrain_subdir,omitempty"`
	TopicsFile       string `json:"topics_file"`
	NFCTaxonomyFile  string `json:"nfc_taxonomy_file"`
	EntityRulesFile  string `json:"entity_rules_file"`
	RootsConfigFile  string `json:"roots_config_file,omitempty"`
	CreatedAtUTC     string `json:"created_at_utc"`
	InferCommand     string `json:"infer_command"`
}

// BundleManifest lists files copied into the Hugging Face model staging directory.
type BundleManifest struct {
	Provider     string   `json:"provider"`
	StagingDir   string   `json:"staging_dir"`
	CopiedFiles  []string `json:"copied_files"`
	CreatedAtUTC string   `json:"created_at_utc"`
}

// StageModel copies trained weights, vocab, and inference configs into destDir for Hugging Face upload.
// Source code and compiled binaries are not copied.
func StageModel(cfg *config.Config, destDir string) (*BundleManifest, error) {
	if strings.TrimSpace(destDir) == "" {
		return nil, fmt.Errorf("staging directory must not be empty")
	}
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return nil, fmt.Errorf("create staging dir %q: %w", destDir, err)
	}

	ckptRel := "checkpoints/" + filepath.Base(cfg.Paths.CheckpointDir)
	var copied []string

	ckptFiles, err := copyDirFiles(cfg.Paths.CheckpointDir, filepath.Join(destDir, ckptRel))
	if err != nil {
		return nil, fmt.Errorf("copy checkpoint dir %q: %w", cfg.Paths.CheckpointDir, err)
	}
	copied = append(copied, ckptFiles...)

	configFiles := []struct {
		src string
		dst string
	}{
		{cfg.Paths.TopicsFile, filepath.Join(destDir, "config", filepath.Base(cfg.Paths.TopicsFile))},
		{cfg.Paths.NFCTaxonomyFile, filepath.Join(destDir, "config", filepath.Base(cfg.Paths.NFCTaxonomyFile))},
		{cfg.Paths.EntityRulesFile, filepath.Join(destDir, "config", filepath.Base(cfg.Paths.EntityRulesFile))},
	}
	if strings.TrimSpace(cfg.Embedding.RootsConfigFile) != "" {
		configFiles = append(configFiles, struct{ src, dst string }{
			cfg.Embedding.RootsConfigFile,
			filepath.Join(destDir, "config", filepath.Base(cfg.Embedding.RootsConfigFile)),
		})
	}
	for _, f := range configFiles {
		if strings.TrimSpace(f.src) == "" {
			continue
		}
		if err := copyFile(f.src, f.dst); err != nil {
			return nil, err
		}
		rel, _ := filepath.Rel(destDir, f.dst)
		copied = append(copied, filepath.ToSlash(rel))
	}

	pretrainRel := ""
	if strings.TrimSpace(cfg.Embedding.PretrainCheckpointDir) != "" {
		if st, err := os.Stat(cfg.Embedding.PretrainCheckpointDir); err == nil && st.IsDir() {
			pretrainRel = "checkpoints/" + filepath.Base(cfg.Embedding.PretrainCheckpointDir)
			pf, err := copyDirFiles(cfg.Embedding.PretrainCheckpointDir, filepath.Join(destDir, pretrainRel))
			if err != nil {
				return nil, fmt.Errorf("copy pretrain dir %q: %w", cfg.Embedding.PretrainCheckpointDir, err)
			}
			copied = append(copied, pf...)
		}
	}

	spec := InferenceSpec{
		Framework:        "gomlx",
		Provider:         cfg.Embedding.Provider,
		BackendDefault:   cfg.Runtime.Backend,
		CheckpointSubdir: ckptRel,
		VocabFile:        filepath.ToSlash(filepath.Join(ckptRel, cfg.Tokenizer.VocabFilename)),
		PretrainSubdir:   pretrainRel,
		TopicsFile:       filepath.ToSlash(filepath.Join("config", filepath.Base(cfg.Paths.TopicsFile))),
		NFCTaxonomyFile:  filepath.ToSlash(filepath.Join("config", filepath.Base(cfg.Paths.NFCTaxonomyFile))),
		EntityRulesFile:  filepath.ToSlash(filepath.Join("config", filepath.Base(cfg.Paths.EntityRulesFile))),
		CreatedAtUTC:     time.Now().UTC().Format(time.RFC3339),
		InferCommand:     "opinion-analyzer infer --config config/config.yaml --checkpoint-dir " + ckptRel + " --text \"...\"",
	}
	if spec.PretrainSubdir != "" {
		spec.RootsConfigFile = filepath.ToSlash(filepath.Join("config", filepath.Base(cfg.Embedding.RootsConfigFile)))
	}
	specPath := filepath.Join(destDir, "inference.json")
	if err := writeJSON(specPath, spec); err != nil {
		return nil, err
	}
	copied = append(copied, "inference.json")

	card := modelCard(cfg, spec)
	if err := os.WriteFile(filepath.Join(destDir, "README.md"), []byte(card), 0o644); err != nil {
		return nil, fmt.Errorf("write model card: %w", err)
	}
	copied = append(copied, "README.md")

	manifest := &BundleManifest{
		Provider:     cfg.Embedding.Provider,
		StagingDir:   destDir,
		CopiedFiles:  copied,
		CreatedAtUTC: spec.CreatedAtUTC,
	}
	if err := writeJSON(filepath.Join(destDir, "bundle_manifest.json"), manifest); err != nil {
		return nil, err
	}
	return manifest, nil
}

func copyDirFiles(srcDir, dstDir string) ([]string, error) {
	entries, err := os.ReadDir(srcDir)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dstDir, 0o755); err != nil {
		return nil, err
	}
	var rels []string
	base := filepath.Base(dstDir)
	parent := filepath.Base(filepath.Dir(dstDir))
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasPrefix(name, ".") {
			continue
		}
		if strings.HasPrefix(name, "checkpoint-") && strings.HasSuffix(name, ".json") {
			binName := strings.TrimSuffix(name, ".json") + ".bin"
			if _, err := os.Stat(filepath.Join(srcDir, binName)); err != nil {
				continue
			}
		}
		src := filepath.Join(srcDir, name)
		dst := filepath.Join(dstDir, name)
		if err := copyFile(src, dst); err != nil {
			return nil, err
		}
		rels = append(rels, filepath.ToSlash(filepath.Join(parent, base, name)))
	}
	if len(rels) == 0 {
		return nil, fmt.Errorf("no files found in %q", srcDir)
	}
	return rels, nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("open %q: %w", src, err)
	}
	defer func() { _ = in.Close() }()
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return fmt.Errorf("mkdir for %q: %w", dst, err)
	}
	out, err := os.Create(dst)
	if err != nil {
		return fmt.Errorf("create %q: %w", dst, err)
	}
	defer func() { _ = out.Close() }()
	if _, err := io.Copy(out, in); err != nil {
		return fmt.Errorf("copy %q -> %q: %w", src, dst, err)
	}
	return out.Close()
}

func writeJSON(path string, v any) error {
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, raw, 0o644)
}

func modelCard(cfg *config.Config, spec InferenceSpec) string {
	var b strings.Builder
	b.WriteString("---\n")
	b.WriteString("library_name: gomlx\n")
	b.WriteString("language:\n- ko\n")
	b.WriteString("pipeline_tag: text-classification\n")
	b.WriteString("tags:\n- korean\n- nfc\n- gomlx\n- opinion-mining\n")
	b.WriteString("---\n\n")
	fmt.Fprintf(&b, "# GoMLX Korean Opinion (`%s`)\n\n", spec.Provider)
	b.WriteString("Trained GoMLX weights for KLUE-YNAT topic, NSMC sentiment, and Nuclear Feeling Checklist (16 labels).\n\n")
	b.WriteString("**Source code lives on GitHub** (`github.com/Mireu-Lab/gomlx-korean-opinion`). This Hub repo holds checkpoints, `vocab.json`, and taxonomy YAML only.\n\n")
	b.WriteString("## Inference\n\n```bash\n")
	b.WriteString("hf download ${HF_MODEL_REPO} --local-dir ./hf_model\n")
	fmt.Fprintf(&b, "go run ./cmd/opinion-analyzer infer --config config/config.yaml --checkpoint-dir ./hf_model/%s --text \"금리 인상으로 가계 부담이 커진다\"\n", spec.CheckpointSubdir)
	b.WriteString("```\n\n")
	fmt.Fprintf(&b, "- provider: `%s`\n- default backend: `%s`\n- vocab: `%s`\n", spec.Provider, spec.BackendDefault, spec.VocabFile)
	if spec.PretrainSubdir != "" {
		fmt.Fprintf(&b, "- KO-Hub pretrain: `%s`\n", spec.PretrainSubdir)
	}
	b.WriteString("\nDo not commit this directory to GitHub. Publish with `./scripts/publish_hf.sh model`.\n")
	return b.String()
}
