package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/config"
)

// Tokenizer converts Korean/ASCII text into fixed-length integer token ID sequences.
type Tokenizer struct {
	Cfg      config.TokenizerConfig `json:"cfg"`
	Vocab    map[string]int32       `json:"vocab"`
	RevVocab []string               `json:"rev_vocab"`
}

// Normalize lowercases ASCII letters, strips punctuation into spaces, and preserves Hangul and alphanumeric characters.
func Normalize(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	lastSpace := true
	for _, r := range strings.ToLower(s) {
		if isKeepRune(r) {
			b.WriteRune(r)
			lastSpace = false
		} else if !lastSpace {
			b.WriteByte(' ')
			lastSpace = true
		}
	}
	return strings.TrimSpace(b.String())
}

func isKeepRune(r rune) bool {
	if (r >= 0xAC00 && r <= 0xD7A3) || (r >= 0x3130 && r <= 0x318F) || (r >= 0x1100 && r <= 0x11FF) {
		return true
	}
	if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
		return true
	}
	return unicode.IsLetter(r) || unicode.IsDigit(r)
}

// SplitKoreanParticle separates a trailing Korean particle from a Hangul word when the remaining
// stem has at least minStemRunes Hangul characters (preventing false subword cross-boundary bigrams like 단색화+가 -> 2:화가).
func SplitKoreanParticle(word string, particles []string, minStemRunes int) (stem, particle string) {
	if len(particles) == 0 {
		return word, ""
	}
	if minStemRunes < 1 {
		minStemRunes = 2
	}
	runes := []rune(word)
	if len(runes) <= minStemRunes {
		return word, ""
	}
	for _, r := range runes {
		if r < 0xAC00 || r > 0xD7A3 {
			return word, ""
		}
	}
	sorted := make([]string, len(particles))
	copy(sorted, particles)
	sort.Slice(sorted, func(i, j int) bool {
		ri, rj := len([]rune(sorted[i])), len([]rune(sorted[j]))
		if ri != rj {
			return ri > rj
		}
		return sorted[i] < sorted[j]
	})
	for _, p := range sorted {
		if p == "" {
			continue
		}
		pRunes := []rune(p)
		if len(runes)-len(pRunes) < minStemRunes {
			continue
		}
		if strings.HasSuffix(word, p) {
			return string(runes[:len(runes)-len(pRunes)]), p
		}
	}
	return word, ""
}

// ExtractWordTokens produces sentence-order word/particle tokens (without char n-grams) for sequential encoders.
func ExtractWordTokens(text string, cfg config.TokenizerConfig) []string {
	norm := Normalize(text)
	if norm == "" {
		return nil
	}
	fields := strings.Fields(norm)
	out := make([]string, 0, len(fields)*2)
	for _, w := range fields {
		if cfg.SplitParticles {
			stem, part := SplitKoreanParticle(w, cfg.KoreanParticles, cfg.ParticleMinStemRunes)
			out = append(out, "w:"+stem)
			if part != "" {
				out = append(out, "p:"+part)
			}
		} else {
			out = append(out, "w:"+w)
		}
	}
	return out
}

// ExtractTokens produces whole-word tokens followed by optional character bigrams and trigrams.
// When SplitParticles is enabled, character n-grams are derived solely from word stems to prevent cross-boundary collisions.
func ExtractTokens(text string, cfg config.TokenizerConfig) []string {
	norm := Normalize(text)
	if norm == "" {
		return nil
	}
	fields := strings.Fields(norm)
	stems := make([]string, 0, len(fields))
	tokens := make([]string, 0, len(fields)*4)
	for _, w := range fields {
		if cfg.SplitParticles {
			stem, part := SplitKoreanParticle(w, cfg.KoreanParticles, cfg.ParticleMinStemRunes)
			stems = append(stems, stem)
			tokens = append(tokens, "w:"+stem)
			if part != "" {
				tokens = append(tokens, "p:"+part)
			}
		} else {
			stems = append(stems, w)
			tokens = append(tokens, "w:"+w)
		}
	}
	if cfg.UseCharBigrams {
		for _, stem := range stems {
			runes := []rune(stem)
			for i := 0; i+2 <= len(runes); i++ {
				tokens = append(tokens, "2:"+string(runes[i:i+2]))
			}
		}
	}
	if cfg.UseCharTrigrams {
		for _, stem := range stems {
			runes := []rune(stem)
			for i := 0; i+3 <= len(runes); i++ {
				tokens = append(tokens, "3:"+string(runes[i:i+3]))
			}
		}
	}
	return tokens
}

// Fit builds a frequency-ordered vocabulary from a corpus of texts.
func Fit(texts []string, cfg config.TokenizerConfig) *Tokenizer {
	counts := make(map[string]int, 65536)
	for _, text := range texts {
		for _, tok := range ExtractTokens(text, cfg) {
			counts[tok]++
		}
	}

	type kv struct {
		tok   string
		count int
	}
	items := make([]kv, 0, len(counts))
	for tok, cnt := range counts {
		if cnt >= cfg.MinFreq {
			items = append(items, kv{tok: tok, count: cnt})
		}
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].count != items[j].count {
			return items[i].count > items[j].count
		}
		return items[i].tok < items[j].tok
	})

	maxEntries := cfg.VocabSize
	if len(items)+2 < maxEntries {
		maxEntries = len(items) + 2
	}
	vocab := make(map[string]int32, maxEntries)
	rev := make([]string, 0, maxEntries)

	vocab[cfg.PadToken] = 0
	rev = append(rev, cfg.PadToken)
	vocab[cfg.UnkToken] = 1
	rev = append(rev, cfg.UnkToken)

	for _, it := range items {
		if len(rev) >= cfg.VocabSize {
			break
		}
		if _, exists := vocab[it.tok]; exists {
			continue
		}
		id := int32(len(rev))
		vocab[it.tok] = id
		rev = append(rev, it.tok)
	}

	return &Tokenizer{
		Cfg:      cfg,
		Vocab:    vocab,
		RevVocab: rev,
	}
}

// VocabLen returns the actual vocabulary size including PAD and UNK.
func (t *Tokenizer) VocabLen() int {
	return len(t.RevVocab)
}

// Encode converts text into a fixed-length int32 token ID slice padded with 0.
func (t *Tokenizer) Encode(text string) []int32 {
	return t.encodeSlice(ExtractTokens(text, t.Cfg), t.Cfg.MaxSeqLen)
}

// EncodeWordSeq converts text into a fixed-length sentence-order word/particle token ID sequence.
func (t *Tokenizer) EncodeWordSeq(text string) []int32 {
	seqLen := t.Cfg.WordSeqLen
	if seqLen <= 0 {
		seqLen = t.Cfg.MaxSeqLen
	}
	return t.encodeSlice(ExtractWordTokens(text, t.Cfg), seqLen)
}

func (t *Tokenizer) encodeSlice(extracted []string, seqLen int) []int32 {
	out := make([]int32, seqLen)
	unkID := int32(1)
	if id, ok := t.Vocab[t.Cfg.UnkToken]; ok {
		unkID = id
	}
	limit := len(extracted)
	if limit > seqLen {
		limit = seqLen
	}
	for i := 0; i < limit; i++ {
		if id, ok := t.Vocab[extracted[i]]; ok {
			out[i] = id
		} else {
			out[i] = unkID
		}
	}
	return out
}

// Save writes the tokenizer configuration and vocabulary to disk as JSON.
func (t *Tokenizer) Save(path string) error {
	data, err := json.Marshal(t)
	if err != nil {
		return fmt.Errorf("marshal tokenizer: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write tokenizer %q: %w", path, err)
	}
	return nil
}

// Load reads a serialized tokenizer from disk.
func Load(path string) (*Tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer %q: %w", path, err)
	}
	var t Tokenizer
	if err := json.Unmarshal(raw, &t); err != nil {
		return nil, fmt.Errorf("parse tokenizer %q: %w", path, err)
	}
	if len(t.RevVocab) < 2 || len(t.Vocab) < 2 {
		return nil, fmt.Errorf("tokenizer %q has invalid vocabulary", path)
	}
	return &t, nil
}
