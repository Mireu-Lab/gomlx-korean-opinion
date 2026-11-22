package graphkg

import (
	"archive/zip"
	"bufio"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/logging"
)

// HanjaEntry represents one Hanja character parsed from Unihan with Korean reading and English glosses.
type HanjaEntry struct {
	Char       string
	CodePoint  rune
	Readings   []string
	Definition string
	Glosses    []string
}

// UnihanLexicon indexes Hanja characters by Hangul syllable reading.
type UnihanLexicon struct {
	ByHangul map[string][]HanjaEntry
	ByChar   map[string]HanjaEntry
}

// OMWConcept represents an aligned multilingual WordNet synset across KO, EN, JA, and ZH.
type OMWConcept struct {
	SynsetID string
	KO       []string
	EN       []string
	JA       []string
	ZH       []string
}

type sourceDownloadSpec struct {
	name string
	url  string
	path string
}

// EnsureSources downloads missing Unihan and OMW multilingual files into cfg.CacheDir.
func EnsureSources(cfg *RootsConfig, log *logging.DualLogger) error {
	if err := os.MkdirAll(cfg.CacheDir, 0755); err != nil {
		return fmt.Errorf("create cache dir %q: %w", cfg.CacheDir, err)
	}
	specs := []sourceDownloadSpec{
		{name: "Unicode Unihan", url: cfg.Sources.UnihanZipURL, path: filepath.Join(cfg.CacheDir, cfg.Sources.UnihanZipFile)},
		{name: "OMW Korean Wikt", url: cfg.Sources.OMWKorWiktURL, path: filepath.Join(cfg.CacheDir, cfg.Sources.OMWKorWiktFile)},
		{name: "OMW Korean CLDR", url: cfg.Sources.OMWKorCLDRURL, path: filepath.Join(cfg.CacheDir, cfg.Sources.OMWKorCLDRFile)},
		{name: "OMW English", url: cfg.Sources.OMWEngURL, path: filepath.Join(cfg.CacheDir, cfg.Sources.OMWEngFile)},
		{name: "OMW Japanese", url: cfg.Sources.OMWJpnURL, path: filepath.Join(cfg.CacheDir, cfg.Sources.OMWJpnFile)},
		{name: "OMW Chinese", url: cfg.Sources.OMWCmnURL, path: filepath.Join(cfg.CacheDir, cfg.Sources.OMWCmnFile)},
	}

	client := &http.Client{Timeout: time.Duration(cfg.HTTPTimeoutSec) * time.Second}
	for _, sp := range specs {
		if st, err := os.Stat(sp.path); err == nil && st.Size() > 0 {
			log.Info("graphkg.sources", "%s already cached at %s (%d bytes)", sp.name, sp.path, st.Size())
			continue
		}
		log.Info("graphkg.sources", "downloading %s from %s -> %s", sp.name, sp.url, sp.path)
		if err := downloadWithRetry(client, sp.url, sp.path, cfg.HTTPRetries, log); err != nil {
			return fmt.Errorf("download %s: %w", sp.name, err)
		}
	}
	return nil
}

func downloadWithRetry(client *http.Client, url, destPath string, maxRetries int, log *logging.DualLogger) error {
	var lastErr error
	for attempt := 1; attempt <= maxRetries; attempt++ {
		if err := downloadOnce(client, url, destPath); err != nil {
			lastErr = err
			log.Warn("graphkg.sources", "attempt %d/%d failed for %s: %v", attempt, maxRetries, url, err)
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

	tmpPath := destPath + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}
	n, copyErr := io.Copy(f, resp.Body)
	closeErr := f.Close()
	if copyErr != nil {
		_ = os.Remove(tmpPath)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmpPath)
		return closeErr
	}
	if n == 0 {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("downloaded empty file from %s", url)
	}
	if err := os.Rename(tmpPath, destPath); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	return nil
}

// ParseUnihanZip extracts kHangul readings and kDefinition glosses from Unihan.zip.
func ParseUnihanZip(zipPath string) (*UnihanLexicon, error) {
	zr, err := zip.OpenReader(zipPath)
	if err != nil {
		return nil, fmt.Errorf("open Unihan zip %q: %w", zipPath, err)
	}
	defer func() { _ = zr.Close() }()

	byChar := make(map[string]*HanjaEntry, 32768)
	var readingsFound bool

	for _, f := range zr.File {
		base := filepath.Base(f.Name)
		if base != "Unihan_Readings.txt" {
			continue
		}
		readingsFound = true
		rc, err := f.Open()
		if err != nil {
			return nil, fmt.Errorf("open %s in %q: %w", f.Name, zipPath, err)
		}
		err = parseUnihanReadingsStream(rc, byChar)
		_ = rc.Close()
		if err != nil {
			return nil, fmt.Errorf("parse %s: %w", f.Name, err)
		}
	}
	if !readingsFound {
		return nil, fmt.Errorf("Unihan_Readings.txt not found inside %q", zipPath)
	}

	lex := &UnihanLexicon{
		ByHangul: make(map[string][]HanjaEntry, 4096),
		ByChar:   make(map[string]HanjaEntry, len(byChar)),
	}
	for ch, entry := range byChar {
		entry.Glosses = extractGlossTokens(entry.Definition)
		lex.ByChar[ch] = *entry
		for _, hangul := range entry.Readings {
			lex.ByHangul[hangul] = append(lex.ByHangul[hangul], *entry)
		}
	}

	// Prioritize common CJK Unified Ideographs with concise definitions.
	for hangul, entries := range lex.ByHangul {
		sort.Slice(entries, func(i, j int) bool {
			ri := rankHanjaEntry(entries[i])
			rj := rankHanjaEntry(entries[j])
			if ri != rj {
				return ri < rj
			}
			return entries[i].CodePoint < entries[j].CodePoint
		})
		lex.ByHangul[hangul] = entries
	}
	return lex, nil
}

func parseUnihanReadingsStream(r io.Reader, byChar map[string]*HanjaEntry) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "\t", 3)
		if len(parts) < 3 {
			continue
		}
		field := strings.TrimSpace(parts[1])
		if field != "kHangul" && field != "kDefinition" {
			continue
		}
		cp, ok := parseUnicodeCodePoint(parts[0])
		if !ok {
			continue
		}
		ch := string(cp)
		entry := byChar[ch]
		if entry == nil {
			entry = &HanjaEntry{Char: ch, CodePoint: cp}
			byChar[ch] = entry
		}
		val := strings.TrimSpace(parts[2])
		if field == "kDefinition" {
			entry.Definition = val
			continue
		}
		// kHangul tokens look like "가:0E" or "화:01E"; keep unique Hangul syllables.
		seen := make(map[string]struct{}, len(entry.Readings))
		for _, existing := range entry.Readings {
			seen[existing] = struct{}{}
		}
		for _, tok := range strings.Fields(val) {
			syll := tok
			if idx := strings.IndexByte(tok, ':'); idx >= 0 {
				syll = tok[:idx]
			}
			syll = strings.TrimSpace(syll)
			if syll == "" {
				continue
			}
			if _, ok := seen[syll]; !ok {
				seen[syll] = struct{}{}
				entry.Readings = append(entry.Readings, syll)
			}
		}
	}
	return scanner.Err()
}

func parseUnicodeCodePoint(s string) (rune, bool) {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "U+") {
		return 0, false
	}
	v, err := strconv.ParseInt(strings.TrimPrefix(s, "U+"), 16, 32)
	if err != nil || v <= 0 {
		return 0, false
	}
	return rune(v), true
}

func rankHanjaEntry(e HanjaEntry) int {
	score := 0
	if e.CodePoint < 0x4E00 || e.CodePoint > 0x9FFF {
		score += 100
	}
	if len(e.Glosses) == 0 {
		score += 50
	}
	return score
}

func extractGlossTokens(def string) []string {
	if strings.TrimSpace(def) == "" {
		return nil
	}
	// Split Unihan definition clauses by ';' or ',' and keep concise 1-2 word English glosses.
	clauses := strings.FieldsFunc(strings.ToLower(def), func(r rune) bool {
		return r == ';' || r == ','
	})
	var out []string
	seen := make(map[string]struct{})
	for _, cl := range clauses {
		cl = strings.TrimSpace(cl)
		if cl == "" || strings.HasPrefix(cl, "see ") || strings.HasPrefix(cl, "surname") {
			continue
		}
		// Strip parenthetical hints.
		if idx := strings.IndexByte(cl, '('); idx >= 0 {
			cl = strings.TrimSpace(cl[:idx])
		}
		words := strings.Fields(cl)
		if len(words) == 0 || len(words) > 2 {
			continue
		}
		// Strip leading "to " on verb glosses.
		if len(words) == 2 && words[0] == "to" {
			words = words[1:]
		}
		token := cleanASCIIWord(strings.Join(words, "_"))
		if len(token) < 2 {
			continue
		}
		if _, ok := seen[token]; ok {
			continue
		}
		seen[token] = struct{}{}
		out = append(out, token)
		if len(out) >= 3 {
			break
		}
	}
	return out
}

func cleanASCIIWord(s string) string {
	var b strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '_' {
			b.WriteRune(r)
		}
	}
	return strings.Trim(b.String(), "_")
}

// ParseOMWTab parses an Open Multilingual WordNet TSV file into synsetID -> []lemma.
func ParseOMWTab(path string) (map[string][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open OMW tab %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	synsets := make(map[string][]string, 16384)
	seenPerSynset := make(map[string]map[string]struct{}, 16384)

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := strings.TrimRight(scanner.Text(), "\r\n")
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Split(line, "\t")
		if len(parts) < 3 {
			continue
		}
		synID := strings.TrimSpace(parts[0])
		rel := strings.TrimSpace(parts[1])
		if rel != "lemma" && !strings.HasSuffix(rel, ":lemma") {
			continue
		}
		lemma := cleanOMWLemma(parts[2])
		if synID == "" || lemma == "" {
			continue
		}
		seen := seenPerSynset[synID]
		if seen == nil {
			seen = make(map[string]struct{}, 4)
			seenPerSynset[synID] = seen
		}
		if _, ok := seen[lemma]; ok {
			continue
		}
		seen[lemma] = struct{}{}
		synsets[synID] = append(synsets[synID], lemma)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan OMW tab %q: %w", path, err)
	}
	return synsets, nil
}

func cleanOMWLemma(s string) string {
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, "+", "")
	if s == "" {
		return ""
	}
	// Collapse internal whitespace to underscore for multi-word English lemmas.
	fields := strings.Fields(strings.ToLower(s))
	if len(fields) == 0 {
		return ""
	}
	joined := strings.Join(fields, "_")
	for _, r := range joined {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			return joined
		}
	}
	return ""
}

// MergeOMWSynsets joins Korean Wiktionary + CLDR lemmas with English, Japanese, and Chinese synset lemmas.
func MergeOMWSynsets(korWikt, korCLDR, eng, jpn, cmn map[string][]string) []OMWConcept {
	korMerged := make(map[string][]string, len(korWikt)+len(korCLDR))
	addMap := func(src map[string][]string) {
		for synID, lemmas := range src {
			existing := korMerged[synID]
			seen := make(map[string]struct{}, len(existing))
			for _, w := range existing {
				seen[w] = struct{}{}
			}
			for _, w := range lemmas {
				if _, ok := seen[w]; !ok {
					seen[w] = struct{}{}
					existing = append(existing, w)
				}
			}
			korMerged[synID] = existing
		}
	}
	addMap(korWikt)
	addMap(korCLDR)

	synIDs := make([]string, 0, len(korMerged))
	for synID := range korMerged {
		synIDs = append(synIDs, synID)
	}
	sort.Strings(synIDs)

	concepts := make([]OMWConcept, 0, len(synIDs))
	for _, synID := range synIDs {
		ko := korMerged[synID]
		en := eng[synID]
		ja := jpn[synID]
		zh := cmn[synID]
		if len(en) == 0 && len(ja) == 0 && len(zh) == 0 {
			continue
		}
		concepts = append(concepts, OMWConcept{
			SynsetID: synID,
			KO:       ko,
			EN:       en,
			JA:       ja,
			ZH:       zh,
		})
	}
	return concepts
}
