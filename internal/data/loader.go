package data

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/parquet-go/parquet-go"
)

// Record represents a unified text item from NSMC, KLUE-YNAT, or external article inputs.
type Record struct {
	ID             string
	Text           string
	Source         string
	Date           string
	URL            string
	TopicID        int
	HasTopic       bool
	SentimentLabel float32
	HasSentiment   bool
}

type ynatParquetRow struct {
	Guid  string `parquet:"guid"`
	Title string `parquet:"title"`
	Label int64  `parquet:"label"`
	URL   string `parquet:"url"`
	Date  string `parquet:"date"`
}

// LoadNSMC parses an NSMC TSV file (header: id\tdocument\tlabel).
func LoadNSMC(path string) ([]Record, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("open NSMC %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	records := make([]Record, 0, 150000)
	skipped := 0
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := strings.TrimRight(scanner.Text(), "\r\n")
		if lineNum == 1 && strings.HasPrefix(line, "id\t") {
			continue
		}
		if strings.TrimSpace(line) == "" {
			skipped++
			continue
		}
		parts := strings.Split(line, "\t")
		if len(parts) < 3 {
			skipped++
			continue
		}
		id := strings.TrimSpace(parts[0])
		doc := strings.TrimSpace(strings.Join(parts[1:len(parts)-1], " "))
		lblStr := strings.TrimSpace(parts[len(parts)-1])
		if doc == "" {
			skipped++
			continue
		}
		lbl, err := strconv.Atoi(lblStr)
		if err != nil || (lbl != 0 && lbl != 1) {
			skipped++
			continue
		}
		records = append(records, Record{
			ID:             id,
			Text:           doc,
			Source:         "nsmc",
			TopicID:        -1,
			HasTopic:       false,
			SentimentLabel: float32(lbl),
			HasSentiment:   true,
		})
	}
	if err := scanner.Err(); err != nil {
		return nil, skipped, fmt.Errorf("scan NSMC %q: %w", path, err)
	}
	return records, skipped, nil
}

// LoadYNAT parses a KLUE-YNAT Parquet file using pure-Go parquet-go.
func LoadYNAT(path string) ([]Record, error) {
	rows, err := parquet.ReadFile[ynatParquetRow](path)
	if err != nil {
		return nil, fmt.Errorf("read YNAT parquet %q: %w", path, err)
	}
	records := make([]Record, 0, len(rows))
	for i, r := range rows {
		title := strings.TrimSpace(r.Title)
		if title == "" {
			continue
		}
		id := strings.TrimSpace(r.Guid)
		if id == "" {
			id = fmt.Sprintf("ynat-%d", i)
		}
		records = append(records, Record{
			ID:       id,
			Text:     title,
			Source:   "ynat",
			Date:     strings.TrimSpace(r.Date),
			URL:      strings.TrimSpace(r.URL),
			TopicID:  int(r.Label),
			HasTopic: true,
		})
	}
	return records, nil
}

// LoadCustomArticles loads articles from .jsonl, .ndjson, .tsv, or .csv files.
func LoadCustomArticles(path string) ([]Record, error) {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".jsonl", ".ndjson":
		return loadCustomJSONL(path)
	case ".tsv":
		return loadCustomDelimited(path, '\t')
	case ".csv":
		return loadCustomDelimited(path, ',')
	default:
		return nil, fmt.Errorf("unsupported article input format %q (expected .jsonl, .ndjson, .tsv, or .csv)", ext)
	}
}

func loadCustomJSONL(path string) ([]Record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open jsonl %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var records []Record
	idx := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var obj map[string]any
		if err := json.Unmarshal([]byte(line), &obj); err != nil {
			return nil, fmt.Errorf("parse jsonl line %d in %q: %w", idx+1, path, err)
		}
		id := firstString(obj, "id", "guid")
		if id == "" {
			id = fmt.Sprintf("article-%d", idx+1)
		}
		title := firstString(obj, "title", "headline")
		body := firstString(obj, "text", "content", "document", "body")
		combined := strings.TrimSpace(strings.Join(nonEmpty(title, body), " "))
		if combined == "" {
			continue
		}
		records = append(records, Record{
			ID:      id,
			Text:    combined,
			Source:  "article",
			Date:    firstString(obj, "date", "published_at"),
			URL:     firstString(obj, "url", "link"),
			TopicID: -1,
		})
		idx++
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan jsonl %q: %w", path, err)
	}
	return records, nil
}

func loadCustomDelimited(path string, sep rune) ([]Record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open delimited %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	r := csv.NewReader(f)
	r.Comma = sep
	r.LazyQuotes = true
	r.FieldsPerRecord = -1

	header, err := r.Read()
	if err != nil {
		return nil, fmt.Errorf("read header of %q: %w", path, err)
	}
	colIdx := make(map[string]int, len(header))
	for i, h := range header {
		colIdx[strings.ToLower(strings.TrimSpace(strings.TrimPrefix(h, "\ufeff")))] = i
	}

	var records []Record
	rowNum := 0
	for {
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read row %d in %q: %w", rowNum+1, path, err)
		}
		rowNum++
		id := colVal(row, colIdx, "id", "guid")
		if id == "" {
			id = fmt.Sprintf("article-%d", rowNum)
		}
		title := colVal(row, colIdx, "title", "headline")
		body := colVal(row, colIdx, "text", "content", "document", "body")
		combined := strings.TrimSpace(strings.Join(nonEmpty(title, body), " "))
		if combined == "" {
			continue
		}
		records = append(records, Record{
			ID:      id,
			Text:    combined,
			Source:  "article",
			Date:    colVal(row, colIdx, "date", "published_at"),
			URL:     colVal(row, colIdx, "url", "link"),
			TopicID: -1,
		})
	}
	return records, nil
}

func firstString(m map[string]any, keys ...string) string {
	for _, k := range keys {
		if v, ok := m[k]; ok && v != nil {
			s := strings.TrimSpace(fmt.Sprint(v))
			if s != "" && s != "<nil>" {
				return s
			}
		}
	}
	return ""
}

func colVal(row []string, colIdx map[string]int, keys ...string) string {
	for _, k := range keys {
		if idx, ok := colIdx[k]; ok && idx < len(row) {
			s := strings.TrimSpace(row[idx])
			if s != "" {
				return s
			}
		}
	}
	return ""
}

func nonEmpty(items ...string) []string {
	out := make([]string, 0, len(items))
	for _, s := range items {
		s = strings.TrimSpace(s)
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}
