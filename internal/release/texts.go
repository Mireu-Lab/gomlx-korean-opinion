package release

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/data"
)

// CollectInferRecords builds inference records from a raw --text value and/or an input file.
// File formats reuse LoadCustomArticles (.jsonl/.csv/.tsv). A .txt file is treated as one document per non-empty line.
func CollectInferRecords(rawText, inputPath string) ([]data.Record, error) {
	var recs []data.Record
	if t := strings.TrimSpace(rawText); t != "" {
		recs = append(recs, data.Record{ID: "text-0", Text: t, Source: "infer", TopicID: -1})
	}
	if strings.TrimSpace(inputPath) == "" {
		if len(recs) == 0 {
			return nil, fmt.Errorf("provide --text and/or --input for infer")
		}
		return recs, nil
	}

	lower := strings.ToLower(inputPath)
	switch {
	case strings.HasSuffix(lower, ".txt"):
		lines, err := readNonEmptyLines(inputPath)
		if err != nil {
			return nil, err
		}
		for i, line := range lines {
			recs = append(recs, data.Record{
				ID:      fmt.Sprintf("line-%d", i),
				Text:    line,
				Source:  "infer",
				TopicID: -1,
			})
		}
	default:
		loaded, err := data.LoadCustomArticles(inputPath)
		if err != nil {
			return nil, err
		}
		recs = append(recs, loaded...)
	}
	if len(recs) == 0 {
		return nil, fmt.Errorf("no texts to infer")
	}
	return recs, nil
}

func readNonEmptyLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open infer input %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	var lines []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		lines = append(lines, line)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("read infer input %q: %w", path, err)
	}
	return lines, nil
}
