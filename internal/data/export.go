package data

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/parquet-go/parquet-go"

	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/nfc"
	"github.com/Mireu-Lab/gomlx-korean-opinion/internal/tokenizer"
)

// PreprocessedRow defines the Hugging Face Datasets-compatible Parquet schema for preprocessed pipeline splits.
type PreprocessedRow struct {
	ID                 string  `parquet:"id"`
	Split              string  `parquet:"split"`
	Source             string  `parquet:"source"`
	Text               string  `parquet:"text"`
	Date               string  `parquet:"date"`
	URL                string  `parquet:"url"`
	HasTopic           bool    `parquet:"has_topic"`
	TopicID            int32   `parquet:"topic_id"`
	TopicLabel         string  `parquet:"topic_label"`
	HasSentiment       bool    `parquet:"has_sentiment"`
	SentimentLabel     float32 `parquet:"sentiment_label"`
	WordTokens         string  `parquet:"word_tokens"`
	TokenIDs           string  `parquet:"token_ids"`
	WordTokenIDs       string  `parquet:"word_token_ids"`
	NFCActiveLabels    string  `parquet:"nfc_active_labels"`
	NFCMatchedCount    int32   `parquet:"nfc_matched_count"`
	NFCHomonymGuardHit bool    `parquet:"nfc_homonym_guard_hit"`
	NFCSupervisedMask  float32 `parquet:"nfc_supervised_mask"`
	NFCTargetsJSON     string  `parquet:"nfc_targets_json"`
	NFCScoresJSON      string  `parquet:"nfc_scores_json"`
}

// ExportSplitStats summarizes row and supervision counts for one exported Parquet split.
type ExportSplitStats struct {
	Split              string         `json:"split"`
	File               string         `json:"file"`
	Rows               int            `json:"rows"`
	TopicLabeled       int            `json:"topic_labeled"`
	SentimentLabeled   int            `json:"sentiment_labeled"`
	NFCActive          int            `json:"nfc_active"`
	NFCSupervised      int            `json:"nfc_supervised"`
	HomonymGuardHits   int            `json:"homonym_guard_hits"`
	MeanWordTokenCount float64        `json:"mean_word_token_count"`
	PerFeelingHits     map[string]int `json:"per_feeling_hits"`
}

// ExportDatasetSummary captures global metadata and per-split statistics for a staged HF Dataset export.
type ExportDatasetSummary struct {
	CreatedAtUTC   string                      `json:"created_at_utc"`
	VocabSize      int                         `json:"vocab_size"`
	MaxSeqLen      int                         `json:"max_seq_len"`
	WordSeqLen     int                         `json:"word_seq_len"`
	SplitParticles bool                        `json:"split_particles"`
	ParticleCount  int                         `json:"particle_count"`
	GraphFile      string                      `json:"graph_file,omitempty"`
	GraphSizeBytes int64                       `json:"graph_size_bytes,omitempty"`
	Splits         map[string]ExportSplitStats `json:"splits"`
}

// ExportPreprocessedSplit writes preprocessed multi-task and NFC-annotated examples to a Parquet shard.
func ExportPreprocessedSplit(
	outPath string,
	splitName string,
	examples []EncodedExample,
	tok *tokenizer.Tokenizer,
	tax *nfc.Taxonomy,
) (ExportSplitStats, error) {
	if err := os.MkdirAll(filepath.Dir(outPath), 0755); err != nil {
		return ExportSplitStats{}, fmt.Errorf("create export dir for %q: %w", outPath, err)
	}

	feelings := tax.Feelings
	perFeeling := make(map[string]int, len(feelings))
	for _, f := range feelings {
		perFeeling[f.KoName] = 0
	}

	rows := make([]PreprocessedRow, len(examples))
	var topicCnt, sentCnt, nfcActiveCnt, nfcSupCnt, guardCnt int
	var totalWordTokens int

	for i, ex := range examples {
		rec := ex.Record
		topicLabel := ""
		if rec.HasTopic {
			topicCnt++
			if tdef, ok := tax.TopicByID(rec.TopicID); ok {
				topicLabel = tdef.KoName
			}
		}
		if rec.HasSentiment {
			sentCnt++
		}
		if ex.NFCMatched > 0 {
			nfcActiveCnt++
		}
		if ex.NFCMask > 0 {
			nfcSupCnt++
		}
		if ex.GuardTriggered {
			guardCnt++
		}

		activeNames := make([]string, 0, ex.NFCMatched)
		for idx, val := range ex.NFCTargets {
			if val > 0 && idx < len(feelings) {
				name := feelings[idx].KoName
				activeNames = append(activeNames, name)
				perFeeling[name]++
			}
		}

		wToks := tokenizer.ExtractWordTokens(rec.Text, tok.Cfg)
		totalWordTokens += len(wToks)

		tokenIDsJSON, err := marshalCompactJSON(ex.Tokens)
		if err != nil {
			return ExportSplitStats{}, fmt.Errorf("marshal token_ids at row %d: %w", i, err)
		}
		wordTokenIDsJSON, err := marshalCompactJSON(ex.WordTokens)
		if err != nil {
			return ExportSplitStats{}, fmt.Errorf("marshal word_token_ids at row %d: %w", i, err)
		}
		nfcTargetsJSON, err := marshalCompactJSON(ex.NFCTargets)
		if err != nil {
			return ExportSplitStats{}, fmt.Errorf("marshal nfc_targets at row %d: %w", i, err)
		}
		nfcScoresJSON, err := marshalCompactJSON(ex.NFCScores)
		if err != nil {
			return ExportSplitStats{}, fmt.Errorf("marshal nfc_scores at row %d: %w", i, err)
		}

		rows[i] = PreprocessedRow{
			ID:                 rec.ID,
			Split:              splitName,
			Source:             rec.Source,
			Text:               rec.Text,
			Date:               rec.Date,
			URL:                rec.URL,
			HasTopic:           rec.HasTopic,
			TopicID:            int32(rec.TopicID),
			TopicLabel:         topicLabel,
			HasSentiment:       rec.HasSentiment,
			SentimentLabel:     rec.SentimentLabel,
			WordTokens:         strings.Join(wToks, " "),
			TokenIDs:           tokenIDsJSON,
			WordTokenIDs:       wordTokenIDsJSON,
			NFCActiveLabels:    strings.Join(activeNames, ","),
			NFCMatchedCount:    int32(ex.NFCMatched),
			NFCHomonymGuardHit: ex.GuardTriggered,
			NFCSupervisedMask:  ex.NFCMask,
			NFCTargetsJSON:     nfcTargetsJSON,
			NFCScoresJSON:      nfcScoresJSON,
		}
	}

	if err := parquet.WriteFile(outPath, rows); err != nil {
		return ExportSplitStats{}, fmt.Errorf("write parquet %q: %w", outPath, err)
	}

	meanWordTokens := 0.0
	if len(examples) > 0 {
		meanWordTokens = float64(totalWordTokens) / float64(len(examples))
	}

	return ExportSplitStats{
		Split:              splitName,
		File:               filepath.ToSlash(outPath),
		Rows:               len(rows),
		TopicLabeled:       topicCnt,
		SentimentLabeled:   sentCnt,
		NFCActive:          nfcActiveCnt,
		NFCSupervised:      nfcSupCnt,
		HomonymGuardHits:   guardCnt,
		MeanWordTokenCount: meanWordTokens,
		PerFeelingHits:     perFeeling,
	}, nil
}

// WriteDatasetSummary writes ExportDatasetSummary as formatted JSON.
func WriteDatasetSummary(path string, summary ExportDatasetSummary) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("create summary dir for %q: %w", path, err)
	}
	raw, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal dataset summary: %w", err)
	}
	if err := os.WriteFile(path, append(raw, '\n'), 0644); err != nil {
		return fmt.Errorf("write dataset summary %q: %w", path, err)
	}
	return nil
}

func marshalCompactJSON(v any) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(b), nil
}
