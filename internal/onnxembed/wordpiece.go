package onnxembed

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"unicode"
)

// WordPieceTokenizer implements pure-Go BERT WordPiece tokenization from a Hugging Face vocab.txt file.
type WordPieceTokenizer struct {
	vocab map[string]int64
	clsID int64
	sepID int64
	padID int64
	unkID int64
}

// LoadWordPiece loads a vocab.txt file where each line index corresponds to the token ID.
func LoadWordPiece(vocabPath string) (*WordPieceTokenizer, error) {
	f, err := os.Open(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("open wordpiece vocab %q: %w", vocabPath, err)
	}
	defer func() { _ = f.Close() }()

	vocab := make(map[string]int64, 32768)
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var idx int64
	for scanner.Scan() {
		tok := strings.TrimRight(scanner.Text(), "\r\n")
		if tok != "" {
			if _, exists := vocab[tok]; !exists {
				vocab[tok] = idx
			}
		}
		idx++
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan wordpiece vocab %q: %w", vocabPath, err)
	}
	if len(vocab) == 0 {
		return nil, fmt.Errorf("empty wordpiece vocab %q", vocabPath)
	}

	lookupSpecial := func(tok string, fallback int64) int64 {
		if id, ok := vocab[tok]; ok {
			return id
		}
		return fallback
	}

	return &WordPieceTokenizer{
		vocab: vocab,
		padID: lookupSpecial("[PAD]", 0),
		unkID: lookupSpecial("[UNK]", 1),
		clsID: lookupSpecial("[CLS]", 2),
		sepID: lookupSpecial("[SEP]", 3),
	}, nil
}

// VocabSize returns the number of entries loaded from vocab.txt.
func (w *WordPieceTokenizer) VocabSize() int {
	return len(w.vocab)
}

// Encode converts text into fixed-length BERT WordPiece input_ids, attention_mask, and token_type_ids.
func (w *WordPieceTokenizer) Encode(text string, maxSeqLen int) (inputIDs, attentionMask, tokenTypeIDs []int64) {
	inputIDs = make([]int64, maxSeqLen)
	attentionMask = make([]int64, maxSeqLen)
	tokenTypeIDs = make([]int64, maxSeqLen)
	for i := range inputIDs {
		inputIDs[i] = w.padID
	}
	if maxSeqLen < 2 {
		return inputIDs, attentionMask, tokenTypeIDs
	}

	words := basicPreTokenize(text)
	maxContent := maxSeqLen - 2
	subIDs := make([]int64, 0, maxContent)
	for _, word := range words {
		pieces := w.tokenizeWord(word)
		for _, pid := range pieces {
			if len(subIDs) >= maxContent {
				break
			}
			subIDs = append(subIDs, pid)
		}
		if len(subIDs) >= maxContent {
			break
		}
	}

	pos := 0
	inputIDs[pos] = w.clsID
	attentionMask[pos] = 1
	pos++
	for _, pid := range subIDs {
		inputIDs[pos] = pid
		attentionMask[pos] = 1
		pos++
	}
	inputIDs[pos] = w.sepID
	attentionMask[pos] = 1
	return inputIDs, attentionMask, tokenTypeIDs
}

func (w *WordPieceTokenizer) tokenizeWord(word string) []int64 {
	runes := []rune(word)
	if len(runes) == 0 {
		return nil
	}
	if len(runes) > 100 {
		return []int64{w.unkID}
	}
	out := make([]int64, 0, len(runes))
	start := 0
	for start < len(runes) {
		end := len(runes)
		var matchedID int64
		matched := false
		for end > start {
			sub := string(runes[start:end])
			if start > 0 {
				sub = "##" + sub
			}
			if id, ok := w.vocab[sub]; ok {
				matchedID = id
				matched = true
				break
			}
			end--
		}
		if !matched {
			return []int64{w.unkID}
		}
		out = append(out, matchedID)
		start = end
	}
	return out
}

func basicPreTokenize(text string) []string {
	var tokens []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			tokens = append(tokens, cur.String())
			cur.Reset()
		}
	}
	for _, r := range text {
		switch {
		case unicode.IsSpace(r) || unicode.IsControl(r):
			flush()
		case isPunctuationOrSymbol(r) || isCJKIdeograph(r):
			flush()
			tokens = append(tokens, string(r))
		default:
			cur.WriteRune(r)
		}
	}
	flush()
	return tokens
}

func isPunctuationOrSymbol(r rune) bool {
	return unicode.IsPunct(r) || unicode.IsSymbol(r)
}

func isCJKIdeograph(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) ||
		(r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0xF900 && r <= 0xFAFF)
}
