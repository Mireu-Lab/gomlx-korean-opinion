package nfc

import (
	"path/filepath"
	"testing"
)

func TestLoadTaxonomyAndWeakAnnotate(t *testing.T) {
	tax, err := LoadTaxonomy(
		filepath.Join("..", "..", "config", "topics.yaml"),
		filepath.Join("..", "..", "config", "nfc_taxonomy.yaml"),
	)
	if err != nil {
		t.Fatalf("LoadTaxonomy failed: %v", err)
	}
	if tax.NumTopics() != 7 {
		t.Fatalf("expected 7 KLUE-YNAT topics, got %d", tax.NumTopics())
	}
	if tax.NumFeelings() != 16 {
		t.Fatalf("expected 16 Nuclear Feeling Checklist items, got %d", tax.NumFeelings())
	}

	targets, scores, matched := tax.WeakAnnotate("물가 부담과 고금리 압박에 서민들 분노와 불안 확산")
	if matched == 0 {
		t.Fatalf("expected weak annotation matches on Korean headline")
	}
	// Expect burden (0), anger (8), anxiety (12) to be active.
	for _, id := range []int{0, 8, 12} {
		if targets[id] != 1.0 || scores[id] <= 0 {
			f, _ := tax.FeelingByID(id)
			t.Fatalf("expected feeling %s (id=%d) to trigger, target=%v score=%v", f.KoName, id, targets[id], scores[id])
		}
	}

	_, _, noneMatched := tax.WeakAnnotate("오늘 날씨가 맑고 산책하기 좋습니다")
	if noneMatched != 0 {
		t.Fatalf("expected 0 matches on neutral sentence, got %d", noneMatched)
	}
}
