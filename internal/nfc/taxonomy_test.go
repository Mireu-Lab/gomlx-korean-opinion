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

	// Regression tests for homonym disambiguation ("화가") and rare feelings ("외로움", "소외").
	for _, artHeadline := range []string{
		"남춘모 부조적 단색화가 펼쳐 보이는 자연",
		"차병원 한국여류화가협회와 소녀가장 돕기 자선전시회 개최",
	} {
		tArt, _, _ := tax.WeakAnnotate(artHeadline)
		if tArt[8] != 0 {
			t.Fatalf("homonym regression: %q should not trigger anger (id=8), got %v", artHeadline, tArt[8])
		}
	}

	tAnger, _, _ := tax.WeakAnnotate("시민들 화가 났다 거센 분노")
	if tAnger[8] != 1.0 {
		t.Fatalf("expected true anger context to trigger anger (id=8)")
	}
	tLonely, _, _ := tax.WeakAnnotate("독거노인 1인 가구 고독감 심화")
	if tLonely[4] != 1.0 {
		t.Fatalf("expected loneliness (id=4) to trigger")
	}
	tAlien, _, _ := tax.WeakAnnotate("비수도권 농어촌 소외지역 낙후 불평등")
	if tAlien[14] != 1.0 {
		t.Fatalf("expected alienation (id=14) to trigger")
	}
}
