package cluster

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"

	"gopkg.in/yaml.v3"
)

// EntityRulesConfig holds externalized Korean NER patterns and stopwords loaded from YAML.
type EntityRulesConfig struct {
	Organizations []string `yaml:"organizations"`
	OrgSuffixes   []string `yaml:"org_suffixes"`
	Locations     []string `yaml:"locations"`
	PersonTitles  []string `yaml:"person_titles"`
	WorkTriggers  []string `yaml:"work_triggers"`
	PublicIssues  []string `yaml:"public_issues"`
	Stopwords     []string `yaml:"stopwords"`
}

// EntityExtractor extracts WORK, ORG, LOC, PER, and ISSUE anchors using YAML rules.
type EntityExtractor struct {
	orgs         []string
	orgSuffixes  []string
	locations    []string
	personTitles []string
	workTriggers []string
	publicIssues []string
	stopwords    map[string]struct{}
}

// LoadEntityExtractor reads entity rules from entity_rules.yaml.
func LoadEntityExtractor(path string) (*EntityExtractor, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read entity rules %q: %w", path, err)
	}
	var cfg EntityRulesConfig
	if err := yaml.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("parse entity rules %q: %w", path, err)
	}

	stopMap := make(map[string]struct{}, len(cfg.Stopwords))
	for _, sw := range cfg.Stopwords {
		norm := strings.ToLower(strings.TrimSpace(sw))
		if norm != "" {
			stopMap[norm] = struct{}{}
		}
	}

	return &EntityExtractor{
		orgs:         normalizeRuleList(cfg.Organizations),
		orgSuffixes:  normalizeRuleList(cfg.OrgSuffixes),
		locations:    normalizeRuleList(cfg.Locations),
		personTitles: normalizeRuleList(cfg.PersonTitles),
		workTriggers: normalizeRuleList(cfg.WorkTriggers),
		publicIssues: normalizeRuleList(cfg.PublicIssues),
		stopwords:    stopMap,
	}, nil
}

func normalizeRuleList(items []string) []string {
	seen := make(map[string]struct{}, len(items))
	out := make([]string, 0, len(items))
	for _, s := range items {
		norm := strings.ToLower(strings.TrimSpace(s))
		if norm == "" {
			continue
		}
		if _, ok := seen[norm]; ok {
			continue
		}
		seen[norm] = struct{}{}
		out = append(out, norm)
	}
	// Longer patterns first so multi-char stems take precedence.
	sort.Slice(out, func(i, j int) bool {
		if len([]rune(out[i])) != len([]rune(out[j])) {
			return len([]rune(out[i])) > len([]rune(out[j]))
		}
		return out[i] < out[j]
	})
	return out
}

// Stopwords returns the externalized stopword set for c-TF-IDF and topic coherence.
func (e *EntityExtractor) Stopwords() map[string]struct{} {
	return e.stopwords
}

// Extract identifies named entities and public issues in a Korean news headline or movie review.
func (e *EntityExtractor) Extract(text string) []Entity {
	lower := strings.ToLower(strings.TrimSpace(text))
	if lower == "" {
		return nil
	}

	var found []Entity
	seen := make(map[string]struct{})
	add := func(typ, val string) {
		val = strings.TrimSpace(val)
		if val == "" {
			return
		}
		key := typ + ":" + val
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		found = append(found, Entity{
			Type:      typ,
			Value:     val,
			Canonical: val,
		})
	}

	hasWorkCue := false
	for _, cue := range e.workTriggers {
		if strings.Contains(lower, cue) {
			hasWorkCue = true
			break
		}
	}

	// 1. Quoted works/titles (`《...》`, `「...」`, `『...』`, `'...'`, `"..."`).
	for _, pair := range [][2]string{{"《", "》"}, {"「", "」"}, {"『", "』"}, {"<", ">"}, {"'", "'"}, {"\"", "\""}} {
		for _, q := range extractDelimited(text, pair[0], pair[1]) {
			rLen := len([]rune(q))
			if rLen < 2 || rLen > 18 {
				continue
			}
			if pair[0] == "'" || pair[0] == "\"" {
				if !hasWorkCue && rLen > 10 {
					continue
				}
			}
			add("WORK", strings.ToLower(q))
		}
	}

	// 2. Organizations from curated gazetteer + suffix rules.
	for _, org := range e.orgs {
		if strings.Contains(lower, org) {
			add("ORG", org)
		}
	}
	fields := strings.Fields(lower)
	for _, tok := range fields {
		clean := trimPunctRunes(tok)
		if clean == "" {
			continue
		}
		for _, suf := range e.orgSuffixes {
			if strings.HasSuffix(clean, suf) {
				stem := strings.TrimSuffix(clean, suf)
				if len([]rune(stem)) >= 2 && isHangulWord(stem) {
					add("ORG", clean)
					break
				}
			}
		}
	}

	// 3. Locations.
	for _, loc := range e.locations {
		if strings.Contains(lower, loc) {
			add("LOC", loc)
		}
	}

	// 4. Person names anchored by role titles (`봉준호 감독`, `이창용 총재`, etc.).
	for idx, tok := range fields {
		clean := trimPunctRunes(tok)
		if clean == "" {
			continue
		}
		for _, title := range e.personTitles {
			if strings.HasPrefix(clean, title) && idx > 0 {
				prev := trimPunctRunes(fields[idx-1])
				if isLikelyKoreanPersonName(prev) {
					add("PER", prev+" "+title)
					break
				}
			}
			if strings.HasSuffix(clean, title) {
				stem := strings.TrimSuffix(clean, title)
				if isLikelyKoreanPersonName(stem) {
					add("PER", stem+" "+title)
					break
				}
			}
		}
	}

	// 5. Cross-domain public issues.
	for _, issue := range e.publicIssues {
		if strings.Contains(lower, issue) {
			add("ISSUE", issue)
		}
	}

	return found
}

// BuildEntityGroups groups documents by shared entity anchors and highlights news <-> review bridges.
func BuildEntityGroups(docs []ClusterDoc) (allGroups []EntityGroup, crossBridges []EntityGroup) {
	type accum struct {
		typ          string
		entity       string
		total        int
		ynat         int
		nsmc         int
		sumIntensity float64
		topicCounts  map[string]int
		nfcCounts    map[string]int
		sampleTitles []string
	}

	groups := make(map[string]*accum)
	for _, doc := range docs {
		for _, ent := range doc.Entities {
			key := ent.Type + ":" + ent.Canonical
			acc, ok := groups[key]
			if !ok {
				acc = &accum{
					typ:         ent.Type,
					entity:      ent.Canonical,
					topicCounts: make(map[string]int),
					nfcCounts:   make(map[string]int),
				}
				groups[key] = acc
			}
			acc.total++
			if doc.Record.Source == "ynat" {
				acc.ynat++
			} else {
				acc.nsmc++
			}
			acc.sumIntensity += doc.Scored.SocialOpinionIntensity
			acc.topicCounts[doc.Scored.TopicKoName]++
			acc.nfcCounts[doc.Scored.PrimaryNFC.KoName]++
			if len(acc.sampleTitles) < 4 {
				acc.sampleTitles = append(acc.sampleTitles, fmt.Sprintf("[%s] %s", doc.Record.Source, doc.Record.Text))
			}
		}
	}

	allGroups = make([]EntityGroup, 0, len(groups))
	for _, acc := range groups {
		if acc.total < 2 {
			continue
		}
		eg := EntityGroup{
			Type:          acc.typ,
			Entity:        acc.entity,
			TotalCount:    acc.total,
			YNATCount:     acc.ynat,
			NSMCCount:     acc.nsmc,
			CrossDomain:   acc.ynat > 0 && acc.nsmc > 0,
			DominantTopic: topCountKey(acc.topicCounts),
			DominantNFC:   topCountKey(acc.nfcCounts),
			MeanIntensity: acc.sumIntensity / float64(acc.total),
			SampleTitles:  acc.sampleTitles,
		}
		allGroups = append(allGroups, eg)
		if eg.CrossDomain {
			crossBridges = append(crossBridges, eg)
		}
	}

	sort.Slice(allGroups, func(i, j int) bool {
		if allGroups[i].TotalCount != allGroups[j].TotalCount {
			return allGroups[i].TotalCount > allGroups[j].TotalCount
		}
		return allGroups[i].Entity < allGroups[j].Entity
	})
	sort.Slice(crossBridges, func(i, j int) bool {
		mi := min(crossBridges[i].YNATCount, crossBridges[i].NSMCCount)
		mj := min(crossBridges[j].YNATCount, crossBridges[j].NSMCCount)
		if mi != mj {
			return mi > mj
		}
		if crossBridges[i].TotalCount != crossBridges[j].TotalCount {
			return crossBridges[i].TotalCount > crossBridges[j].TotalCount
		}
		return crossBridges[i].Entity < crossBridges[j].Entity
	})
	return allGroups, crossBridges
}

func extractDelimited(s, open, close string) []string {
	var out []string
	pos := 0
	for pos < len(s) {
		start := strings.Index(s[pos:], open)
		if start < 0 {
			break
		}
		startAbs := pos + start + len(open)
		end := strings.Index(s[startAbs:], close)
		if end < 0 {
			break
		}
		sub := strings.TrimSpace(s[startAbs : startAbs+end])
		if sub != "" {
			out = append(out, sub)
		}
		pos = startAbs + end + len(close)
	}
	return out
}

func trimPunctRunes(s string) string {
	return strings.TrimFunc(s, func(r rune) bool {
		return unicode.IsPunct(r) || unicode.IsSymbol(r)
	})
}

func isHangulWord(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < 0xAC00 || r > 0xD7A3 {
			return false
		}
	}
	return true
}

func isLikelyKoreanPersonName(s string) bool {
	runes := []rune(s)
	if len(runes) < 2 || len(runes) > 3 || !isHangulWord(s) {
		return false
	}
	last := runes[len(runes)-1]
	switch last {
	case '다', '는', '은', '을', '를', '가', '이', '에', '서', '로', '와', '과', '도', '만', '된', '한', '적', '할', '될', '인', '의', '고', '며', '면', '지', '던':
		return false
	}
	return true
}

func topCountKey(m map[string]int) string {
	bestKey := "-"
	bestCount := -1
	for k, cnt := range m {
		if cnt > bestCount || (cnt == bestCount && k < bestKey) {
			bestCount = cnt
			bestKey = k
		}
	}
	return bestKey
}
