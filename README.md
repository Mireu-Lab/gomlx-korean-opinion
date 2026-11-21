# GoMLX ReLU Multi-Task Korean Social Opinion, Nuclear Feeling & Hybrid Clustering Pipeline

**NSMC**(네이버 영화 리뷰 감성 코퍼스)와 **KLUE-YNAT**(연합뉴스 뉴스 토픽 분류) 데이터셋을 결합해, **GoMLX (`v0.28.38`) ReLU 멀티태스크 신경망**으로 기사별 주제·여론 감성 및 **핵심감정 체크리스트(Nuclear Feeling Checklist, 16개 핵심감정 & 8개 학술 요인군)**를 분류·정렬하고, 문체가 다른 뉴스와 리뷰를 **3가지 다차원 클러스터링 방식 및 3단계 검증 프레임워크**로 그룹화·검증하는 순수 Go(Zero Python) 파이프라인입니다.

> **상세 설계 매뉴얼**: 전체 아키텍처, 텐서/손실 수식, HDBSCAN/NER/계층적 주제 설계, CLI 런북 및 실측 벤치마크는 [`docs/PROJECT_DESIGN_MANUAL.md`](docs/PROJECT_DESIGN_MANUAL.md)에서 확인할 수 있습니다.

---

## 1. 아키텍처 개요

### A. GoMLX ReLU Multi-Task Network (`internal/model/relu_nlp.go`)
* **입력**: 한국어 단어 + 음절 2-gram / 3-gram 토큰 ID `[B, T]`
* **인코더**: `Embedding` → PAD 마스킹 **Mean Pooling ‖ Max Pooling** → Shared **ReLU MLP** (`hidden_dims`, Dropout) → L2 정규화 **공유 의미 임베딩(`SemanticEmbed`)**
* **3개 멀티태스크 헤드**:
  1. **KLUE-YNAT 주제 헤드 (7개 클래스)**: `IT과학`, `경제`, `사회`, `생활문화`, `세계`, `스포츠`, `정치` (`Masked Sparse Cross-Entropy`)
  2. **NSMC 여론 감성 헤드 (이진 극성)**: 긍정/부정 확률 (`Masked Binary Cross-Entropy with Logits`)
  3. **Nuclear Feeling Checklist 헤드 (16개 다중 라벨)**: 공유 표현 + 주제·감성 은닉 특성을 결합해 16개 핵심감정 예측 (`Positive-Weighted Masked Multi-Label BCE`)
* **핵심감정 체크리스트 16개 항목 (`config/nfc_taxonomy.yaml`)**:
  `부담감`, `경쟁심`, `억울함`, `열등감`, `외로움`, `그리움`, `질투`, `두려움`, `화`, `무기력`, `허무`, `슬픔`, `불안`, `공포`, `소외`, `적개심` (김경민 2019 타당화 연구의 **8개 학술 요인군** 매핑 포함)

### B. 뉴스 + 리뷰 혼합 데이터 3가지 그룹화(Clustering) & 3단계 검증 (`internal/cluster/`)
1. **방식 ① 하이브리드 임베딩 기반 HDBSCAN 밀도 클러스터링 (`hdbscan.go`)**
   * GoMLX L2 정규화 공유 잠재 임베딩 + 주제 확률 + NFC 감정 벡터 + 감성 극성을 결합한 하이브리드 코사인 공간 위에서 순수 Go **HDBSCAN**(Core Distance → Mutual Reachability Prim MST → Condensed Tree → EOM 선택)으로 군집 개수 `K` 사전 지정 없이 자동 그룹화 및 노이즈(`-1`) 필터링.
2. **방식 ② 개체명(NER)·공적 이슈 앵커 그룹화 (`entity.go`)**
   * `config/entity_rules.yaml` 기반으로 `WORK`, `ORG`, `LOC`, `PER`, `ISSUE` 엔티티를 추출하여 뉴스 기사와 영화 리뷰를 동일 엔티티 브리지로 연결.
3. **방식 ③ 계층적 주제 명명 & 선택적 LLM 보정 (`hierarchy.go`)**
   * 군집별 **c-TF-IDF 상위 10개 키워드**와 중심(Medoid) 뉴스+리뷰 대표 문서를 추출하고 `[상위: 주제] - [중위: 요인군/핵심감정] - [하위: 엔티티·핵심어]` 계층 라벨 생성(환경변수 설정 시 OpenAI 호환 LLM 정제 지원).
4. **3단계 검증 (`validate.go`)**
   * **1단계 수학적 검증**: Cosine **Silhouette Score** + **Dunn Index** (Centroid & Strict)
   * **2단계 키워드 일관성 검증**: 정보 이론 기반 **NPMI** 및 **UMass Topic Coherence**
   * **3단계 Gold Standard 인간 검증**: 군집별 층화 샘플 `outputs/human_eval_sample.csv` + `outputs/human_eval_rubric.md` 생성 및 `eval-human` CLI로 **정합성(Agreement Rate) / Cohen's Kappa** 자동 산출

### C. Zero Hardcoding & Dual Logging
* 모든 URL, 경로, 하이퍼파라미터, 손실 가중치, 정렬 기준, 토픽, 핵심감정 사전, NER 규칙, 클러스터링/검증 파라미터는 [`config/config.yaml`](config/config.yaml), [`config/topics.yaml`](config/topics.yaml), [`config/nfc_taxonomy.yaml`](config/nfc_taxonomy.yaml), [`config/entity_rules.yaml`](config/entity_rules.yaml) 및 환경변수/CLI 플래그로 제어됩니다.
* 단계별 진행 상태는 `logs/status.log`(Jenkins 스타일), 상세 로그는 `logs/processing.log`(Log4j 스타일)에 이중 기록됩니다.

---

## 2. 실행 방법

### 전체 파이프라인 실행 (`download` → `train` → `eval` → `analyze` → `cluster`)
```bash
./scripts/run_pipeline.sh pipeline
```

### 개별 단계 실행
```bash
# 데이터셋 다운로드 / 학습 / 평가
go run ./cmd/opinion-analyzer download --config config/config.yaml
go run ./cmd/opinion-analyzer train --config config/config.yaml --steps 1500
go run ./cmd/opinion-analyzer eval --config config/config.yaml

# 기사별 사회적 여론 감정·핵심감정 분류 및 정렬
go run ./cmd/opinion-analyzer analyze --config config/config.yaml --sort-by topic_nfc_intensity

# 뉴스 + 리뷰 혼합 다차원 클러스터링 및 3단계 검증
go run ./cmd/opinion-analyzer cluster --config config/config.yaml

# 전문가 인간 검증 CSV 채점 (정합성 & Cohen's Kappa)
go run ./cmd/opinion-analyzer eval-human --config config/config.yaml --input outputs/human_eval_sample.csv
```

### 지원 정렬 모드 (`--sort-by`)
* `topic_nfc_intensity`: 기사 주제(`IT과학`~`정치`) → 주 핵심감정(0~15) → 사회적 여론 강도 내림차순
* `nfc_intensity`: 주 핵심감정 → 사회적 여론 강도 내림차순 → 주제
* `intensity_desc`: 전체 사회적 여론 강도 내림차순
* `sentiment_neg_desc`: 부정 여론 확률 내림차순 → 사회적 여론 강도

---

## 3. 생성 산출물 (`outputs/`)

* `articles_sorted.json` / `articles_sorted.csv` / `social_opinion_report.md`: 기사별 주제·감성·Nuclear Feeling Checklist 분류, `7 × 16` 교차 매트릭스 및 정렬 리포트
* `clusters_report.json` / `clustering_validation_report.md`: 3가지 클러스터링 결과 및 Silhouette, Dunn Index, NPMI/UMass Coherence 검증 스코어카드
* `human_eval_sample.csv` / `human_eval_rubric.md`: 전문가 인간 검증용 층화 샘플 및 평가 가이드
