# GoMLX ReLU 기반 한국어 사회적 여론·핵심감정 분류 및 다차원 뉴스·리뷰 클러스터링 프로젝트 설계 매뉴얼

* **Project Manager**: `limmireu1214@gmail.com`
* **Implementation Owner**: `limmireu1214@gmail.com`
* **Module Path**: `github.com/Mireu-Lab/gomlx-korean-opinion`
* **Hugging Face Dataset**: [`Mireu-Lab/gomlx-korean-opinion-dataset`](https://huggingface.co/datasets/Mireu-Lab/gomlx-korean-opinion-dataset)
* **Hugging Face Model Hub**: [`Mireu-Lab/gomlx-korean-opinion`](https://huggingface.co/Mireu-Lab/gomlx-korean-opinion)
* **Google Docs 논문 초안**: [`GoMLX KO-Hub 논문 초안`](https://docs.google.com/document/d/1Lx5mZmT5xWHzr1CIL6H2AduXl7gQm07TCul_xntxR3c/edit)
* **Tech Stack**: Go `1.27` · GoMLX `v0.28.38` · `onnx-gomlx v0.5.35` · `gomlx/compute v0.1.32` · `gomlx/go-xla v0.4.32` · `parquet-go v0.32.1` · `yaml.v3 v3.0.1`
* **작성 기준일 (UTC)**: 2026-11-22 (`v0.2.0` 6종 아키텍처 확장 반영)

---

## 1. 문서 개요 및 프로젝트 설계 원칙

### 1.1 해결하고자 하는 문제
본 프로젝트는 **스타일과 어휘 분포가 상이한 두 한국어 말뭉치**를 하나의 통합 파이프라인에서 결합하여 사회적 여론을 입체적으로 분석·정렬·군집화하는 것을 목표로 합니다.

1. **기사 주제 분류 (KLUE-YNAT)**: 객관적·사실 전달 중심의 연합뉴스 헤드라인 7개 주제(`IT과학`, `경제`, `사회`, `생활문화`, `세계`, `스포츠`, `정치`) 분류
2. **여론 극성 판별 (NSMC)**: 주관적·구어체·감성 중심의 네이버 영화 리뷰 기반 긍정/부정 여론 극성 학습
3. **Nuclear Feeling Checklist 기반 핵심감정 분류 및 여론 정렬**: 김경민(2006/2019)의 **16개 핵심감정 체크리스트** 및 **8개 학술 요인군**을 약지도 다중 라벨(Multi-label)로 학습하고, 기사별 사회적 여론 강도(Social Opinion Intensity)를 정량화하여 정렬
4. **이질적 도메인(뉴스 ↔ 리뷰) 통합 클러스터링 및 3단계 검증**: 표면적 문체 차이를 넘어 의미 임베딩(`HDBSCAN`), 고유명사·이슈 앵커(`NER`), 계층적 주제(`c-TF-IDF` / 선택적 LLM)로 통합 군집화하고 수학적·일관성·인간 검증을 수행

### 1.2 핵심 엔지니어링 원칙
* **Pure Go (Zero Python)**: 데이터 다운로드, Parquet 파싱, 한국어 토크나이징, GoMLX 신경망 학습·추론, HDBSCAN 클러스터링, 평가·리포트 생성까지 전 과정을 순수 Go로 구현하여 단일 바이너리(`bin/opinion-analyzer`)로 실행합니다.
* **Zero Hardcoding**: 모든 URL, 파일 경로, 토크나이저 규격, 모델 구조, 손실 가중치, 정렬 기준, 7개 주제 정의, 16개 핵심감정 어휘 사전, NER 규칙, 클러스터링 및 검증 임계값은 `config/*.yaml`, 환경변수, CLI 플래그로만 주입합니다.
* **Dual Logging Standard**: Jenkins 스타일의 파이프라인 단계 로그(`logs/status.log`)와 Log4j 스타일의 상세 구조화 로그(`logs/processing.log`)를 동시에 기록합니다.
* **Verification First**: 모든 모듈의 단위 테스트(`go test`)와 실제 NSMC + KLUE-YNAT 데이터셋 기반 엔드투엔드 검증을 통과한 수치만을 기준으로 동작을 보장합니다.

---

## 2. 전체 시스템 아키텍처 및 데이터 흐름

### 2.1 엔드투엔드 파이프라인 흐름

```mermaid
flowchart TD
    CFG["외부 설정 로드 (config.yaml / topics.yaml / nfc_taxonomy.yaml / entity_rules.yaml)"] --> DL["1. 데이터 적재 (NSMC TSV + KLUE-YNAT Parquet)"]
    DL --> TOK["2. 한국어 단어 + 음절 2/3-gram 토크나이저 학습 및 인코딩"]
    TOK --> WEAK["3. Nuclear Feeling Checklist 16종 약지도 라벨링 + 부정 캘리브레이션"]
    WEAK --> TRAIN["4. GoMLX ReLU 멀티태스크 학습 (gobackend / OpenXLA PJRT)"]
    TRAIN --> CKPT["체크포인트 및 vocab.json 저장"]
    CKPT --> EVAL["5. 통합 평가 (YNAT Accuracy/F1, NSMC Accuracy/F1, NFC Recall@3)"]
    CKPT --> ANALYZE["6. 기사별 여론·핵심감정 분류 및 다중 모드 정렬 (Sorter)"]
    CKPT --> CLUSTER["7. 뉴스+리뷰 혼합 3방식 클러스터링 및 3단계 검증"]
    ANALYZE --> OUT1["산출물 (articles_sorted.json / articles_sorted.csv / social_opinion_report.md)"]
    CLUSTER --> OUT2["산출물 (clusters_report.json / clustering_validation_report.md / human_eval_sample.csv)"]
    CFG -.-> LOG["이중 로거 (logs/status.log + logs/processing.log)"]
```

### 2.2 OpenXLA / PJRT 백엔드 및 JAX 생태계 호환 구조
* **공유 컴파일러 기반**: JAX가 연산 그래프를 OpenXLA의 중간 표현인 **StableHLO**로 내리고 **PJRT** 런타임을 통해 CPU/GPU/TPU에서 실행하듯, GoMLX 역시 `gomlx/compute` 추상화 위에서 **OpenXLA/PJRT 백엔드(`gomlx/go-xla`)**와 순수 Go **SimpleGo 백엔드(`gobackend`, `"go"`)**를 선택적으로 구동합니다.
* **빌드 분리 설계**:
  * `internal/model/backend.go`: 기본 `"go"` 백엔드(`github.com/gomlx/compute/gobackend`) 항상 등록
  * `internal/model/backend_xla.go`: `//go:build !noxla` 태그를 통해 Linux/macOS/Windows에서 `github.com/gomlx/go-xla/compute/xla/autoinstall` 등록
* **실행 전환**: 코드 수정 없이 `config/config.yaml`의 `runtime.backend`, `GOMLX_BACKEND`/`NFC_BACKEND` 환경변수, 또는 `--backend` 플래그(`go`, `xla`, `xla:cpu`, `xla:cuda`)로 백엔드를 전환할 수 있습니다.
* **JAX 상호운용성 경로**: JAX에서 학습한 모델을 OpenXLA/PJRT 기반 Go 런타임에서 서빙해야 하는 경우, 계산 그래프는 StableHLO IR로, 가중치는 중립 텐서 포맷(Safetensors/ONNX/raw array)으로 분리 내보내기하여 연동하는 구조를 따릅니다.

---

## 3. 저장소 구조 및 패키지 모듈 설계

### 3.1 디렉터리 구조

```text
gomlx-korean-opinion/
├── cmd/
│   └── opinion-analyzer/
│       └── main.go                 # CLI 진입점 (10개 서브커맨드 + onnx_backbone_relu 분기)
├── config/
│   ├── config.yaml                 # 런타임, 경로, 데이터셋, 토크나이저(조사분리), 6종 모델/ONNX, 학습(CORAL), 정렬, 클러스터링 설정
│   ├── multilingual_roots.yaml     # KO-Hub 4축 다국어 뿌리(Unihan + OMW KO/EN/JA/ZH) 그래프·사전학습 설정
│   ├── topics.yaml                 # KLUE-YNAT 7개 주제 메타데이터 및 정렬 순서
│   ├── nfc_taxonomy.yaml           # Nuclear Feeling Checklist 16개 핵심감정, 8요인군, 어휘/동음이의어 가드 사전
│   └── entity_rules.yaml           # NER 규칙(기관/지역/인명직함/작품트리거/공적이슈) 및 불용어
├── docs/
│   ├── PROJECT_DESIGN_MANUAL.md    # 본 설계 매뉴얼
│   └── references.csv              # 원천 검증 완료 선행연구 및 언어학 참고문헌 18편
├── internal/
│   ├── config/                     # YAML 파싱, 6종 provider 및 ONNX 설정 검증, 환경변수 오버라이드
│   ├── logging/                    # Jenkins status.log + Log4j processing.log 이중 로거
│   ├── nfc/                        # Topic/NFC 분류체계 로더 및 WeakAnnotateDetailed 동음이의어 가드
│   ├── graphkg/                    # Unihan/OMW 다국어 뿌리 다운로더 및 이종 CSR 지식 그래프 빌더
│   ├── pretrain/                   # Context-Aware Soft Edge InfoNCE + 동음이의어 분리 + Student 비대칭 증류
│   ├── tokenizer/                  # 한국어 정규화, 조사 26종 경계 분리(SplitKoreanParticle), 어순 시퀀스 추출
│   ├── data/                       # HTTP 다운로더, NSMC/YNAT/커스텀 로더, 듀얼입력 Dataset, Parquet exporter(export.go)
│   ├── onnxembed/                  # 순수 Go WordPiece 토크나이저 + onnx-gomlx KLUE-RoBERTa 임포트 및 임베딩 캐시
│   ├── model/                      # 6종 GoMLX ReLU 인코더(encoders.go/relu_nlp.go/onnx_head.go), CORAL+마스킹 손실, 정적 추론
│   ├── sorter/                     # NFC 보정, 여론 강도 스코어링, 4종 정렬기, 교차 매트릭스/리포트
│   └── cluster/                    # 순수 Go Power-Iteration PCA(pca.go), HDBSCAN, NER 앵커, c-TF-IDF, 3단계 검증
├── scripts/
│   ├── run_pipeline.sh             # 단일 모델 Bash 파이프라인 실행기 (set -euo pipefail)
│   ├── benchmark_all_models.sh     # 6종 아키텍처 일괄 실데이터 벤치마크 러너
│   └── publish_hf.sh               # 전처리 Parquet 데이터셋 및 학습 모델 Hugging Face Hub 업로드 스크립트
├── go.mod
├── go.sum
└── README.md
```

### 3.2 패키지별 책임 및 주요 심볼

| 패키지 | 파일 | 핵심 심볼 및 역할 | 주요 의존 |
|---|---|---|---|
| `cmd/opinion-analyzer` | `main.go` | `runCommand`, `stageDownload`, `stageTrain`, `stageEval`, `stageAnalyze`, `stageCluster`, `stageEvalHuman` | 전체 `internal/*` |
| `internal/config` | `config.go` | `Config`, `Load`, `applyEnvOverrides`, `Validate`, `ValidSortModes` | `yaml.v3` |
| `internal/logging` | `logger.go` | `DualLogger`, `NewDualLogger`, `Status`, `Debug`/`Info`/`Warn`/`Error` | 표준 라이브러리 |
| `internal/nfc` | `taxonomy.go` | `Taxonomy`, `TopicDef`, `FeelingDef`, `LoadTaxonomy`, `WeakAnnotate`, `SortedTopicsByDisplayOrder` | `yaml.v3` |
| `internal/tokenizer` | `tokenizer.go` | `Tokenizer`, `Normalize`, `ExtractTokens`, `Fit`, `Encode`, `Save`/`Load` | `internal/config` |
| `internal/data` | `downloader.go`, `loader.go`, `dataset.go` | `EnsureDatasets`, `LoadNSMC`, `LoadYNAT`, `LoadCustomArticles`, `PrepareExamples`, `MultiTaskTrainDataset` | `parquet-go`, `gomlx/ml/train` |
| `internal/model` | `relu_nlp.go`, `loss.go`, `train.go`, `infer.go` | `BuildGraphWithEmbedding`, `BuildLogits`, `BuildMultiTaskLoss`, `Train`, `Evaluate`, `Predictor`, `LoadPredictor` | `gomlx`, `compute` |
| `internal/sorter` | `sorter.go` | `ScoreArticles`, `SortArticles`, `BuildReport`, `WriteOutputs` | `internal/nfc`, `internal/data` |
| `internal/cluster` | `hdbscan.go`, `entity.go`, `hierarchy.go`, `validate.go`, `report.go` | `BuildHybridVectors`, `RunHDBSCAN`, `EntityExtractor`, `BuildEntityGroups`, `ComputeClusterTopKeywords`, `ComputeSilhouette`, `ComputeDunnIndex`, `ComputeTopicCoherence`, `ExportHumanEvalSample`, `ScoreHumanEvalCSV`, `RunClusteringPipeline` | `internal/sorter`, `internal/tokenizer`, `internal/nfc` |

---

## 4. 데이터 파이프라인 및 한국어 토크나이저 설계

### 4.1 입력 데이터셋 스키마 및 파서
1. **NSMC (`internal/data/loader.go` -> `LoadNSMC`)**
   * 포맷: UTF-8 TSV (`id \t document \t label`)
   * `label`: `0`(부정), `1`(긍정). 공백 문서 및 손상 행은 스킵 카운트로 추적합니다 (`TopicID = -1`, `HasTopic = false`, `HasSentiment = true`).
2. **KLUE-YNAT (`internal/data/loader.go` -> `LoadYNAT`)**
   * 포맷: Hugging Face Parquet (`guid`, `title`, `label`, `url`, `date`)
   * 순수 Go `parquet.ReadFile[ynatParquetRow]`로 직접 디코딩합니다 (`HasTopic = true`, `HasSentiment = false`).
3. **외부 기사 입력 (`LoadCustomArticles`)**
   * `.jsonl`/`.ndjson`, `.csv`, `.tsv` 지원 (`id`/`guid`, `title`/`headline`, `text`/`content`/`document`/`body`, `date`/`published_at`, `url`/`link` 필드 자동 매핑).

### 4.2 단어 우선 + 음절 n-gram 한국어 토크나이저
* **정규화 (`tokenizer.Normalize`)**: ASCII 소문자화, 한글 음절(`0xAC00..0xD7A3`)·자모·영숫자를 제외한 문장부호/제어문자 공백 치환.
* **토큰 추출 순서 (`tokenizer.ExtractTokens`)**:
  1. `"w:<어절>"` 전체 어절 토큰을 먼저 추출
  2. `use_char_bigrams: true` 시 `"2:<2음절>"` 추출
  3. `use_char_trigrams: true` 시 `"3:<3음절>"` 추출
  * **설계 의도**: 고정 길이(`max_seq_len: 48`) 절단 시 앞부분 n-gram 때문에 문장 후반 어절이 잘리는 현상을 방지하고, 헤드라인과 긴 영화 리뷰 모두 문맥 전체를 우선 보존하면서 형태소 분석기 없이도 한국어 복합어/활용형 일반화를 확보합니다.
* **특수 ID**: `<PAD> = 0`, `<UNK> = 1`, 빈도 순 상위 `vocab_size` (`20,000`)까지 `vocab.json`에 영속화합니다.

### 4.3 약지도 라벨링 및 균형 멀티태스크 배칭
* **약지도 + 부정 캘리브레이션 (`PrepareExamples`)**:
  * `WeakAnnotate(text)` 결과 1개 이상 핵심감정이 매칭되면 `NFCMask = 1.0`.
  * 매칭이 0건일 때에도 긍정 리뷰(`HasSentiment && SentimentLabel >= 0.5`)는 `nfc_negative_calibration_prob (0.35)` 확률로, 중립/비매칭 뉴스는 `0.15 * 0.35` 확률로 `NFCMask = 1.0` + 영벡터(`0`) 타깃을 부여합니다. 이는 모델이 모든 입력에 높은 핵심감정 점수를 출력하는 과잉 활성(Over-triggering)을 방지합니다.
* **균형 미니배치 (`MultiTaskTrainDataset`)**:
  * 매 배치(`B = 64`)를 `ynat_batch_ratio (0.45)`와 `nsmc_batch_ratio (0.55)` 비율로 조합하고, 각 도메인 슬롯 내부의 `nfc_positive_sample_ratio (0.50)`를 NFC 활성 샘플에서 추출한 뒤 행 순서를 셔플합니다. 이를 통해 16개 희소 핵심감정과 두 도메인 간 그래디언트 균형을 유지합니다.

---

## 5. GoMLX ReLU 멀티태스크 신경망 및 손실 함수 설계

### 5.1 신경망 계층 구조 및 텐서 형상

```mermaid
flowchart LR
    IN["tokens int32 (B, 48)"] --> MASK["padMask (tokens != 0)"]
    IN --> EMB["Embedding (B, 48, 64)"]
    MASK --> MEAN["Masked MeanPool (B, 64)"]
    MASK --> MAXP["Masked MaxPool -1e9 (B, 64)"]
    EMB --> MEAN
    EMB --> MAXP
    MEAN --> CAT1["Concatenate (B, 128)"]
    MAXP --> CAT1
    CAT1 --> TR0["Dense 128 + ReLU + Dropout 0.2"]
    TR0 --> TR1["Dense 64 + ReLU + Dropout 0.2"]
    TR1 --> L2["L2Norm SemanticEmbed (B, 64)"]
    TR1 --> TH["Topic Dense 64 + ReLU"]
    TR1 --> SH["Sentiment Dense 64 + ReLU"]
    TH --> TOUT["Topic Logits (B, 7)"]
    SH --> SOUT["Sentiment Logit (B, 1)"]
    TR1 --> CAT2["Concatenate (B, 192)"]
    TH --> CAT2
    SH --> CAT2
    CAT2 --> NH["NFC Dense 64 + ReLU + Dropout 0.2"]
    NH --> NOUT["NFC Logits (B, 16)"]
```

* **핵심 구현 포인트 (`internal/model/relu_nlp.go`)**:
  * **Zero-Division / All-PAD 방어**: `safeLengths = Max(lengths, 1.0)`으로 평균 풀링 나눗셈 안전성을 보장하고, 최대 풀링은 PAD 위치에 `-1e9`를 채운 뒤 `lengths == 0`인 빈 입력 행은 0으로 마스킹합니다.
  * **조건부 핵심감정 헤드**: NFC 은닉층이 공유 표현 `h` 뿐 아니라 주제 은닉 `topicH`와 감성 은닉 `sentH`를 함께 결합(`Concatenate [B, 192]`)하여, 동일 단어라도 뉴스 주제와 부정/긍정 맥락에 따라 핵심감정을 조건부로 추론합니다.
  * **Scope 정합성**: 학습(`train.NewTrainer`)과 추론(`model.NewExec`) 모두 `scope.In("relu_nlp")` 하위 경로를 공유하여 체크포인트 저장·복원 시 변수 경로가 100% 일치합니다. 추론기(`Predictor`)는 `scope.Store().SetTraining(g, false)`를 명시해 드롭아웃을 비활성화합니다.
  * **정적 배치 패딩**: `PredictExamples`는 마지막 잔여 배치를 `eval_batch_size (128)` 크기로 패딩해 단일 그래프만 컴파일하고 결과 슬라이스만 잘라 반환합니다.

### 5.2 NaN-Safe 마스킹 멀티태스크 손실 함수 (`internal/model/loss.go`)
하나의 배치에 YNAT(토픽만 보유)과 NSMC(감성만 보유)가 섞여 있으므로 각 태스크는 전용 마스크(`topicMask`, `sentMask`, `nfcMask`)를 통해 해당 샘플에 대해서만 평균 손실을 계산합니다. 분모는 항상 `max(sum(mask), mask_epsilon)`으로 클램핑합니다.

1. **주제 손실 (Masked Sparse Categorical Cross-Entropy)**:
   $$\mathcal{L}_{\text{topic}} = \frac{\sum_{b=1}^{B} m^{\text{topic}}_b \cdot \left(-\sum_{k=0}^{6} \mathbf{1}[y^{\text{topic}}_b = k] \log \text{Softmax}(z^{\text{topic}}_b)_k\right)}{\max\left(\sum_{b=1}^{B} m^{\text{topic}}_b,\, \varepsilon\right)}$$
2. **감성 손실 (Masked Numerically-Stable BCE with Logits)**:
   $$\ell^{\text{sent}}_b = \max(z_b, 0) - z_b y_b + \log(1 + e^{-|z_b|}), \quad \mathcal{L}_{\text{sent}} = \frac{\sum_{b} m^{\text{sent}}_b \ell^{\text{sent}}_b}{\max(\sum_{b} m^{\text{sent}}_b, \varepsilon)}$$
3. **핵심감정 손실 (Positive-Weighted Masked Multi-Label BCE)**:
   * 다중 라벨 희소성(16개 중 평균 1~2개 활성)을 보정하기 위해 양성 가중치 `nfc_pos_weight = 6.0`을 적용합니다.
   $$\ell^{\text{nfc}}_{b,c} = w_{\text{pos}} y_{b,c}\left(\max(-z_{b,c}, 0) + \log(1+e^{-|z_{b,c}|})\right) + (1-y_{b,c})\left(\max(z_{b,c}, 0) + \log(1+e^{-|z_{b,c}|})\right)$$
   $$\mathcal{L}_{\text{nfc}} = \frac{\sum_{b=1}^{B} m^{\text{nfc}}_b \left(\frac{1}{16}\sum_{c=0}^{15} \ell^{\text{nfc}}_{b,c}\right)}{\max(\sum_{b=1}^{B} m^{\text{nfc}}_b, \varepsilon)}$$
4. **총 손실**:
   $$\mathcal{L}_{\text{total}} = w_{\text{topic}}\mathcal{L}_{\text{topic}} + w_{\text{sent}}\mathcal{L}_{\text{sent}} + w_{\text{nfc}}\mathcal{L}_{\text{nfc}} \quad (w_{\text{topic}}=1.0,\; w_{\text{sent}}=1.0,\; w_{\text{nfc}}=1.5)$$

---

## 6. Nuclear Feeling Checklist 분류체계 및 기사 여론 정렬 알고리즘

### 6.1 16개 핵심감정 & 8개 학술 요인군 참조표 (`config/nfc_taxonomy.yaml`)

| ID | 코드 | 한국어명 | 영문명 | 학술 요인군 (`factor_group`) | 사회 여론 가중치 (`social_weight`) | 극성 친화도 |
|---:|---|---|---|---|---:|---|
| 0 | `burden` | 부담감 | Burden | 무기력/압박 | 0.95 | `negative` |
| 1 | `competitiveness` | 경쟁심 | Competitiveness | 경쟁심 | 0.85 | `competitive` |
| 2 | `unfairness` | 억울함 | Unfairness | 적개심/억울 | 1.15 | `negative` |
| 3 | `inferiority` | 열등감 | Inferiority | 소외/위축 | 0.85 | `negative` |
| 4 | `loneliness` | 외로움 | Loneliness | 소외 | 0.90 | `negative` |
| 5 | `longing` | 그리움 | Longing | 화/친밀감 | 0.75 | `mixed` |
| 6 | `jealousy` | 질투 | Jealousy | 질투/경쟁 | 0.85 | `negative` |
| 7 | `fear` | 두려움 | Fear | 공포 | 1.05 | `negative` |
| 8 | `anger` | 화 | Anger | 화/친밀감 | 1.20 | `negative` |
| 9 | `helplessness` | 무기력 | Helplessness | 무기력 | 1.00 | `negative` |
| 10 | `emptiness` | 허무 | Emptiness | 무기력 | 0.90 | `negative` |
| 11 | `sadness` | 슬픔 | Sadness | 소외/슬픔 | 1.05 | `negative` |
| 12 | `anxiety` | 불안 | Anxiety | 불안 | 1.10 | `negative` |
| 13 | `terror` | 공포 | Terror | 공포 | 1.15 | `negative` |
| 14 | `alienation` | 소외 | Alienation | 소외 | 1.00 | `negative` |
| 15 | `hostility` | 적개심 | Hostility | 적개심 | 1.20 | `negative` |

### 6.2 보정 점수, 여론 강도 스코어링 및 정렬 모드 (`internal/sorter/sorter.go`)
* **보정 핵심감정 점수 (Calibrated Score)**:
  $$s_c = \frac{w_{\text{neural}} \cdot p^{\text{neural}}_c + w_{\text{lexical}} \cdot s^{\text{lex}}_c}{w_{\text{neural}} + w_{\text{lexical}}} \quad (w_{\text{neural}}=0.85,\; w_{\text{lexical}}=0.15)$$
  여기서 어휘 점수 $s^{\text{lex}}_c = \min(1.0,\, \text{hits}_c / \text{lexical\_saturation})$ ($\text{lexical\_saturation}=2.0$)입니다.
* **감성 판정 라벨**: 긍정 확률 $p_{\text{pos}} \ge 0.60 \rightarrow \text{긍정}$, $p_{\text{pos}} \le 0.40 \rightarrow \text{부정}$, 그 외 $\rightarrow \text{중립}$.
* **사회적 여론 강도 (Social Opinion Intensity)**:
  가중 감정 점수 $ws_c = s_c \cdot \text{social\_weight}_c$를 내림차순 정렬한 뒤 다음과 같이 산출합니다.
  $$I = \frac{0.55 \cdot \max_c(ws_c) + 0.25 \cdot \text{mean}(\text{top3}(ws)) + 0.20 \cdot (1 - p_{\text{pos}})}{0.55 + 0.25 + 0.20}$$
* **지원 정렬 모드 (모든 모드는 최종 동률 시 `ID` 오름차순으로 결정적 정렬)**:
  1. `topic_nfc_intensity`: `Topic.DisplayOrder` 오름차순 → `PrimaryNFC.ID` 오름차순 → `SocialOpinionIntensity` 내림차순
  2. `nfc_intensity`: `PrimaryNFC.ID` 오름차순 → `SocialOpinionIntensity` 내림차순 → `Topic.DisplayOrder` 오름차순
  3. `intensity_desc`: `SocialOpinionIntensity` 내림차순
  4. `sentiment_neg_desc`: `SentimentNegativeProb` 내림차순 → `SocialOpinionIntensity` 내림차순

---

## 7. 뉴스+리뷰 혼합 3가지 다차원 클러스터링 및 3단계 검증 설계

### 7.1 방식 ①: 하이브리드 임베딩 + 순수 Go HDBSCAN (`internal/cluster/hdbscan.go`)
* **하이브리드 벡터 구성 (`BuildHybridVectors`)**:
  문체 차이를 상쇄하면서 주제·정서 맥락을 보존하기 위해 4개 L2 정규화 서브벡터를 가중 결합한 뒤 재정규화합니다.
  $$\mathbf{v}_{\text{hybrid}} = \text{L2Norm}\Big(\sqrt{0.40}\,\hat{\mathbf{e}}_{\text{latent}} \;\parallel\; \sqrt{0.30}\,\hat{\mathbf{p}}_{\text{topic}} \;\parallel\; \sqrt{0.20}\,\hat{\mathbf{s}}_{\text{nfc}} \;\parallel\; \sqrt{0.10}\,\hat{\mathbf{p}}_{\text{sent}}\Big)$$
* **순수 Go HDBSCAN 6단계 알고리즘 (`RunHDBSCAN`)**:
  1. 대칭 코사인 거리 행렬 $d(i,j) = \text{clamp}(1 - \mathbf{v}_i \cdot \mathbf{v}_j,\, 0,\, 2)$ 계산
  2. 각 점의 $k$-최근접 이웃(`min_samples = 8`, 자기 자신 포함 7번째 타 점) 거리로 `core_dist[i]` 산출
  3. 상호 도달 가능 거리 $\text{mrd}(i,j) = \max(\text{core\_dist}[i], \text{core\_dist}[j], d(i,j))$ 기반 $\mathcal{O}(N^2)$ Prim 최소 신장 트리(MST) 구축
  4. 간선 오름차순 Union-Find로 이진 단일연결 덴드로그램(`2N - 1` 노드) 구성
  5. `min_cluster_size = 15` 기준으로 하향식 축약 트리(Condensed Tree) 생성 및 밀도 $\lambda = 1 / \max(\text{mrd}, 10^{-12})$ 기반 안정성(Stability) 계산
  6. 상향식 **Excess of Mass (EOM)** 선택으로 최적 군집을 추출하고, 어떤 선택 군집에도 속하지 못한 이상치/스팸/고립 문서는 `-1`(노이즈)로 자동 필터링

### 7.2 방식 ②: 개체명(NER)·공적 이슈 앵커 그룹화 (`internal/cluster/entity.go`)
`config/entity_rules.yaml`을 로드하여 5종의 구조적 앵커를 추출하고, `ynat`와 `nsmc`가 동시에 1건 이상 포함된 앵커를 **교차 도메인 브리지(Cross-Domain Bridge)**로 식별합니다.
* `WORK`: `《...》`, `「...」`, `『...』`, `<...>` 및 작품 트리거 동반 인용문
* `ORG`: 주요 기업/기관/정당/영화사 목록 + 한국어 기관 접미사(`전자`, `은행`, `위원회`, `연구원`, `엔터` 등)
* `LOC`: 국내외 국가·도시·지역명
* `PER`: `2~3`음절 순수 한글 토큰 + 직함(`감독`, `대통령`, `배우`, `총재`, `의원` 등), 동사·조사 어미(`다/는/은/을/된/한` 등) 오탐 방지 필터 적용
* `ISSUE`: `물가`, `금리`, `AI`, `노동`, `전쟁`, `범죄`, `흥행`, `박스오피스` 등 공적·사회 이슈 키워드

### 7.3 방식 ③: c-TF-IDF 계층적 주제 분류 및 선택적 LLM 정제 (`internal/cluster/hierarchy.go`)
* **Top-10 c-TF-IDF 키워드**:
  $$\text{c-TF-IDF}(w, c) = \text{tf}(w, c) \cdot \log\left(1 + \frac{\bar{A}}{\max(1, f_{\text{global}}(w))}\right)$$
  단, 단일 문서 오탈자·고유 토큰이 대표 키워드를 점유하지 않도록 군집 내 문서 출현 빈도 `df >= 2` 키워드를 우선 정렬합니다.
* **균형 대표 문서(Medoids)**: 군집 센트로이드와의 코사인 거리가 가장 가까운 `ynat`와 `nsmc` 문서를 교차(Interleave) 추출하여 `representative_docs_per_cluster (6)`건 제시합니다.
* **3계층 라벨 생성**: `[상위: 최다 Topic] - [중위: 최다 NFC 요인군/핵심감정] - [하위: 최다 엔티티·대표 키워드]`.
* **선택적 LLM 정제 (`MaybeRefineLabelsWithLLM`)**: `clustering.llm_labeling.enabled: true`이고 환경변수(`NFC_LLM_ENDPOINT`, `NFC_LLM_API_KEY`, `NFC_LLM_MODEL`)가 모두 존재할 때만 외부 OpenAI 호환 엔드포인트로 라벨을 정제합니다.

### 7.4 3단계 클러스터링 검증 프레임워크 (`internal/cluster/validate.go`)

| 검증 단계 | 지표 / 산출물 | 계산 방식 및 목적 | 목표 기준 |
|---|---|---|---|
| **1단계: 수학적 공간 검증** | Cosine **Silhouette Score** | 군집 내 응집도 $a(i)$와 최근접 타 군집 분리도 $b(i)$ 비교: $(b-a)/\max(a,b)$ | `>= 0.50` |
| **1단계: 수학적 공간 검증** | **Dunn Index** (`Centroid` / `Strict`) | 군집 간 최소 거리 대비 군집 내 최대/평균 직경 비율 | 값이 클수록 분리도 우수 |
| **2단계: 키워드 일관성** | **NPMI** & **UMass** Topic Coherence | 상위 10 키워드 쌍의 말뭉치 내 동시 출현 확률 기반 정보이론 응집도 | NPMI 높고 UMass가 0에 가까울수록 우수 |
| **3단계: 전문가 인간 검증** | `human_eval_sample.csv` + `eval-human` | 군집/노이즈 층화 추출(`150`건) 후 인간 평가 정합성 및 **Cohen's Kappa** 산출 | 정합성 `>= 80%` |

---

## 8. 외부 설정 사전, CLI 및 운영 런북

### 8.1 설정 우선순위 및 주요 환경변수
* **우선순위**: `CLI Flags` > `Environment Variables` > `config/config.yaml`
* **지원 환경변수**: `GOMLX_BACKEND`, `NFC_BACKEND`, `NFC_DATA_DIR`, `NFC_CHECKPOINT_DIR`, `NFC_OUTPUT_DIR`, `NFC_LOG_DIR`, `NFC_TOPICS_FILE`, `NFC_TAXONOMY_FILE`, `NFC_ENTITY_RULES_FILE`, `NFC_SORT_MODE`, `NFC_TRAIN_STEPS`, `NFC_BATCH_SIZE`, `NFC_CLUSTER_MIN_SIZE`, `NFC_CLUSTER_MIN_SAMPLES`, `NFC_CLUSTER_YNAT_SAMPLES`, `NFC_CLUSTER_NSMC_SAMPLES`, `NFC_HUMAN_EVAL_SAMPLES`.

### 8.2 CLI 서브커맨드 및 실행 방법

```bash
# 1. 전체 파이프라인 (download -> train -> eval -> analyze -> cluster)
./scripts/run_pipeline.sh pipeline

# 2. 개별 단계 실행
./bin/opinion-analyzer download --config config/config.yaml
./bin/opinion-analyzer train    --config config/config.yaml --steps 1500 --batch-size 64
./bin/opinion-analyzer eval     --config config/config.yaml
./bin/opinion-analyzer analyze  --config config/config.yaml --sort-by topic_nfc_intensity
./bin/opinion-analyzer cluster  --config config/config.yaml

# 3. 커스텀 기사(.jsonl / .csv / .tsv) 분석 및 정렬
./bin/opinion-analyzer analyze --config config/config.yaml --input my_articles.jsonl --sort-by intensity_desc

# 4. 전문가 인간 검증 CSV 채점
./bin/opinion-analyzer eval-human --config config/config.yaml --input outputs/human_eval_sample.csv
```

### 8.3 이중 로깅 포맷 (`internal/logging/logger.go`)
* `logs/status.log`: `[RFC3339] [STAGE] [STATUS] message` (예: `[TRAIN] [SUCCESS]`, `[CLUSTER] [SUCCESS]`)
* `logs/processing.log`: `[RFC3339] [LEVEL] [component] message` (`DEBUG` / `INFO` / `WARN` / `ERROR`)

---

## 9. 아키텍처 의사결정 기록(ADR), 실측 벤치마크 및 확장 로드맵

### 9.1 주요 아키텍처 의사결정 (ADR)
1. **ADR-001 (순수 Go + GoMLX `gobackend` 기본 / OpenXLA `xla:cpu` 가속 구조)**: C/Python 런타임 의존 없이 단일 바이너리 이식성을 확보하면서도 `--backend xla:cpu` 전환 시 PJRT JIT 컴파일로 `32.6x`(CNN 기준 `554ms -> 17ms`) 가속을 확보.
2. **ADR-002 (한국어 조사 경계 분리 + 어순 시퀀스 이중 인코딩)**: `SplitKoreanParticle`(`min_stem_runes=2`, 조사 26종)으로 `"단색화가"`를 `w:단색화 p:가`로 분리해 교차 경계 음절 2-gram(`2:화가`) 오탐을 차단하고, Bag-of-Tokens `[B, 48]`과 어순 Word Sequence `[B, 20]`을 동시 제공.
3. **ADR-003 (KO-Hub 4축 다국어 뿌리 그래프 + Context-Aware Soft Edge 비대칭 증류)**: `Unihan`(한자) + `OMW/WordNet`(KO/EN/JA/ZH) + 말뭉치로 구축한 `74,578`노드/`374,577`엣지 CSR 그래프를 학습 시에만 문맥 어텐션(`Soft Edge`)과 InfoNCE + 동음이의어 분리 마진으로 증류하고, 추론 시에는 한국어 단독 Student 인코더만 실행.
4. **ADR-004 (동음이의어 가드 Hard-Negative 캘리브레이션 + CORAL 사용역 정렬)**: `WeakAnnotateDetailed`로 가드 발동 문서를 명시적 부정 감독(`nfc_supervised_mask=1.0`)으로 학습하고, YNAT(뉴스)↔NSMC(리뷰) 임베딩 1차/2차 통계량을 맞추는 Deep CORAL 손실(`domain_align=0.15`)을 결합.
5. **ADR-005 (6종 멀티아키텍처 비교 + 순수 Go PCA + HDBSCAN)**: `scratch_relu`, `ko_hub_graph`, `ko_hub_cnn_relu`, `ko_hub_transformer_relu`(RoPE+RMSNorm), `ko_hub_lstm_relu`(Ragged BiLSTM), `onnx_backbone_relu`(KLUE-RoBERTa ONNX via `onnx-gomlx`)를 동일 조건으로 비교하고, 128d 임베딩을 순수 Go Power-Iteration PCA(`24d`)로 축소해 HDBSCAN 거리 집중을 완화.
6. **ADR-006 (Parquet 전처리 익스포트 + Hugging Face Hub 아카이빙)**: `export-dataset` 커맨드로 `254,777`행 전처리 Parquet 샤드(어순 토큰·조사분리·가드 플래그 포함)를 생성하고, `scripts/publish_hf.sh`로 HF Dataset/Model Hub에 재현 배포.

### 9.2 실제 검증 완료 6종 아키텍처 벤치마크 (`xla:cpu` 및 `v0.1.1` 대비)
2 vCPU / 3.8 GiB RAM Linux amd64 환경에서 실제 데이터(`NSMC Train 149,995 / Test 49,997`, `KLUE-YNAT Train 45,678 / Val 9,107`)를 대상으로 검증한 실측 결과입니다. 참고 문헌 및 언어학 근거 18편은 [`docs/references.csv`](references.csv)에 수록되어 있습니다.

> **평가 범위 주석**: 네이티브 GoMLX 5종은 **전체 평가셋**(`YNAT n=9,107`, `NSMC n=49,997`, `NFC supervised=26,461 / active=19,650`)을 평가했습니다. `onnx_backbone_relu`(`leewaay/klue-roberta-base-klueNLI-klueSTS-MSL128` 768d 동결 임베딩)는 CPU 인코딩 시간(`~15m 46s`)상 도메인당 `max_train=2,000` / `max_eval=1,000`(`NFC supervised=781 / active=612`) 서브샘플 조건에서 학습·평가했으며, HDBSCAN 군집화는 전 모델 공통 `YNAT 800 + NSMC 400 = 1,200`건입니다.

| 모델 (`embedding.provider`) | Steps / 스텝속도 / 학습시간 | PCA (`128→24d`) | YNAT Acc / Macro-F1 | NSMC Acc / F1 | NFC Recall@3 / Micro-F1 / 활성 | HDBSCAN 군집 (교차 / 노이즈) | Silhouette / Dunn_C / Mean NPMI |
|---|---:|---:|---:|---:|---:|---:|---:|
| *(`v0.1.1` 구형 `scratch_relu`, `go`)* | *1,500 / 36ms / 1m01s* | *No* | *71.14% / 0.7100* | *82.53% / 0.8309* | *89.03% / 0.4529 / 15/16 (`외로움 0`)* | *8 (7 / 47.3%)* | *0.7877 / 4.5249 / -0.3219* |
| **`scratch_relu`** *(조사분리+가드+CORAL, `xla:cpu`)* | 2,500 / `4ms` / `15s` | 0 | 74.76% / 0.7462 | 83.39% / 0.8355 | 93.81% / 0.5438 / **16/16** (`외167·소62`) | 8 (7 / 46.5%) | 0.6786 / 2.9383 / -0.3167 |
| **`ko_hub_graph`** *(`xla:cpu`)* | 2,500 / `5ms` / `17s` | 24d (91.74%) | 74.82% / 0.7474 | 83.52% / 0.8346 | **95.04% / 0.6487 / 16/16** (`외336·소310`) | 6 (3 / 49.2%) | 0.4928 / 2.4043 / -0.3411 |
| **`ko_hub_cnn_relu`** *(`xla:cpu`)* | 2,500 / `17ms` / `1m52s` | 24d (91.14%) | **75.91% / 0.7575** | 83.58% / **0.8359** | 94.67% / 0.6311 / **16/16** (`외94·소189`) | 7 (4 / 52.8%) | 0.6776 / 4.0664 / -0.3297 |
| **`ko_hub_transformer_relu`** *(2L RoPE, `xla:cpu`)* | 2,500 / `180ms` / `6m14s` | 24d (93.83%) | 74.67% / 0.7461 | **83.60% / 0.8359** | 93.78% / 0.5711 / **16/16** (`외16·소111`) | **7 (7 / 62.5%)** | **0.6938 / 4.1647** / -0.3155 |
| **`ko_hub_lstm_relu`** *(Ragged BiLSTM, `xla:cpu`)* | 1,500 / `237ms` / `6m55s` | 24d (92.42%) | 72.66% / 0.7272 | 82.84% / 0.8315 | 92.39% / 0.5316 / **16/16** (`외29·소8`) | 8 (5 / 51.9%) | 0.6586 / 2.9131 / -0.3164 |
| **`onnx_backbone_relu`** *(RoBERTa 768d 동결, `xla:cpu`)* | 1,500 / `3ms` / `23s` (+enc `15m46s`) | 24d (84.58%) | 67.00% / 0.6738 *(n=1k)* | 80.70% / 0.8153 *(n=1k)* | 73.53% / 0.4297 / **16/16** *(n=612, 외8·소37)* | 7 (7 / 54.3%) | 0.5357 / 2.1630 / **-0.2861** |

* **동음이의어 회귀 검증 (`단색화가`, `여류화가협회`)**:
  * `"남춘모 부조적 단색화가 펼쳐 보이는 자연"`: `v0.1.1`의 `화(분노)` 오탐이 **전 6종 모델에서 완전히 해소**되었으며 `ko_hub_cnn_relu`·`ko_hub_transformer_relu`·`ko_hub_lstm_relu`·`onnx_backbone_relu` 모두 주제 **`생활문화`**를 정확히 복원.
  * `"차병원 한국여류화가협회와 소녀가장 돕기 자선전시회 개최"`: `scratch_relu` 단독에서는 `화:0.659` 오탐이 잔존했으나 `ko_hub_graph`·`ko_hub_cnn_relu`·`ko_hub_transformer_relu`에서 주 감정 `화`가 제거되었고, `onnx_backbone_relu`는 `사회(0.9826) / 긍정(0.9995) / nfc_activated=false`로 가장 정확히 판별.
* **단위/통합 테스트**: `go test -count=1 ./...` 기준 **11/11 패키지 PASS** (`cluster`, `config`, `data`, `graphkg`, `logging`, `model`, `nfc`, `onnxembed`, `pretrain`, `sorter`, `tokenizer`), 교차 도메인 NER 브리지 **19개**(ONNX 1k 서브샘플 기준 **15개**) 및 인간 검증 층화 샘플 **150건** 생성 확인.

### 9.3 향후 개선 및 운영 확장 로드맵
1. **형태소 용언 활용 확장**: 체언 조사 분리(`SplitKoreanParticle`)에 이어 용언 어간/어미 및 불규칙 음운 변동(`외롭다 → 외로워`) 분해를 추가해 형태론적 일반화 강화.
2. **전문가 Gold 평가셋(300~500건) 구축 및 `eval-human`**: 약지도 키워드 평가를 넘어 인간 합의(`Cohen's Kappa`) 기반의 정량 평가셋 확충 및 반어/경험자 분리.
3. **클라우드 Spot GPU 배치 & GitFlow v0.2.0 릴리즈**: `gcp-calculator-mcp` 비용 산정 기반 GCP Spot VM 장기 학습 및 `feature -> develop -> main (v0.2.0)` 병합.
