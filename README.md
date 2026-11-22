# GoMLX Korean Social Opinion, KO-Hub Multilingual Roots & Hybrid Clustering Pipeline

**NSMC**(네이버 영화 리뷰 감성 코퍼스)와 **KLUE-YNAT**(연합뉴스 뉴스 토픽 분류) 데이터셋을 결합해, **GoMLX (`v0.28.38`) + OpenXLA (`xla:cpu`) + `onnx-gomlx` (`v0.5.35`)** 기반 **6종 신경망 아키텍처**로 기사 주제·여론 감성·**핵심감정 체크리스트(Nuclear Feeling Checklist, 16종 & 8요인군)**를 분류·정량화·정렬하고, 문체가 다른 뉴스와 리뷰를 **의미(HDBSCAN) · 엔티티(NER) · 계층 토픽(`c-TF-IDF`) 3방식 & 3단계 검증**으로 그룹화하는 **순수 Go(Zero Python)** 파이프라인입니다.

* **상세 설계 매뉴얼**: [`docs/PROJECT_DESIGN_MANUAL.md`](docs/PROJECT_DESIGN_MANUAL.md)
* **검증된 선행연구·언어학 참고문헌 CSV (18편)**: [`docs/references.csv`](docs/references.csv)
* **Hugging Face 전처리 데이터셋 (254,777 rows, Parquet)**: [`Mireu-Lab/gomlx-korean-opinion-dataset`](https://huggingface.co/datasets/Mireu-Lab/gomlx-korean-opinion-dataset)
* **Hugging Face 학습 모델 허브 (6종 모델 + Pretrain)**: [`Mireu-Lab/gomlx-korean-opinion`](https://huggingface.co/Mireu-Lab/gomlx-korean-opinion)
* **Google Docs 논문 초안**: [`GoMLX KO-Hub 논문 초안`](https://docs.google.com/document/d/1Lx5mZmT5xWHzr1CIL6H2AduXl7gQm07TCul_xntxR3c/edit)

---

## 1. 아키텍처 개요

### A. 한국어 중심 허브(KO-Hub) & 선택 가능한 6종 GoMLX 인코더 (`internal/graphkg`, `internal/pretrain`, `internal/model`)
* **한국어 형태소·가드 전처리**: 26종 한국어 조사 경계 분리(`SplitKoreanParticle`, `단색화+가` → `w:단색화 p:가`)로 교차 경계 2-gram(`2:화가`) 오탐을 방지하고, `WeakAnnotateDetailed` 동음이의어 가드로 Hard-Negative 캘리브레이션(`4,694`건)을 수행합니다.
* **KO-Hub 4축 다국어 뿌리 그래프 + 비대칭 증류**: Unicode `Unihan`(한자) + `OMW/WordNet`(KO/EN/JA/ZH) + 말뭉치 기반 CSR 지식 그래프(`74,578` 노드 / `374,577` 엣지) 위에서 Context-Aware Soft Edge InfoNCE 대조 사전학습(`600` steps)을 수행하고, 추론 시에는 한국어 단독 Student만 실행합니다.
* **6종 아키텍처 (`--provider`) & `CORAL` 사용역 정렬**:
  1. `scratch_relu`: 조사 분리 어휘+음절 n-gram + Mean/Max Pooling + ReLU MLP
  2. `ko_hub_graph`: KO-Hub Student 전이 + Residual ReLU 어댑터
  3. `ko_hub_cnn_relu`: KO-Hub + 어순 Multi-Scale Conv1D (`k=[2,3,4]`) + AttentionPool + ReLU
  4. `ko_hub_transformer_relu`: KO-Hub + 어순 2-Layer RoPE Multi-Head Self-Attention + RMSNorm + ReLU FFN + AttentionPool
  5. `ko_hub_lstm_relu`: KO-Hub + 어순 Ragged Bidirectional LSTM + AttentionPool + ReLU
  6. `onnx_backbone_relu`: `onnx-gomlx` 기반 [`leewaay/klue-roberta-base-klueNLI-klueSTS-MSL128`](https://huggingface.co/leewaay/klue-roberta-base-klueNLI-klueSTS-MSL128) (768d) 임베딩 + GoMLX ReLU 멀티태스크 헤드
* **3개 멀티태스크 + `CORAL` 도메인 정렬**: KLUE-YNAT 주제(7종) · NSMC 감성(2종) · 조건부 NFC(16종) 손실과 뉴스↔리뷰 `domain_align (0.15)` 손실을 동시 최적화합니다.

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

### 6종 모델 일괄 벤치마크 & 전처리 Parquet/HF Hub 배포
```bash
# KO-Hub 4축 그래프 구축 & 대조 사전학습
go run ./cmd/opinion-analyzer build-graph --config config/config.yaml
go run ./cmd/opinion-analyzer pretrain --config config/config.yaml --backend xla:cpu --steps 600

# 6종 아키텍처 전체 벤치마크 (xla:cpu)
./scripts/benchmark_all_models.sh

# 전처리 4-Split Parquet 데이터셋 익스포트 & Hugging Face Hub 업로드
go run ./cmd/opinion-analyzer export-dataset --config config/config.yaml --output-dir data/hf_dataset_staging
./scripts/publish_hf.sh dataset
```

### 단일 모델 파이프라인 & 개별 단계 실행
```bash
./scripts/run_pipeline.sh pipeline

go run ./cmd/opinion-analyzer train --config config/config.yaml --backend xla:cpu --provider ko_hub_cnn_relu --steps 2500
go run ./cmd/opinion-analyzer analyze --config config/config.yaml --backend xla:cpu --provider ko_hub_cnn_relu --sort-by topic_nfc_intensity
go run ./cmd/opinion-analyzer cluster --config config/config.yaml --backend xla:cpu --provider ko_hub_cnn_relu --pca-dim 24
go run ./cmd/opinion-analyzer eval-human --config config/config.yaml --input outputs/human_eval_sample.csv
```

---

## 3. 실측 6종 벤치마크 요약 (`xla:cpu`, 2 vCPU / 3.8 GiB RAM)

> 네이티브 GoMLX 5종은 전체 평가셋(`YNAT n=9,107`, `NSMC n=49,997`)으로 평가하였으며, `onnx_backbone_relu`(KLUE-RoBERTa 768d 동결 임베딩)는 CPU 인코딩 시간(`15m 46s`)상 도메인당 `max_train=2,000 / max_eval=1,000` 서브샘플 조건에서 평가했습니다.

| 모델 (`provider`) | YNAT Acc / F1 | NSMC Acc / F1 | NFC Recall@3 / Micro-F1 | 활성 NFC | HDBSCAN Silhouette / Dunn_C / NPMI |
|---|---:|---:|---:|---:|---:|
| `scratch_relu` | 74.76% / 0.7462 | 83.39% / 0.8355 | 93.81% / 0.5438 | 16/16 | 0.6786 / 2.9383 / -0.3167 |
| `ko_hub_graph` | 74.82% / 0.7474 | 83.52% / 0.8346 | **95.04% / 0.6487** | 16/16 | 0.4928 / 2.4043 / -0.3411 |
| `ko_hub_cnn_relu` | **75.91% / 0.7575** | 83.58% / **0.8359** | 94.67% / 0.6311 | 16/16 | 0.6776 / 4.0664 / -0.3297 |
| `ko_hub_transformer_relu` | 74.67% / 0.7461 | **83.60% / 0.8359** | 93.78% / 0.5711 | 16/16 | **0.6938 / 4.1647** / -0.3155 |
| `ko_hub_lstm_relu` | 72.66% / 0.7272 | 82.84% / 0.8315 | 92.39% / 0.5316 | 16/16 | 0.6586 / 2.9131 / -0.3164 |
| `onnx_backbone_relu` *(n=1k)* | 67.00% / 0.6738 | 80.70% / 0.8153 | 73.53% / 0.4297 | 16/16 | 0.5357 / 2.1630 / **-0.2861** |

## 4. 생성 산출물 (`outputs/`)

* `articles_sorted.json` / `articles_sorted.csv` / `social_opinion_report.md`: 기사별 주제·감성·Nuclear Feeling Checklist 분류, `7 × 16` 교차 매트릭스 및 정렬 리포트
* `clusters_report.json` / `clustering_validation_report.md`: 3가지 클러스터링 결과 및 Silhouette, Dunn Index, NPMI/UMass Coherence 검증 스코어카드
* `human_eval_sample.csv` / `human_eval_rubric.md`: 전문가 인간 검증용 층화 샘플 및 평가 가이드
