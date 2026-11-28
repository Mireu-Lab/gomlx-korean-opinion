# Gold Standard 전문가 인간 검증 (Human Evaluation) 가이드

- **추출 샘플 수**: 150 건 (군집별 층화 추출)
- **평가 파일**: `human_eval_sample.csv`
- **목표 기준**: AI 군집/주제 분류와 인간 평가자 기준 정합성 **80% 이상** (`human_agree = 1` 비율)

## 평가 방법
1. `human_eval_sample.csv` 를 스프레드시트 또는 편집기로 엽니다.
2. 각 행의 `title`(원문)을 읽고, AI가 배정한 `hierarchical_label` / `predicted_topic` / `predicted_nfc` 가 문맥상 타당한지 판단합니다.
3. `human_agree` 컬럼에 타당하면 `1`, 부적합하면 `0` 을 입력합니다.
4. 부적합(`0`)인 경우 `human_corrected_topic` 에 올바른 주제/군집명을 기입합니다.
5. 저장 후 아래 명령으로 정합성(Agreement Rate)과 Cohen's Kappa를 자동 산출합니다:

```bash
go run ./cmd/opinion-analyzer eval-human --config config/config.yaml --input outputs/human_eval_sample.csv
```
