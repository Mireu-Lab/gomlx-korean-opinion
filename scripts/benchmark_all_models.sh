#!/usr/bin/env bash
# ==============================================================================
# Multi-Architecture Benchmark Runner (All 6 Korean Opinion & NFC Models)
# Conforms to Limmireu Pipeline Global Rules (set -euo pipefail, Zero Hardcoding)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/config.yaml}"
BIN_PATH="${BIN_PATH:-${PROJECT_ROOT}/bin/opinion-analyzer}"
BACKEND="${GOMLX_BACKEND:-xla:cpu}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/benchmark}"

PRETRAIN_STEPS="${PRETRAIN_STEPS:-600}"
SCRATCH_STEPS="${SCRATCH_STEPS:-2500}"
KOHUB_STEPS="${KOHUB_STEPS:-2500}"
CNN_STEPS="${CNN_STEPS:-2500}"
TRANSFORMER_STEPS="${TRANSFORMER_STEPS:-2500}"
LSTM_STEPS="${LSTM_STEPS:-1500}"
ONNX_STEPS="${ONNX_STEPS:-1500}"
PCA_DIM="${PCA_DIM:-24}"
RUN_ONNX="${RUN_ONNX:-true}"

mkdir -p "${LOG_DIR}"
STATUS_LOG="${LOG_DIR}/status.log"
PROC_LOG="${LOG_DIR}/processing.log"

log_status() {
  local stage="$1"
  local state="$2"
  shift 2
  local ts
  ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf '[%s] [%s] [%s] %s\n' "${ts}" "${stage}" "${state}" "$*" | tee -a "${STATUS_LOG}" "${PROC_LOG}"
}

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${BIN_PATH}")"
log_status "BUILD" "START" "building ${BIN_PATH}"
go build -o "${BIN_PATH}" ./cmd/opinion-analyzer
log_status "BUILD" "SUCCESS" "binary ready at ${BIN_PATH}"

log_status "GRAPH" "START" "building KO-Hub heterogeneous CSR graph"
"${BIN_PATH}" build-graph --config "${CONFIG_PATH}" >> "${PROC_LOG}" 2>&1
log_status "GRAPH" "SUCCESS" "KO-Hub graph built"

log_status "PRETRAIN" "START" "running KO-Hub contrastive pretraining (steps=${PRETRAIN_STEPS})"
"${BIN_PATH}" pretrain --config "${CONFIG_PATH}" --backend "${BACKEND}" --steps "${PRETRAIN_STEPS}" >> "${PROC_LOG}" 2>&1
log_status "PRETRAIN" "SUCCESS" "KO-Hub pretraining complete"

run_model() {
  local provider="$1"
  local steps="$2"
  local pca="$3"
  local ckpt="${CKPT_ROOT}/bench_${provider}"
  local out="${OUT_ROOT}/bench_${provider}"
  local run_log="${LOG_DIR}/bench_${provider}.log"

  log_status "BENCH_${provider}" "START" "backend=${BACKEND} steps=${steps} pca=${pca}"
  "${BIN_PATH}" pipeline \
    --config "${CONFIG_PATH}" \
    --backend "${BACKEND}" \
    --provider "${provider}" \
    --checkpoint-dir "${ckpt}" \
    --output-dir "${out}" \
    --steps "${steps}" \
    --pca-dim "${pca}" > "${run_log}" 2>&1
  cat "${run_log}" >> "${PROC_LOG}"

  local eval_line cluster_line
  eval_line="$(grep '\[EVAL\] \[SUCCESS\]' "${run_log}" | tail -n 1 || true)"
  cluster_line="$(grep '\[CLUSTER\] \[SUCCESS\]' "${run_log}" | tail -n 1 || true)"
  log_status "BENCH_${provider}" "SUCCESS" "${eval_line} | ${cluster_line}"
}

run_model "scratch_relu" "${SCRATCH_STEPS}" "0"
run_model "ko_hub_graph" "${KOHUB_STEPS}" "${PCA_DIM}"
run_model "ko_hub_cnn_relu" "${CNN_STEPS}" "${PCA_DIM}"
run_model "ko_hub_transformer_relu" "${TRANSFORMER_STEPS}" "${PCA_DIM}"
run_model "ko_hub_lstm_relu" "${LSTM_STEPS}" "${PCA_DIM}"

if [[ "${RUN_ONNX}" == "true" ]]; then
  run_model "onnx_backbone_relu" "${ONNX_STEPS}" "${PCA_DIM}"
fi

log_status "BENCHMARK_ALL" "SUCCESS" "all requested models completed; logs in ${LOG_DIR}"
