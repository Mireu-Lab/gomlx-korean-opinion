#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/config.yaml}"
BIN_PATH="${BIN_PATH:-${PROJECT_ROOT}/bin/opinion-analyzer}"
COMMAND="${1:-pipeline}"
if [[ $# -gt 0 ]]; then
  shift
fi

log_step() {
  local level="$1"
  shift
  printf '[%s] [%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${level}" "$*"
}

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${BIN_PATH}")"

log_step "INFO" "Building opinion-analyzer binary -> ${BIN_PATH}"
go build -o "${BIN_PATH}" ./cmd/opinion-analyzer

CMD_ARGS=(--config "${CONFIG_PATH}")
if [[ -n "${GOMLX_BACKEND:-}" ]]; then
  CMD_ARGS+=(--backend "${GOMLX_BACKEND}")
fi
if [[ -n "${SORT_MODE:-}" ]]; then
  CMD_ARGS+=(--sort-by "${SORT_MODE}")
fi
if [[ -n "${TRAIN_STEPS:-}" ]]; then
  CMD_ARGS+=(--steps "${TRAIN_STEPS}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  CMD_ARGS+=(--batch-size "${BATCH_SIZE}")
fi
if [[ -n "${INPUT_FILE:-}" ]]; then
  CMD_ARGS+=(--input "${INPUT_FILE}")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  CMD_ARGS+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${ANALYZE_LIMIT:-}" ]]; then
  CMD_ARGS+=(--limit "${ANALYZE_LIMIT}")
fi

log_step "INFO" "Executing command '${COMMAND}' with config=${CONFIG_PATH}"
"${BIN_PATH}" "${COMMAND}" "${CMD_ARGS[@]}" "$@"
log_step "INFO" "Pipeline command '${COMMAND}' completed successfully."
