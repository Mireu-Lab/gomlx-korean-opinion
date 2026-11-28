#!/usr/bin/env bash
# Download Hugging Face model weights into a local directory for infer.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HF_BIN="${HF_BIN:-hf}"
HF_MODEL_REPO="${HF_MODEL_REPO:-Mireu-Lab/gomlx-korean-opinion}"
LOCAL_DIR="${HF_LOCAL_DIR:-${PROJECT_ROOT}/data/hf_model}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/hf_publish}"

mkdir -p "${LOG_DIR}" "${LOCAL_DIR}"
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

if ! command -v "${HF_BIN}" >/dev/null 2>&1; then
  log_status "HF_FETCH" "FAILED" "hf CLI not found: ${HF_BIN}"
  exit 1
fi

log_status "HF_FETCH" "START" "downloading ${HF_MODEL_REPO} -> ${LOCAL_DIR}"
"${HF_BIN}" download "${HF_MODEL_REPO}" --repo-type model --local-dir "${LOCAL_DIR}" >> "${PROC_LOG}" 2>&1
log_status "HF_FETCH" "SUCCESS" "weights at ${LOCAL_DIR}. Infer with --checkpoint-dir pointing at checkpoints/<provider>"
