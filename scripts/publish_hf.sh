#!/usr/bin/env bash
# ==============================================================================
# Hugging Face Hub Publisher (Datasets & Model Hub)
# Conforms to Limmireu Pipeline Global Rules (set -euo pipefail, Zero Hardcoding)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET="${1:-all}"
HF_BIN="${HF_BIN:-hf}"
HF_NAMESPACE="${HF_NAMESPACE:-Mireu-Lab}"
HF_DATASET_REPO="${HF_DATASET_REPO:-${HF_NAMESPACE}/gomlx-korean-opinion-dataset}"
HF_MODEL_REPO="${HF_MODEL_REPO:-${HF_NAMESPACE}/gomlx-korean-opinion}"
HF_VISIBILITY="${HF_VISIBILITY:-public}"
DATASET_STAGING_DIR="${DATASET_STAGING_DIR:-${PROJECT_ROOT}/data/hf_dataset_staging}"
MODEL_STAGING_DIR="${MODEL_STAGING_DIR:-${PROJECT_ROOT}/data/hf_model_staging}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/hf_publish}"
HF_COMMIT_MSG="${HF_COMMIT_MSG:-}"

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

if ! command -v "${HF_BIN}" >/dev/null 2>&1; then
  log_status "HF_AUTH" "FAILED" "hf CLI binary not found: ${HF_BIN}"
  exit 1
fi

log_status "HF_AUTH" "START" "verifying active Hugging Face Hub session"
WHOAMI="$("${HF_BIN}" auth whoami 2>&1)"
printf '%s\n' "${WHOAMI}" >> "${PROC_LOG}"
log_status "HF_AUTH" "SUCCESS" "authenticated on Hugging Face Hub"

VISIBILITY_FLAG="--public"
if [[ "${HF_VISIBILITY}" == "private" ]]; then
  VISIBILITY_FLAG="--private"
elif [[ "${HF_VISIBILITY}" != "public" ]]; then
  log_status "CONFIG" "FAILED" "HF_VISIBILITY must be public or private, got ${HF_VISIBILITY}"
  exit 1
fi

publish_repo() {
  local repo_type="$1"
  local repo_id="$2"
  local staging_dir="$3"
  local default_msg="$4"
  local stage_tag="HF_${repo_type^^}"

  if [[ ! -d "${staging_dir}" ]]; then
    log_status "${stage_tag}" "FAILED" "staging directory does not exist: ${staging_dir}"
    exit 1
  fi
  if [[ ! -f "${staging_dir}/README.md" ]]; then
    log_status "${stage_tag}" "FAILED" "missing required README.md card in ${staging_dir}"
    exit 1
  fi

  local repo_url="https://huggingface.co/${repo_id}"
  if [[ "${repo_type}" == "dataset" ]]; then
    repo_url="https://huggingface.co/datasets/${repo_id}"
  fi
  local commit_msg="${HF_COMMIT_MSG:-${default_msg}}"
  log_status "${stage_tag}" "START" "ensuring ${repo_type} repo ${repo_id} (${HF_VISIBILITY}) and uploading ${staging_dir}"
  "${HF_BIN}" repos create "${repo_id}" --type "${repo_type}" "${VISIBILITY_FLAG}" --exist-ok >> "${PROC_LOG}" 2>&1
  "${HF_BIN}" upload "${repo_id}" "${staging_dir}" . --type "${repo_type}" --commit-message "${commit_msg}" >> "${PROC_LOG}" 2>&1
  log_status "${stage_tag}" "SUCCESS" "published ${repo_url}"
}

case "${TARGET}" in
  dataset)
    publish_repo "dataset" "${HF_DATASET_REPO}" "${DATASET_STAGING_DIR}" "Upload preprocessed KLUE-YNAT + NSMC + KO-Hub Parquet dataset and vocab/graph metadata"
    ;;
  model)
    publish_repo "model" "${HF_MODEL_REPO}" "${MODEL_STAGING_DIR}" "Upload validated GoMLX multi-architecture Korean opinion & NFC models"
    ;;
  all)
    publish_repo "dataset" "${HF_DATASET_REPO}" "${DATASET_STAGING_DIR}" "Upload preprocessed KLUE-YNAT + NSMC + KO-Hub Parquet dataset and vocab/graph metadata"
    publish_repo "model" "${HF_MODEL_REPO}" "${MODEL_STAGING_DIR}" "Upload validated GoMLX multi-architecture Korean opinion & NFC models"
    ;;
  *)
    printf 'Usage: %s [dataset|model|all]\n' "$0" >&2
    exit 2
    ;;
esac
