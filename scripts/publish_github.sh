#!/usr/bin/env bash
# Push source only to GitHub. Weights, data, logs, and binaries stay out via .gitignore.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/github_publish}"
REMOTE="${GIT_REMOTE:-origin}"
BRANCH="${GIT_BRANCH:-}"

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
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  log_status "GIT" "FAILED" "not a git repository"
  exit 1
fi

if [[ -z "${BRANCH}" ]]; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
fi

log_status "GIT" "START" "source-only push remote=${REMOTE} branch=${BRANCH}"
git check-ignore -v bin/opinion-analyzer data/checkpoints 2>>"${PROC_LOG}" || true
git status --short | tee -a "${PROC_LOG}"

if git ls-files | grep -E '(^bin/|^data/|^checkpoints/|^logs/|^outputs/)' >/dev/null; then
  log_status "GIT" "FAILED" "tracked artifacts under bin/data/checkpoints/logs/outputs; untrack them before pushing source"
  exit 1
fi

if ! git remote get-url "${REMOTE}" >/dev/null 2>&1; then
  log_status "GIT" "FAILED" "remote ${REMOTE} is not configured"
  exit 1
fi

git push -u "${REMOTE}" "${BRANCH}" | tee -a "${PROC_LOG}"
log_status "GIT" "SUCCESS" "pushed source to $(git remote get-url "${REMOTE}") ${BRANCH}"
echo "Model weights: run 'go run ./cmd/opinion-analyzer export-model' then './scripts/publish_hf.sh model'"
