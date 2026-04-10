#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

ACTION="${1:-run}"
SYSTEM_BASE_URL="${SYSTEM_BASE_URL:-http://127.0.0.1:8000}"
SYSTEM_SUMMARY_PATH="${SYSTEM_SUMMARY_PATH:-deploy_artifacts/paper_balanced/seed/seed42/MAML_CWRU_STFT_5w5s_source012_target3_labels0123456789/compression_summary.json}"
FALLBACK_SUMMARY_PATH="deploy_artifacts/MAML_CWRU_STFT_5w5s_source012_target3_labels0123456789/compression_summary.json"
OUTPUT_PATH="${OUTPUT_PATH:-logs/thesis_tables/paper_balanced/system_benchmark.json}"
REQUEST_COUNT="${REQUEST_COUNT:-100}"
LABELS="${LABELS:-0,1,2,3,4}"

resolve_summary_path() {
  if [[ -f "${ROOT_DIR}/${SYSTEM_SUMMARY_PATH}" ]]; then
    printf '%s\n' "${SYSTEM_SUMMARY_PATH}"
    return 0
  fi
  if [[ -f "${ROOT_DIR}/${FALLBACK_SUMMARY_PATH}" ]]; then
    printf '%s\n' "${FALLBACK_SUMMARY_PATH}"
    return 0
  fi
  echo "Unable to resolve system benchmark summary path." >&2
  return 1
}

run_benchmark() {
  local summary_path
  summary_path="$(resolve_summary_path)"
  echo
  echo "=== System Benchmark ==="
  echo "Base URL: ${SYSTEM_BASE_URL}"
  echo "Summary: ${summary_path}"
  echo "Output: ${OUTPUT_PATH}"
  echo
  "${PYTHON_BIN}" test_layer/system_benchmark.py \
    --base_url "${SYSTEM_BASE_URL}" \
    --summary_path "${summary_path}" \
    --prefer_int8 \
    --source cwru \
    --domain 3 \
    --labels "${LABELS}" \
    --request_count "${REQUEST_COUNT}" \
    --interval 0 \
    --output_path "${OUTPUT_PATH}"
}

case "${ACTION}" in
  clean)
    rm -f "${ROOT_DIR}/${OUTPUT_PATH}"
    ;;
  restart)
    rm -f "${ROOT_DIR}/${OUTPUT_PATH}"
    run_benchmark
    ;;
  run)
    run_benchmark
    ;;
  *)
    echo "Usage: bash test_layer/run_system_benchmark.sh [clean|run|restart]" >&2
    exit 1
    ;;
esac
