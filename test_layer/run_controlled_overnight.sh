#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

RUN_ROOT="${ROOT_DIR}/logs/overnight_runs/controlled"
LATEST_DIR="${RUN_ROOT}/latest"
TABLES_DIR="${ROOT_DIR}/logs/thesis_tables/controlled"
STDOUT_LOG="${RUN_ROOT}/overnight_stdout.log"
PID_FILE="${RUN_ROOT}/overnight_pipeline.pid"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"

clean_outputs() {
  pkill -f "python .*overnight_pipeline.py" >/dev/null 2>&1 || true
  pkill -f "python .*train.py" >/dev/null 2>&1 || true
  pkill -f "python .*deploy.py" >/dev/null 2>&1 || true
  shopt -s nullglob
  rm -rf "${RUN_ROOT}" "${TABLES_DIR}" "${ROOT_DIR}/logs/thesis_runs/latest"
  rm -f "${ROOT_DIR}/logs/thesis_benchmark_rows.csv"
  rm -rf \
    "${ROOT_DIR}"/deploy_artifacts/CNN_CWRU_FFT_* \
    "${ROOT_DIR}"/deploy_artifacts/CNN_CWRU_STFT_* \
    "${ROOT_DIR}"/deploy_artifacts/CNN_CWRU_WT_* \
    "${ROOT_DIR}"/deploy_artifacts/MAML_CWRU_FFT_* \
    "${ROOT_DIR}"/deploy_artifacts/MAML_CWRU_STFT_* \
    "${ROOT_DIR}"/deploy_artifacts/MAML_CWRU_WT_* \
    "${ROOT_DIR}"/deploy_artifacts/ProtoNet_CWRU_FFT_* \
    "${ROOT_DIR}"/deploy_artifacts/ProtoNet_CWRU_STFT_* \
    "${ROOT_DIR}"/deploy_artifacts/ProtoNet_CWRU_WT_*
  rm -f \
    "${ROOT_DIR}"/checkpoints/CNN_CWRU_FFT_*_best.pt \
    "${ROOT_DIR}"/checkpoints/CNN_CWRU_FFT_*_history.json \
    "${ROOT_DIR}"/checkpoints/CNN_CWRU_STFT_*_best.pt \
    "${ROOT_DIR}"/checkpoints/CNN_CWRU_STFT_*_history.json \
    "${ROOT_DIR}"/checkpoints/CNN_CWRU_WT_*_best.pt \
    "${ROOT_DIR}"/checkpoints/CNN_CWRU_WT_*_history.json \
    "${ROOT_DIR}"/checkpoints/MAML_CWRU_FFT_*_best.pt \
    "${ROOT_DIR}"/checkpoints/MAML_CWRU_FFT_*_history.json \
    "${ROOT_DIR}"/checkpoints/MAML_CWRU_STFT_*_best.pt \
    "${ROOT_DIR}"/checkpoints/MAML_CWRU_STFT_*_history.json \
    "${ROOT_DIR}"/checkpoints/MAML_CWRU_WT_*_best.pt \
    "${ROOT_DIR}"/checkpoints/MAML_CWRU_WT_*_history.json \
    "${ROOT_DIR}"/checkpoints/ProtoNet_CWRU_FFT_*_best.pt \
    "${ROOT_DIR}"/checkpoints/ProtoNet_CWRU_FFT_*_history.json \
    "${ROOT_DIR}"/checkpoints/ProtoNet_CWRU_STFT_*_best.pt \
    "${ROOT_DIR}"/checkpoints/ProtoNet_CWRU_STFT_*_history.json \
    "${ROOT_DIR}"/checkpoints/ProtoNet_CWRU_WT_*_best.pt \
    "${ROOT_DIR}"/checkpoints/ProtoNet_CWRU_WT_*_history.json
}

start_run() {
  mkdir -p "${RUN_ROOT}"
  "${PYTHON_BIN}" -c "import torch, learn2learn, onnxruntime, numpy" >/dev/null
  {
    echo "started_at=$(date -Iseconds)"
    echo "python_bin=${PYTHON_BIN}"
    "${PYTHON_BIN}" -V
    "${PYTHON_BIN}" -c "import sys; print(sys.executable)"
  } > "${RUN_ROOT}/environment.txt"
  nohup "${PYTHON_BIN}" "${ROOT_DIR}/test_layer/overnight_pipeline.py" \
    --preset overnight_controlled \
    --max_attempts "${MAX_ATTEMPTS}" \
    --manifest_dir "logs/overnight_runs/controlled/latest" \
    --tables_dir "logs/thesis_tables/controlled" \
    > "${STDOUT_LOG}" 2>&1 < /dev/null &
  echo $! > "${PID_FILE}"
  echo "PID=$(cat "${PID_FILE}")"
}

show_status() {
  if [[ -f "${PID_FILE}" ]]; then
    echo "PID=$(cat "${PID_FILE}")"
    ps -fp "$(cat "${PID_FILE}")" || true
  else
    echo "PID file not found."
  fi
  if compgen -G "${LATEST_DIR}/logs/*/*/*.log" > /dev/null; then
    tail -n 40 "$(ls -t ${LATEST_DIR}/logs/*/*/*.log | head -n 1)"
  else
    echo "No experiment logs yet."
  fi
}

ACTION="${1:-restart}"
case "${ACTION}" in
  clean)
    clean_outputs
    ;;
  start)
    start_run
    ;;
  status)
    show_status
    ;;
  restart)
    clean_outputs
    start_run
    ;;
  *)
    echo "Usage: bash test_layer/run_controlled_overnight.sh [restart|start|clean|status]" >&2
    exit 1
    ;;
esac
