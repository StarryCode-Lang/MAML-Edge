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

ACTION="${1:-restart}"
REL_ROOT="paper_balanced/ablation"
RUN_ROOT="${ROOT_DIR}/logs/paper_runs/balanced/ablation"
LOG_ROOT="${RUN_ROOT}/logs"
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints/${REL_ROOT}"
COMPRESSION_OUTPUT_ROOT="${ROOT_DIR}/deploy_artifacts/${REL_ROOT}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"
MONITOR_TAIL_LINES="${MONITOR_TAIL_LINES:-4}"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-42}"
TEST_TASK_NUM_OVERRIDE="${TEST_TASK_NUM_OVERRIDE:-200}"
DEPLOYMENT_LABEL_SAMPLING="${DEPLOYMENT_LABEL_SAMPLING:-random_per_episode}"
TOTAL_STEPS=2
COMPLETED_STEPS=0
RUN_STARTED_EPOCH=0

format_duration() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf '%02d:%02d:%02d' "${hours}" "${minutes}" "${seconds}"
}

render_progress_bar() {
  local completed="$1"
  local total="$2"
  local width=24
  local filled=0
  if [[ "${total}" -gt 0 ]]; then
    filled=$((completed * width / total))
  fi
  local empty=$((width - filled))
  printf '['
  printf '%*s' "${filled}" '' | tr ' ' '#'
  printf '%*s' "${empty}" '' | tr ' ' '-'
  printf ']'
}

print_live_log_update() {
  local log_path="$1"
  local step_name="$2"
  local elapsed=$(( $(date +%s) - RUN_STARTED_EPOCH ))
  local live_lines
  live_lines="$(grep -v '^$' "${log_path}" | tail -n "${MONITOR_TAIL_LINES}" || true)"
  if [[ -z "${live_lines}" ]]; then
    return 0
  fi
  echo
  printf '[Live %s] %s\n' "$(format_duration "${elapsed}")" "${step_name}"
  while IFS= read -r line; do
    [[ -n "${line}" ]] && printf '  %s\n' "${line}"
  done <<< "${live_lines}"
}

monitor_running_command() {
  local cmd_pid="$1"
  local log_path="$2"
  local step_name="$3"
  local last_line_count=-1
  while kill -0 "${cmd_pid}" >/dev/null 2>&1; do
    sleep "${MONITOR_INTERVAL}"
    if ! kill -0 "${cmd_pid}" >/dev/null 2>&1; then
      break
    fi
    if [[ ! -f "${log_path}" ]]; then
      continue
    fi
    local current_line_count
    current_line_count="$(wc -l < "${log_path}" 2>/dev/null || echo 0)"
    if [[ "${current_line_count}" != "${last_line_count}" ]]; then
      print_live_log_update "${log_path}" "${step_name}"
      last_line_count="${current_line_count}"
    fi
  done
}

clean_outputs() {
  rm -rf "${RUN_ROOT}" "${CHECKPOINT_ROOT}" "${COMPRESSION_OUTPUT_ROOT}"
}

run_logged_command() {
  local step_name="$1"
  shift
  local log_path="${LOG_ROOT}/${step_name}.log"
  mkdir -p "$(dirname "${log_path}")"
  local display_index=$((COMPLETED_STEPS + 1))
  local percent=$((display_index * 100 / TOTAL_STEPS))
  echo
  printf '%s %02d/%02d (%3d%%)  ablation\n' \
    "$(render_progress_bar "${COMPLETED_STEPS}" "${TOTAL_STEPS}")" \
    "${display_index}" \
    "${TOTAL_STEPS}" \
    "${percent}"
  echo "Step: ${step_name}"
  echo "Log : ${log_path}"
  {
    echo
    echo "=== $(date -Iseconds) ==="
    printf '$ '
    printf '%q ' "$@"
    printf '\n'
  } >> "${log_path}"

  set +e
  (
    cd "${ROOT_DIR}"
    "$@"
  ) >> "${log_path}" 2>&1 &
  local cmd_pid=$!
  monitor_running_command "${cmd_pid}" "${log_path}" "${step_name}"
  wait "${cmd_pid}"
  local rc=$?
  set -e

  COMPLETED_STEPS=$((COMPLETED_STEPS + 1))
  printf 'Result: [%s]  completed=%d/%d  elapsed=%s  step=%s\n' \
    "$([[ "${rc}" -eq 0 ]] && echo OK || echo FAIL)" \
    "${COMPLETED_STEPS}" \
    "${TOTAL_STEPS}" \
    "$(format_duration $(( $(date +%s) - RUN_STARTED_EPOCH )))" \
    "${step_name}"
  return "${rc}"
}

run_suite() {
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${CHECKPOINT_ROOT}" "${COMPRESSION_OUTPUT_ROOT}"
  RUN_STARTED_EPOCH="$(date +%s)"
  echo
  echo "=== Compression Ablation Run ==="
  echo "Root: ${ROOT_DIR}"
  echo "Python: ${PYTHON_BIN}"
  echo "Checkpoints: ${CHECKPOINT_ROOT}"
  echo "Deploy artifacts: ${COMPRESSION_OUTPUT_ROOT}"
  echo "Seed: ${EXPERIMENT_SEED}"
  echo "Deployment label sampling: ${DEPLOYMENT_LABEL_SAMPLING}"
  echo

  run_logged_command "MAML_CWRU_STFT_5w5s_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm maml \
    --dataset CWRU --preprocess STFT --ways 5 --shots 5 --query_shots 5 \
    --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 \
    --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 \
    --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false \
    --onnx_opset 17 --iters 80 --meta_batch_size 16 --train_task_num 100 \
    --test_task_num "${TEST_TASK_NUM_OVERRIDE}" --compression_finetune_iters 80 \
    --compression_meta_batch_size 8 --seed "${EXPERIMENT_SEED}" \
    --checkpoint_path "${CHECKPOINT_ROOT}" --compression_output_path "${COMPRESSION_OUTPUT_ROOT}" \
    --deployment_label_sampling "${DEPLOYMENT_LABEL_SAMPLING}"

  run_logged_command "ProtoNet_CWRU_FFT_5w5s5q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm protonet \
    --dataset CWRU --preprocess FFT --ways 5 --shots 5 --query_shots 5 \
    --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 \
    --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 \
    --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false \
    --onnx_opset 17 --iters 400 --meta_batch_size 16 --train_task_num 100 \
    --test_task_num "${TEST_TASK_NUM_OVERRIDE}" --compression_finetune_iters 80 \
    --compression_meta_batch_size 8 --seed "${EXPERIMENT_SEED}" \
    --checkpoint_path "${CHECKPOINT_ROOT}" --compression_output_path "${COMPRESSION_OUTPUT_ROOT}" \
    --deployment_label_sampling "${DEPLOYMENT_LABEL_SAMPLING}"
}

case "${ACTION}" in
  clean)
    clean_outputs
    ;;
  restart)
    clean_outputs
    run_suite
    ;;
  run)
    run_suite
    ;;
  *)
    echo "Usage: bash test_layer/run_compression_ablation.sh [clean|run|restart]" >&2
    exit 1
    ;;
esac
