#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

RUN_NAMESPACE="${RUN_NAMESPACE:-controlled}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/logs/overnight_runs/${RUN_NAMESPACE}}"
LATEST_DIR="${LATEST_DIR:-${RUN_ROOT}/latest}"
LOG_ROOT="${LOG_ROOT:-${LATEST_DIR}/logs}"
TABLES_DIR="${TABLES_DIR:-${ROOT_DIR}/logs/thesis_tables/${RUN_NAMESPACE}}"
BENCHMARK_ROWS_OUTPUT="${BENCHMARK_ROWS_OUTPUT:-${TABLES_DIR}/benchmark_rows.csv}"
SUMMARY_GLOB="${SUMMARY_GLOB:-deploy_artifacts/*/compression_summary.json}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT_DIR}/checkpoints}"
COMPRESSION_OUTPUT_ROOT="${COMPRESSION_OUTPUT_ROOT:-${ROOT_DIR}/deploy_artifacts}"
CURRENT_STEP_FILE="${RUN_ROOT}/current_step.txt"
STATUS_FILE="${RUN_ROOT}/step_status.tsv"
FAILED_STEPS_FILE="${RUN_ROOT}/failed_steps.txt"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-42}"
TRAIN_DOMAINS_OVERRIDE="${TRAIN_DOMAINS_OVERRIDE:-0,1,2}"
TEST_DOMAIN_OVERRIDE="${TEST_DOMAIN_OVERRIDE:-3}"
FAULT_LABELS_OVERRIDE="${FAULT_LABELS_OVERRIDE:-0,1,2,3,4,5,6,7,8,9}"
TEST_TASK_NUM_OVERRIDE="${TEST_TASK_NUM_OVERRIDE:-}"
DEPLOYMENT_LABEL_SAMPLING="${DEPLOYMENT_LABEL_SAMPLING:-random_per_episode}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
PLANNED_TRAIN_STEPS=27
PLANNED_AUX_STEPS=2
TOTAL_PLANNED_STEPS=$((PLANNED_TRAIN_STEPS + PLANNED_AUX_STEPS))
COMPLETED_STEPS=0
FAILED_STEP_COUNT=0
RUN_STARTED_EPOCH=0
MONITOR_INTERVAL="${MONITOR_INTERVAL:-15}"
MONITOR_TAIL_LINES="${MONITOR_TAIL_LINES:-4}"

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

print_run_header() {
  echo
  echo "=== Controlled Overnight Run ==="
  echo "Root: ${ROOT_DIR}"
  echo "Python: ${PYTHON_BIN}"
  echo "Run namespace: ${RUN_NAMESPACE}"
  echo "Planned steps: ${TOTAL_PLANNED_STEPS} (train=${PLANNED_TRAIN_STEPS}, analysis=${PLANNED_AUX_STEPS})"
  echo "Logs: ${LOG_ROOT}"
  echo "Tables: ${TABLES_DIR}"
  echo "Checkpoints: ${CHECKPOINT_ROOT}"
  echo "Deploy artifacts: ${COMPRESSION_OUTPUT_ROOT}"
  echo "Seed: ${EXPERIMENT_SEED}"
  echo "Domains: source=${TRAIN_DOMAINS_OVERRIDE} target=${TEST_DOMAIN_OVERRIDE}"
  echo "Deployment label sampling: ${DEPLOYMENT_LABEL_SAMPLING}"
  echo "Live monitor: every ${MONITOR_INTERVAL}s, tail ${MONITOR_TAIL_LINES} lines"
  echo
}

announce_step_start() {
  local step_kind="$1"
  local preprocess="$2"
  local algorithm="$3"
  local step_name="$4"
  local log_path="$5"
  local display_index=$((COMPLETED_STEPS + 1))
  local percent=$((display_index * 100 / TOTAL_PLANNED_STEPS))
  echo
  printf '%s %02d/%02d (%3d%%)  %-10s  %s / %s\n' \
    "$(render_progress_bar "${COMPLETED_STEPS}" "${TOTAL_PLANNED_STEPS}")" \
    "${display_index}" \
    "${TOTAL_PLANNED_STEPS}" \
    "${percent}" \
    "${step_kind}" \
    "${preprocess}" \
    "${algorithm}"
  echo "Step: ${step_name}"
  echo "Log : ${log_path}"
}

announce_step_end() {
  local rc="$1"
  local step_name="$2"
  COMPLETED_STEPS=$((COMPLETED_STEPS + 1))
  if [[ "${rc}" -ne 0 ]]; then
    FAILED_STEP_COUNT=$((FAILED_STEP_COUNT + 1))
  fi
  local elapsed=$(( $(date +%s) - RUN_STARTED_EPOCH ))
  local status_label="OK"
  if [[ "${rc}" -ne 0 ]]; then
    status_label="FAIL"
  fi
  printf 'Result: [%s]  completed=%d/%d  failed=%d  elapsed=%s  step=%s\n' \
    "${status_label}" \
    "${COMPLETED_STEPS}" \
    "${TOTAL_PLANNED_STEPS}" \
    "${FAILED_STEP_COUNT}" \
    "$(format_duration "${elapsed}")" \
    "${step_name}"
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
  pkill -f "python .*train.py" >/dev/null 2>&1 || true
  pkill -f "python .*deploy.py" >/dev/null 2>&1 || true
  shopt -s nullglob
  rm -rf "${RUN_ROOT}" "${TABLES_DIR}" "${ROOT_DIR}/logs/thesis_runs/latest"
  rm -f "${ROOT_DIR}/logs/thesis_benchmark_rows.csv"
  if [[ "${CHECKPOINT_ROOT}" != "${ROOT_DIR}/checkpoints" || "${COMPRESSION_OUTPUT_ROOT}" != "${ROOT_DIR}/deploy_artifacts" ]]; then
    rm -rf "${CHECKPOINT_ROOT}" "${COMPRESSION_OUTPUT_ROOT}"
    return 0
  fi
  rm -f \
    "${ROOT_DIR}"/logs/CNN_CWRU_*.log \
    "${ROOT_DIR}"/logs/MAML_CWRU_*.log \
    "${ROOT_DIR}"/logs/ProtoNet_CWRU_*.log
  rm -rf \
    "${COMPRESSION_OUTPUT_ROOT}"/CNN_CWRU_FFT_* \
    "${COMPRESSION_OUTPUT_ROOT}"/CNN_CWRU_STFT_* \
    "${COMPRESSION_OUTPUT_ROOT}"/CNN_CWRU_WT_* \
    "${COMPRESSION_OUTPUT_ROOT}"/MAML_CWRU_FFT_* \
    "${COMPRESSION_OUTPUT_ROOT}"/MAML_CWRU_STFT_* \
    "${COMPRESSION_OUTPUT_ROOT}"/MAML_CWRU_WT_* \
    "${COMPRESSION_OUTPUT_ROOT}"/ProtoNet_CWRU_FFT_* \
    "${COMPRESSION_OUTPUT_ROOT}"/ProtoNet_CWRU_STFT_* \
    "${COMPRESSION_OUTPUT_ROOT}"/ProtoNet_CWRU_WT_*
  rm -f \
    "${CHECKPOINT_ROOT}"/CNN_CWRU_FFT_*_best.pt \
    "${CHECKPOINT_ROOT}"/CNN_CWRU_FFT_*_history.json \
    "${CHECKPOINT_ROOT}"/CNN_CWRU_STFT_*_best.pt \
    "${CHECKPOINT_ROOT}"/CNN_CWRU_STFT_*_history.json \
    "${CHECKPOINT_ROOT}"/CNN_CWRU_WT_*_best.pt \
    "${CHECKPOINT_ROOT}"/CNN_CWRU_WT_*_history.json \
    "${CHECKPOINT_ROOT}"/MAML_CWRU_FFT_*_best.pt \
    "${CHECKPOINT_ROOT}"/MAML_CWRU_FFT_*_history.json \
    "${CHECKPOINT_ROOT}"/MAML_CWRU_STFT_*_best.pt \
    "${CHECKPOINT_ROOT}"/MAML_CWRU_STFT_*_history.json \
    "${CHECKPOINT_ROOT}"/MAML_CWRU_WT_*_best.pt \
    "${CHECKPOINT_ROOT}"/MAML_CWRU_WT_*_history.json \
    "${CHECKPOINT_ROOT}"/ProtoNet_CWRU_FFT_*_best.pt \
    "${CHECKPOINT_ROOT}"/ProtoNet_CWRU_FFT_*_history.json \
    "${CHECKPOINT_ROOT}"/ProtoNet_CWRU_STFT_*_best.pt \
    "${CHECKPOINT_ROOT}"/ProtoNet_CWRU_STFT_*_history.json \
    "${CHECKPOINT_ROOT}"/ProtoNet_CWRU_WT_*_best.pt \
    "${CHECKPOINT_ROOT}"/ProtoNet_CWRU_WT_*_history.json
}

prepare_run_dirs() {
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${TABLES_DIR}"
  : > "${STATUS_FILE}"
  : > "${FAILED_STEPS_FILE}"
  {
    echo "started_at=$(date -Iseconds)"
    echo "python_bin=${PYTHON_BIN}"
    "${PYTHON_BIN}" -V
    "${PYTHON_BIN}" -c "import sys; print(sys.executable)"
  } > "${RUN_ROOT}/environment.txt"
}

run_logged_command() {
  local preprocess="$1"
  local algorithm="$2"
  local step_name="$3"
  local step_kind="${4:-step}"
  local count_progress="${5:-1}"
  shift 5

  local log_path="${LOG_ROOT}/${preprocess}/${algorithm}/${step_name}.log"
  mkdir -p "$(dirname "${log_path}")"
  echo "${step_name}" > "${CURRENT_STEP_FILE}"
  if [[ "${count_progress}" == "1" ]]; then
    announce_step_start "${step_kind}" "${preprocess}" "${algorithm}" "${step_name}" "${log_path}"
  else
    echo
    echo "[Recovery] ${step_name}"
    echo "Log : ${log_path}"
  fi
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

  {
    echo
    echo "[return_code] ${rc}"
  } >> "${log_path}"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "${preprocess}" "${algorithm}" "${step_name}" "${rc}" >> "${STATUS_FILE}"
  if [[ "${count_progress}" == "1" ]]; then
    announce_step_end "${rc}" "${step_name}"
  else
    if [[ "${rc}" -eq 0 ]]; then
      echo "[Recovery OK] ${step_name}"
    else
      echo "[Recovery FAIL] ${step_name}"
    fi
  fi
  return "${rc}"
}

append_log_path_if_missing() {
  local log_dir="$1"
  shift
  local -a command_args=("$@")
  local argument
  for argument in "${command_args[@]}"; do
    if [[ "${argument}" == "--log_path" ]]; then
      printf '%s\n' "${command_args[@]}"
      return 0
    fi
  done
  command_args+=("--log_path" "${log_dir}")
  printf '%s\n' "${command_args[@]}"
}

upsert_command_arg() {
  local array_name="$1"
  local flag="$2"
  local value="$3"
  local -n command_ref="${array_name}"
  local index=0
  while [[ "${index}" -lt "${#command_ref[@]}" ]]; do
    if [[ "${command_ref[${index}]}" == "${flag}" ]]; then
      if [[ $((index + 1)) -lt "${#command_ref[@]}" ]]; then
        command_ref[$((index + 1))]="${value}"
      else
        command_ref+=("${value}")
      fi
      return 0
    fi
    index=$((index + 1))
  done
  command_ref+=("${flag}" "${value}")
}

compact_csv_digits() {
  local value="$1"
  printf '%s' "${value//,/}"
}

resolve_experiment_title() {
  local experiment_title="$1"
  local compact_domains
  local compact_labels
  compact_domains="$(compact_csv_digits "${TRAIN_DOMAINS_OVERRIDE}")"
  compact_labels="$(compact_csv_digits "${FAULT_LABELS_OVERRIDE}")"
  experiment_title="$(printf '%s\n' "${experiment_title}" | sed -E "s/_source[0-9]+_target/_source${compact_domains}_target/")"
  experiment_title="$(printf '%s\n' "${experiment_title}" | sed -E "s/_target[0-9]+_labels/_target${TEST_DOMAIN_OVERRIDE}_labels/")"
  experiment_title="$(printf '%s\n' "${experiment_title}" | sed -E "s/_labels[0-9,]+$/_labels${compact_labels}/")"
  printf '%s\n' "${experiment_title}"
}

rewrite_train_command() {
  local array_name="$1"
  local -n command_ref="${array_name}"
  upsert_command_arg "${array_name}" "--seed" "${EXPERIMENT_SEED}"
  upsert_command_arg "${array_name}" "--train_domains" "${TRAIN_DOMAINS_OVERRIDE}"
  upsert_command_arg "${array_name}" "--test_domain" "${TEST_DOMAIN_OVERRIDE}"
  upsert_command_arg "${array_name}" "--fault_labels" "${FAULT_LABELS_OVERRIDE}"
  upsert_command_arg "${array_name}" "--checkpoint_path" "${CHECKPOINT_ROOT}"
  upsert_command_arg "${array_name}" "--compression_output_path" "${COMPRESSION_OUTPUT_ROOT}"
  upsert_command_arg "${array_name}" "--deployment_label_sampling" "${DEPLOYMENT_LABEL_SAMPLING}"
  if [[ -n "${TEST_TASK_NUM_OVERRIDE}" ]]; then
    upsert_command_arg "${array_name}" "--test_task_num" "${TEST_TASK_NUM_OVERRIDE}"
  fi
}

run_train_step() {
  local preprocess="$1"
  local algorithm="$2"
  local base_experiment_title="$3"
  shift 3

  local experiment_title
  experiment_title="$(resolve_experiment_title "${base_experiment_title}")"
  local summary_path="${COMPRESSION_OUTPUT_ROOT}/${experiment_title}/compression_summary.json"
  local step_log_dir="${LOG_ROOT}/${preprocess}/${algorithm}"
  local attempt=1
  local success=1
  local -a train_command=()
  mapfile -t train_command < <(append_log_path_if_missing "${step_log_dir}" "$@")
  rewrite_train_command train_command

  while [[ "${attempt}" -le "${MAX_ATTEMPTS}" ]]; do
    if run_logged_command "${preprocess}" "${algorithm}" "${experiment_title}" "train" "1" "${train_command[@]}"; then
      if [[ -f "${summary_path}" ]]; then
        success=0
        break
      fi
    fi
    attempt=$((attempt + 1))
  done

  if [[ "${success}" -ne 0 && ! -f "${summary_path}" ]]; then
    local -a deploy_command=("${train_command[0]}" "deploy_layer/deploy.py" "${train_command[@]:4}")
    if run_logged_command "${preprocess}" "${algorithm}" "${experiment_title}__deploy_recovery" "recovery" "0" "${deploy_command[@]}"; then
      if [[ -f "${summary_path}" ]]; then
        success=0
      fi
    fi
  fi

  if [[ "${success}" -ne 0 ]]; then
    echo "${experiment_title}" >> "${FAILED_STEPS_FILE}"
  fi
  return "${success}"
}

run_aux_step() {
  local group_name="$1"
  local step_name="$2"
  shift 2
  if ! run_logged_command "test_layer" "${group_name}" "${step_name}" "analysis" "1" "$@"; then
    echo "${step_name}" >> "${FAILED_STEPS_FILE}"
    return 1
  fi
}

run_matrix_commands() {
  local failures=0

  run_train_step "fft" "cnn" "CNN_CWRU_FFT_5w5s5q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm cnn --dataset CWRU --preprocess FFT --ways 5 --shots 5 --query_shots 5 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --epochs 40 --batch_size 64 --finetune_epochs 15 --finetune_lr 0.0003 --test_task_num 50 --compression_finetune_iters 80 || failures=$((failures + 1))
  run_train_step "fft" "cnn" "CNN_CWRU_FFT_5w10s10q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm cnn --dataset CWRU --preprocess FFT --ways 5 --shots 10 --query_shots 10 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --epochs 40 --batch_size 64 --finetune_epochs 15 --finetune_lr 0.0003 --test_task_num 50 --compression_finetune_iters 80 || failures=$((failures + 1))
  run_train_step "fft" "cnn" "CNN_CWRU_FFT_5w15s15q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm cnn --dataset CWRU --preprocess FFT --ways 5 --shots 15 --query_shots 15 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --epochs 40 --batch_size 64 --finetune_epochs 15 --finetune_lr 0.0003 --test_task_num 50 --compression_finetune_iters 80 || failures=$((failures + 1))

  run_train_step "fft" "maml" "MAML_CWRU_FFT_5w5s_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm maml --dataset CWRU --preprocess FFT --ways 5 --shots 5 --query_shots 5 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 400 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))
  run_train_step "fft" "maml" "MAML_CWRU_FFT_5w10s_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm maml --dataset CWRU --preprocess FFT --ways 5 --shots 10 --query_shots 10 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 400 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))
  run_train_step "fft" "maml" "MAML_CWRU_FFT_5w15s_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm maml --dataset CWRU --preprocess FFT --ways 5 --shots 15 --query_shots 15 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 400 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))

  run_train_step "fft" "protonet" "ProtoNet_CWRU_FFT_5w5s5q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm protonet --dataset CWRU --preprocess FFT --ways 5 --shots 5 --query_shots 5 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 400 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))
  run_train_step "fft" "protonet" "ProtoNet_CWRU_FFT_5w10s10q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm protonet --dataset CWRU --preprocess FFT --ways 5 --shots 10 --query_shots 10 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 400 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))
  run_train_step "fft" "protonet" "ProtoNet_CWRU_FFT_5w15s15q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm protonet --dataset CWRU --preprocess FFT --ways 5 --shots 15 --query_shots 15 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 400 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))

  run_train_step "stft" "cnn" "CNN_CWRU_STFT_5w5s5q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm cnn --dataset CWRU --preprocess STFT --ways 5 --shots 5 --query_shots 5 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --epochs 30 --batch_size 64 --finetune_epochs 15 --finetune_lr 0.0003 --test_task_num 50 --compression_finetune_iters 80 || failures=$((failures + 1))
  run_train_step "stft" "cnn" "CNN_CWRU_STFT_5w10s10q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm cnn --dataset CWRU --preprocess STFT --ways 5 --shots 10 --query_shots 10 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --epochs 30 --batch_size 64 --finetune_epochs 15 --finetune_lr 0.0003 --test_task_num 50 --compression_finetune_iters 80 || failures=$((failures + 1))
  run_train_step "stft" "cnn" "CNN_CWRU_STFT_5w15s15q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm cnn --dataset CWRU --preprocess STFT --ways 5 --shots 15 --query_shots 15 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --epochs 30 --batch_size 64 --finetune_epochs 15 --finetune_lr 0.0003 --test_task_num 50 --compression_finetune_iters 80 || failures=$((failures + 1))

  run_train_step "stft" "maml" "MAML_CWRU_STFT_5w5s_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm maml --dataset CWRU --preprocess STFT --ways 5 --shots 5 --query_shots 5 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 80 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))
  run_train_step "stft" "maml" "MAML_CWRU_STFT_5w10s_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm maml --dataset CWRU --preprocess STFT --ways 5 --shots 10 --query_shots 10 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 80 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))
  run_train_step "stft" "maml" "MAML_CWRU_STFT_5w15s_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm maml --dataset CWRU --preprocess STFT --ways 5 --shots 15 --query_shots 15 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 80 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))

  run_train_step "stft" "protonet" "ProtoNet_CWRU_STFT_5w5s5q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm protonet --dataset CWRU --preprocess STFT --ways 5 --shots 5 --query_shots 5 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 80 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))
  run_train_step "stft" "protonet" "ProtoNet_CWRU_STFT_5w10s10q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm protonet --dataset CWRU --preprocess STFT --ways 5 --shots 10 --query_shots 10 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 80 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))
  run_train_step "stft" "protonet" "ProtoNet_CWRU_STFT_5w15s15q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm protonet --dataset CWRU --preprocess STFT --ways 5 --shots 15 --query_shots 15 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 80 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))

  run_train_step "wt" "cnn" "CNN_CWRU_WT_5w5s5q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm cnn --dataset CWRU --preprocess WT --ways 5 --shots 5 --query_shots 5 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --epochs 30 --batch_size 64 --finetune_epochs 15 --finetune_lr 0.0003 --test_task_num 50 --compression_finetune_iters 80 || failures=$((failures + 1))
  run_train_step "wt" "cnn" "CNN_CWRU_WT_5w10s10q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm cnn --dataset CWRU --preprocess WT --ways 5 --shots 10 --query_shots 10 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --epochs 30 --batch_size 64 --finetune_epochs 15 --finetune_lr 0.0003 --test_task_num 50 --compression_finetune_iters 80 || failures=$((failures + 1))
  run_train_step "wt" "cnn" "CNN_CWRU_WT_5w15s15q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm cnn --dataset CWRU --preprocess WT --ways 5 --shots 15 --query_shots 15 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --epochs 30 --batch_size 64 --finetune_epochs 15 --finetune_lr 0.0003 --test_task_num 50 --compression_finetune_iters 80 || failures=$((failures + 1))

  run_train_step "wt" "maml" "MAML_CWRU_WT_5w5s_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm maml --dataset CWRU --preprocess WT --ways 5 --shots 5 --query_shots 5 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 80 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))
  run_train_step "wt" "maml" "MAML_CWRU_WT_5w10s_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm maml --dataset CWRU --preprocess WT --ways 5 --shots 10 --query_shots 10 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 80 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))
  run_train_step "wt" "maml" "MAML_CWRU_WT_5w15s_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm maml --dataset CWRU --preprocess WT --ways 5 --shots 15 --query_shots 15 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 80 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))

  run_train_step "wt" "protonet" "ProtoNet_CWRU_WT_5w5s5q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm protonet --dataset CWRU --preprocess WT --ways 5 --shots 5 --query_shots 5 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 80 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))
  run_train_step "wt" "protonet" "ProtoNet_CWRU_WT_5w10s10q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm protonet --dataset CWRU --preprocess WT --ways 5 --shots 10 --query_shots 10 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 80 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))
  run_train_step "wt" "protonet" "ProtoNet_CWRU_WT_5w15s15q_source012_target3_labels0123456789" \
    "${PYTHON_BIN}" train.py --mode train --algorithm protonet --dataset CWRU --preprocess WT --ways 5 --shots 15 --query_shots 15 --train_domains 0,1,2 --test_domain 3 --fault_labels 0,1,2,3,4,5,6,7,8,9 --runtime_backend onnxruntime --enable_compression true --prune_ratio 0.4 --plot false --log true --cuda true --calibration_size 64 --enable_qat_recovery false --onnx_opset 17 --iters 80 --meta_batch_size 16 --train_task_num 100 --test_task_num 50 --compression_finetune_iters 80 --compression_meta_batch_size 8 || failures=$((failures + 1))

  run_aux_step "aggregation" "benchmark_rows_export" \
    "${PYTHON_BIN}" test_layer/result_aggregator.py \
    --summary_glob "${SUMMARY_GLOB}" \
    --output_format csv \
    --output_path "${BENCHMARK_ROWS_OUTPUT}" || failures=$((failures + 1))

  run_aux_step "aggregation" "thesis_tables_export" \
    "${PYTHON_BIN}" test_layer/thesis_tables.py \
    --summary_glob "${SUMMARY_GLOB}" \
    --output_dir "${TABLES_DIR}" \
    --allow_missing || failures=$((failures + 1))

  return "${failures}"
}

run_internal() {
  prepare_run_dirs
  RUN_STARTED_EPOCH="$(date +%s)"
  print_run_header
  "${PYTHON_BIN}" -c "import torch, learn2learn, onnxruntime, numpy" >/dev/null
  if run_matrix_commands; then
    echo
    echo "Run finished successfully."
    exit 0
  fi
  echo
  echo "Run finished with failures. See ${FAILED_STEPS_FILE}."
  exit 1
}

start_run() {
  run_internal
}

ACTION="${1:-restart}"
case "${ACTION}" in
  clean)
    clean_outputs
    ;;
  restart)
    clean_outputs
    start_run
    ;;
  run)
    start_run
    ;;
  *)
    echo "Usage: bash test_layer/run_controlled_overnight.sh [clean|run|restart]" >&2
    exit 1
    ;;
esac
