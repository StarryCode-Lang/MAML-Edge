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

RUN_ROOT="${ROOT_DIR}/logs/overnight_runs/controlled"
LATEST_DIR="${RUN_ROOT}/latest"
LOG_ROOT="${LATEST_DIR}/logs"
TABLES_DIR="${ROOT_DIR}/logs/thesis_tables/controlled"
CURRENT_STEP_FILE="${RUN_ROOT}/current_step.txt"
STATUS_FILE="${RUN_ROOT}/step_status.tsv"
FAILED_STEPS_FILE="${RUN_ROOT}/failed_steps.txt"
COMMAND_PLAN_FILE="${RUN_ROOT}/command_plan.sh"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
PLAN_ONLY=0

clean_outputs() {
  pkill -f "python .*train.py" >/dev/null 2>&1 || true
  pkill -f "python .*deploy.py" >/dev/null 2>&1 || true
  pkill -f "bash .*run_controlled_overnight.sh" >/dev/null 2>&1 || true
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

record_command_plan_header() {
  cat <<EOF
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="\${ROOT_DIR:-$(printf '%q' "${ROOT_DIR}")}"
PYTHON_BIN="\${PYTHON_BIN:-$(printf '%q' "${PYTHON_BIN}")}"

cd "\${ROOT_DIR}"

EOF
}

print_command() {
  local step_name="$1"
  shift
  printf '# %s\n' "${step_name}"
  printf '%q ' "$@"
  printf '\n\n'
}

run_logged_command() {
  local preprocess="$1"
  local algorithm="$2"
  local step_name="$3"
  shift 3

  if [[ "${PLAN_ONLY}" == "1" ]]; then
    print_command "${step_name}" "$@"
    return 0
  fi

  local log_path="${LOG_ROOT}/${preprocess}/${algorithm}/${step_name}.log"
  mkdir -p "$(dirname "${log_path}")"
  echo "${step_name}" > "${CURRENT_STEP_FILE}"
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
  ) >> "${log_path}" 2>&1
  local rc=$?
  set -e

  {
    echo
    echo "[return_code] ${rc}"
  } >> "${log_path}"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "${preprocess}" "${algorithm}" "${step_name}" "${rc}" >> "${STATUS_FILE}"
  return "${rc}"
}

run_train_step() {
  local preprocess="$1"
  local algorithm="$2"
  local experiment_title="$3"
  shift 3

  local summary_path="${ROOT_DIR}/deploy_artifacts/${experiment_title}/compression_summary.json"
  local attempt=1
  local success=1

  while [[ "${attempt}" -le "${MAX_ATTEMPTS}" ]]; do
    if run_logged_command "${preprocess}" "${algorithm}" "${experiment_title}" "$@"; then
      if [[ -f "${summary_path}" ]]; then
        success=0
        break
      fi
    fi
    attempt=$((attempt + 1))
  done

  if [[ "${success}" -ne 0 && ! -f "${summary_path}" ]]; then
    local -a train_command=("$@")
    local -a deploy_command=("${train_command[0]}" "deploy_layer/deploy.py" "${train_command[@]:4}")
    if run_logged_command "${preprocess}" "${algorithm}" "${experiment_title}__deploy_recovery" "${deploy_command[@]}"; then
      if [[ -f "${summary_path}" ]]; then
        success=0
      fi
    fi
  fi

  if [[ "${PLAN_ONLY}" == "0" && "${success}" -ne 0 ]]; then
    echo "${experiment_title}" >> "${FAILED_STEPS_FILE}"
  fi
  return "${success}"
}

run_aux_step() {
  local group_name="$1"
  local step_name="$2"
  shift 2
  if ! run_logged_command "test_layer" "${group_name}" "${step_name}" "$@"; then
    if [[ "${PLAN_ONLY}" == "0" ]]; then
      echo "${step_name}" >> "${FAILED_STEPS_FILE}"
    fi
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
    --summary_glob deploy_artifacts/*/compression_summary.json \
    --output_format csv \
    --output_path logs/thesis_tables/controlled/benchmark_rows.csv || failures=$((failures + 1))

  run_aux_step "aggregation" "thesis_tables_export" \
    "${PYTHON_BIN}" test_layer/thesis_tables.py \
    --summary_glob deploy_artifacts/*/compression_summary.json \
    --output_dir logs/thesis_tables/controlled \
    --allow_missing || failures=$((failures + 1))

  return "${failures}"
}

write_command_plan() {
  record_command_plan_header > "${COMMAND_PLAN_FILE}"
  PLAN_ONLY=1
  run_matrix_commands >> "${COMMAND_PLAN_FILE}"
  PLAN_ONLY=0
  chmod +x "${COMMAND_PLAN_FILE}"
}

run_internal() {
  prepare_run_dirs
  write_command_plan
  "${PYTHON_BIN}" -c "import torch, learn2learn, onnxruntime, numpy" >/dev/null
  if run_matrix_commands; then
    exit 0
  fi
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
  print)
    mkdir -p "${RUN_ROOT}"
    write_command_plan
    cat "${COMMAND_PLAN_FILE}"
    ;;
  restart)
    clean_outputs
    start_run
    ;;
  run)
    start_run
    ;;
  *)
    echo "Usage: bash test_layer/run_controlled_overnight.sh [clean|print|run|restart]" >&2
    exit 1
    ;;
esac
