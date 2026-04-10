#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-restart}"
SEED_LIST="${SEED_LIST:-43 44}"
TEST_TASK_NUM_OVERRIDE="${TEST_TASK_NUM_OVERRIDE:-200}"
DEPLOYMENT_LABEL_SAMPLING="${DEPLOYMENT_LABEL_SAMPLING:-random_per_episode}"

run_seed_matrix() {
  local action="$1"
  local seed="$2"
  local rel_root="paper_balanced/seed/seed${seed}"
  local run_root="${ROOT_DIR}/logs/paper_runs/balanced/seed/seed${seed}"
  RUN_NAMESPACE="${rel_root}" \
  RUN_ROOT="${run_root}" \
  CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints/${rel_root}" \
  COMPRESSION_OUTPUT_ROOT="${ROOT_DIR}/deploy_artifacts/${rel_root}" \
  TABLES_DIR="${ROOT_DIR}/logs/thesis_tables/${rel_root}" \
  BENCHMARK_ROWS_OUTPUT="${ROOT_DIR}/logs/thesis_tables/${rel_root}/benchmark_rows.csv" \
  SUMMARY_GLOB="deploy_artifacts/${rel_root}/*/compression_summary.json" \
  EXPERIMENT_SEED="${seed}" \
  TEST_TASK_NUM_OVERRIDE="${TEST_TASK_NUM_OVERRIDE}" \
  DEPLOYMENT_LABEL_SAMPLING="${DEPLOYMENT_LABEL_SAMPLING}" \
  bash "${SCRIPT_DIR}/run_controlled_overnight.sh" "${action}"
}

case "${ACTION}" in
  clean|run|restart)
    for seed in ${SEED_LIST}; do
      run_seed_matrix "${ACTION}" "${seed}"
    done
    ;;
  *)
    echo "Usage: bash test_layer/run_seed_extension.sh [clean|run|restart]" >&2
    exit 1
    ;;
esac
