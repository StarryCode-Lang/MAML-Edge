#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-restart}"
DOMAIN_SPLITS="${DOMAIN_SPLITS:-013:2 023:1 123:0}"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-42}"
TEST_TASK_NUM_OVERRIDE="${TEST_TASK_NUM_OVERRIDE:-200}"
DEPLOYMENT_LABEL_SAMPLING="${DEPLOYMENT_LABEL_SAMPLING:-random_per_episode}"

digits_to_csv() {
  local digits="$1"
  local result=""
  local index
  for ((index = 0; index < ${#digits}; index++)); do
    if [[ -n "${result}" ]]; then
      result+=","
    fi
    result+="${digits:index:1}"
  done
  printf '%s\n' "${result}"
}

run_domain_matrix() {
  local action="$1"
  local compact_train_domains="$2"
  local test_domain="$3"
  local train_domains_csv
  train_domains_csv="$(digits_to_csv "${compact_train_domains}")"
  local rel_root="paper_balanced/domain/source${compact_train_domains}_target${test_domain}"
  local run_root="${ROOT_DIR}/logs/paper_runs/balanced/domain/source${compact_train_domains}_target${test_domain}"
  RUN_NAMESPACE="${rel_root}" \
  RUN_ROOT="${run_root}" \
  CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints/${rel_root}" \
  COMPRESSION_OUTPUT_ROOT="${ROOT_DIR}/deploy_artifacts/${rel_root}" \
  TABLES_DIR="${ROOT_DIR}/logs/thesis_tables/${rel_root}" \
  BENCHMARK_ROWS_OUTPUT="${ROOT_DIR}/logs/thesis_tables/${rel_root}/benchmark_rows.csv" \
  SUMMARY_GLOB="deploy_artifacts/${rel_root}/*/compression_summary.json" \
  EXPERIMENT_SEED="${EXPERIMENT_SEED}" \
  TRAIN_DOMAINS_OVERRIDE="${train_domains_csv}" \
  TEST_DOMAIN_OVERRIDE="${test_domain}" \
  TEST_TASK_NUM_OVERRIDE="${TEST_TASK_NUM_OVERRIDE}" \
  DEPLOYMENT_LABEL_SAMPLING="${DEPLOYMENT_LABEL_SAMPLING}" \
  bash "${SCRIPT_DIR}/run_controlled_overnight.sh" "${action}"
}

case "${ACTION}" in
  clean|run|restart)
    for split in ${DOMAIN_SPLITS}; do
      compact_train_domains="${split%%:*}"
      test_domain="${split##*:}"
      run_domain_matrix "${ACTION}" "${compact_train_domains}" "${test_domain}"
    done
    ;;
  *)
    echo "Usage: bash test_layer/run_domain_extension.sh [clean|run|restart]" >&2
    exit 1
    ;;
esac
