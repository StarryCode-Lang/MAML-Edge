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
RUN_SYSTEM_BENCHMARK="${RUN_SYSTEM_BENCHMARK:-1}"

case "${ACTION}" in
  clean|run|restart)
    bash "${SCRIPT_DIR}/run_controlled_overnight.sh" "${ACTION}"
    bash "${SCRIPT_DIR}/run_seed_extension.sh" "${ACTION}"
    bash "${SCRIPT_DIR}/run_domain_extension.sh" "${ACTION}"
    bash "${SCRIPT_DIR}/run_compression_ablation.sh" "${ACTION}"
    if [[ "${RUN_SYSTEM_BENCHMARK}" != "0" ]]; then
      bash "${SCRIPT_DIR}/run_system_benchmark.sh" "${ACTION}"
    fi
    if [[ "${ACTION}" != "clean" ]]; then
      "${PYTHON_BIN}" "${SCRIPT_DIR}/paper_tables.py" \
        --system_benchmark_path "logs/thesis_tables/paper_balanced/system_benchmark.json" \
        --output_dir "logs/thesis_tables/paper_balanced"
    fi
    ;;
  *)
    echo "Usage: bash test_layer/run_paper_suite.sh [clean|run|restart]" >&2
    exit 1
    ;;
esac
