#!/usr/bin/env bash
# Reproduce EvoMemState on MemSyco-Bench with a single method and answer model.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "${ROOT}/.env"
  set +a
fi
BENCH="${ROOT}/datasets/MemSyco-Bench"
PY="${PY:-python3}"
MODE="${MODE:-full}"
DRY_RUN="${DRY_RUN:-0}"
TASKS="${TASKS:-personalized_memory_use valid_memory_selection memory_evidence_conflict contextual_scope_control objective_fact_judgment}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/evomemstate_memsyco}"
TOP_K="${TOP_K:-10}"
WORKERS="${WORKERS:-8}"
PREBUILD_WORKERS="${PREBUILD_WORKERS:-8}"
EVAL_CURRENT_DATE="${EVAL_CURRENT_DATE:-2025-06-01}"
case "${MODE}" in
  smoke) LIMIT=3 ;;
  pilot) LIMIT=20 ;;
  full) LIMIT=0 ;;
  *) echo "MODE must be smoke, pilot, or full" >&2; exit 2 ;;
esac
if [ ! -d "${BENCH}/evaluation" ]; then
  echo "MemSyco-Bench is missing at ${BENCH}" >&2
  exit 2
fi
if [ "${DRY_RUN}" != "1" ]; then
  : "${MEMORY_API_KEY:?Set MEMORY_API_KEY}"
  : "${MEMORY_BASE_URL:?Set MEMORY_BASE_URL}"
  : "${GEN_B_MODEL:?Set GEN_B_MODEL}"
  : "${GEN_B_BASE_URL:?Set GEN_B_BASE_URL}"
  : "${GEN_B_API_KEY:?Set GEN_B_API_KEY}"
  : "${JUDGE_MODEL:?Set JUDGE_MODEL}"
  : "${REASONING_STATE_MODEL:?Set REASONING_STATE_MODEL to the trained policy}"
  : "${REASONING_STATE_BASE_URL:?Set REASONING_STATE_BASE_URL}"
  : "${REASONING_STATE_API_KEY:?Set REASONING_STATE_API_KEY}"
fi
JUDGE_BASE_URL="${JUDGE_BASE_URL:-${MEMORY_BASE_URL:-}}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${MEMORY_API_KEY:-}}"
export EVOMEMSTATE_DECOMPOSE=1 EVOMEMSTATE_REASONING_STATE_ENABLED=1
print_dry_run() {
  local arg hide=0
  printf 'DRY RUN:'
  for arg in "$@"; do
    if [ "${hide}" = "1" ]; then
      printf ' [REDACTED]'
      hide=0
      continue
    fi
    case "${arg}" in
      --api-key|--judge-api-key|--memory-api-key) hide=1 ;;
    esac
    printf ' %q' "${arg}"
  done
  printf '\n'
}
mkdir -p "${RUN_ROOT}"
PREBUILD=("${PY}" -m evomemstate.prebuild_graphs --tasks "$(echo "${TASKS}" | tr ' ' ',')" --limit "${LIMIT}" --workers "${PREBUILD_WORKERS}" --model "${MEMORY_LLM_MODEL:-deepseek-v4-flash}")
if [ "${DRY_RUN}" = "1" ]; then
  print_dry_run "${PREBUILD[@]}"
else
  (cd "${ROOT}" && "${PREBUILD[@]}")
fi
for task in ${TASKS}; do
  out="${RUN_ROOT}/${task}/results.json"
  mkdir -p "$(dirname "${out}")"
  if [ -f "${out}" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "skip existing ${out}"
    continue
  fi
  cmd=("${PY}" evaluation/run_task.py "${task}"
    --memory-method EvoMemState
    --memory-baseline-config "${BENCH}/baselines/configs/EvoMemState.json"
    --memory-api-key "${MEMORY_API_KEY:-}"
    --memory-base-url "${MEMORY_BASE_URL:-}"
    --memory-llm-model "${MEMORY_LLM_MODEL:-deepseek-v4-flash}"
    --memory-top-k "${TOP_K}"
    --memory-save-root "${RUN_ROOT}/${task}/memory"
    --model "${GEN_B_MODEL:-qwen3-8b}"
    --base-url "${GEN_B_BASE_URL:-}"
    --api-key "${GEN_B_API_KEY:-}"
    --judge-model "${JUDGE_MODEL:-deepseek-v4-flash}"
    --judge-base-url "${JUDGE_BASE_URL:-}"
    --judge-api-key "${JUDGE_API_KEY:-}"
    --workers "${WORKERS}"
    --current-date "${EVAL_CURRENT_DATE}"
    --output "${out}")
  if [ "${LIMIT}" -gt 0 ]; then cmd+=(--limit "${LIMIT}"); fi
  if [ "${DRY_RUN}" = "1" ]; then
    print_dry_run "${cmd[@]}"
  else
    (cd "${BENCH}" && "${cmd[@]}")
  fi
done
