#!/usr/bin/env bash
# Build EvoMemState memory graphs for LoCoMo-Refined, answer every question,
# and score the predictions with LoCoMo-Refined's own (unmodified) evaluator.
#
# Usage
#   ./scripts/run_locomo.sh                          # full run, all conversations/questions
#   LOCOMO_LIMIT=5 ./scripts/run_locomo.sh            # smoke test: first 5 questions
#   LOCOMO_SAMPLE_IDS=conv-26 ./scripts/run_locomo.sh # restrict to one conversation
#   LOCOMO_METRICS="f1 bleu" ./scripts/run_locomo.sh  # skip the llm judge for a quick check
#   DRY_RUN=1 ./scripts/run_locomo.sh                 # print commands, call nothing

set -u
set -o pipefail

FW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"

LOCOMO_ROOT_DEFAULT="$(cd "${FW_ROOT}/../LoCoMo_refined" 2>/dev/null && pwd || true)"
LOCOMO_BENCHMARK_ROOT="${LOCOMO_BENCHMARK_ROOT:-${LOCOMO_ROOT_DEFAULT}}"

if [ -z "${LOCOMO_BENCHMARK_ROOT}" ] || [ ! -d "${LOCOMO_BENCHMARK_ROOT}" ]; then
  echo "error: LoCoMo_refined not found. Set LOCOMO_BENCHMARK_ROOT explicitly." >&2
  exit 1
fi

if ! command -v "${PY}" >/dev/null 2>&1; then
  echo "error: Python interpreter not found: ${PY}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Load EvoMemState's .env. Shell-exported values win over the file, matching
# the runner's precedence rule.
# ---------------------------------------------------------------------------
ENV_FILE="${ENV_FILE:-${FW_ROOT}/.env}"
if [ -f "${ENV_FILE}" ]; then
  _OVERRIDABLE="MEMORY_API_KEY MEMORY_BASE_URL MEMORY_LLM_MODEL
    EMBEDDING_API_KEY EMBEDDING_MODEL EMBEDDING_BASE_URL EMBEDDING_DIMS
    EMBEDDING_WEIGHT EMBEDDING_ENABLED"
  for _v in ${_OVERRIDABLE}; do
    eval "_cur=\${${_v}-__unset__}"
    eval "__PRE_${_v}=\"\${_cur}\""
  done
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
  for _v in ${_OVERRIDABLE}; do
    eval "_pre=\${__PRE_${_v}}"
    if [ "${_pre}" != "__unset__" ]; then
      export "${_v}=${_pre}"
    fi
    unset "__PRE_${_v}"
  done
fi

if [ -z "${MEMORY_API_KEY:-}" ] || [ -z "${MEMORY_BASE_URL:-}" ]; then
  echo "error: MEMORY_API_KEY and MEMORY_BASE_URL must be set (see ${ENV_FILE})." >&2
  exit 1
fi

QUESTIONS_PATH="${LOCOMO_BENCHMARK_ROOT}/data/public/questions.jsonl"
CONVERSATIONS_PATH="${LOCOMO_BENCHMARK_ROOT}/data/public/conversations.jsonl"
PREDICTIONS_PATH="${LOCOMO_PREDICTIONS_PATH:-${LOCOMO_BENCHMARK_ROOT}/outputs/predictions.jsonl}"

RUN_ARGS=(
  --questions-path "${QUESTIONS_PATH}"
  --conversations-path "${CONVERSATIONS_PATH}"
  --output-path "${PREDICTIONS_PATH}"
)
[ -n "${LOCOMO_LIMIT:-}" ] && RUN_ARGS+=(--limit "${LOCOMO_LIMIT}")
[ -n "${LOCOMO_SAMPLE_IDS:-}" ] && RUN_ARGS+=(--sample-ids "${LOCOMO_SAMPLE_IDS}")
[ -n "${WORKERS:-}" ] && RUN_ARGS+=(--workers "${WORKERS}")

echo "[1/2] Building graphs + answering questions..." >&2
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "${PY} -m evomemstate.run_locomo ${RUN_ARGS[*]}"
else
  ( cd "${FW_ROOT}" && "${PY}" -m evomemstate.run_locomo "${RUN_ARGS[@]}" )
fi
STATUS=$?
if [ "${STATUS}" -ne 0 ]; then
  echo "error: run_locomo.py failed (exit ${STATUS})" >&2
  exit "${STATUS}"
fi

echo "[2/2] Scoring predictions with LoCoMo-Refined's evaluator..." >&2
# Always pass --metrics explicitly (never leave EVAL_ARGS empty): run_eval.sh
# does "${EVAL_ARGS[@]}" unguarded, which macOS's bash 3.2 rejects under set -u
# when the array has zero elements.
EVAL_ARGS=(--metrics)
if [ -n "${LOCOMO_METRICS:-}" ]; then
  for _m in ${LOCOMO_METRICS}; do
    EVAL_ARGS+=("${_m}")
  done
else
  EVAL_ARGS+=(f1 bleu llm)
fi
[ -n "${LOCOMO_EVAL_CONCURRENCY:-}" ] && EVAL_ARGS+=(--concurrency "${LOCOMO_EVAL_CONCURRENCY}")

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "LOCOMO_PREDICTIONS_PATH=${PREDICTIONS_PATH} ${LOCOMO_BENCHMARK_ROOT}/scripts/run_eval.sh ${EVAL_ARGS[*]}"
else
  ( cd "${LOCOMO_BENCHMARK_ROOT}" && LOCOMO_PREDICTIONS_PATH="${PREDICTIONS_PATH}" ./scripts/run_eval.sh "${EVAL_ARGS[@]}" )
fi
