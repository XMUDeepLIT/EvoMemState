#!/usr/bin/env bash
# Build relational memory, generate a Reasoning State, and score LoCoMo answers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "${ROOT}/.env"
  set +a
fi
if [ "${DRY_RUN:-0}" != "1" ]; then
  : "${REASONING_STATE_MODEL:?Set REASONING_STATE_MODEL to the trained policy}"
  : "${REASONING_STATE_BASE_URL:?Set REASONING_STATE_BASE_URL}"
  : "${REASONING_STATE_API_KEY:?Set REASONING_STATE_API_KEY}"
fi
export EVOMEMSTATE_REASONING_STATE_ENABLED=1
exec bash "${ROOT}/scripts/run_locomo.sh" "$@"
