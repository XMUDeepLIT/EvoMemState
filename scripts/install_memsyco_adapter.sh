#!/usr/bin/env bash
# Register EvoMemState with the external MemSyco-Bench harness.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${ROOT}/datasets/MemSyco-Bench"
if [ ! -f "${BENCH}/baselines/registry.py" ]; then
  echo "Clone MemSyco-Bench into ${BENCH} first" >&2
  exit 2
fi
cat > "${BENCH}/baselines/evomemstate.py" <<'PY_ADAPTER'
"""MemSyco-Bench bridge for the EvoMemState method."""
from evomemstate.memsyco_adapter import METHOD, METHODS, build_context
PY_ADAPTER
mkdir -p "${BENCH}/baselines/configs"
cp "${ROOT}/config/EvoMemState.json" "${BENCH}/baselines/configs/EvoMemState.json"
python3 - "${BENCH}/baselines/registry.py" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
import_line = "from . import evomemstate\n"
entry = '_BUILDERS["EvoMemState"] = evomemstate.build_context\n'
if import_line not in source:
    anchor = "from .base import BaselineContext, BaselineEvalConfig\n"
    if anchor not in source:
        raise SystemExit("Unsupported MemSyco registry layout: missing base import")
    source = source.replace(anchor, anchor + import_line, 1)
if entry not in source:
    anchor = "BASELINE_METHODS = tuple(_BUILDERS)"
    if anchor not in source:
        raise SystemExit("Unsupported MemSyco registry layout: missing method table")
    source = source.replace(anchor, entry + "\n" + anchor, 1)
path.write_text(source, encoding="utf-8")
print("EvoMemState adapter registered")
PY
