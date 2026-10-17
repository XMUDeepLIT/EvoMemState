"""Pre-build EvoMemState relational memory graphs for MemTrapBench.

Usage
-----
    EVOMEMSTATE_GRAPH_ROOT=outputs/memtrap_graphs_ds_flash \\
      python -m evomemstate.prebuild_memtrap --datasets all --workers 8

    # smoke
    python -m evomemstate.prebuild_memtrap --datasets poison --limit 3 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split("#", 1)[0]
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    _load_dotenv(REPO_ROOT / ".env")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from evomemstate.memtrap_adapter import (  # noqa: WPS433
        DATASETS,
        _build_settings,
        _graph_root,
        _load_or_build_graph,
        graph_dir_for,
        iter_all_samples,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset names, or 'all'.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Rows per dataset; 0 = all.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--api-key", default=os.environ.get("MEMORY_API_KEY", ""))
    parser.add_argument("--base-url", default=os.environ.get("MEMORY_BASE_URL", ""))
    parser.add_argument(
        "--model",
        default=os.environ.get("MEMORY_LLM_MODEL", "") or "deepseek-v4-flash",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many graphs are missing without calling any API.",
    )
    parser.add_argument(
        "--graph-root",
        default="",
        help="Override EVOMEMSTATE_GRAPH_ROOT (default: outputs/memtrap_graphs_v1).",
    )
    args = parser.parse_args()

    if args.graph_root.strip():
        os.environ["EVOMEMSTATE_GRAPH_ROOT"] = args.graph_root.strip()
    # Sensible default for the flash build tree (matches memsyco naming).
    if not os.environ.get("EVOMEMSTATE_GRAPH_ROOT", "").strip():
        os.environ["EVOMEMSTATE_GRAPH_ROOT"] = str(
            REPO_ROOT / "outputs" / "memtrap_graphs_ds_flash"
        )

    datasets = (
        list(DATASETS)
        if args.datasets.strip() == "all"
        else [t.strip() for t in args.datasets.split(",") if t.strip()]
    )
    unknown = [t for t in datasets if t not in DATASETS]
    if unknown:
        print("Unknown dataset(s): %s" % ", ".join(unknown))
        return 2

    eval_config = SimpleNamespace(
        method="EvoMemState",
        api_key=args.api_key or None,
        base_url=args.base_url or None,
        llm_model=args.model,
    )
    settings = _build_settings(eval_config)
    root = _graph_root()

    samples = iter_all_samples(datasets, limit=args.limit)
    pending: List[Dict[str, Any]] = []
    done_existing = 0
    for row in samples:
        sk = row["sample_key"]
        if (graph_dir_for(sk) / "ontology.json").exists():
            done_existing += 1
            continue
        pending.append(row)

    print(
        "model=%s chunk_mode=%s slot_policy=%s version=%s"
        % (
            settings["model"],
            settings["chunk_mode"],
            settings["slot_policy"],
            settings["version"],
        )
    )
    print("graph root: %s" % root)
    print(
        "samples: %d | already built: %d | to build: %d"
        % (len(samples), done_existing, len(pending))
    )

    if args.dry_run:
        return 0
    if not pending:
        print("Nothing to build.")
        return 0
    if not (args.api_key and args.base_url):
        print("ERROR: need --api-key/--base-url (or MEMORY_API_KEY/MEMORY_BASE_URL).")
        return 2

    lock = threading.Lock()
    state = {"ok": 0, "fail": 0}
    failures: List[str] = []
    started = time.time()
    n_pending = len(pending)
    if n_pending <= 20:
        progress_every = 1
    elif n_pending <= 100:
        progress_every = 5
    elif n_pending <= 500:
        progress_every = 10
    else:
        progress_every = 25
    workers = max(1, args.workers)
    print(
        "building %d graphs with %d workers (progress every %d) ..."
        % (n_pending, workers, progress_every),
        flush=True,
    )

    def _one(row: Dict[str, Any]) -> Tuple[str, bool, str]:
        sk = row["sample_key"]
        try:
            _load_or_build_graph(
                row.get("context_history"),
                eval_config,
                sample_key_str=sk,
            )
            return sk, True, ""
        except Exception as exc:  # noqa: BLE001
            return sk, False, "%s: %s\n%s" % (type(exc).__name__, exc, traceback.format_exc())

    def _report(sk: str, *, force: bool = False) -> None:
        total = state["ok"] + state["fail"]
        if not force and total % progress_every != 0 and total != n_pending:
            return
        elapsed = max(1e-6, time.time() - started)
        rate = total / elapsed
        eta = (n_pending - total) / rate if rate > 0 else 0.0
        print(
            "[%d/%d] last=%s ok=%d fail=%d rate=%.2f/s eta=%.0fs"
            % (total, n_pending, sk, state["ok"], state["fail"], rate, eta),
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, row): row["sample_key"] for row in pending}
        for fut in as_completed(futs):
            sk, ok, err = fut.result()
            with lock:
                if ok:
                    state["ok"] += 1
                else:
                    state["fail"] += 1
                    failures.append("%s | %s" % (sk, err.splitlines()[0] if err else "?"))
                    print("[fail] %s\n%s" % (sk, err), flush=True)
                _report(sk)

    print(
        "DONE ok=%d fail=%d elapsed=%.1fs root=%s"
        % (state["ok"], state["fail"], time.time() - started, root),
        flush=True,
    )
    if failures:
        print("failures (%d):" % len(failures))
        for line in failures[:20]:
            print("  %s" % line)
    return 0 if state["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
