"""Pre-build EvoMemState relational memory graphs for MemSyco-Bench.

Why this exists
---------------
Graph construction costs API calls, while retrieval is CPU-side. This command
prepares EvoMemState graphs in parallel so benchmark evaluation can reuse them.
The cache key excludes the method name because the graph belongs to dialogue
content.

Cache-key correctness
---------------------
The key must be computed from the SAME string the evaluator will later pass as
`prior_dialogue`. That string is not `row["dialogue"]`: the evaluator converts
canonical rows via `to_eval_row` and then calls
`format_memory_prior_dialogue_from_row`, which stops at the query turn and
prefixes each turn with "User: " / "Assistant: ". This module imports both of
those functions rather than reimplementing them, so the keys cannot drift.

Usage
-----
    python -m evomemstate.prebuild_graphs --tasks all --limit 0 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = REPO_ROOT / "datasets" / "MemSyco-Bench"

TASKS = (
    "personalized_memory_use",
    "valid_memory_selection",
    "memory_evidence_conflict",
    "contextual_scope_control",
    "objective_fact_judgment",
)


def _ensure_paths() -> None:
    for path in (REPO_ROOT, BENCH_ROOT, BENCH_ROOT / "evaluation"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from .env, without overriding what is already set.

    Nothing in the repo calls load_dotenv(). scripts/run_evomemstate_memsyco.sh sources
    .env itself, so the wrapper works -- but running this module directly, which
    the docstring above tells you to do, would abort on a missing
    MEMORY_API_KEY even though the key is sitting in .env. Load it here so both
    entry points behave the same.

    Already-exported variables win, so a one-off override still works.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        # .env carries trailing comments like `WORKERS=30   # 评测并发`; keeping
        # them would turn the value into a string int() chokes on.
        value = value.split("#", 1)[0]
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _load_rows(task: str, limit: int) -> List[Dict[str, Any]]:
    path = BENCH_ROOT / "data" / ("%s.jsonl" % task)
    if not path.exists():
        raise FileNotFoundError("Missing dataset file: %s" % path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _dialogues(task: str, limit: int) -> List[Tuple[str, str]]:
    """Return (sample_id, prior_dialogue) exactly as the evaluator will see it."""
    _ensure_paths()
    from _dataset_compat import to_eval_row
    from _objective_base import format_memory_prior_dialogue_from_row

    out: List[Tuple[str, str]] = []
    for row in _load_rows(task, limit):
        eval_row = to_eval_row(row)
        dialogue = format_memory_prior_dialogue_from_row(eval_row)
        if dialogue.strip():
            out.append((str(row.get("id", "")), dialogue))
    return out


def main() -> int:
    # Must precede argparse: the --api-key/--base-url defaults below read
    # os.environ at parser-construction time, so loading .env afterwards would
    # be too late to supply them.
    _load_dotenv(REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="all", help="Comma-separated task names, or 'all'.")
    parser.add_argument("--limit", type=int, default=0, help="Rows per task; 0 = all.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--api-key", default=os.environ.get("MEMORY_API_KEY", ""))
    parser.add_argument("--base-url", default=os.environ.get("MEMORY_BASE_URL", ""))
    parser.add_argument(
        "--model",
        default=os.environ.get("MEMORY_LLM_MODEL", "") or "deepseek-v4-flash",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many graphs are missing without calling any API.",
    )
    args = parser.parse_args()

    tasks = TASKS if args.tasks.strip() == "all" else tuple(
        t.strip() for t in args.tasks.split(",") if t.strip()
    )
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        print("Unknown task(s): %s" % ", ".join(unknown))
        return 2

    _ensure_paths()
    from baselines.base import BaselineEvalConfig
    from evomemstate.memsyco_adapter import _graph_key, _graph_root, _build_settings, _load_or_build_graph

    eval_config = BaselineEvalConfig(
        method="EvoMemState",
        top_k=args.top_k,
        api_key=args.api_key or None,
        base_url=args.base_url or None,
        llm_model=args.model,
    )
    settings = _build_settings(eval_config)
    root = _graph_root()

    # Deduplicate across tasks: the same dialogue can appear in more than one
    # task file, and it only needs one graph.
    pending: List[Tuple[str, str, str]] = []
    seen_keys: set = set()
    done_existing = 0
    for task in tasks:
        for sample_id, dialogue in _dialogues(task, args.limit):
            key = _graph_key(dialogue, settings)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if (root / key / "ontology.json").exists():
                done_existing += 1
                continue
            pending.append((task, sample_id, dialogue))

    print("model=%s chunk_mode=%s slot_policy=%s" % (
        settings["model"], settings["chunk_mode"], settings["slot_policy"]))
    print("graph root: %s" % root)
    print("unique dialogues: %d | already built: %d | to build: %d"
          % (len(seen_keys), done_existing, len(pending)))

    if args.dry_run:
        return 0
    if not pending:
        print("Nothing to build.")
        return 0
    if not (args.api_key and args.base_url):
        print("ERROR: need --api-key/--base-url (or MEMORY_API_KEY/MEMORY_BASE_URL).")
        return 2

    lock = threading.Lock()
    state = {"ok": 0, "fail": 0, "in_flight": 0}
    failures: List[str] = []
    started = time.time()
    n_pending = len(pending)
    # Smoke/pilot used to look "stuck": progress only printed every 25 graphs, so
    # a 15-graph smoke run stayed silent until the very end.
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

    def _report(task: str, sample_id: str, *, force: bool = False) -> None:
        total = state["ok"] + state["fail"]
        if not force and total % progress_every != 0 and total != n_pending:
            return
        rate = total / max(1e-6, time.time() - started)
        remaining = (n_pending - total) / rate if rate > 0 else 0
        print(
            "  [%d/%d] ok=%d fail=%d in_flight=%d | last=%s/%s | ~%.1f min left"
            % (
                total,
                n_pending,
                state["ok"],
                state["fail"],
                state["in_flight"],
                task,
                sample_id,
                remaining / 60,
            ),
            flush=True,
        )

    def _work(item: Tuple[str, str, str]) -> None:
        task, sample_id, dialogue = item
        with lock:
            state["in_flight"] += 1
        try:
            _load_or_build_graph(dialogue, eval_config)
            with lock:
                state["ok"] += 1
                state["in_flight"] -= 1
                _report(task, sample_id)
        except Exception as exc:  # noqa: BLE001 - one bad row must not stop the sweep
            with lock:
                state["fail"] += 1
                state["in_flight"] -= 1
                failures.append("%s/%s: %s: %s" % (task, sample_id, type(exc).__name__, exc))
                _report(task, sample_id, force=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_work, item) for item in pending]
        for future in as_completed(futures):
            future.result()

    elapsed = (time.time() - started) / 60
    print("done in %.1f min: ok=%d fail=%d" % (elapsed, state["ok"], state["fail"]), flush=True)
    if failures:
        print("first failures:", flush=True)
        for line in failures[:10]:
            print("  " + line, flush=True)
    # A few failures are tolerable (those samples fall back to empty context),
    # but a wholesale failure means the run should not proceed. Use an absolute
    # floor so a tiny leftover rebuild (e.g. 1/2) cannot abort a nearly-complete
    # cache under the relative 5% rule.
    if state["fail"] and state["fail"] > max(5, int(len(pending) * 0.05)):
        print("ERROR: >5%% of graphs failed to build.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
