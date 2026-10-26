#!/usr/bin/env python3
"""Backfill TensorBoard reasoning_state metrics from step_info JSONL.

Reads train/eval step_info written by ``evomemstate.slime_step_info`` and logs
the same scalars as online training (``reasoning_state/mean_delta``, …).

Example::

    python3 scripts/backfill_reasoning_state_tb.py \\
      --step-info outputs/slime_saved/evomemstate-reasoning-state-qwen3-4b/step_info \\
      --tb-dir outputs/slime_saved/tb_logs/evomemstate-reasoning_state-qwen3-0.6B-v3/reasoning_state_offline
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_STEP_RE = re.compile(r"step_(\d+)\.jsonl$")
_VAL_RE = re.compile(r"step_(\d+)_valStep_\d+\.jsonl$")


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def aggregate_records(records: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Same metric names as ``slime_step_info._aggregate_reasoning_state_metrics``."""
    by_key: Dict[str, List[float]] = {
        "new_acc": [],
        "delta_acc": [],
        "reward": [],
        "score_with": [],
        "score_without": [],
    }
    by_source_delta: Dict[str, List[float]] = defaultdict(list)
    delta_pos = delta_zero = n_with_delta = 0

    for rec in records:
        # Flattened fields preferred; fall back to nested debug.
        debug = rec.get("metadata", {}).get("reasoning_state_reward_debug") if isinstance(rec.get("metadata"), dict) else None
        if not isinstance(debug, dict):
            debug = {}
        vals: Dict[str, float] = {}
        for key in by_key:
            raw = rec.get(key)
            if raw is None and key in debug:
                raw = debug.get(key)
            if isinstance(raw, (int, float)):
                vals[key] = float(raw)
        if not vals:
            continue
        for key, bucket in by_key.items():
            if key in vals:
                bucket.append(vals[key])
        if "delta_acc" in vals:
            d = vals["delta_acc"]
            n_with_delta += 1
            if d > 1e-6:
                delta_pos += 1
            if abs(d) < 1e-6:
                delta_zero += 1
            src = str(rec.get("source") or (debug.get("source") if isinstance(debug, dict) else None) or "unknown")
            by_source_delta[src].append(d)

    if not any(by_key.values()):
        return {}

    metrics: Dict[str, float] = {}
    prefix = "reasoning_state"
    if by_key["delta_acc"]:
        metrics[f"{prefix}/mean_delta"] = _mean(by_key["delta_acc"])
    if by_key["new_acc"]:
        metrics[f"{prefix}/mean_new_acc"] = _mean(by_key["new_acc"])
    if by_key["reward"]:
        metrics[f"{prefix}/mean_reward"] = _mean(by_key["reward"])
    if by_key["score_with"]:
        metrics[f"{prefix}/mean_score_with"] = _mean(by_key["score_with"])
    if by_key["score_without"]:
        metrics[f"{prefix}/mean_score_without"] = _mean(by_key["score_without"])
    if n_with_delta:
        metrics[f"{prefix}/delta_pos_frac"] = delta_pos / n_with_delta
        metrics[f"{prefix}/delta_zero_frac"] = delta_zero / n_with_delta
    for src, deltas in by_source_delta.items():
        safe = src.replace("/", "_")
        metrics[f"{prefix}/mean_delta/{safe}"] = _mean(deltas)
    return metrics


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def collect_train(step_info: Path) -> List[Tuple[int, Dict[str, float]]]:
    exp = step_info / "expdata"
    out: List[Tuple[int, Dict[str, float]]] = []
    if not exp.is_dir():
        return out
    for path in sorted(exp.glob("step_*.jsonl")):
        m = _STEP_RE.search(path.name)
        if not m:
            continue
        step = int(m.group(1))
        metrics = aggregate_records(load_jsonl(path))
        if metrics:
            out.append((step, metrics))
    return out


def collect_eval(step_info: Path) -> List[Tuple[int, Dict[str, float]]]:
    val = step_info / "valdata"
    out: List[Tuple[int, Dict[str, float]]] = []
    if not val.is_dir():
        return out
    # Prefer one file per train step (latest valStep if duplicates).
    by_step: Dict[int, Path] = {}
    for path in sorted(val.glob("step_*_valStep_*.jsonl")):
        m = _VAL_RE.search(path.name)
        if not m:
            continue
        by_step[int(m.group(1))] = path
    for step, path in sorted(by_step.items()):
        metrics = aggregate_records(load_jsonl(path))
        if metrics:
            # Match online eval prefix.
            metrics = {("eval/" + k if not k.startswith("eval/") else k): v for k, v in metrics.items()}
            out.append((step, metrics))
    return out


def write_tb(tb_dir: Path, train: List[Tuple[int, Dict[str, float]]], evals: List[Tuple[int, Dict[str, float]]]) -> None:
    from torch.utils.tensorboard import SummaryWriter

    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))
    for step, metrics in train:
        for key, value in metrics.items():
            writer.add_scalar(key, value, step)
    for step, metrics in evals:
        for key, value in metrics.items():
            writer.add_scalar(key, value, step)
    writer.flush()
    writer.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step-info",
        type=Path,
        default=Path("outputs/slime_saved/evomemstate-reasoning-state-qwen3-4b/step_info"),
        help="Directory containing expdata/ and valdata/",
    )
    parser.add_argument(
        "--tb-dir",
        type=Path,
        default=None,
        help="TensorBoard log dir (default: <experiment>/tb_logs sibling under tb_logs/<name>/reasoning_state_offline)",
    )
    args = parser.parse_args(argv)

    step_info = args.step_info.resolve()
    if args.tb_dir is None:
        # Prefer existing experiment TB root: .../tb_logs/<exp>/reasoning_state_offline
        exp_name = step_info.parent.name
        tb_dir = Path("outputs/slime_saved/tb_logs") / exp_name / "reasoning_state_offline"
    else:
        tb_dir = args.tb_dir.resolve()

    train = collect_train(step_info)
    evals = collect_eval(step_info)
    if not train and not evals:
        print("No metrics found under", step_info)
        return 1

    write_tb(tb_dir, train, evals)
    print("Wrote %d train steps + %d eval steps → %s" % (len(train), len(evals), tb_dir))
    if train:
        s0, m0 = train[0]
        s1, m1 = train[-1]
        print(
            "  train[%d]: mean_delta=%.4f mean_new=%.4f | train[%d]: mean_delta=%.4f mean_new=%.4f"
            % (
                s0,
                m0.get("reasoning_state/mean_delta", float("nan")),
                m0.get("reasoning_state/mean_new_acc", float("nan")),
                s1,
                m1.get("reasoning_state/mean_delta", float("nan")),
                m1.get("reasoning_state/mean_new_acc", float("nan")),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
