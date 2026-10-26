"""Tests for evomemstate.slime_step_info."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("slime.utils.types")

from evomemstate.slime_step_info import sample_to_record, write_step_info_samples
from slime.utils.types import Sample


def test_sample_to_record_and_write(tmp_path: Path):
    args = SimpleNamespace(
        step_info_path=str(tmp_path),
        eval_interval=20,
        rollout_batch_size=32,
        n_samples_per_prompt=8,
        global_batch_size=256,
        wandb_always_use_train_step=False,
        reward_key=None,
    )
    sample = Sample(
        prompt="Memory context:\n...\n\nClaim: test",
        response="Verdict: REFUTED\nExplanation: no.",
        label={"type": "verdict_match", "target": "REFUTED"},
        reward=1.0,
        metadata={"source": "scifact", "task": "verdict"},
    )
    record = sample_to_record(sample, rollout_id=3, step=3, args=args)
    assert record["reward"] == 1.0
    assert record["reward_spec"]["type"] == "verdict_match"

    write_step_info_samples(args, [sample], rollout_id=3, evaluation=False)
    out = tmp_path / "expdata" / "step_003.jsonl"
    assert out.is_file()
    line = json.loads(out.read_text(encoding="utf-8").strip())
    assert line["response"].startswith("Verdict:")


def test_aggregate_reasoning_state_metrics():
    from evomemstate.slime_step_info import _aggregate_reasoning_state_metrics

    s1 = Sample(
        prompt="p",
        response="h1",
        metadata={
            "source": "scifact",
            "reasoning_state_reward_debug": {
                "new_acc": 1.0,
                "delta_acc": 0.0,
                "reward": 1.0,
                "score_with": 1.0,
                "score_without": 1.0,
            },
        },
    )
    s2 = Sample(
        prompt="p",
        response="h2",
        metadata={
            "source": "personamem_v2",
            "reasoning_state_reward_debug": {
                "new_acc": 0.2,
                "delta_acc": 0.1,
                "reward": 0.3,
                "score_with": 0.2,
                "score_without": 0.1,
            },
        },
    )
    m = _aggregate_reasoning_state_metrics([s1, s2])
    assert abs(m["reasoning_state/mean_delta"] - 0.05) < 1e-9
    assert abs(m["reasoning_state/mean_new_acc"] - 0.6) < 1e-9
    assert abs(m["reasoning_state/mean_reward"] - 0.65) < 1e-9
    assert m["reasoning_state/delta_pos_frac"] == 0.5
    assert m["reasoning_state/delta_zero_frac"] == 0.5
    assert abs(m["reasoning_state/mean_delta/scifact"] - 0.0) < 1e-9
    assert abs(m["reasoning_state/mean_delta/personamem_v2"] - 0.1) < 1e-9


def test_log_reasoning_state_metrics_tensorboard(monkeypatch):
    import sys
    import types

    from evomemstate.slime_step_info import _log_reasoning_state_metrics

    logged = {}

    def fake_log(args, metrics, step_key):
        logged["metrics"] = metrics
        logged["step_key"] = step_key

    logging_utils = types.SimpleNamespace(log=fake_log)
    observability = types.SimpleNamespace(logging_utils=logging_utils)
    slime_pkg = types.ModuleType("slime")
    slime_pkg.observability = observability  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slime", slime_pkg)
    monkeypatch.setitem(sys.modules, "slime.observability", observability)
    monkeypatch.setitem(sys.modules, "slime.observability.logging_utils", logging_utils)

    args = SimpleNamespace(
        use_tensorboard=True,
        rollout_batch_size=64,
        n_samples_per_prompt=8,
        global_batch_size=512,
        wandb_always_use_train_step=False,
    )
    sample = Sample(
        prompt="p",
        response="h",
        metadata={"reasoning_state_reward_debug": {"new_acc": 0.5, "delta_acc": 0.2, "reward": 0.7}},
    )
    _log_reasoning_state_metrics(args, [sample], rollout_id=10, evaluation=False)
    assert logged["step_key"] == "rollout/step"
    assert abs(logged["metrics"]["reasoning_state/mean_delta"] - 0.2) < 1e-9
    assert logged["metrics"]["rollout/step"] == 10
