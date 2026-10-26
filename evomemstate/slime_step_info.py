"""Save per-step Reasoning State policy rollout debug JSONL (train / eval).

Layout (mirrors search ``reward_info``):

    {step_info_path}/expdata/step_{step:03d}.jsonl      # train
    {step_info_path}/valdata/step_{step:03d}_valStep_{eval_interval:03d}.jsonl  # eval

Wire via::

    --step-info-path /path/to/step_info
    --rollout-all-samples-process-path evomemstate.slime_step_info.process_rollout_samples
    --rollout-function-path evomemstate.slime_step_info.generate_rollout_with_step_info
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from slime.rollout.base_types import RolloutFnEvalOutput
    from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _compute_rollout_step(args, rollout_id: int) -> int:
    try:
        from slime.observability.metric_utils import compute_rollout_step

        return int(compute_rollout_step(args, rollout_id))
    except ImportError:
        if getattr(args, "wandb_always_use_train_step", False):
            gbs = int(getattr(args, "global_batch_size", 1) or 1)
            batch = int(getattr(args, "rollout_batch_size", 1) or 1)
            nsp = int(getattr(args, "n_samples_per_prompt", 1) or 1)
            return rollout_id * batch * nsp // gbs
        return int(rollout_id)


def _step_info_root(args) -> Optional[Path]:
    raw = getattr(args, "step_info_path", None) or os.environ.get("EVOMEMSTATE_STEP_INFO_PATH")
    if not raw:
        return None
    return Path(str(raw))


def _reward_value(args, sample: Sample) -> Any:
    if sample.reward is None:
        return None
    try:
        return sample.get_reward_value(args)
    except Exception:
        return sample.reward


def sample_to_record(
    sample: Sample,
    *,
    rollout_id: int,
    step: int,
    evaluation: bool = False,
    dataset_name: str = "",
    args=None,
) -> Dict[str, Any]:
    label = sample.label
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    reward_spec = label if isinstance(label, dict) else metadata.get("reward_spec")
    record: Dict[str, Any] = {
        "rollout_id": rollout_id,
        "step": step,
        "evaluation": evaluation,
        "dataset": dataset_name,
        "index": sample.index,
        "prompt": sample.prompt,
        "response": sample.response,
        "reward": _reward_value(args, sample) if args is not None else sample.reward,
        "status": str(sample.status),
        "label": label,
        "reward_spec": reward_spec,
        "response_length": sample.response_length,
        "metadata": metadata,
    }
    # Flatten common fields from aligned JSONL if present in metadata.
    for key in ("source", "task", "condition", "source_group_id", "id", "question", "memory_body"):
        if key in metadata:
            record[key] = metadata[key]
    debug = metadata.get("reasoning_state_reward_debug")
    if isinstance(debug, dict):
        record["score_with"] = debug.get("score_with")
        record["score_without"] = debug.get("score_without")
        record["new_acc"] = debug.get("new_acc")
        record["delta_acc"] = debug.get("delta_acc")
        record["penalty_acc"] = debug.get("penalty_acc")
        record["reasoning_state"] = debug.get("reasoning_state")
        record["reasoning_state_thought"] = debug.get("reasoning_state_thought")
        record["think_mode"] = debug.get("think_mode")
        record["raw_reward"] = debug.get("raw_reward")
        record["raw_new_acc"] = debug.get("raw_new_acc")
        record["raw_delta_acc"] = debug.get("raw_delta_acc")
        record["format_violations"] = debug.get("format_violations") or []
        record["answer_with"] = debug.get("answer_with")
        record["answer_without"] = debug.get("answer_without")
        record["judge_contains_answer"] = debug.get("judge_contains_answer")
        record["judge_answer_tag"] = debug.get("judge_answer_tag")
        record["judge_parse_ok"] = debug.get("judge_parse_ok")
        record["judge_penalize"] = debug.get("judge_penalize")
    return record


def _step_file_path(args, rollout_id: int, evaluation: bool) -> Optional[Path]:
    root = _step_info_root(args)
    if root is None:
        return None
    step = _compute_rollout_step(args, rollout_id)
    if evaluation:
        eval_interval = int(getattr(args, "eval_interval", 0) or 0)
        subdir = root / "valdata"
        filename = "step_%s_valStep_%s.jsonl" % (str(step).zfill(3), str(eval_interval).zfill(3))
    else:
        subdir = root / "expdata"
        filename = "step_%s.jsonl" % str(step).zfill(3)
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / filename


def _flatten_groups(groups: Iterable[Any]) -> List["Sample"]:
    from slime.utils.types import Sample

    samples: List["Sample"] = []
    for group in groups:
        if isinstance(group, list) and group and isinstance(group[0], list):
            for nested in group:
                if isinstance(nested, list):
                    samples.extend(nested)
                else:
                    samples.append(nested)
        elif isinstance(group, list):
            samples.extend(group)
        elif isinstance(group, Sample):
            samples.append(group)
    return samples


def _reasoning_state_component_values(sample: Sample) -> Dict[str, Optional[float]]:
    """Read gated reasoning_state reward components from sample metadata."""
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    debug = meta.get("reasoning_state_reward_debug")
    if not isinstance(debug, dict):
        return {}
    out: Dict[str, Optional[float]] = {}
    for key in ("new_acc", "delta_acc", "reward", "score_with", "score_without"):
        val = debug.get(key)
        if isinstance(val, (int, float)):
            out[key] = float(val)
    return out


def _aggregate_reasoning_state_metrics(samples: List[Sample]) -> Dict[str, float]:
    """Batch means / fractions for reasoning_state RL components (train + eval)."""
    if not samples:
        return {}

    by_key: Dict[str, List[float]] = {
        "new_acc": [],
        "delta_acc": [],
        "reward": [],
        "score_with": [],
        "score_without": [],
    }
    by_source_delta: Dict[str, List[float]] = {}
    delta_pos = delta_zero = n_with_delta = 0

    for sample in samples:
        vals = _reasoning_state_component_values(sample)
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
            meta = sample.metadata if isinstance(sample.metadata, dict) else {}
            src = str(meta.get("source") or "unknown")
            by_source_delta.setdefault(src, []).append(d)

    if not any(by_key.values()):
        return {}

    def mean(xs: List[float]) -> float:
        return sum(xs) / len(xs)

    metrics: Dict[str, float] = {}
    prefix = "reasoning_state"
    if by_key["delta_acc"]:
        metrics[f"{prefix}/mean_delta"] = mean(by_key["delta_acc"])
    if by_key["new_acc"]:
        metrics[f"{prefix}/mean_new_acc"] = mean(by_key["new_acc"])
    if by_key["reward"]:
        metrics[f"{prefix}/mean_reward"] = mean(by_key["reward"])
    if by_key["score_with"]:
        metrics[f"{prefix}/mean_score_with"] = mean(by_key["score_with"])
    if by_key["score_without"]:
        metrics[f"{prefix}/mean_score_without"] = mean(by_key["score_without"])
    if n_with_delta:
        metrics[f"{prefix}/delta_pos_frac"] = delta_pos / n_with_delta
        metrics[f"{prefix}/delta_zero_frac"] = delta_zero / n_with_delta
    for src, deltas in by_source_delta.items():
        safe = src.replace("/", "_")
        metrics[f"{prefix}/mean_delta/{safe}"] = mean(deltas)
    return metrics


def _log_reasoning_state_metrics(args, samples: List[Sample], *, rollout_id: int, evaluation: bool) -> None:
    if not getattr(args, "use_tensorboard", False):
        return
    metrics = _aggregate_reasoning_state_metrics(samples)
    if not metrics:
        return
    try:
        from slime.observability.logging_utils import log as slime_log
    except ImportError:
        return

    step = _compute_rollout_step(args, rollout_id)
    if evaluation:
        metrics = {("eval/" + k if not k.startswith("eval/") else k): v for k, v in metrics.items()}
        metrics["eval/step"] = step
        slime_log(args, metrics, step_key="eval/step")
    else:
        metrics["rollout/step"] = step
        slime_log(args, metrics, step_key="rollout/step")
    logger.info(
        "reasoning_state_tb rollout=%s eval=%s step=%s mean_delta=%s mean_new=%s",
        rollout_id,
        evaluation,
        step,
        metrics.get("reasoning_state/mean_delta", metrics.get("eval/reasoning_state/mean_delta")),
        metrics.get("reasoning_state/mean_new_acc", metrics.get("eval/reasoning_state/mean_new_acc")),
    )


def write_step_info_samples(
    args,
    samples: List[Sample],
    *,
    rollout_id: int,
    evaluation: bool = False,
    dataset_name: str = "",
) -> None:
    path = _step_file_path(args, rollout_id, evaluation)
    if path is None or not samples:
        return
    step = _compute_rollout_step(args, rollout_id)
    with path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            record = sample_to_record(
                sample,
                rollout_id=rollout_id,
                step=step,
                evaluation=evaluation,
                dataset_name=dataset_name,
                args=args,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d step_info records to %s", len(samples), path)
    _log_reasoning_state_metrics(args, samples, rollout_id=rollout_id, evaluation=evaluation)


def process_rollout_samples(args, all_groups: List[Any], data_source) -> None:
    """``--rollout-all-samples-process-path`` hook after train rollout."""
    rollout_id = 0
    try:
        from slime.rollout.sample_hooks import _current_rollout_id

        if _current_rollout_id is not None:
            rollout_id = _current_rollout_id
    except ImportError:
        pass
    samples = _flatten_groups(all_groups)
    write_step_info_samples(args, samples, rollout_id=rollout_id, evaluation=False)


def _save_eval_output(args, rollout_id: int, output: "RolloutFnEvalOutput") -> None:
    from slime.rollout.base_types import RolloutFnEvalOutput

    data = output.data if isinstance(output, RolloutFnEvalOutput) else {}
    for dataset_name, payload in data.items():
        samples = payload.get("samples") or []
        write_step_info_samples(
            args,
            samples,
            rollout_id=rollout_id,
            evaluation=True,
            dataset_name=str(dataset_name),
        )


def generate_rollout_with_step_info(args, rollout_id: int, data_source, evaluation: bool = False):
    """Wrapper around default slime rollout; saves eval step_info after eval.

    slime's ``eval_rollout`` asserts ``not args.group_rm``. ReasoningState RL uses
    ``--group-rm`` for faster batched answer scoring in training, so we
    temporarily clear the flag for evaluation only.
    """
    from slime.rollout.sglang_rollout import generate_rollout as base_generate_rollout

    saved_group_rm = bool(getattr(args, "group_rm", False))
    if evaluation and saved_group_rm:
        args.group_rm = False
    try:
        output = base_generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
    finally:
        if evaluation and saved_group_rm:
            args.group_rm = True
    if evaluation:
        from slime.rollout.base_types import RolloutFnEvalOutput

        if isinstance(output, RolloutFnEvalOutput):
            _save_eval_output(args, rollout_id, output)
        elif isinstance(output, dict):
            _save_eval_output(args, rollout_id, RolloutFnEvalOutput(data=output))
    return output
