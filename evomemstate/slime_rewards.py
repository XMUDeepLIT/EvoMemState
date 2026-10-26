"""Slime custom reward for EvoMemState ``rl_training_dataed`` JSONL.

Wire with::

    --custom-rm-path evomemstate.slime_rewards.reward_from_spec
    --label-key reward_spec
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, TYPE_CHECKING

from .text_clean import strip_generated_state as strip_answer_for_reward

if TYPE_CHECKING:
    from slime.utils.types import Sample

_VERDICT_RE = re.compile(
    r"verdict\s*:\s*(SUPPORTED|REFUTED)\b",
    re.IGNORECASE,
)

_F1_SCORE: Callable[[str, str], Tuple[float, float, float]] | None = None


def _load_f1_score() -> Callable[[str, str], Tuple[float, float, float]]:
    """Load slime's f1.py without importing rm_hub (avoids optional deps)."""
    global _F1_SCORE
    if _F1_SCORE is not None:
        return _F1_SCORE

    candidates = [
        Path(__file__).resolve().parents[1] / "slime" / "slime" / "rollout" / "rm_hub" / "f1.py",
    ]
    for f1_path in candidates:
        if f1_path.is_file():
            spec = importlib.util.spec_from_file_location("evomemstate_slime_f1", f1_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _F1_SCORE = module.f1_score
                return _F1_SCORE

    from slime.rollout.rm_hub.f1 import f1_score as fallback

    _F1_SCORE = fallback
    return _F1_SCORE


def _get_spec(sample: "Sample") -> Dict[str, Any]:
    label = sample.label
    if isinstance(label, dict):
        return label
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    nested = metadata.get("reward_spec")
    if isinstance(nested, dict):
        return nested
    raise ValueError(
        "Expected reward_spec dict on sample.label (set --label-key reward_spec) "
        "or sample.metadata['reward_spec']"
    )


def parse_verdict(response: str) -> str | None:
    match = _VERDICT_RE.search(response or "")
    return match.group(1).upper() if match else None


def verdict_match_reward(response: str, target: str) -> float:
    answer = strip_answer_for_reward(response)
    pred = parse_verdict(answer)
    if pred is None:
        pred = parse_verdict(response)
    if pred is None:
        return 0.0
    return 1.0 if pred == str(target).upper() else 0.0


def token_f1_reward(response: str, target: str) -> float:
    answer = strip_answer_for_reward(response)
    f1, _, _ = _load_f1_score()(answer or response, target)
    return float(f1)


async def reward_from_spec(args, sample: "Sample", **kwargs) -> float:
    """Dispatch on ``reward_spec.type``: ``verdict_match`` or ``token_f1``."""
    spec = _get_spec(sample)
    rtype = str(spec.get("type") or "").strip()
    target = str(spec.get("target") or "")

    if rtype == "verdict_match":
        return verdict_match_reward(sample.response, target)
    if rtype == "token_f1":
        return token_f1_reward(sample.response, target)

    raise NotImplementedError(f"Unknown reward_spec.type: {rtype!r}")
