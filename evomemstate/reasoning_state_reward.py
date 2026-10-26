"""Reasoning State policy reward (legacy ``reasoning_state`` API and metric names).

The reward compares a frozen answer model with and without the generated state;
the remaining ``reasoning_state`` identifiers are retained for training compatibility.

Per sample::

    new_acc  = score(with_reasoning_state)                 # logged only
    delta    = score(with_reasoning_state) - score(no_reasoning_state)  # format-gated (0 on bad format)
    penalty  = -REASONING_STATE_ANSWER_JUDGE_PENALTY if judge says reasoning_state
               contains answer statements else 0   (default 0.1)
             + -REASONING_STATE_FORMAT_EMPTY_PENALTY if reasoning_state is empty after strip
               (default 0.05; avoids silence beating mild negative delta)
    reward   = delta + penalty                  # logged scalar

Advantages (``post_process_reasoning_state_rewards``). Default ``REASONING_STATE_ADVANTAGE=grpo``::

    adv = group_whiten(delta) + group_whiten(penalty)

``REASONING_STATE_ADVANTAGE=baseline_anchored`` does **not** subtract the group mean.
``delta`` is already ``S_i - S_0`` (shared no-reasoning_state baseline). Standard GRPO
centering cancels ``S_0`` and reduces to comparing reasoning_states with each other::

    ( (S_i - S_0) - mean_j(S_j - S_0) ) / std = (S_i - mean S) / std

Baseline-anchored advantage keeps the baseline in the gradient::

    adv_i = (S_i - S_0) / (std({S_j - S_0}) + eps)
          + penalty_i / (std(penalty) + eps)

Penalty uses the same no-center scale so a constant format/judge penalty is
not wiped out by group centering. If a group's std is ~0, that channel is
passed through raw (denominator 1) instead of dividing by ``eps``.

Wire with::

    --custom-rm-path evomemstate.reasoning_state_reward.reward_from_reasoning_state
    --custom-reward-post-process-path evomemstate.reasoning_state_reward.post_process_reasoning_state_rewards
    --group-rm
    --label-key reward_spec

Sample.metadata must contain ``memory_body``, ``question``, ``answer_system``,
``answer_user_template`` (and optionally ``has_supersede``). Frozen answer model
is reached via OpenAI-compatible HTTP (default GEN_B / ANSWER_* env).

Performance notes
-----------------
- Uses ``AsyncOpenAI`` + semaphore (``REASONING_STATE_ANSWER_CONCURRENCY``, default 64).
- Within one sample, with-reasoning_state / no-reasoning_state are fetched concurrently.
- Identical requests (esp. shared no-reasoning_state baselines in a GRPO group) are
  single-flighted + disk-cached so 8 samples for one prompt share one no_reasoning_state call.
- Optional LLM judge (``REASONING_STATE_ANSWER_JUDGE=1``, default on): same frozen 8B scores
  whether the reasoning_state contains answer statements; ``yes`` adds a fixed penalty
  (default -0.1), whitened separately from delta.
- Empty / unusable reasoning_states (``empty`` format violation) skip answer calls, zero
  delta, and apply ``REASONING_STATE_FORMAT_EMPTY_PENALTY`` (also via ``penalty_acc``).
- Think-mode (``REASONING_STATE_THINK_MODE=1`` or ``metadata.think_mode``): extract
  ``<reasoning_state>`` via ``extract_reasoning_state_and_thought`` before format gate / scoring;
  ``think_unclosed`` / ``missing_reasoning_state_tag`` are format violations.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from slime.utils.types import Sample

from .reasoning_state_policy import extract_reasoning_state_and_thought
from .reasoning_state_format import apply_reasoning_state_format_gate
from .prompting import REASONING_STATE_INTRO, compose_reasoning_state
from .text_clean import strip_generated_state as strip_answer_for_reward
from .slime_rewards import token_f1_reward, verdict_match_reward

logger = logging.getLogger(__name__)

_CACHE_LOCK = threading.Lock()
_MEMORY_CACHE: Dict[str, str] = {}

_ASYNC_CLIENT = None
_ASYNC_CLIENT_LOCK = asyncio.Lock()
_SEM: Optional[asyncio.Semaphore] = None
_INFLIGHT: Dict[str, asyncio.Future] = {}
_INFLIGHT_LOCK = asyncio.Lock()


def _env(name: str, *fallbacks: str, default: str = "") -> str:
    for key in (name,) + fallbacks:
        value = os.environ.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def _answer_client_config() -> Dict[str, str]:
    return {
        "model": _env("ANSWER_MODEL", "REASONING_STATE_ANSWER_MODEL", "GEN_B_MODEL", default="qwen3-8b"),
        "base_url": _env(
            "ANSWER_BASE_URL",
            "REASONING_STATE_ANSWER_BASE_URL",
            "GEN_B_BASE_URL",
            default="http://127.0.0.1:8800/v1",
        ),
        "api_key": _env(
            "ANSWER_API_KEY",
            "REASONING_STATE_ANSWER_API_KEY",
            "GEN_B_API_KEY",
            default="EMPTY",
        ),
    }


def _concurrency() -> int:
    return max(1, int(_env("REASONING_STATE_ANSWER_CONCURRENCY", default="64")))


def _max_tokens() -> int:
    return max(32, int(_env("REASONING_STATE_ANSWER_MAX_TOKENS", default="512")))


def _cache_dir() -> Path:
    raw = _env("REASONING_STATE_ANSWER_CACHE_DIR", default="")
    if raw:
        path = Path(raw)
    else:
        path = Path(__file__).resolve().parents[1] / "outputs" / "reasoning_state_answer_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_cached(key: str) -> Optional[str]:
    with _CACHE_LOCK:
        if key in _MEMORY_CACHE:
            return _MEMORY_CACHE[key]
    path = _cache_dir() / ("%s.json" % key)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        text = str(data.get("text") or "")
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    with _CACHE_LOCK:
        _MEMORY_CACHE[key] = text
    return text


def _store_cached(key: str, text: str) -> None:
    with _CACHE_LOCK:
        _MEMORY_CACHE[key] = text
    path = _cache_dir() / ("%s.json" % key)
    try:
        path.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.warning("reasoning_state_answer cache write failed: %s", exc)


def _get_semaphore() -> asyncio.Semaphore:
    global _SEM
    if _SEM is None:
        _SEM = asyncio.Semaphore(_concurrency())
    return _SEM


async def _get_async_client():
    global _ASYNC_CLIENT
    async with _ASYNC_CLIENT_LOCK:
        if _ASYNC_CLIENT is not None:
            return _ASYNC_CLIENT
        from openai import AsyncOpenAI

        cfg = _answer_client_config()
        # Raise connection pool so many concurrent reward calls do not queue on httpx.
        _ASYNC_CLIENT = AsyncOpenAI(
            api_key=cfg["api_key"] or "EMPTY",
            base_url=cfg["base_url"],
            max_retries=2,
            timeout=120.0,
        )
        return _ASYNC_CLIENT


def _completion_cache_key(model: str, messages: list, extra_body: Optional[dict] = None) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "extra_body": extra_body or {},
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


async def call_frozen_answer_async(
    messages: list,
    *,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    extra_body: Optional[dict] = None,
) -> str:
    """Async chat completion with disk cache + in-flight singleflight."""
    cfg = _answer_client_config()
    key = _completion_cache_key(cfg["model"], messages, extra_body)
    cached = _load_cached(key)
    if cached is not None:
        return cached

    async with _INFLIGHT_LOCK:
        existing = _INFLIGHT.get(key)
        if existing is not None:
            waiter = existing
        else:
            loop = asyncio.get_running_loop()
            waiter = loop.create_future()
            _INFLIGHT[key] = waiter
            existing = None

    if existing is not None:
        return await asyncio.shield(waiter)

    tokens = _max_tokens() if max_tokens is None else max_tokens
    try:
        async with _get_semaphore():
            client = await _get_async_client()
            kwargs: Dict[str, Any] = {
                "model": cfg["model"],
                "messages": messages,
                "temperature": temperature,
                "max_tokens": tokens,
            }
            if extra_body:
                kwargs["extra_body"] = extra_body
            response = await client.chat.completions.create(**kwargs)
        text = (response.choices[0].message.content or "") if response.choices else ""
        _store_cached(key, text)
        if not waiter.done():
            waiter.set_result(text)
        return text
    except Exception as exc:
        if not waiter.done():
            waiter.set_exception(exc)
        raise
    finally:
        async with _INFLIGHT_LOCK:
            if _INFLIGHT.get(key) is waiter:
                _INFLIGHT.pop(key, None)


def call_frozen_answer(
    messages: list,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    extra_body: Optional[dict] = None,
) -> str:
    """Sync wrapper (tests / offline). Prefer ``call_frozen_answer_async`` in training."""
    cfg = _answer_client_config()
    cached_key = _completion_cache_key(cfg["model"], messages, extra_body)
    cached = _load_cached(cached_key)
    if cached is not None:
        return cached
    from openai import OpenAI

    client = OpenAI(api_key=cfg["api_key"] or "EMPTY", base_url=cfg["base_url"])
    kwargs: Dict[str, Any] = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(**kwargs)
    text = (response.choices[0].message.content or "") if response.choices else ""
    _store_cached(cached_key, text)
    return text


def _meta(sample: "Sample") -> Dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return metadata


def _reward_spec(sample: "Sample") -> Dict[str, Any]:
    label = sample.label
    if isinstance(label, dict) and label.get("type"):
        return label
    nested = _meta(sample).get("reward_spec")
    if isinstance(nested, dict):
        return nested
    raise ValueError("Missing reward_spec on sample.label / metadata")


def _field(sample: "Sample", key: str, default: str = "") -> str:
    meta = _meta(sample)
    if key in meta and meta[key] is not None:
        return str(meta[key])
    return default


def build_answer_messages(
    *,
    context: str,
    question: str,
    answer_system: str,
    answer_user_template: str,
) -> list:
    system = (answer_system or "").replace("{context}", context).replace("{question}", question)
    user = (
        (answer_user_template or "Memory context:\n{context}\n\nQuestion: {question}")
        .replace("{context}", context)
        .replace("{question}", question)
    )
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def context_with_reasoning_state(memory_body: str, reasoning_state: str, *, has_supersede: bool = False) -> str:
    reasoning_state = compose_reasoning_state(
        strip_answer_for_reward(reasoning_state) or (reasoning_state or "").strip(),
        include_latest=bool(has_supersede),
        include_scope=True,
    )
    body = (memory_body or "").strip()
    if not reasoning_state:
        return body
    if body.startswith(REASONING_STATE_INTRO):
        return reasoning_state
    return "%s\n\n%s" % (body, reasoning_state)


def score_answer(response: str, spec: Dict[str, Any]) -> float:
    rtype = str(spec.get("type") or "").strip()
    target = str(spec.get("target") or "")
    if rtype == "verdict_match":
        return float(verdict_match_reward(response, target))
    if rtype == "token_f1":
        return float(token_f1_reward(response, target))
    raise NotImplementedError("Unknown reward_spec.type: %r" % rtype)


def _pair_messages(sample: "Sample", reasoning_state: str) -> Tuple[list, list, Dict[str, Any]]:
    meta = _meta(sample)
    memory_body = _field(sample, "memory_body") or str(meta.get("memory_body") or "")
    question = _field(sample, "question") or str(meta.get("question") or "")
    answer_system = _field(sample, "answer_system") or str(meta.get("answer_system") or "")
    answer_user_template = _field(sample, "answer_user_template") or str(
        meta.get("answer_user_template") or ""
    )
    has_supersede = bool(meta.get("has_supersede"))
    spec = _reward_spec(sample)

    ctx_with = context_with_reasoning_state(memory_body, reasoning_state, has_supersede=has_supersede)
    ctx_without = memory_body.strip()
    msgs_with = build_answer_messages(
        context=ctx_with,
        question=question,
        answer_system=answer_system,
        answer_user_template=answer_user_template,
    )
    msgs_without = build_answer_messages(
        context=ctx_without,
        question=question,
        answer_system=answer_system,
        answer_user_template=answer_user_template,
    )
    return msgs_with, msgs_without, spec


def _compose_reward(score_with: float, score_without: float) -> Tuple[float, float, float]:
    """Return ``(new_acc, delta_acc, reward)`` with ``reward = delta_acc`` only."""
    new_acc = float(score_with)
    delta_acc = float(score_with) - float(score_without)
    return new_acc, delta_acc, delta_acc


def score_pair(
    sample: "Sample",
    reasoning_state: str,
    *,
    answer_fn=None,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """Sync path for unit tests; training uses ``score_pair_async``.

    Returns ``(reward, score_with, score_without, debug)`` where
    ``reward = delta_acc = score_with - score_without``.
    """
    answer_fn = answer_fn or call_frozen_answer
    msgs_with, msgs_without, spec = _pair_messages(sample, reasoning_state)
    ans_with = answer_fn(msgs_with)
    ans_without = answer_fn(msgs_without)
    score_with = score_answer(ans_with, spec)
    score_without = score_answer(ans_without, spec)
    new_acc, delta_acc, reward = _compose_reward(score_with, score_without)
    debug = {
        "score_with": score_with,
        "score_without": score_without,
        "new_acc": new_acc,
        "delta_acc": delta_acc,
        "reward": reward,
        "reasoning_state": strip_answer_for_reward(reasoning_state) or reasoning_state,
        "answer_with": ans_with,
        "answer_without": ans_without,
    }
    return reward, score_with, score_without, debug


async def score_pair_async(
    sample: "Sample",
    reasoning_state: str,
) -> Tuple[float, float, float, Dict[str, Any]]:
    msgs_with, msgs_without, spec = _pair_messages(sample, reasoning_state)
    ans_with, ans_without = await asyncio.gather(
        call_frozen_answer_async(msgs_with),
        call_frozen_answer_async(msgs_without),
    )
    score_with = score_answer(ans_with, spec)
    score_without = score_answer(ans_without, spec)
    new_acc, delta_acc, reward = _compose_reward(score_with, score_without)
    debug = {
        "score_with": score_with,
        "score_without": score_without,
        "new_acc": new_acc,
        "delta_acc": delta_acc,
        "reward": reward,
        "reasoning_state": strip_answer_for_reward(reasoning_state) or reasoning_state,
        "answer_with": ans_with,
        "answer_without": ans_without,
    }
    return reward, score_with, score_without, debug


def _judge_penalty_magnitude() -> float:
    """Absolute penalty applied when judge says reasoning_state contains an answer."""
    try:
        return max(0.0, float(_env("REASONING_STATE_ANSWER_JUDGE_PENALTY", default="0.1")))
    except ValueError:
        return 0.1


def _attach_debug(sample: "Sample", debug: Dict[str, Any]) -> None:
    ans_with = debug.get("answer_with")
    ans_without = debug.get("answer_without")
    payload = {
        "score_with": debug.get("score_with"),
        "score_without": debug.get("score_without"),
        "new_acc": debug.get("new_acc"),
        "delta_acc": debug.get("delta_acc"),
        "penalty_acc": debug.get("penalty_acc", 0.0),
        "reward": debug.get("reward"),
        "raw_reward": debug.get("raw_reward", debug.get("reward")),
        "raw_new_acc": debug.get("raw_new_acc", debug.get("new_acc")),
        "raw_delta_acc": debug.get("raw_delta_acc", debug.get("delta_acc")),
        "format_violations": debug.get("format_violations") or [],
        "reasoning_state": debug.get("reasoning_state"),
        # Frozen-answer replies (strip CoT for readability in step_info).
        "answer_with": strip_answer_for_reward(ans_with) if ans_with else ans_with,
        "answer_without": strip_answer_for_reward(ans_without) if ans_without else ans_without,
        "answer_with_raw": ans_with,
        "answer_without_raw": ans_without,
        "judge_contains_answer": debug.get("judge_contains_answer"),
        "judge_answer_tag": debug.get("judge_answer_tag"),
        "judge_parse_ok": debug.get("judge_parse_ok"),
        "judge_penalize": debug.get("judge_penalize"),
        "judge_raw": debug.get("judge_raw"),
    }
    if isinstance(sample.metadata, dict):
        sample.metadata = dict(sample.metadata)
        sample.metadata["reasoning_state_reward_debug"] = payload
    else:
        sample.metadata = {"reasoning_state_reward_debug": payload}


def _delta_from_sample(sample: "Sample") -> float:
    """Read format-gated delta (not including answer-leak penalty)."""
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    debug = meta.get("reasoning_state_reward_debug") if isinstance(meta, dict) else None
    if isinstance(debug, dict) and debug.get("delta_acc") is not None:
        return float(debug["delta_acc"])
    reward = sample.reward
    if isinstance(reward, dict):
        if reward.get("delta_acc") is not None:
            return float(reward["delta_acc"])
        return float(reward.get("reward", 0.0) or 0.0)
    return float(reward or 0.0)


def _penalty_from_sample(sample: "Sample") -> float:
    """Read answer-leak penalty (0 or negative)."""
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    debug = meta.get("reasoning_state_reward_debug") if isinstance(meta, dict) else None
    if isinstance(debug, dict) and debug.get("penalty_acc") is not None:
        return float(debug["penalty_acc"])
    return 0.0


def _reward_groups(values: Sequence[float], args) -> List[List[float]]:
    """Split a flat reward list into GRPO prompt groups (slime's reshape)."""
    flat = list(values)
    n = int(getattr(args, "n_samples_per_prompt", 1) or 1)
    rbs = int(getattr(args, "rollout_batch_size", 0) or 0)
    if n > 1 and len(flat) == n * rbs:
        return [flat[i : i + n] for i in range(0, len(flat), n)]
    if n > 1 and len(flat) % n == 0:
        return [flat[i : i + n] for i in range(0, len(flat), n)]
    return [flat]


def _group_std(group: Sequence[float]) -> float:
    """Unbiased std, matching ``torch.std`` default (ddof=1)."""
    if len(group) <= 1:
        return 0.0
    mean = sum(group) / len(group)
    var = sum((x - mean) * (x - mean) for x in group) / (len(group) - 1)
    return var ** 0.5


def _advantage_mode() -> str:
    """``grpo`` (center) or ``baseline_anchored`` (keep S_0).

    Set ``REASONING_STATE_ADVANTAGE=baseline_anchored``. Aliases: ``anchored``, ``baseline``.
    """
    raw = _env("REASONING_STATE_ADVANTAGE", default="grpo").strip().lower().replace("-", "_")
    if raw in {"baseline_anchored", "anchored", "baseline", "delta"}:
        return "baseline_anchored"
    return "grpo"


def _scale_group(group: Sequence[float], *, center: bool, normalize_std: bool) -> List[float]:
    """Scale one prompt-group.

    ``center=True`` is standard GRPO: ``(x - mean) / (std + eps)``.
    ``center=False`` is baseline-anchored: ``x / (std + eps)``. A near-zero
    std passes the raw values through so a shared non-zero delta vs ``S_0``
    stays non-zero, without dividing by ``eps``.
    """
    if not group:
        return []
    if center:
        mean = sum(group) / len(group)
        values = [x - mean for x in group]
        if not normalize_std or len(values) <= 1:
            return values
        std = _group_std(group)
        if std < 1e-8:
            return [0.0 for _ in values]
        return [x / (std + 1e-6) for x in values]

    if not normalize_std or len(group) <= 1:
        return [float(x) for x in group]
    std = _group_std(group)
    if std < 1e-8:
        return [float(x) for x in group]
    return [float(x) / (std + 1e-6) for x in group]


def _group_scale(values: Sequence[float], args, *, center: bool) -> List[float]:
    normalize_std = bool(getattr(args, "grpo_std_normalization", True))
    out: List[float] = []
    for group in _reward_groups(values, args):
        out.extend(_scale_group(group, center=center, normalize_std=normalize_std))
    return out


def _group_whiten(values: Sequence[float], args) -> List[float]:
    """Per-group (x - mean) [/ std], matching slime's default GRPO reshape."""
    return _group_scale(values, args, center=True)


def post_process_reasoning_state_rewards(args, samples: Sequence["Sample"]):
    """Turn per-sample delta/penalty into GRPO advantages.

    Returns ``(raw_rewards, advantages)``.

    Default (``REASONING_STATE_ADVANTAGE=grpo``)::

        raw_rewards[i] = delta_i + penalty_i
        advantages[i]  = whiten(delta)_i + whiten(penalty)_i

    ``REASONING_STATE_ADVANTAGE=baseline_anchored`` skips the group-mean subtraction, so
    ``delta = S - S_0`` is not cancelled::

        advantages[i] = scale(delta)_i + scale(penalty)_i
        scale(x)_i    = x_i / (std(x) + eps)
    """
    deltas = [_delta_from_sample(sample) for sample in samples]
    penalties = [_penalty_from_sample(sample) for sample in samples]
    center = _advantage_mode() != "baseline_anchored"
    adv_delta = _group_scale(deltas, args, center=center)
    adv_penalty = _group_scale(penalties, args, center=center)
    advantages = [d + p for d, p in zip(adv_delta, adv_penalty)]
    raw_rewards = [d + p for d, p in zip(deltas, penalties)]
    logger.info(
        "reasoning_state_advantage mode=%s n=%d mean_delta=%.4f mean_adv=%.4f",
        "grpo" if center else "baseline_anchored",
        len(samples),
        (sum(deltas) / len(deltas)) if deltas else 0.0,
        (sum(advantages) / len(advantages)) if advantages else 0.0,
    )
    return raw_rewards, advantages


def _think_mode_enabled(sample: "Sample") -> bool:
    """REASONING_STATE_THINK_MODE env overrides; else use sample.metadata['think_mode']."""
    raw = _env("REASONING_STATE_THINK_MODE", default="").lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(_meta(sample).get("think_mode"))


def _rollout_reasoning_state_for_reward(sample: "Sample") -> Tuple[str, str, List[str]]:
    """Return ``(reasoning_state, thought, extraction_notes)`` from rollout text.

    Think-mode: extract ``<reasoning_state>`` (and optional ``<think>``). Notes such as
    ``think_unclosed`` / ``missing_reasoning_state_tag`` are unioned with format violations.
    """
    raw = sample.response or ""
    if not _think_mode_enabled(sample):
        return raw, "", []
    reasoning_state, thought, notes = extract_reasoning_state_and_thought(raw)
    return (reasoning_state or "").strip(), (thought or "").strip(), list(notes or [])


async def _reward_one(sample: "Sample") -> float:
    reasoning_state, thought, extract_notes = _rollout_reasoning_state_for_reward(sample)
    # Format gate first: skip expensive answer calls on bad reasoning_states.
    # Empty reasoning_states get an explicit negative format penalty (not reward=0), so
    # silence cannot dominate mild negative deltas.
    from .reasoning_state_format import format_penalty_for_violations, reasoning_state_format_violations

    violations = list(dict.fromkeys(list(extract_notes) + reasoning_state_format_violations(reasoning_state)))
    if violations:
        format_penalty = float(format_penalty_for_violations(violations))
        total = format_penalty  # delta gated to 0
        debug = {
            "score_with": None,
            "score_without": None,
            "new_acc": 0.0,
            "delta_acc": 0.0,
            "penalty_acc": format_penalty,
            "reward": total,
            "raw_reward": None,
            "raw_new_acc": None,
            "raw_delta_acc": None,
            "format_violations": list(violations),
            "reasoning_state": strip_answer_for_reward(reasoning_state) or reasoning_state,
            "reasoning_state_thought": thought,
            "think_mode": _think_mode_enabled(sample),
            "answer_with": None,
            "answer_without": None,
            "judge_contains_answer": None,
            "judge_penalize": False,
            "judge_parse_ok": False,
        }
        _attach_debug(sample, debug)
        logger.info(
            "reasoning_state_reward id=%s format_gate violations=%s penalty=%.4f total=%.4f",
            getattr(sample, "index", None),
            ",".join(violations),
            format_penalty,
            total,
        )
        return total

    from .reasoning_state_judge import judge_enabled, judge_reasoning_state_contains_answer_async

    question = _field(sample, "question") or str(_meta(sample).get("question") or "")
    if judge_enabled():
        (reward, score_with, score_without, debug), judge = await asyncio.gather(
            score_pair_async(sample, reasoning_state),
            judge_reasoning_state_contains_answer_async(question, reasoning_state),
        )
    else:
        reward, score_with, score_without, debug = await score_pair_async(sample, reasoning_state)
        judge = {
            "raw": "",
            "answer_tag": "",
            "contains_answer": None,
            "parse_ok": False,
            "penalize": False,
            "skipped": True,
        }

    raw_new = float(debug["new_acc"])
    raw_delta = float(debug["delta_acc"])
    gated, violations, raw = apply_reasoning_state_format_gate(reward, reasoning_state)
    if violations:
        gated_new, gated_delta = 0.0, 0.0
    else:
        gated_new, gated_delta = raw_new, raw_delta

    # LLM judge: answer-like reasoning_state → fixed penalty (default -0.1). Does NOT zero delta.
    # Fail-open when parse/API fails. Format-gate failures skip judge penalty.
    judge_penalize = bool(judge.get("penalize")) and not violations
    format_penalty = float(format_penalty_for_violations(violations))
    penalty = (-_judge_penalty_magnitude() if judge_penalize else 0.0) + format_penalty
    if judge_penalize:
        violations = list(violations) + ["answer_in_reasoning_state"]

    total = float(gated_delta) + float(penalty)
    debug["raw_reward"] = raw
    debug["raw_new_acc"] = raw_new
    debug["raw_delta_acc"] = raw_delta
    debug["new_acc"] = gated_new
    debug["delta_acc"] = gated_delta
    debug["penalty_acc"] = penalty
    debug["reward"] = total
    debug["format_violations"] = violations
    debug["reasoning_state"] = strip_answer_for_reward(reasoning_state) or reasoning_state
    debug["reasoning_state_thought"] = thought
    debug["think_mode"] = _think_mode_enabled(sample)
    debug["judge_contains_answer"] = judge.get("contains_answer")
    debug["judge_answer_tag"] = judge.get("answer_tag")
    debug["judge_parse_ok"] = judge.get("parse_ok")
    debug["judge_penalize"] = judge_penalize
    debug["judge_raw"] = judge.get("raw")
    _attach_debug(sample, debug)
    logger.info(
        "reasoning_state_reward id=%s with=%.4f without=%.4f new=%.4f delta=%.4f penalty=%.4f "
        "total=%.4f judge=%s",
        getattr(sample, "index", None),
        score_with,
        score_without,
        raw_new,
        gated_delta,
        penalty,
        total,
        judge.get("contains_answer"),
    )
    return total


async def reward_from_reasoning_state(
    args,
    sample: Union["Sample", Sequence["Sample"]],
    **kwargs,
) -> Union[float, List[float]]:
    """Custom slime RM.

    Supports both:
    - per-sample (``async_rm``): ``Sample`` → ``float``
    - batched / ``--group-rm`` (``batched_async_rm``): ``list[Sample]`` → ``list[float]``

    Scalar is ``delta + penalty``. Advantages = whiten(delta)+whiten(penalty)
    in ``post_process_reasoning_state_rewards``.
    """
    del args, kwargs
    if isinstance(sample, (list, tuple)):
        samples = list(sample)
        results = await asyncio.gather(*[_reward_one(s) for s in samples])
        return list(results)
    return await _reward_one(sample)
