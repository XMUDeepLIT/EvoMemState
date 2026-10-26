"""Generate a request-specific Reasoning State from a Relational State."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .reasoning_state_prompt import (
    extract_reasoning_state_and_thought,
    reasoning_state_user_prompt,
    select_reasoning_state_system_prompt,
)
from .prompting import EMPTY_CONTEXT, render_context, render_memory_body, select_render_mode
from .retrieval import RelationalState
from .text_clean import strip_generated_state

logger = logging.getLogger(__name__)


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def reasoning_state_enabled() -> bool:
    return any(
        _truthy(k)
        for k in (
            "EVOMEMSTATE_REASONING_STATE_ENABLED",
            "MEMSYCO_REASONING_STATE_ENABLED",
        )
    )


def _env(name: str, *fallbacks: str, default: str = "") -> str:
    for key in (name,) + fallbacks:
        value = os.environ.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def reasoning_state_client_config() -> Dict[str, str]:
    return {
        "model": _env(
            "REASONING_STATE_MODEL", "GEN_B_MODEL", "ANSWER_MODEL", default="qwen3-8b"
        ),
        "base_url": _env(
            "REASONING_STATE_BASE_URL",
            "GEN_B_BASE_URL",
            "ANSWER_BASE_URL",
            default="http://127.0.0.1:8800/v1",
        ),
        "api_key": _env(
            "REASONING_STATE_API_KEY", "GEN_B_API_KEY", "ANSWER_API_KEY", default="EMPTY"
        ),
    }


def _max_tokens() -> int:
    default = "1536" if _think_mode() else "512"
    return max(32, int(_env("REASONING_STATE_MAX_TOKENS", default=default)))


def _temperature() -> float:
    return float(_env("REASONING_STATE_TEMPERATURE", default="0.3"))


def _enable_thinking() -> bool:
    # Native chat-template thinking; complementary to prompt-level <think>/<reasoning_state>.
    return _truthy("REASONING_STATE_ENABLE_THINKING") or _think_mode()


def _think_mode() -> bool:
    """Prompt-level <think>/<reasoning_state> contract (REASONING_STATE_SYSTEM_PROMPT_THINK)."""
    return _truthy("REASONING_STATE_THINK_MODE") or _truthy("REASONING_STATE_ENABLE_THINKING")


def _memory_only() -> bool:
    """Omit question/claim from the Reasoning State generator user prompt."""
    return _truthy("REASONING_STATE_MEMORY_ONLY") or _truthy("REASONING_STATE_MEMORY_ONLY")


def _channel_needs() -> bool:
    """Inject fact_need / preference_need into the reasoning_state user prompt."""
    return _truthy("REASONING_STATE_CHANNEL_NEEDS") or _truthy("REASONING_STATE_CHANNEL_NEEDS")


def _cache_dir() -> Optional[Path]:
    raw = _env("REASONING_STATE_CACHE_DIR", default="")
    if not raw:
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(cache_dir: Path, *, model: str, messages: list) -> Path:
    key = hashlib.sha256(
        json.dumps(
            {"model": model, "messages": messages},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return cache_dir / ("%s.json" % key)


def generate_reasoning_state(
    question: str,
    memory_body: str,
    *,
    task: str = "",
    cache_dir: Optional[Path] = None,
    fact_need: Optional[float] = None,
    preference_need: Optional[float] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Call chat model for routing bullets. Empty reasoning_state on failure."""
    cfg = reasoning_state_client_config()
    think_mode = _think_mode()
    memory_only = _memory_only()
    channel_needs = _channel_needs()
    meta: Dict[str, Any] = {
        "online_reasoning_state": True,
        "reasoning_state_model": cfg["model"],
        "reasoning_state_base_url": cfg["base_url"],
        "reasoning_state_ok": False,
        "reasoning_state_chars": 0,
        "reasoning_state_error": "",
        "reasoning_state_cache_hit": False,
        "think_mode": think_mode,
        "memory_only": memory_only,
        "channel_needs": channel_needs,
        "reasoning_state_thought": "",
        "reasoning_state_format_violations": [],
        "fact_need": None if fact_need is None else float(fact_need),
        "preference_need": None if preference_need is None else float(preference_need),
    }
    body = (memory_body or "").strip()
    if not body or body == EMPTY_CONTEXT:
        meta["reasoning_state_error"] = "empty_memory_body"
        return "", meta

    system_prompt = select_reasoning_state_system_prompt(
        think_mode=think_mode,
        memory_only=memory_only,
        channel_needs=channel_needs,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": reasoning_state_user_prompt(
                question or "",
                body,
                task=task,
                memory_only=memory_only,
                channel_needs=channel_needs,
                fact_need=fact_need if channel_needs else None,
                preference_need=preference_need if channel_needs else None,
            ),
        },
    ]

    disk = cache_dir if cache_dir is not None else _cache_dir()
    if disk is not None:
        disk.mkdir(parents=True, exist_ok=True)
        path = _cache_path(disk, model=cfg["model"], messages=messages)
        if path.is_file():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                text = str(cached.get("text") or "").strip()
                if text:
                    meta["reasoning_state_ok"] = True
                    meta["reasoning_state_chars"] = len(text)
                    meta["reasoning_state_cache_hit"] = True
                    meta["reasoning_state_thought"] = str(cached.get("thought") or "")
                    return text, meta
            except (OSError, json.JSONDecodeError, TypeError):
                pass

    try:
        from openai import OpenAI
    except ImportError as exc:
        meta["reasoning_state_error"] = "openai_missing:%s" % exc
        return "", meta

    client = OpenAI(api_key=cfg["api_key"] or "EMPTY", base_url=cfg["base_url"].rstrip("/"))
    kwargs: Dict[str, Any] = {
        "model": cfg["model"],
        "messages": messages,
        "max_tokens": _max_tokens(),
        "temperature": _temperature(),
    }
    # Always set enable_thinking explicitly (True for think-mode, False otherwise).
    kwargs["extra_body"] = {
        "chat_template_kwargs": {"enable_thinking": bool(_enable_thinking())}
    }

    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            try:
                resp = client.chat.completions.create(**kwargs)
            except TypeError:
                kwargs.pop("extra_body", None)
                resp = client.chat.completions.create(**kwargs)
            raw = ""
            reasoning_content = ""
            if resp.choices:
                message = resp.choices[0].message
                raw = message.content or ""
                reasoning_content = str(
                    getattr(message, "reasoning_content", None) or ""
                )
            if think_mode:
                reasoning_state, thought, notes = extract_reasoning_state_and_thought(
                    raw, reasoning_content=reasoning_content
                )
                text = (reasoning_state or "").strip()
                meta["reasoning_state_thought"] = thought
                meta["reasoning_state_format_violations"] = list(notes)
                if not text:
                    meta["reasoning_state_error"] = "empty_reasoning_state_after_extract"
                    if notes:
                        meta["reasoning_state_error"] += ":" + ",".join(notes)
                    return "", meta
            else:
                text = (strip_generated_state(raw) or "").strip()
                if not text:
                    meta["reasoning_state_error"] = "empty_response"
                    return "", meta
            meta["reasoning_state_ok"] = True
            meta["reasoning_state_chars"] = len(text)
            if disk is not None:
                try:
                    _cache_path(disk, model=cfg["model"], messages=messages).write_text(
                        json.dumps(
                            {
                                "text": text,
                                "thought": meta.get("reasoning_state_thought") or "",
                                "raw": raw,
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                except OSError as exc:
                    logger.warning("online_reasoning_state cache write failed: %s", exc)
            return text, meta
        except Exception as exc:  # noqa: BLE001 — network / provider
            last_error = exc
            time.sleep(min(4, 2**attempt))

    meta["reasoning_state_error"] = "%s: %s" % (
        type(last_error).__name__ if last_error else "error",
        last_error,
    )
    logger.warning("online_reasoning_state failed: %s", meta["reasoning_state_error"])
    return "", meta


def render_with_reasoning_state(
    result: RelationalState,
    question: str,
    *,
    flat: bool = False,
    task: str = "",
    cache_dir: Optional[Path] = None,
    force: Optional[bool] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Render context; optionally replace fixed rules with a live model Reasoning State."""
    enabled = reasoning_state_enabled() if force is None else bool(force)
    meta: Dict[str, Any] = {
        "online_reasoning_state": enabled,
        "reasoning_state_ok": False,
        "reasoning_state_fallback_fixed": False,
        "render_mode": select_render_mode(result),
    }
    if not enabled:
        return render_context(result, include_intents=False, flat=flat), meta

    memory_body = render_memory_body(
        result, include_intents=False, mode=meta["render_mode"]
    )
    reasoning_state, reasoning_state_meta = generate_reasoning_state(
        question,
        memory_body,
        task=task,
        cache_dir=cache_dir,
        fact_need=float(result.fact_need),
        preference_need=float(result.preference_need),
    )
    meta.update(reasoning_state_meta)
    if reasoning_state:
        text = render_context(
            result, include_intents=False, flat=flat, reasoning_state=reasoning_state
        )
        meta["generated_reasoning_state"] = reasoning_state
        return text, meta

    meta["reasoning_state_fallback_fixed"] = True
    return render_context(result, include_intents=False, flat=flat), meta
