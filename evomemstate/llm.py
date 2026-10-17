from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _extract_json(text: str) -> Dict[str, Any]:
    value = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    value = value.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    candidates: List[tuple[int, int, Dict[str, Any]]] = []
    for start, char in enumerate(value):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(value)):
            current = value[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    raw = value[start : end + 1]
                    try:
                        item = json.loads(raw)
                    except json.JSONDecodeError:
                        break
                    if isinstance(item, dict):
                        schema_keys = sum(
                            key in item for key in ("concept_edits", "memories", "relations")
                        )
                        candidates.append((schema_keys, len(raw), item))
                    break
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]
    raise ValueError("LLM response does not contain a complete JSON object")


class OpenAIJSONClient:
    """Small OpenAI-compatible JSON client with deterministic disk caching."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        cache_dir: Optional[Path] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("API key is empty")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: pip install -e .") from exc
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
        self.cache_dir = cache_dir
        self.temperature = temperature
        self.max_retries = max(1, max_retries)
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, messages: List[Dict[str, str]]) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(
            json.dumps(
                {"model": self.model, "messages": messages, "temperature": self.temperature},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / (key + ".json")

    def complete_json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        cache_path = self._cache_path(messages)
        if cache_path is not None and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict):
                return cached

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                }
                # Local Qwen3: disable thinking so JSON rerank stays short/parseable.
                if "qwen3" in (self.model or "").lower():
                    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                try:
                    response = self.client.chat.completions.create(
                        **kwargs, response_format={"type": "json_object"}
                    )
                except Exception as format_error:
                    message = str(format_error).lower()
                    if "response_format" not in message and "json" not in message:
                        raise
                    response = self.client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content or ""
                payload = _extract_json(text)
                if cache_path is not None:
                    cache_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                return payload
            except Exception as exc:  # network/provider path
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(2**attempt)
        raise RuntimeError("LLM JSON request failed after retries") from last_error


class EvaJSONClient:
    """Synchronous JSON adapter for the EVA backend used by personalized_harness."""

    def __init__(
        self,
        *,
        model: str = "deepseek-v4-flash",
        cache_dir: Optional[Path] = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        max_retries: int = 4,
        json_repair_attempts: int = 2,
        timeout: float = 600.0,
        harness_root: Optional[Path] = None,
    ) -> None:
        root = harness_root or (Path(__file__).resolve().parents[2] / "personalized_harness")
        if not root.exists():
            raise RuntimeError("personalized_harness was not found at %s" % root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        os.environ.setdefault("EVA_USAGE_LOG", "0")
        try:
            from personalized_harness.utils.eva_client import EvaClient
        except ImportError as exc:
            raise RuntimeError("Unable to import the EVA client from personalized_harness") from exc
        self.model = model
        self.client = EvaClient(timeout=timeout, max_retries=max_retries)
        self.cache_dir = cache_dir
        self.temperature = temperature
        self.max_tokens = max(1, int(max_tokens))
        self.json_repair_attempts = max(0, int(json_repair_attempts))
        self.failure_dir = cache_dir.parent / "failed_responses" if cache_dir is not None else None
        self._stats_lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "model": model,
            "requests": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "json_repairs": 0,
            "json_failures": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "duration_sec": 0.0,
        }
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        if self.failure_dir is not None:
            self.failure_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, messages: List[Dict[str, str]]) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(
            json.dumps(
                {
                    "provider": "eva",
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / (key + ".json")

    def _call(self, messages: List[Dict[str, str]]) -> tuple[str, Dict[str, Any]]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = response.choices[0].message.content or response.choices[0].message.reasoning_content or ""
        usage = {
            "prompt_tokens": int(response.usage.prompt_tokens),
            "completion_tokens": int(response.usage.completion_tokens),
            "total_tokens": int(response.usage.total_tokens),
            "cache_hit_tokens": int(response.usage.cache_hit_tokens),
            "cache_write_tokens": int(response.usage.cache_write_tokens),
            "cost_usd": float(response.cost_usd),
            "duration_sec": float(response.duration),
        }
        with self._stats_lock:
            self._stats["api_calls"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self._stats[key] += usage[key]
            self._stats["cost_usd"] += usage["cost_usd"]
            self._stats["duration_sec"] += usage["duration_sec"]
        return text, usage

    @staticmethod
    def _repair_messages(text: str, error: Exception) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Repair an invalid or truncated JSON response. Return JSON only. "
                    "The root must be one object containing concept_edits, memories, and relations arrays. "
                    "Preserve the original facts, IDs, evidence quotes, and intended relations. "
                    "Remove commentary and fix syntax or complete truncated closing content."
                ),
            },
            {
                "role": "user",
                "content": "Parser error: %s\n\nInvalid response:\n%s" % (error, text[:60000]),
            },
        ]

    def _write_failure(
        self,
        messages: List[Dict[str, str]],
        responses: List[str],
        error: Exception,
    ) -> None:
        if self.failure_dir is None:
            return
        key = hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        path = self.failure_dir / (key + ".json")
        path.write_text(
            json.dumps(
                {
                    "model": self.model,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "responses": responses,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def complete_json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        cache_path = self._cache_path(messages)
        if cache_path is not None and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            payload = cached.get("payload") if isinstance(cached, dict) else None
            if isinstance(payload, dict):
                with self._stats_lock:
                    self._stats["requests"] += 1
                    self._stats["cache_hits"] += 1
                return payload

        with self._stats_lock:
            self._stats["requests"] += 1
        request_messages = messages
        responses: List[str] = []
        usages: List[Dict[str, Any]] = []
        last_error: Exception = ValueError("No EVA response")
        payload: Optional[Dict[str, Any]] = None
        for attempt in range(self.json_repair_attempts + 1):
            text, usage = self._call(request_messages)
            responses.append(text)
            usages.append(usage)
            try:
                payload = _extract_json(text)
                if attempt:
                    with self._stats_lock:
                        self._stats["json_repairs"] += 1
                break
            except Exception as exc:
                last_error = exc
                if attempt < self.json_repair_attempts:
                    request_messages = self._repair_messages(text, exc)

        if payload is None:
            with self._stats_lock:
                self._stats["json_failures"] += 1
            self._write_failure(messages, responses, last_error)
            raise RuntimeError(
                "EVA returned invalid JSON after %d repair attempts: %s"
                % (self.json_repair_attempts, last_error)
            ) from last_error

        aggregate_usage = {
            key: sum(float(item.get(key, 0)) for item in usages)
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cache_hit_tokens",
                "cache_write_tokens",
                "cost_usd",
                "duration_sec",
            )
        }
        if cache_path is not None:
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {"payload": payload, "usage": aggregate_usage, "repair_calls": len(usages) - 1},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary.replace(cache_path)
        return payload

    def usage_summary(self) -> Dict[str, Any]:
        with self._stats_lock:
            result = dict(self._stats)
        result["cost_usd"] = round(float(result["cost_usd"]), 8)
        result["duration_sec"] = round(float(result["duration_sec"]), 3)
        return result


class AsyncEvaJSONClient(EvaJSONClient):
    """Async EVA adapter with bounded concurrency and the same cache/repair semantics."""

    def __init__(
        self,
        *,
        model: str = "deepseek-v4-flash",
        cache_dir: Optional[Path] = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        max_retries: int = 4,
        json_repair_attempts: int = 2,
        timeout: float = 600.0,
        concurrency: int = 12,
        harness_root: Optional[Path] = None,
    ) -> None:
        root = harness_root or (Path(__file__).resolve().parents[2] / "personalized_harness")
        if not root.exists():
            raise RuntimeError("personalized_harness was not found at %s" % root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        os.environ.setdefault("EVA_USAGE_LOG", "0")
        try:
            from personalized_harness.utils.eva_client import AsyncEvaClient
        except ImportError as exc:
            raise RuntimeError("Unable to import AsyncEvaClient from personalized_harness") from exc
        self.model = model
        self.client = AsyncEvaClient(timeout=timeout, max_retries=max_retries)
        self.cache_dir = cache_dir
        self.temperature = temperature
        self.max_tokens = max(1, int(max_tokens))
        self.json_repair_attempts = max(0, int(json_repair_attempts))
        self.failure_dir = cache_dir.parent / "failed_responses" if cache_dir is not None else None
        self.concurrency = max(1, int(concurrency))
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._stats_lock = threading.Lock()
        self._stats = {
            "model": model,
            "requests": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "json_repairs": 0,
            "json_failures": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "duration_sec": 0.0,
            "concurrency": self.concurrency,
        }
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        if self.failure_dir is not None:
            self.failure_dir.mkdir(parents=True, exist_ok=True)

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency)
        return self._semaphore

    async def _call_async(self, messages: List[Dict[str, str]]) -> tuple[str, Dict[str, Any]]:
        async with self._get_semaphore():
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        text = response.choices[0].message.content or response.choices[0].message.reasoning_content or ""
        usage = {
            "prompt_tokens": int(response.usage.prompt_tokens),
            "completion_tokens": int(response.usage.completion_tokens),
            "total_tokens": int(response.usage.total_tokens),
            "cache_hit_tokens": int(response.usage.cache_hit_tokens),
            "cache_write_tokens": int(response.usage.cache_write_tokens),
            "cost_usd": float(response.cost_usd),
            "duration_sec": float(response.duration),
        }
        with self._stats_lock:
            self._stats["api_calls"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self._stats[key] += usage[key]
            self._stats["cost_usd"] += usage["cost_usd"]
            self._stats["duration_sec"] += usage["duration_sec"]
        return text, usage

    async def complete_json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        cache_path = self._cache_path(messages)
        if cache_path is not None and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            payload = cached.get("payload") if isinstance(cached, dict) else None
            if isinstance(payload, dict):
                with self._stats_lock:
                    self._stats["requests"] += 1
                    self._stats["cache_hits"] += 1
                return payload

        with self._stats_lock:
            self._stats["requests"] += 1
        request_messages = messages
        responses: List[str] = []
        usages: List[Dict[str, Any]] = []
        last_error: Exception = ValueError("No EVA response")
        payload: Optional[Dict[str, Any]] = None
        for attempt in range(self.json_repair_attempts + 1):
            text, usage = await self._call_async(request_messages)
            responses.append(text)
            usages.append(usage)
            try:
                payload = _extract_json(text)
                if attempt:
                    with self._stats_lock:
                        self._stats["json_repairs"] += 1
                break
            except Exception as exc:
                last_error = exc
                if attempt < self.json_repair_attempts:
                    request_messages = self._repair_messages(text, exc)

        if payload is None:
            with self._stats_lock:
                self._stats["json_failures"] += 1
            self._write_failure(messages, responses, last_error)
            raise RuntimeError(
                "EVA returned invalid JSON after %d repair attempts: %s"
                % (self.json_repair_attempts, last_error)
            ) from last_error

        aggregate_usage = {
            key: sum(float(item.get(key, 0)) for item in usages)
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cache_hit_tokens",
                "cache_write_tokens",
                "cost_usd",
                "duration_sec",
            )
        }
        if cache_path is not None:
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {"payload": payload, "usage": aggregate_usage, "repair_calls": len(usages) - 1},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary.replace(cache_path)
        return payload
