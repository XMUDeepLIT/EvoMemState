"""Embedding client for retrieval scoring.

Why this exists
---------------
``ontology._similarity`` is Jaccard over character bigrams plus a substring
bonus. It cannot see that "Pacific Island music theory" and "Indian classical
music theory" are the same *kind* of thing about the same user -- they share no
salient terms, so it scores them near zero. That is exactly the case MemSyco's
``valid_memory_selection`` is built around, so lexical-only retrieval is
handicapped on the task the ontology is supposed to win.

Scope: RETRIEVAL ONLY
---------------------
This is deliberately not wired into graph construction. Build-time similarity
feeds ``local_branch`` and ``apply_delta``, and both are inputs to the graph
cache key. Changing them would invalidate every cached graph and force a full
rebuild, while also making the extraction stage non-deterministic with respect
to an external service. The graph stays lexical and reproducible; only ranking
over an already-built graph becomes semantic.

Failure policy
--------------
Every failure path degrades to lexical scoring rather than raising. A missing
API key, a dead endpoint, or a malformed response must not abort an evaluation
run -- it should produce a run that is merely no better than the old behaviour,
and say so in the audit record.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence


DEFAULT_MODEL = "baai/bge-m3"
DEFAULT_DIMS = 1024
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Requests are batched; oversized batches are the common cause of 400s.
MAX_BATCH = 64


def _l2_normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return [0.0] * len(vector)
    return [value / norm for value in vector]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity. Vectors are stored pre-normalized, so this is a dot."""
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


class EmbeddingClient:
    """Batched, disk-cached embedding client over an OpenAI-compatible endpoint.

    The cache is keyed on (model, text) and shared across arms and tasks, so the
    1550-question corpus is embedded once no matter how many arms run. Cache
    entries are plain JSON files -- inspectable, and safe to delete.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str = "",
        base_url: str = DEFAULT_BASE_URL,
        dims: int = DEFAULT_DIMS,
        cache_dir: Optional[Path] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("Embedding API key is empty")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is installed
            raise RuntimeError("Install dependencies with: pip install -e .") from exc

        self.model = model
        self.dims = int(dims)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            max_retries=max_retries,
        )
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        # In-process memo on top of the disk cache: a single retrieval call asks
        # for the same question text once per memory, and disk round-trips add up.
        self._memo: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

        self.n_api_calls = 0
        self.n_cache_hits = 0
        self.n_failures = 0

    # -- cache ---------------------------------------------------------------
    def _cache_path(self, text: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(
            ("%s\0%s" % (self.model, text)).encode("utf-8")
        ).hexdigest()
        # Shard by prefix: a flat directory with 1550+ entries is slow to list
        # and unpleasant to inspect.
        shard = self.cache_dir / key[:2]
        shard.mkdir(parents=True, exist_ok=True)
        return shard / (key + ".json")

    def _read_cache(self, text: str) -> Optional[List[float]]:
        path = self._cache_path(text)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            vector = payload.get("vector")
            if isinstance(vector, list) and vector:
                return [float(value) for value in vector]
        except Exception:  # noqa: BLE001 - a corrupt cache entry is just a miss
            return None
        return None

    def _write_cache(self, text: str, vector: List[float]) -> None:
        path = self._cache_path(text)
        if path is None:
            return
        try:
            # Write-then-rename so a killed process cannot leave a half file
            # that later reads as corrupt.
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"model": self.model, "vector": vector}),
                encoding="utf-8",
            )
            tmp.replace(path)
        except Exception:  # noqa: BLE001 - caching is an optimization
            pass

    # -- embedding -----------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> List[Optional[List[float]]]:
        """Embed texts, returning None per text whose embedding is unavailable.

        Callers must treat None as "fall back to lexical" rather than as a zero
        vector: a zero vector would silently rank that memory last, which is a
        scoring decision disguised as a failure.
        """
        cleaned = [(text or "").strip() for text in texts]
        results: List[Optional[List[float]]] = [None] * len(cleaned)

        pending: List[str] = []
        pending_index: Dict[str, List[int]] = {}

        for index, text in enumerate(cleaned):
            if not text:
                continue
            with self._lock:
                memo_hit = self._memo.get(text)
            if memo_hit is not None:
                results[index] = memo_hit
                self.n_cache_hits += 1
                continue
            disk_hit = self._read_cache(text)
            if disk_hit is not None:
                with self._lock:
                    self._memo[text] = disk_hit
                results[index] = disk_hit
                self.n_cache_hits += 1
                continue
            if text not in pending_index:
                pending_index[text] = []
                pending.append(text)
            pending_index[text].append(index)

        for start in range(0, len(pending), MAX_BATCH):
            batch = pending[start : start + MAX_BATCH]
            vectors = self._embed_batch(batch)
            for text, vector in zip(batch, vectors):
                if vector is None:
                    continue
                with self._lock:
                    self._memo[text] = vector
                self._write_cache(text, vector)
                for index in pending_index[text]:
                    results[index] = vector

        return results

    def _embed_batch(self, batch: List[str]) -> List[Optional[List[float]]]:
        if not batch:
            return []
        try:
            response = self.client.embeddings.create(model=self.model, input=batch)
            self.n_api_calls += 1
        except Exception:  # noqa: BLE001 - degrade to lexical, never abort a run
            self.n_failures += 1
            return [None] * len(batch)

        # Providers are not required to preserve input order; index is authoritative.
        by_index: Dict[int, List[float]] = {}
        for item in getattr(response, "data", []) or []:
            try:
                by_index[int(item.index)] = _l2_normalize(list(item.embedding))
            except Exception:  # noqa: BLE001 - skip a malformed row
                continue
        return [by_index.get(position) for position in range(len(batch))]

    def embed_one(self, text: str) -> Optional[List[float]]:
        return self.embed([text])[0]

    def stats(self) -> Dict[str, int]:
        return {
            "api_calls": self.n_api_calls,
            "cache_hits": self.n_cache_hits,
            "failures": self.n_failures,
        }


def embedding_client_from_env(cache_dir: Optional[Path] = None) -> Optional[EmbeddingClient]:
    """Build a client from EMBEDDING_* env vars, or None when unconfigured.

    Returning None rather than raising is intentional: embeddings are an
    enhancement to retrieval, so an unset key should mean "run lexically", not
    "fail the experiment".
    """
    if os.environ.get("EMBEDDING_ENABLED", "1").strip().lower() in {"0", "false", "no"}:
        return None
    api_key = os.environ.get("EMBEDDING_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        return EmbeddingClient(
            model=os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            api_key=api_key,
            base_url=os.environ.get("EMBEDDING_BASE_URL", DEFAULT_BASE_URL).strip()
            or DEFAULT_BASE_URL,
            dims=int(os.environ.get("EMBEDDING_DIMS", str(DEFAULT_DIMS)) or DEFAULT_DIMS),
            cache_dir=cache_dir,
        )
    except Exception:  # noqa: BLE001 - misconfiguration degrades, never aborts
        return None
