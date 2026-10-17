"""MemTrapBench adapter for EvoMemState's graph and retrieval components.

The benchmark method is ``EvoMemState``.

MemTrap is closer to MemSyco than LoCoMo: one ``context_history`` dialogue +
one ``final_trigger`` query. Graphs are built from history only (never the
trigger), then retrieved and paired with an online Reasoning State at eval time.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

GRAPH_VERSION = "v1"
DEFAULT_CHUNK_MODE = "whole_dialogue"
DEFAULT_SLOT_POLICY = "compatible"
DEFAULT_CONCEPT_TOP_K = 8
DEFAULT_MEMORY_TOP_L = 40

DATASETS = (
    "poison",
    "number_game",
    "hurt",
    "hallucination",
    "inertia",
    "unclear",
)

_BUILD_LOCKS: Dict[str, threading.Lock] = {}
_BUILD_LOCKS_GUARD = threading.Lock()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bench_root() -> Path:
    override = os.environ.get("MEMTRAP_BENCHMARK_ROOT", "").strip()
    if override:
        return Path(override)
    return _project_root() / "datasets" / "MemTrapBench"


def _graph_root() -> Path:
    override = os.environ.get("EVOMEMSTATE_GRAPH_ROOT", "").strip()
    if override:
        return Path(override)
    return _project_root() / "outputs" / "memtrap_graphs_v1"


def dataset_paths() -> Dict[str, Path]:
    """Official MemTrapBench eval JSON paths (mirrors runners/eval/eval_common)."""
    root = _bench_root()
    return {
        "poison": root / "memtrapbench" / "safety" / "poison_200.json",
        "number_game": root / "memtrapbench" / "cognitive_bias" / "number_game_200.json",
        "hurt": root / "memtrapbench" / "trauma" / "hurt_150.json",
        "hallucination": root / "memtrapbench" / "task_boundary" / "hallucination_150.json",
        "inertia": root / "memtrapbench" / "cognitive_bias" / "Inertia_150.json",
        "unclear": root / "memtrapbench" / "task_boundary" / "unclear_200.json",
    }


def load_dataset(dataset: str, *, limit: int = 0) -> List[Dict[str, Any]]:
    paths = dataset_paths()
    if dataset not in paths:
        raise ValueError("unknown MemTrap dataset: %s (want %s)" % (dataset, ", ".join(DATASETS)))
    path = paths[dataset]
    if not path.exists():
        raise FileNotFoundError("missing MemTrap file: %s" % path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("%s is not a JSON array" % path)
    if limit and limit > 0:
        rows = rows[:limit]
    return rows


def sample_key(dataset: str, sample_id: str) -> str:
    """Stable directory / join key: ``{dataset}__{id}``."""
    return "%s__%s" % (dataset, sample_id)


def parse_sample_key(key: str) -> Tuple[str, str]:
    if "__" not in key:
        raise ValueError("bad sample_key (want dataset__id): %s" % key)
    dataset, sample_id = key.split("__", 1)
    return dataset, sample_id


def history_to_messages(context_history: Any) -> List[Dict[str, str]]:
    """Convert MemTrap ``context_history`` into extractor messages."""
    messages: List[Dict[str, str]] = []
    if not isinstance(context_history, list):
        return messages
    for index, item in enumerate(context_history, start=1):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        messages.append(
            {
                "turn_id": "ctx_%06d" % index,
                "role": role,
                "content": content,
            }
        )
    return messages


def history_to_prior_dialogue(context_history: Any) -> str:
    """Human-readable prior dialogue (audit / AdaptiveMem export)."""
    lines: List[str] = []
    for msg in history_to_messages(context_history):
        label = {
            "user": "User",
            "assistant": "Assistant",
            "system": "System",
        }.get(msg["role"], msg["role"].capitalize())
        lines.append("%s: %s" % (label, msg["content"]))
    return "\n".join(lines)


def _build_settings(eval_config: Any = None) -> Dict[str, Any]:
    model = None
    if eval_config is not None:
        model = getattr(eval_config, "llm_model", None)
    model = model or os.environ.get("MEMORY_LLM_MODEL") or "deepseek-v4-flash"
    return {
        "model": str(model),
        "chunk_mode": os.environ.get("EVOMEMSTATE_CHUNK_MODE", DEFAULT_CHUNK_MODE),
        "slot_policy": os.environ.get("EVOMEMSTATE_SLOT_POLICY", DEFAULT_SLOT_POLICY),
        "concept_top_k": int(
            os.environ.get("EVOMEMSTATE_CONCEPT_TOP_K", str(DEFAULT_CONCEPT_TOP_K))
        ),
        "memory_top_l": int(
            os.environ.get("EVOMEMSTATE_MEMORY_TOP_L", str(DEFAULT_MEMORY_TOP_L))
        ),
        "version": os.environ.get("EVOMEMSTATE_GRAPH_VERSION", GRAPH_VERSION),
        "bench": "memtrap",
    }


def _graph_key(messages: List[Dict[str, str]], settings: Dict[str, Any]) -> str:
    """Content hash of history messages + build settings (method excluded)."""
    normalized = [
        {"role": m["role"], "content": " ".join(m["content"].split())}
        for m in messages
    ]
    payload = {
        "model": settings["model"],
        "chunk_mode": settings["chunk_mode"],
        "slot_policy": settings["slot_policy"],
        "concept_top_k": settings["concept_top_k"],
        "memory_top_l": settings["memory_top_l"],
        "version": settings["version"],
        "bench": settings.get("bench", "memtrap"),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()[:24]


def _get_embedding_client():
    from evomemstate.embedding import embedding_client_from_env

    cache = _graph_root() / "embedding_cache"
    return embedding_client_from_env(cache_dir=cache)


def _lock_for(key: str) -> threading.Lock:
    with _BUILD_LOCKS_GUARD:
        lock = _BUILD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _BUILD_LOCKS[key] = lock
        return lock


def _llm_client(eval_config: Any, *, cache_dir: Path):
    from evomemstate.llm import OpenAIJSONClient

    api_key = getattr(eval_config, "api_key", None) or os.environ.get("MEMORY_API_KEY") or ""
    base_url = getattr(eval_config, "base_url", None) or os.environ.get("MEMORY_BASE_URL") or ""
    model = (
        getattr(eval_config, "llm_model", None)
        or os.environ.get("MEMORY_LLM_MODEL")
        or "deepseek-v4-flash"
    )
    if not api_key or not base_url:
        raise RuntimeError(
            "MEMORY_API_KEY / MEMORY_BASE_URL (or eval_config equivalents) required to build graphs"
        )
    return OpenAIJSONClient(
        model=str(model),
        api_key=str(api_key),
        base_url=str(base_url),
        cache_dir=cache_dir,
    )


def graph_dir_for(sample_key_str: str) -> Path:
    return _graph_root() / sample_key_str


def _messages_to_session(
    messages: List[Dict[str, str]],
    *,
    user_id: str,
) -> Dict[str, Any]:
    return {
        "session_id": "memtrap_whole",
        "timestamp": "",
        "user_id": user_id,
        "messages": list(messages),
    }


def _settings_fresh(existing: Dict[str, Any], settings: Dict[str, Any], content_key: str) -> bool:
    if not existing:
        return False
    if existing.get("graph_key") != content_key:
        return False
    for field in ("model", "chunk_mode", "slot_policy", "concept_top_k", "memory_top_l", "version"):
        if existing.get(field) != settings.get(field):
            return False
    return True


def _load_or_build_graph(
    context_history: Any,
    eval_config: Any,
    *,
    sample_key_str: str,
):
    """Load cached ontology.json or build once under a per-sample lock."""
    from evomemstate.builder import MemoryBuilder
    from evomemstate.extractor import OntologyGuidedExtractor
    from evomemstate.ontology import OntologyState, OntologyStore

    settings = _build_settings(eval_config)
    messages = history_to_messages(context_history)
    content_key = _graph_key(messages, settings)
    graph_dir = graph_dir_for(sample_key_str)
    state_path = graph_dir / "ontology.json"
    settings_path = graph_dir / "build_settings.json"
    user_id = "memtrap:%s" % sample_key_str

    if state_path.exists() and settings_path.exists():
        try:
            existing = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if _settings_fresh(existing, settings, content_key):
            return OntologyStore.load(state_path, user_id, slot_policy=settings["slot_policy"])

    lock = _lock_for(sample_key_str)
    with lock:
        if state_path.exists() and settings_path.exists():
            try:
                existing = json.loads(settings_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            if _settings_fresh(existing, settings, content_key):
                return OntologyStore.load(state_path, user_id, slot_policy=settings["slot_policy"])

        graph_dir.mkdir(parents=True, exist_ok=True)
        store = OntologyStore(
            OntologyState.empty(user_id),
            slot_policy=settings["slot_policy"],
        )
        client = _llm_client(eval_config, cache_dir=graph_dir / "llm_cache")
        builder = MemoryBuilder(
            store,
            OntologyGuidedExtractor(
                client,
                top_k=settings["concept_top_k"],
                memory_top_l=settings["memory_top_l"],
            ),
        )
        session = _messages_to_session(messages, user_id=user_id)
        if session["messages"]:
            try:
                builder.build_session(session)
            except Exception as exc:  # noqa: BLE001
                print(
                    "[memtrap] build_session failed for %s: %s: %s"
                    % (sample_key_str, type(exc).__name__, exc),
                    flush=True,
                )
                (graph_dir / "build_error.txt").write_text(
                    "%s: %s\n" % (type(exc).__name__, exc),
                    encoding="utf-8",
                )
        store.save(state_path)
        settings_path.write_text(
            json.dumps(
                {
                    **settings,
                    "graph_key": content_key,
                    "sample_key": sample_key_str,
                    "n_messages": len(messages),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return store


def load_graph_only(sample_key_str: str, eval_config: Any = None):
    """Load a prebuilt graph; raise if missing (handoff path — no rebuild)."""
    from evomemstate.ontology import OntologyStore

    settings = _build_settings(eval_config)
    graph_dir = graph_dir_for(sample_key_str)
    state_path = graph_dir / "ontology.json"
    if not state_path.exists():
        raise FileNotFoundError(
            "missing MemTrap graph for %s at %s (run prebuild_memtrap first)"
            % (sample_key_str, state_path)
        )
    user_id = "memtrap:%s" % sample_key_str
    return OntologyStore.load(state_path, user_id, slot_policy=settings["slot_policy"])


def build_context(
    context_history: Any,
    user_question: str,
    eval_config: Any,
    *,
    sample_key_str: str,
    allow_build: bool = False,
):
    """Retrieve (+ optional online reasoning_state via harness) and return BaselineContext."""
    from baselines.base import BaselineContext
    from baselines.common import jsonable_memories
    from evomemstate.harness import HarnessContext, run_context_pipeline

    method = str(getattr(eval_config, "method", None) or "EvoMemState")
    settings = _build_settings(eval_config)
    messages = history_to_messages(context_history)
    content_key = _graph_key(messages, settings)
    if allow_build:
        store = _load_or_build_graph(
            context_history, eval_config, sample_key_str=sample_key_str
        )
    else:
        store = load_graph_only(sample_key_str, eval_config)

    root = _graph_root()
    prior = history_to_prior_dialogue(context_history)
    ctx = HarnessContext(
        prior_dialogue=prior,
        user_question=user_question or "",
        eval_config=eval_config,
        method=method,
        sample_key=sample_key_str,
        store=store,
        graph_key=content_key,
        build_settings=settings,
        llm_client_factory=_llm_client,
        embedding_client_factory=_get_embedding_client,
        graph_root=root,
    )
    run_context_pipeline(ctx)

    return BaselineContext(
        context_text=ctx.context_text,
        retrieved_memories=jsonable_memories(ctx.relational_state_payload),
        user_id="memtrap:%s" % sample_key_str,
        save_dir=str(root / sample_key_str),
        method=method,
        top_k=ctx.top_k,
    )


def iter_all_samples(
    datasets: Optional[List[str]] = None,
    *,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    """Flatten selected datasets into rows with ``dataset`` / ``sample_key`` fields."""
    wanted = list(datasets) if datasets else list(DATASETS)
    out: List[Dict[str, Any]] = []
    for ds in wanted:
        for row in load_dataset(ds, limit=limit):
            sid = str(row.get("id") or "")
            if not sid:
                continue
            item = dict(row)
            item["dataset"] = ds
            item["sample_key"] = sample_key(ds, sid)
            item["final_trigger"] = str(row.get("final_trigger") or "")
            out.append(item)
    return out


METHOD = "EvoMemState"
METHODS = (METHOD,)

__all__ = [
    "DATASETS",
    "GRAPH_VERSION",
    "METHOD",
    "METHODS",
    "build_context",
    "dataset_paths",
    "graph_dir_for",
    "history_to_messages",
    "history_to_prior_dialogue",
    "iter_all_samples",
    "load_dataset",
    "load_graph_only",
    "parse_sample_key",
    "sample_key",
    "_build_settings",
    "_get_embedding_client",
    "_graph_key",
    "_graph_root",
    "_load_or_build_graph",
]
