"""Build EvoMemState memory contexts for the MemSyco-Bench harness."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Graph cache identity. Bumping GRAPH_VERSION forces a clean rebuild without
# overwriting older trees (key includes this field).
# ---------------------------------------------------------------------------
GRAPH_VERSION = "v3"
DEFAULT_CHUNK_MODE = "whole_dialogue"
DEFAULT_SLOT_POLICY = "compatible"
DEFAULT_CONCEPT_TOP_K = 8
DEFAULT_MEMORY_TOP_L = 40

_BUILD_LOCKS: Dict[str, threading.Lock] = {}
_BUILD_LOCKS_GUARD = threading.Lock()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _graph_root() -> Path:
    override = os.environ.get("EVOMEMSTATE_GRAPH_ROOT", "").strip()
    if override:
        return Path(override)
    return _project_root() / "outputs" / "memsyco_graphs_v3"


def _build_settings(eval_config: Any) -> Dict[str, Any]:
    model = (
        getattr(eval_config, "llm_model", None)
        or os.environ.get("MEMORY_LLM_MODEL")
        or "deepseek-v4-flash"
    )
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
    }


def _graph_key(prior_dialogue: str, settings: Dict[str, Any]) -> str:
    """Cache key for the EvoMemState graph."""
    normalized = " ".join((prior_dialogue or "").split())
    payload = {
        "model": settings["model"],
        "chunk_mode": settings["chunk_mode"],
        "slot_policy": settings["slot_policy"],
        "concept_top_k": settings["concept_top_k"],
        "memory_top_l": settings["memory_top_l"],
        "version": settings["version"],
    }
    digest = hashlib.sha256()
    digest.update(normalized.encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()[:24]


def _get_embedding_client():
    from evomemstate.embedding import embedding_client_from_env

    cache = _project_root() / "outputs" / "embedding_cache"
    return embedding_client_from_env(cache_dir=cache)


def _lock_for(key: str) -> threading.Lock:
    with _BUILD_LOCKS_GUARD:
        lock = _BUILD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _BUILD_LOCKS[key] = lock
        return lock


def _parse_messages(prior_dialogue: str) -> List[Dict[str, str]]:
    """Parse MemSyco prior_dialogue into role/content messages with turn_ids."""
    from baselines.common import parse_dialogue_to_messages

    raw = parse_dialogue_to_messages(prior_dialogue)
    messages: List[Dict[str, str]] = []
    for index, item in enumerate(raw, start=1):
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


def _dialogue_to_session(
    prior_dialogue: str,
    *,
    user_id: str,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    messages = _parse_messages(prior_dialogue)
    system_msgs = [m for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]

    chunk_mode = settings.get("chunk_mode") or DEFAULT_CHUNK_MODE
    if chunk_mode != "whole_dialogue":
        # MemSyco path is whole_dialogue; keep a safe fallback of one chunk.
        chunk = system_msgs + conversation
    else:
        chunk = system_msgs + conversation

    return {
        "session_id": "memsyco_whole",
        "timestamp": "",
        "user_id": user_id,
        "messages": chunk,
    }


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
        raise RuntimeError("MEMORY_API_KEY / MEMORY_BASE_URL (or eval_config equivalents) required to build graphs")
    return OpenAIJSONClient(
        model=str(model),
        api_key=str(api_key),
        base_url=str(base_url),
        cache_dir=cache_dir,
    )


def _load_or_build_graph(prior_dialogue: str, eval_config: Any):
    """Load cached ontology.json or build it once under a per-key lock."""
    from evomemstate.builder import MemoryBuilder
    from evomemstate.extractor import OntologyGuidedExtractor
    from evomemstate.ontology import OntologyState, OntologyStore

    settings = _build_settings(eval_config)
    key = _graph_key(prior_dialogue, settings)
    root = _graph_root()
    graph_dir = root / key
    state_path = graph_dir / "ontology.json"
    user_id = "memsyco:%s" % key

    if state_path.exists():
        return OntologyStore.load(state_path, user_id, slot_policy=settings["slot_policy"])

    lock = _lock_for(key)
    with lock:
        if state_path.exists():
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
        session = _dialogue_to_session(prior_dialogue, user_id=user_id, settings=settings)
        if session["messages"]:
            try:
                builder.build_session(session)
            except Exception as exc:
                # Keep evaluation cells alive when a single dialogue yields a hard
                # extraction/apply failure; an empty graph is better than aborting.
                print(
                    "[evomemstate] build_session failed for %s: %s: %s"
                    % (key, type(exc).__name__, exc),
                    flush=True,
                )
                (graph_dir / "build_error.txt").write_text(
                    "%s: %s\n" % (type(exc).__name__, exc),
                    encoding="utf-8",
                )
        store.save(state_path)
        (graph_dir / "build_settings.json").write_text(
            json.dumps({**settings, "graph_key": key}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return store


def _ablation_flags(method: str) -> Tuple[bool, bool]:
    """Return (flat, latest_wins) for a registered method name (legacy helper)."""
    from evomemstate.harness.profile import load_profile

    arm = load_profile().arm_for(method)
    return arm.flat, arm.latest_wins


def _append_audit(record: Dict[str, Any]) -> None:
    path = os.environ.get("EVOMEMSTATE_AUDIT_PATH", "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def build_context(
    prior_dialogue: str,
    user_question: str,
    eval_config: Any,
    *,
    sample_key: str | int | None = None,
):
    """MemSyco harness entry: build/load graph, retrieve, render context_text."""
    from baselines.base import BaselineContext
    from baselines.common import jsonable_memories
    from evomemstate.harness import HarnessContext, run_context_pipeline

    method = str(getattr(eval_config, "method", None) or "EvoMemState")
    settings = _build_settings(eval_config)
    key = _graph_key(prior_dialogue, settings)
    store = _load_or_build_graph(prior_dialogue, eval_config)
    root = _graph_root()

    ctx = HarnessContext(
        prior_dialogue=prior_dialogue,
        user_question=user_question or "",
        eval_config=eval_config,
        method=method,
        sample_key=sample_key,
        store=store,
        graph_key=key,
        build_settings=settings,
        llm_client_factory=_llm_client,
        embedding_client_factory=_get_embedding_client,
        graph_root=root,
    )
    run_context_pipeline(ctx)

    result = ctx.relational_state
    save_dir = str(root / key)
    retrieved = ctx.relational_state_payload

    _append_audit(
        {
            "method": method,
            "sample_key": "" if sample_key is None else str(sample_key),
            "graph_key": key,
            "question": user_question or "",
            "rules_version": ctx.rules_version,
            "rules_profile": ctx.rules_profile,
            "applied_rules": list(ctx.applied_rules),
            "n_memories_total": ctx.graph_stats.get(
                "n_memories_total", len(store.state.memories)
            ),
            "n_edges_supersedes": ctx.graph_stats.get(
                "n_edges_supersedes",
                sum(
                    1
                    for e in store.state.edges.values()
                    if e.edge_type == "SUPERSEDES"
                ),
            ),
            "n_retrieved": len(retrieved),
            "n_facts": len(result.facts) if result else 0,
            "n_preferences": len(result.preferences) if result else 0,
            "n_dropped_superseded": len(result.dropped_superseded) if result else 0,
            "n_demoted_superseded": len(result.demoted_superseded) if result else 0,
            "latest_wins_applied": bool(result.latest_wins_applied) if result else False,
            "latest_wins_mode": getattr(result, "latest_wins_mode", "off") if result else "off",
            "fact_need": float(result.fact_need) if result else 0.0,
            "preference_need": float(result.preference_need) if result else 0.0,
            "has_world_evidence": ctx.has_world_evidence,
            "render_mode": ctx.render_mode,
            "embedding_used": bool(result.embedding_used) if result else False,
            "dropped_superseded": result.dropped_superseded if result else [],
            "demoted_superseded": result.demoted_superseded if result else [],
        }
    )

    return BaselineContext(
        context_text=ctx.context_text,
        retrieved_memories=jsonable_memories(retrieved),
        user_id="memsyco:%s" % key,
        save_dir=save_dir,
        method=method,
        top_k=ctx.top_k,
    )


# Public names expected by prebuild / diagnose / registry.
METHOD = "EvoMemState"
METHODS = (METHOD,)

__all__ = [
    "GRAPH_VERSION",
    "METHOD",
    "METHODS",
    "build_context",
    "_build_settings",
    "_get_embedding_client",
    "_graph_key",
    "_graph_root",
    "_load_or_build_graph",
]
