"""Built-in static rules encoding the EvoMemState method."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from evomemstate.prompting import (
    has_world_evidence,
    render_context,
    select_render_mode,
)
from evomemstate.retrieval import retrieve

from .context import HarnessContext
from .profile import ArmFlags, HarnessProfile
from .rules import register_rule


def _env_decompose_enabled() -> bool:
    return os.environ.get("EVOMEMSTATE_DECOMPOSE", "0").strip() in {
        "1",
        "true",
        "True",
        "yes",
    }


def _resolve_decompose(arm: ArmFlags) -> bool:
    flag = arm.decompose
    if flag == "env":
        return _env_decompose_enabled()
    return bool(flag)


@register_rule("record_graph_stats")
def record_graph_stats(ctx: HarnessContext, profile: HarnessProfile) -> None:
    """on_after_build: snapshot graph size and SUPERSEDES edges for audit."""
    del profile
    store = ctx.store
    if store is None:
        ctx.graph_stats = {}
        return
    ctx.graph_stats = {
        "n_memories_total": len(store.state.memories),
        "n_edges_supersedes": sum(
            1 for edge in store.state.edges.values() if edge.edge_type == "SUPERSEDES"
        ),
    }


@register_rule("resolve_arm_flags")
def resolve_arm_flags(ctx: HarnessContext, profile: HarnessProfile) -> None:
    """on_before_retrieve: apply method arm switches (flat / latest_wins / decompose)."""
    arm = profile.arm_for(ctx.method)
    ctx.flat = arm.flat
    ctx.latest_wins = arm.latest_wins
    ctx.decompose_enabled = _resolve_decompose(arm) and not ctx.flat


@register_rule("configure_retrieval_budget")
def configure_retrieval_budget(ctx: HarnessContext, profile: HarnessProfile) -> None:
    """on_before_retrieve: top-K and fact/preference channel split."""
    del profile
    top_k = max(1, int(getattr(ctx.eval_config, "top_k", None) or ctx.top_k))
    half = max(1, top_k // 2)
    ctx.top_k = top_k
    ctx.fact_top_n = half
    ctx.preference_top_n = top_k - half
    ctx.concept_top_k = int(
        ctx.build_settings.get("concept_top_k")
        or getattr(ctx.eval_config, "concept_top_k", None)
        or 8
    )


@register_rule("attach_decompose_client")
def attach_decompose_client(ctx: HarnessContext, profile: HarnessProfile) -> None:
    """on_before_retrieve: LLM client for intent decomposition (when enabled)."""
    del profile
    ctx.decompose_client = None
    if not ctx.decompose_enabled or ctx.flat:
        return
    factory = ctx.llm_client_factory
    if factory is None:
        return
    graph_root = ctx.graph_root
    cache_dir = None
    if graph_root is not None:
        cache_dir = graph_root / ctx.graph_key / "decompose_cache"
    try:
        ctx.decompose_client = factory(ctx.eval_config, cache_dir=cache_dir)
    except Exception:  # noqa: BLE001 — fall back to lexical retrieval
        ctx.decompose_client = None


@register_rule("attach_embedding_client")
def attach_embedding_client(ctx: HarnessContext, profile: HarnessProfile) -> None:
    """on_before_retrieve: semantic similarity backend for all arms."""
    del profile
    if ctx.embedding_client is not None:
        return
    factory = ctx.embedding_client_factory
    if factory is not None:
        ctx.embedding_client = factory()


@register_rule("run_retrieval")
def run_retrieval(ctx: HarnessContext, profile: HarnessProfile) -> None:
    """Core retrieval step (invoked by pipeline between pre/post retrieve hooks)."""
    del profile
    if ctx.store is None:
        raise RuntimeError("run_retrieval requires ctx.store")
    ctx.relational_state = retrieve(
        ctx.store,
        ctx.user_question or "",
        client=ctx.decompose_client,
        embedding_client=ctx.embedding_client,
        concept_top_k=ctx.concept_top_k,
        fact_top_n=ctx.fact_top_n,
        preference_top_n=ctx.preference_top_n,
        flat=ctx.flat,
        latest_wins=ctx.latest_wins,
    )


@register_rule("compute_generation_signals")
def compute_generation_signals(ctx: HarnessContext, profile: HarnessProfile) -> None:
    """on_after_retrieve: derive presentation regime from retrieval signals."""
    del profile
    result = ctx.relational_state
    if result is None:
        ctx.has_world_evidence = False
        ctx.render_mode = ""
        return
    ctx.has_world_evidence = bool(has_world_evidence(result))
    ctx.render_mode = select_render_mode(result)


@register_rule("build_relational_state_payload")
def build_relational_state_payload(ctx: HarnessContext, profile: HarnessProfile) -> None:
    """on_after_retrieve: JSON-serializable retrieval audit list."""
    del profile
    result = ctx.relational_state
    if result is None:
        ctx.relational_state_payload = []
        return
    items: List[Dict[str, Any]] = []
    for item in result.all_memories():
        memory = item.memory
        items.append(
            {
                "id": memory.memory_id,
                "content": memory.content,
                "score": float(item.score),
                "channel": item.channel,
                "memory_type": memory.memory_type,
                "status": memory.status,
            }
        )
    ctx.relational_state_payload = items


@register_rule("signal_conditioned_render")
def signal_conditioned_render(ctx: HarnessContext, profile: HarnessProfile) -> None:
    """on_before_generation: render context_text from signals (no task routing)."""
    del profile
    result = ctx.relational_state
    if result is None:
        ctx.context_text = ""
        return
    from evomemstate.reasoning_state import reasoning_state_enabled, render_with_reasoning_state

    if reasoning_state_enabled():
        text, meta = render_with_reasoning_state(
            result,
            ctx.user_question or "",
            flat=ctx.flat,
            cache_dir=(
                (ctx.graph_root / ctx.graph_key / "reasoning_state_cache")
                if ctx.graph_root and ctx.graph_key
                else None
            ),
        )
        ctx.context_text = text
        ctx.render_mode = str(meta.get("render_mode") or ctx.render_mode)
        return
    ctx.context_text = render_context(
        result,
        flat=ctx.flat,
        include_intents=False,
    )
