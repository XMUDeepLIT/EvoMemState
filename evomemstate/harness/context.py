"""Mutable state bag flowing through harness lifecycle hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evomemstate.ontology import OntologyStore
from evomemstate.retrieval import RelationalState


@dataclass
class HarnessContext:
    """All inputs and intermediate artifacts for one MemSyco context build."""

    prior_dialogue: str
    user_question: str
    eval_config: Any
    method: str = "EvoMemState"
    sample_key: str | int | None = None

    # Graph / build
    store: Optional[OntologyStore] = None
    graph_key: str = ""
    build_settings: Dict[str, Any] = field(default_factory=dict)

    # Retrieval plan (filled by on_before_retrieve rules)
    flat: bool = False
    latest_wins: bool = True
    decompose_enabled: bool = False
    decompose_client: Any = None
    embedding_client: Any = None
    top_k: int = 10
    fact_top_n: int = 5
    preference_top_n: int = 5
    concept_top_k: int = 8

    # Retrieval output
    relational_state: Optional[RelationalState] = None

    # Generation signals + output
    has_world_evidence: bool = False
    render_mode: str = ""
    context_text: str = ""
    relational_state_payload: List[Dict[str, Any]] = field(default_factory=list)

    # Audit trail (Harness-R1 patch log / SKILLOPT skill version)
    graph_stats: Dict[str, Any] = field(default_factory=dict)
    applied_rules: List[str] = field(default_factory=list)
    rules_version: str = ""
    rules_profile: str = ""

    # Injectable deps for tests (avoid importing adapter internals)
    llm_client_factory: Any = None
    embedding_client_factory: Any = None
    graph_root: Any = None
