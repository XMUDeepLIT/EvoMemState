"""Retrieve a request-specific Relational State from the memory ontology.

Two main paths (``flat`` ablation aside):

**No intent decomposition** (``EVOMEMSTATE_DISABLE_INTENT_DECOMP=1`` or
``client=None``):
    1. embedding similarity retrieves the top-K seed memories
    2. each seed is replaced by one evidence memory: the seed plus the
       SUPERSEDES / CONFLICTS_WITH closure, with the relations written into
       the text
    3. no second score (disputed / supersede penalty / channel need) and no
       per-memory Top-K that could drop one end of a relation

**With intent decomposition** (default when a decomp ``client`` is provided):
    1. LLM intent decomp → expanded query
    2. concept localization + semantic backfill (seeds)
    3. same SUPERSEDES + CONFLICTS_WITH chaining
    4. same simplified score → top-K (or full pool if ``return_all_scored``)

Struct pool path (``EVOMEMSTATE_STRUCT_CANDIDATE_POOL=1``):
    concept-full ∪ semantic top-K as the candidate pool (no latest-wins /
    need re-score). CONFLICTS_WITH counterparts are still pulled in.
    The selected memories and their relations feed the Reasoning State policy.

Ranking similarity for seed + final score is **embedding cosine only** when
an embedding client is available; lexical Jaccard is fallback if embeddings
fail (tests / offline).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .models import Memory
from .ontology import OntologyStore, _similarity


# A memory typed `constraint` is relevant to both channels: it is a hard limit
# on what may be recommended (preference-like) and a durable truth about the
# user (fact-like). It is emitted to both and de-duplicated when counted.
FACT_TYPES = frozenset({"fact", "experience", "constraint"})
PREFERENCE_TYPES = frozenset({"preference", "belief", "constraint"})

DISPUTED_PENALTY = 0.10
# Extra channel need weight when preference_need dominates fact_need.
PREF_DOMINANT_NEED_EXTRA = 0.35
# Kept for older tests/imports; scoring no longer uses these.
SCOPE_MATCH_BONUS = 0.0
CONFIDENCE_WEIGHT = 0.0
WORLD_EVIDENCE_BONUS = 0.0
ANAPHORA_SCOPE_BONUS = 0.0

# Candidate-pool modes (see ``_struct_candidate_pool_enabled``).
POOL_MODE_NEED_SCORED = "need_scored"  # decomp → concept∪backfill → score top-K
POOL_MODE_STRUCT = "struct"  # concept-full ∪ semantic top-K only
POOL_MODE_EMB_SEED = "emb_seed"  # no decomp: embedding seed → relation-complete memory

# Share of the similarity term taken by embedding cosine. 0.0 = pure lexical
# (the pre-embedding behaviour), 1.0 = pure semantic. The default leans on
# embeddings for recall while retaining lexical precision.
DEFAULT_EMBEDDING_WEIGHT = 0.7
DEFAULT_SUPERSEDE_PENALTY = 0.15
DEFAULT_QUERY_ANCHOR_WEIGHT = 0.6
DEFAULT_CHANNEL_NEED_BONUS = 0.5

_ANAPHORA_CUES = (
    "situation we discussed",
    "the situation",
    "this situation",
    "that situation",
    "we discussed",
    "this case",
    "that case",
    "in that",
    "earlier",
    "that time",
    "this time",
    "the context",
    "as discussed",
)


def _intent_decomp_disabled() -> bool:
    """When set, skip LLM intent decomp and concept localization / backfill."""
    return os.environ.get("EVOMEMSTATE_DISABLE_INTENT_DECOMP", "").strip().lower() in {
        "1",
        "true",
        "True",
        "yes",
        "on",
    }


def _embedding_weight() -> float:
    try:
        value = float(os.environ.get("EMBEDDING_WEIGHT", DEFAULT_EMBEDDING_WEIGHT))
    except ValueError:
        return DEFAULT_EMBEDDING_WEIGHT
    return min(1.0, max(0.0, value))


def _supersede_penalty() -> float:
    """Multiplier applied to superseded memories under soft latest-wins.

    ``0`` disables demotion (treated as no penalty). Default ``0.15`` keeps
    replaced preferences out of the top ranks without deleting them.
    """
    try:
        value = float(
            os.environ.get("EVOMEMSTATE_SUPERSEDE_PENALTY", DEFAULT_SUPERSEDE_PENALTY)
        )
    except ValueError:
        return DEFAULT_SUPERSEDE_PENALTY
    return min(1.0, max(0.0, value))


def _query_anchor_weight() -> float:
    try:
        value = float(
            os.environ.get(
                "EVOMEMSTATE_QUERY_ANCHOR_WEIGHT", DEFAULT_QUERY_ANCHOR_WEIGHT
            )
        )
    except ValueError:
        return DEFAULT_QUERY_ANCHOR_WEIGHT
    return min(1.0, max(0.0, value))


def _channel_need_bonus() -> float:
    """How strongly fact_need/preference_need reweight global ranking.

    ``0`` recovers pure similarity (flat-like selection). Default ``0.5`` still
    boosts the needed channel on evid/obj without hard quotas that starve pers.
    """
    try:
        value = float(
            os.environ.get(
                "EVOMEMSTATE_CHANNEL_NEED_BONUS", DEFAULT_CHANNEL_NEED_BONUS
            )
        )
    except ValueError:
        return DEFAULT_CHANNEL_NEED_BONUS
    return min(2.0, max(0.0, value))


def _latest_wins_hard() -> bool:
    return os.environ.get("EVOMEMSTATE_LATEST_WINS_HARD", "").strip().lower() in {
        "1",
        "true",
        "True",
        "yes",
    }


def _struct_candidate_pool_enabled() -> bool:
    """Concept-full ∪ semantic top-K pool; skip latest-wins + need re-score.

    Enable with ``EVOMEMSTATE_STRUCT_CANDIDATE_POOL=1``.

    When off (default), keep the need-weighted global re-score path with
    soft latest-wins. When on, related memories remain available for relation-aware rendering.
    """
    return os.environ.get("EVOMEMSTATE_STRUCT_CANDIDATE_POOL", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _global_reserve_slots(total_k: int) -> int:
    """Deprecated helper kept for tests/callers; selection no longer uses it."""
    if total_k < 2:
        return 0
    return max(total_k // 2, 2)


def _need_multiplier(
    memory: Memory,
    fact_need: float,
    preference_need: float,
    bonus: float,
) -> float:
    if bonus <= 0.0:
        return 1.0
    in_fact = memory.memory_type in FACT_TYPES
    in_pref = memory.memory_type in PREFERENCE_TYPES
    effective_bonus = float(bonus)
    # Preference-led questions (pers-like) need a stronger soft push toward
    # preference/belief memories without restoring hard dual-channel quotas.
    if float(preference_need) > float(fact_need):
        effective_bonus = float(bonus) + PREF_DOMINANT_NEED_EXTRA
    if in_fact and in_pref:
        need = max(fact_need, preference_need)
    elif in_pref:
        need = preference_need
    elif in_fact:
        need = fact_need
    else:
        need = 0.5
    return 1.0 + effective_bonus * max(0.0, float(need))


def _conflict_adjacency(store: OntologyStore) -> Dict[str, Set[str]]:
    """Undirected CONFLICTS_WITH neighbors keyed by memory_id."""
    adj: Dict[str, Set[str]] = {}
    for edge in store.state.edges.values():
        if edge.edge_type != "CONFLICTS_WITH":
            continue
        src, tgt = edge.source_id, edge.target_id
        if not src or not tgt or src == tgt:
            continue
        adj.setdefault(src, set()).add(tgt)
        adj.setdefault(tgt, set()).add(src)
    return adj


def _chain_conflicts(
    store: OntologyStore,
    eligible: List[Memory],
    by_id: Dict[str, Memory],
    result: "RelationalState",
) -> List[Memory]:
    """Pull CONFLICTS_WITH counterparts into the pool (relation-complete bundles).

    Seed retrieval may only hit one side of a conflict. Chaining restores the
    other endpoint so the answer model sees both, without resolving the conflict.
    """
    adj = _conflict_adjacency(store)
    if not adj:
        return eligible
    known = {m.memory_id for m in eligible}
    out = list(eligible)
    # One-hop expansion is enough for undirected conflict pairs.
    for mid in list(known):
        for other_id in adj.get(mid, ()):
            if other_id in known:
                continue
            other = by_id.get(other_id)
            if other is None:
                continue
            out.append(other)
            known.add(other_id)
            result.chained_conflicts.append(
                {
                    "memory_id": mid,
                    "conflicts_with": other_id,
                    "content": (by_id.get(mid).content if by_id.get(mid) else ""),
                    "other_content": other.content,
                }
            )
    return out


def _is_world_evidence_memory(memory: Memory) -> bool:
    if memory.memory_type != "fact":
        return False
    level = str((memory.scope or {}).get("level", "")).strip().lower()
    return level in {"domain", "task"}


def _is_narrow_scope_memory(memory: Memory) -> bool:
    level = str((memory.scope or {}).get("level", "")).strip().lower()
    return level in {"situation", "task"}


def _question_has_anaphora(question: str) -> bool:
    text = (question or "").strip().lower()
    if not text:
        return False
    return any(cue in text for cue in _ANAPHORA_CUES)


def _signal_additive_bonus(
    memory: Memory,
    *,
    question: str,
    fact_need: float,
    anaphora: bool,
) -> float:
    """Small additive boosts from content/question signals (no task names)."""
    bonus = 0.0
    if fact_need >= 0.45 and _is_world_evidence_memory(memory):
        bonus += WORLD_EVIDENCE_BONUS * max(0.0, float(fact_need))
    if anaphora and _is_narrow_scope_memory(memory):
        bonus += ANAPHORA_SCOPE_BONUS
    return bonus



INTENT_PROMPT = """You decompose a user question into retrieval intents.

Return exactly one JSON object:
{
  "intents": ["one short sub-question or information need"],
  "topic_terms": ["salient noun phrase"],
  "fact_need": 0.0,
  "preference_need": 0.0
}

Rules:
1. Produce 1-4 intents. Each is a self-contained information need, not a restatement.
2. topic_terms are the concrete entities, domains, and activities named or implied. 3-8 terms.
3. fact_need is how much answering depends on objective state, history, evidence, or hard constraints (0-1).
4. preference_need is how much the answer should be tailored to the user's tastes or beliefs (0-1).
5. Both may be high. They need not sum to 1; the system will normalize them. Output JSON only."""


# Cheap lexical priors used when intent decomposition is unavailable or unusable.
_FACT_NEED_CUES = (
    "when", "where", "who", "whom", "whose", "which", "how many", "how much",
    "how long", "what date", "what year", "what time", "did ", "was ", "were ",
    "happened", "occurred", "evidence", "fact", "true", "false",
)
_PREF_NEED_CUES = (
    "prefer", "preference", "like", "dislike", "love", "hate", "want",
    "recommend", "suggestion", "should i", "should we", "better for me",
    "taste", "favorite", "favourite", "leaning",
)


@dataclass
class ScoredMemory:
    memory: Memory
    score: float
    channel: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory.memory_id,
            "content": self.memory.content,
            "type": self.memory.memory_type,
            "scope": dict(self.memory.scope),
            "status": self.memory.status,
            "confidence": self.memory.confidence,
            "channel": self.channel,
            "score": round(self.score, 4),
            "evidence_quote": (self.memory.source or {}).get("evidence_quote", ""),
        }


@dataclass
class RelationalState:
    """Memories and relations retrieved for one question from the ontology."""

    intents: List[str] = field(default_factory=list)
    topic_terms: List[str] = field(default_factory=list)
    selected_concept_ids: List[str] = field(default_factory=list)
    facts: List[ScoredMemory] = field(default_factory=list)
    preferences: List[ScoredMemory] = field(default_factory=list)
    dropped_superseded: List[Dict[str, str]] = field(default_factory=list)
    demoted_superseded: List[Dict[str, str]] = field(default_factory=list)
    chained_conflicts: List[Dict[str, str]] = field(default_factory=list)
    assembled_bundles: List[Dict[str, Any]] = field(default_factory=list)
    intent_decomposition_failed: bool = False
    intent_decomposition_used: bool = False
    latest_wins_applied: bool = True
    latest_wins_mode: str = "off"  # "soft" | "hard" | "off"
    embedding_used: bool = False
    fact_need: float = 0.5
    preference_need: float = 0.5
    fact_top_n: int = 0
    preference_top_n: int = 0
    # "need_scored" (default) | "struct" (concept ∪ semantic top-K) | "emb_seed"
    candidate_pool_mode: str = POOL_MODE_NEED_SCORED
    n_concept_memories: int = 0
    n_semantic_backfill: int = 0
    n_embedding_seeds: int = 0

    def all_memories(self) -> List[ScoredMemory]:
        seen: Set[str] = set()
        out: List[ScoredMemory] = []
        for item in list(self.facts) + list(self.preferences):
            if item.memory.memory_id in seen:
                continue
            seen.add(item.memory.memory_id)
            out.append(item)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intents": self.intents,
            "topic_terms": self.topic_terms,
            "selected_concept_ids": self.selected_concept_ids,
            "facts": [item.to_dict() for item in self.facts],
            "preferences": [item.to_dict() for item in self.preferences],
            "dropped_superseded": self.dropped_superseded,
            "demoted_superseded": self.demoted_superseded,
            "chained_conflicts": self.chained_conflicts,
            "assembled_bundles": self.assembled_bundles,
            "intent_decomposition_failed": self.intent_decomposition_failed,
            "intent_decomposition_used": self.intent_decomposition_used,
            "latest_wins_applied": self.latest_wins_applied,
            "latest_wins_mode": self.latest_wins_mode,
            "embedding_used": self.embedding_used,
            "fact_need": round(self.fact_need, 4),
            "preference_need": round(self.preference_need, 4),
            "fact_top_n": self.fact_top_n,
            "preference_top_n": self.preference_top_n,
            "candidate_pool_mode": self.candidate_pool_mode,
            "n_concept_memories": self.n_concept_memories,
            "n_semantic_backfill": self.n_semantic_backfill,
            "n_embedding_seeds": self.n_embedding_seeds,
        }


def _clamp01(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return min(1.0, max(0.0, number))


def _parse_channel_needs(payload: Dict[str, Any]) -> Tuple[float, float]:
    """Read fact_need / preference_need, with bool-field fallback for old caches."""
    if "fact_need" in payload or "preference_need" in payload:
        fact_need = _clamp01(payload.get("fact_need"), 0.5)
        preference_need = _clamp01(payload.get("preference_need"), 0.5)
        return fact_need, preference_need

    # Backward compatible with the previous needs_facts / needs_preferences booleans.
    needs_facts = payload.get("needs_facts")
    needs_preferences = payload.get("needs_preferences")
    if isinstance(needs_facts, bool) or isinstance(needs_preferences, bool):
        fact_need = 0.8 if needs_facts is True else (0.2 if needs_facts is False else 0.5)
        preference_need = (
            0.8 if needs_preferences is True else (0.2 if needs_preferences is False else 0.5)
        )
        return fact_need, preference_need
    return 0.5, 0.5


def estimate_channel_needs(question: str) -> Tuple[float, float]:
    """Lexical prior for channel mix when the LLM decomposition is unavailable."""
    text = (question or "").casefold()
    if not text:
        return 0.5, 0.5
    fact_hits = sum(1 for cue in _FACT_NEED_CUES if cue in text)
    pref_hits = sum(1 for cue in _PREF_NEED_CUES if cue in text)
    if fact_hits == 0 and pref_hits == 0:
        return 0.5, 0.5
    # Laplace smoothing keeps a single cue from collapsing the other channel to 0.
    fact_need = (fact_hits + 1.0) / (fact_hits + pref_hits + 2.0)
    preference_need = 1.0 - fact_need
    return fact_need, preference_need


def allocate_channel_budget(
    total_k: int,
    fact_need: float,
    preference_need: float,
) -> Tuple[int, int]:
    """Split a fixed total budget across fact/preference channels.

    Keeps total context size comparable to the old hard half-split, while
    letting question-conditioned needs move the cut. Each channel keeps at
    least ``max(1, total_k // 5)`` slots when ``total_k >= 2``.
    """
    total_k = max(1, int(total_k))
    fact_need = max(0.0, float(fact_need))
    preference_need = max(0.0, float(preference_need))
    weight_sum = fact_need + preference_need
    if weight_sum <= 0.0:
        fact_share = 0.5
    else:
        fact_share = fact_need / weight_sum

    if total_k == 1:
        return (1, 0) if fact_share >= 0.5 else (0, 1)

    floor = min(max(1, total_k // 5), total_k // 2)
    fact_k = int(round(total_k * fact_share))
    fact_k = max(floor, min(total_k - floor, fact_k))
    return fact_k, total_k - fact_k


def decompose_intent(
    question: str, client: Any = None
) -> Tuple[List[str], List[str], bool, float, float]:
    """Stage 1. Returns intents, topic_terms, failed, fact_need, preference_need.

    Never raises: on any failure the raw question is used as the sole intent and
    channel needs fall back to a lexical prior, so retrieval degrades instead of
    aborting the sample. ``failed`` is True whenever intents are just the raw
    question -- callers use it to suppress echoing the question into the prompt.
    """
    question = (question or "").strip()
    prior_fact, prior_pref = estimate_channel_needs(question)
    if client is None or not question:
        return ([question] if question else []), [], True, prior_fact, prior_pref

    try:
        payload = client.complete_json(
            [
                {"role": "system", "content": INTENT_PROMPT},
                {"role": "user", "content": json.dumps({"question": question}, ensure_ascii=False)},
            ]
        )
        if not isinstance(payload, dict):
            return [question], [], True, prior_fact, prior_pref
        intents = [str(item).strip() for item in payload.get("intents", []) if str(item).strip()]
        terms = [str(item).strip() for item in payload.get("topic_terms", []) if str(item).strip()]
        fact_need, preference_need = _parse_channel_needs(payload)
        if not intents:
            return [question], terms, True, fact_need, preference_need
        return intents[:4], terms[:8], False, fact_need, preference_need
    except Exception:  # noqa: BLE001 - retrieval must survive a bad decomposition
        return [question], [], True, prior_fact, prior_pref


def _supersede_heads(store: OntologyStore) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Map every superseded memory to the memory that replaced it.

    Builds the SUPERSEDES adjacency once. ``ontology._has_supersedes_path``
    rebuilds it on every call, which is fine for validation but wasteful when
    walking many candidates per query.
    """
    replaced_by: Dict[str, str] = {}
    for edge in store.state.edges.values():
        if edge.edge_type == "SUPERSEDES":
            # edge: newer(source) SUPERSEDES older(target)
            replaced_by[edge.target_id] = edge.source_id

    heads: Dict[str, str] = {}
    for memory_id in replaced_by:
        seen: Set[str] = {memory_id}
        current = memory_id
        while current in replaced_by:
            nxt = replaced_by[current]
            if nxt in seen:  # defensive; commit-time validation rejects cycles
                break
            seen.add(nxt)
            current = nxt
        heads[memory_id] = current
    return replaced_by, heads


def _scope_matches(memory: Memory, query_text: str) -> bool:
    value = str((memory.scope or {}).get("value", "")).strip()
    if not value:
        return False
    return _similarity(value, query_text) > 0.0


# `_similarity` returns Jaccard (0..1) plus a 0.25 substring bonus, so its range
# is 0..1.25. Divide before blending, otherwise the lexical term would carry
# more than its nominal share of the weight.
_LEXICAL_MAX = 1.25


class _Similarity:
    """Query↔memory similarity with optional embeddings.

    Production ranking uses ``semantic`` (embedding cosine only). ``blend`` is
    kept for older callers / flat ablation fallbacks. When embeddings are
    unavailable, ``semantic`` falls back to lexical Jaccard so offline tests
    still rank somehow.
    """

    def __init__(
        self,
        query_text: str,
        memories: Sequence[Memory],
        client: Any = None,
        weight: Optional[float] = None,
    ) -> None:
        self.query_text = (query_text or "").strip()
        self.weight = _embedding_weight() if weight is None else weight
        self.query_vector: Optional[List[float]] = None
        self.vectors: Dict[str, List[float]] = {}
        self.enabled = False

        if client is None or self.weight <= 0.0:
            return
        try:
            from .embedding import cosine  # noqa: F401 - presence check

            texts = [self.query_text] + [m.content for m in memories]
            vectors = client.embed(texts)
            self.query_vector = vectors[0]
            for memory, vector in zip(memories, vectors[1:]):
                if vector is not None:
                    self.vectors[memory.memory_id] = vector
            self.enabled = self.query_vector is not None and bool(self.vectors)
        except Exception:  # noqa: BLE001 - retrieval must survive embedding failure
            self.query_vector = None
            self.vectors = {}
            self.enabled = False

    def lexical(self, memory: Memory) -> float:
        return _similarity(self.query_text, memory.content) / _LEXICAL_MAX

    def semantic(self, memory: Memory) -> float:
        """Embedding cosine in [0, 1]; lexical fallback if embeddings missing."""
        vector = self.vectors.get(memory.memory_id) if self.enabled else None
        if vector is None or self.query_vector is None:
            return self.lexical(memory)
        from .embedding import cosine

        return max(0.0, cosine(self.query_vector, vector))

    def blend(self, query_text: str, memory: Memory) -> float:
        # Prefer embedding-only ranking; keep signature for older call sites.
        if query_text and query_text.strip() != self.query_text:
            # Rare: rescoring against a different string — lexical only for that.
            return _similarity(query_text, memory.content) / _LEXICAL_MAX
        return self.semantic(memory)


def _score(
    memory: Memory,
    sim: _Similarity,
    *,
    supersede_factor: float = 1.0,
) -> float:
    """Simplified rank score: emb_sim − disputed, then × supersede demotion.

    Channel-need reweighting is applied by the caller after this returns.
    Scope / confidence / signal additives are intentionally omitted.
    """
    score = float(sim.semantic(memory))
    if memory.status == "disputed":
        score -= DISPUTED_PENALTY
    if supersede_factor != 1.0:
        score *= supersede_factor
    return score


def _channel_for(memory: Memory) -> str:
    if memory.memory_type in PREFERENCE_TYPES and memory.memory_type not in FACT_TYPES:
        return "preference"
    if memory.memory_type in FACT_TYPES and memory.memory_type not in PREFERENCE_TYPES:
        return "fact"
    if memory.memory_type in PREFERENCE_TYPES:
        return "preference"
    return "fact"


def _partition_scored(items: Sequence[ScoredMemory]) -> Tuple[List[ScoredMemory], List[ScoredMemory]]:
    facts: List[ScoredMemory] = []
    preferences: List[ScoredMemory] = []
    for item in items:
        memory = item.memory
        if memory.memory_type in FACT_TYPES:
            facts.append(
                ScoredMemory(memory=memory, score=item.score, channel="fact")
            )
        if memory.memory_type in PREFERENCE_TYPES:
            preferences.append(
                ScoredMemory(memory=memory, score=item.score, channel="preference")
            )
    return facts, preferences


def _state_relation_graph(
    store: OntologyStore,
) -> Tuple[Dict[str, Set[str]], List[Dict[str, str]]]:
    """Undirected closure graph for SUPERSEDES and CONFLICTS_WITH.

    SUPERSEDES stays directed in the edge records (source is the newer memory).
    CONFLICTS_WITH is stored once per unordered pair.
    """
    adjacency: Dict[str, Set[str]] = {}
    relations: List[Dict[str, str]] = []
    seen_conflicts: Set[Tuple[str, str]] = set()
    for edge in store.state.edges.values():
        if edge.edge_type not in {"SUPERSEDES", "CONFLICTS_WITH"}:
            continue
        src, tgt = edge.source_id, edge.target_id
        if not src or not tgt or src == tgt:
            continue
        if edge.edge_type == "CONFLICTS_WITH":
            key = (src, tgt) if src < tgt else (tgt, src)
            if key in seen_conflicts:
                continue
            seen_conflicts.add(key)
        adjacency.setdefault(src, set()).add(tgt)
        adjacency.setdefault(tgt, set()).add(src)
        reason = ""
        if isinstance(edge.metadata, dict):
            reason = str(edge.metadata.get("reason", "")).strip()
        relations.append(
            {
                "type": edge.edge_type,
                "source_id": src,
                "target_id": tgt,
                "reason": reason,
            }
        )
    return adjacency, relations


def _relation_closure(seed_id: str, adjacency: Dict[str, Set[str]]) -> Set[str]:
    seen: Set[str] = set()
    stack = [seed_id]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(adjacency.get(current, ()))
    return seen


def memory_annotation_suffix(memory: Memory, *, supersede_by: str = "") -> str:
    """Same trailing metadata already used for a single retrieved memory."""
    scope = memory.scope or {}
    level = str(scope.get("level", "")).strip()
    value = str(scope.get("value", "")).strip()
    annotations: List[str] = []
    if level:
        annotations.append("scope: %s%s" % (level, (" -- %s" % value) if value else ""))
    time = memory.time or {}
    raw = (time.get("valid_from") or "").strip() or (time.get("observed_at") or "").strip()
    date_part = raw[:10]
    if len(date_part) == 10 and date_part[4] == "-" and date_part[7] == "-":
        annotations.append("date: %s" % date_part)
    if memory.status == "disputed":
        annotations.append("disputed")
    if supersede_by:
        short = " ".join(supersede_by.split())
        if len(short) > 120:
            short = short[:117] + "..."
        annotations.append("superseded by: %s" % short)
    elif memory.status == "superseded":
        annotations.append("superseded")
    if not annotations:
        return ""
    return "  (%s)" % "; ".join(annotations)


def _format_relation_memory(
    seed: Memory,
    members: Sequence[Memory],
    relations: Sequence[Dict[str, str]],
) -> str:
    """Render each related memory with its existing metadata, then the relation."""
    ordered = [seed] + [m for m in members if m.memory_id != seed.memory_id]
    by_member = {m.memory_id: m for m in ordered}
    lines = [
        "%s%s" % (memory.content.strip(), memory_annotation_suffix(memory))
        for memory in ordered
    ]
    written: Set[Tuple[str, str, str]] = set()
    for rel in relations:
        src, tgt = rel["source_id"], rel["target_id"]
        if src not in by_member or tgt not in by_member:
            continue
        kind = rel["type"]
        key = (kind, src, tgt)
        if key in written:
            continue
        written.add(key)
        lines.append(
            'Relation: "%s" %s "%s"'
            % (by_member[src].content.strip(), kind, by_member[tgt].content.strip())
        )
    return "\n".join(lines)


def _bundle_memory(
    seed: Memory,
    members: Sequence[Memory],
    relations: Sequence[Dict[str, str]],
) -> Memory:
    ordered_ids = [seed.memory_id] + sorted(
        m.memory_id for m in members if m.memory_id != seed.memory_id
    )
    return Memory(
        memory_id="bundle:" + "+".join(sorted(m.memory_id for m in members)),
        content=_format_relation_memory(seed, members, relations),
        memory_type=seed.memory_type,
        scope=dict(seed.scope or {}),
        time=dict(seed.time or {}),
        source={
            "bundle": True,
            "seed_id": seed.memory_id,
            "member_ids": ordered_ids,
            "relations": [
                {
                    "type": rel["type"],
                    "source_id": rel["source_id"],
                    "target_id": rel["target_id"],
                }
                for rel in relations
                if rel["source_id"] in {m.memory_id for m in members}
                and rel["target_id"] in {m.memory_id for m in members}
            ],
        },
        # Member status stays inside the text. The unit itself is not demoted.
        status="active",
        parent_concept_id=seed.parent_concept_id,
        confidence=seed.confidence,
    )


def _assemble_relation_bundles(
    store: OntologyStore,
    seeds: Sequence[Memory],
    by_id: Dict[str, Memory],
    sim: _Similarity,
    result: RelationalState,
) -> List[ScoredMemory]:
    """Replace each similarity hit with its relation-complete evidence memory.

    Seeds that fall in the same SUPERSEDES / CONFLICTS_WITH component become
    one memory. Nothing inside that component is dropped by a later score.
    """
    adjacency, relations = _state_relation_graph(store)
    covered: Set[str] = set()
    out: List[ScoredMemory] = []
    for seed in seeds:
        if seed.memory_id in covered:
            continue
        component = _relation_closure(seed.memory_id, adjacency)
        covered.update(component)
        members = [by_id[mid] for mid in component if mid in by_id]
        if len(members) <= 1:
            out.append(
                ScoredMemory(
                    memory=seed,
                    score=float(sim.semantic(seed)),
                    channel=_channel_for(seed),
                )
            )
            continue
        bundled = _bundle_memory(seed, members, relations)
        result.assembled_bundles.append(
            {
                "memory_id": bundled.memory_id,
                "seed_id": seed.memory_id,
                "member_ids": list((bundled.source or {}).get("member_ids") or []),
                "relations": list((bundled.source or {}).get("relations") or []),
            }
        )
        out.append(
            ScoredMemory(
                memory=bundled,
                score=float(sim.semantic(seed)),
                channel=_channel_for(seed),
            )
        )
    return out


def retrieve(
    store: OntologyStore,
    question: str,
    *,
    client: Any = None,
    embedding_client: Any = None,
    concept_top_k: int = 8,
    fact_top_n: int = 8,
    preference_top_n: int = 8,
    flat: bool = False,
    latest_wins: bool = True,
    return_all_scored: bool = False,
) -> RelationalState:
    """Run retrieval over a built ontology.

    **No intent decomp** (``EVOMEMSTATE_DISABLE_INTENT_DECOMP=1`` or ``client=None``):
      embedding similarity top-K → replace each seed with one relation-complete
      memory (SUPERSEDES / CONFLICTS_WITH closure written into the text).
      No score-function filter after that.

    **With intent decomp** (decomp client provided and not disabled):
      intent → concept localize + semantic backfill → same chaining → same score.

    Struct pool (``EVOMEMSTATE_STRUCT_CANDIDATE_POOL=1``): concept∪sim top-K pool
    without latest-wins / need re-score; CONFLICTS_WITH counterparts still pulled.

    ``flat=True``: plain emb similarity over all memories, no structure.

    ``embedding_client`` is used for seed ranking and final scores.
    """
    result = RelationalState()
    struct_pool = (not flat) and _struct_candidate_pool_enabled()
    use_decomp = (
        (not flat)
        and (client is not None)
        and (not _intent_decomp_disabled())
    )
    result.intent_decomposition_used = bool(use_decomp)
    if flat:
        result.candidate_pool_mode = POOL_MODE_NEED_SCORED
    elif struct_pool:
        result.candidate_pool_mode = POOL_MODE_STRUCT
    elif use_decomp:
        result.candidate_pool_mode = POOL_MODE_NEED_SCORED
    else:
        result.candidate_pool_mode = POOL_MODE_EMB_SEED

    hard_wins = (
        (not struct_pool)
        and bool(latest_wins)
        and _latest_wins_hard()
        and not flat
    )
    soft_wins = (
        (not struct_pool)
        and bool(latest_wins)
        and not hard_wins
        and not flat
    )
    if struct_pool or flat or not latest_wins:
        result.latest_wins_applied = False
        result.latest_wins_mode = "off"
    elif hard_wins:
        result.latest_wins_applied = True
        result.latest_wins_mode = "hard"
    else:
        result.latest_wins_applied = True
        result.latest_wins_mode = "soft"

    question = (question or "").strip()
    total_k = max(1, int(fact_top_n) + int(preference_top_n))

    # ---- Stage 1: optional intent decomposition ----------------------------
    if use_decomp:
        intents, topic_terms, failed, fact_need, preference_need = decompose_intent(
            question, client=client
        )
        result.intent_decomposition_failed = failed
    else:
        # No LLM decomp: raw question + lexical channel prior.
        intents = [question] if question else []
        topic_terms = []
        failed = True
        fact_need, preference_need = estimate_channel_needs(question)
        result.intent_decomposition_failed = True

    result.intents = intents
    result.topic_terms = topic_terms
    result.fact_need = fact_need
    result.preference_need = preference_need
    if flat:
        fact_budget, preference_budget = total_k, 0
    else:
        fact_budget, preference_budget = allocate_channel_budget(
            total_k, fact_need, preference_need
        )
    result.fact_top_n = fact_budget
    result.preference_top_n = preference_budget

    expanded_query = " ".join([question] + intents + topic_terms).strip()
    if not expanded_query:
        expanded_query = question

    all_memories = list(store.state.memories.values())
    by_id = {m.memory_id: m for m in all_memories}

    if flat:
        sim = _Similarity(question or expanded_query, all_memories, client=embedding_client)
        result.embedding_used = sim.enabled
        ranked = sorted(
            (
                ScoredMemory(
                    memory=m,
                    score=_score(m, sim),
                    channel="memory",
                )
                for m in all_memories
            ),
            key=lambda item: (-item.score, item.memory.memory_id),
        )
        result.facts = ranked if return_all_scored else ranked[:total_k]
        return result

    # ---- Stage 2: build seed candidates ------------------------------------
    if use_decomp or struct_pool:
        # Concept localization + semantic backfill (only with intent decomp,
        # or when the full structural candidate pool is requested).
        selected = store.select_relevant_concepts(expanded_query, top_k=concept_top_k)
        concept_ids: Set[str] = set()
        for concept_id in selected:
            concept_ids.update(store._ancestor_ids(concept_id))
            concept_ids.update(
                child_id
                for child_id, child in store.state.concepts.items()
                if child.parent_id == concept_id and child.status == "active"
            )
        result.selected_concept_ids = sorted(concept_ids)

        concept_memories = [
            m for m in all_memories if m.parent_concept_id in concept_ids
        ]
        result.n_concept_memories = len(concept_memories)
        seen_ids = {m.memory_id for m in concept_memories}
        rest = [m for m in all_memories if m.memory_id not in seen_ids]
        backfill_limit = total_k if struct_pool else max(total_k, concept_top_k)
        backfilled: List[Memory] = []
        if rest and backfill_limit > 0:
            backfill_sim = _Similarity(expanded_query, rest, client=embedding_client)
            backfilled = sorted(
                rest,
                key=lambda m: (-backfill_sim.semantic(m), m.memory_id),
            )[:backfill_limit]
        result.n_semantic_backfill = len(backfilled)
        candidates = list(concept_memories) + list(backfilled)
        if not candidates:
            if struct_pool:
                result.facts, result.preferences = [], []
                return result
            candidates = all_memories
    else:
        # Similarity top-K, then replace each hit with its relation-complete
        # memory. No disputed / supersede / channel-need score after that.
        seed_sim = _Similarity(question, all_memories, client=embedding_client)
        result.embedding_used = seed_sim.enabled
        seeds = sorted(
            all_memories,
            key=lambda m: (-seed_sim.semantic(m), m.memory_id),
        )[:total_k]
        result.n_embedding_seeds = len(seeds)
        result.latest_wins_applied = False
        result.latest_wins_mode = "off"
        if not seeds:
            result.facts, result.preferences = [], []
            return result
        bundled = _assemble_relation_bundles(store, seeds, by_id, seed_sim, result)
        result.facts, result.preferences = _partition_scored(bundled)
        return result

    # ---- Stage 3: structure-preserving chaining ----------------------------
    demoted_ids: Set[str] = set()
    if struct_pool:
        eligible = list(candidates)
        # Still pull CONFLICTS_WITH counterparts into the struct pool.
        eligible = _chain_conflicts(store, eligible, by_id, result)
    else:
        replaced_by, heads = _supersede_heads(store)
        eligible = []
        for memory in candidates:
            superseded = memory.memory_id in replaced_by or memory.status == "superseded"
            if superseded:
                head_id = heads.get(memory.memory_id, "")
                head = by_id.get(head_id)
                record = {
                    "memory_id": memory.memory_id,
                    "content": memory.content,
                    "superseded_by": head_id,
                    "superseded_by_content": head.content if head else "",
                }
                result.dropped_superseded.append(record)
                if hard_wins:
                    continue
                if soft_wins:
                    demoted_ids.add(memory.memory_id)
                    result.demoted_superseded.append(record)
            eligible.append(memory)

        known = {m.memory_id for m in eligible}
        for entry in result.dropped_superseded:
            head = by_id.get(entry["superseded_by"])
            if (
                head is not None
                and head.memory_id not in known
                and head.status != "superseded"
            ):
                eligible.append(head)
                known.add(head.memory_id)

        # CONFLICTS_WITH: ensure both endpoints of a conflict are present.
        eligible = _chain_conflicts(store, eligible, by_id, result)

    # ---- Stage 4: score + truncate -----------------------------------------
    # Prefer question anchor for final ranking (emb-only). With decomp the
    # expanded query was already used for concept/backfill seeding.
    sim = _Similarity(question or expanded_query, eligible, client=embedding_client)
    result.embedding_used = result.embedding_used or sim.enabled

    scored: List[ScoredMemory] = []
    if struct_pool:
        for memory in eligible:
            base = _score(memory, sim)
            scored.append(
                ScoredMemory(memory=memory, score=base, channel=_channel_for(memory))
            )
        scored.sort(key=lambda item: (-item.score, item.memory.memory_id))
        merged = scored if return_all_scored else scored[:total_k]
    else:
        penalty = _supersede_penalty()
        need_bonus = _channel_need_bonus()

        def _factor(memory: Memory) -> float:
            if memory.memory_id in demoted_ids and penalty > 0.0:
                return penalty
            return 1.0

        for memory in eligible:
            base = _score(memory, sim, supersede_factor=_factor(memory))
            adj = base * _need_multiplier(
                memory, fact_need, preference_need, need_bonus
            )
            scored.append(
                ScoredMemory(memory=memory, score=adj, channel=_channel_for(memory))
            )
        scored.sort(key=lambda item: (-item.score, item.memory.memory_id))
        merged = scored if return_all_scored else scored[:total_k]

    result.facts, result.preferences = _partition_scored(merged)
    return result


# Keep a stable alias for callers / docs that still mention the old name.
retrieve.__doc__ = retrieve.__doc__
