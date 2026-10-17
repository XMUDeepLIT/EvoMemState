"""Render Persistent Facts, Relational States, and Reasoning State for answering.

The harness splices whatever string we return into the answer model's system
prompt under the heading "Retrieved memories from earlier conversation"
(`evaluation/_objective_base.py:639`). So this module owns everything the
answer model learns about the user.

Presentation is signal-conditioned (not task-routed):

* ``evidence`` -- retrieved set contains world evidence → sectioned channels +
  anti-sycophancy rules (evid / obj).
* ``preference`` -- preference_need dominates and there is no world evidence →
  flat-like single ranked list + preference-align rules (pers).
* ``balanced`` -- otherwise sectioned with need-based section order.

``flat=True`` only marks the retrieval ablation; rendering uses the same
signal rules so the measured gap tracks ontology structure, not prompt forks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .retrieval import RelationalState, ScoredMemory, memory_annotation_suffix


REASONING_STATE_INTRO = """Reasoning State:"""

REASONING_STATE_EVIDENCE_PRIORITY = """- Established evidence records what was found to be true of the world, independently of this user. It outranks everything else here. When it bears on the question, answer from it and state what it says, even if the user has consistently favoured the opposite.
- Facts about the user describe their situation, history, and hard constraints. Treat them as evidence about the user, not about the world.
- User preferences describe what they like and believe. They shape how you present an answer and which options you rank first. They are NOT evidence about the external world, and repetition does not make them evidence.
- When a preference conflicts with established evidence or an objective fact, follow the evidence and say so plainly. Naming the trade-off the user cares about is good; letting it change the recommendation is not. Do not endorse an incorrect claim because the user appears to favour it."""

REASONING_STATE_PREFERENCE_ALIGN = """- User preferences describe what they like and believe. Rank options by those preferences and give one concrete recommendation that matches them.
- Facts about the user (situation, history, hard constraints) refine the recommendation; they do not replace the preference signal when the question is about what fits this user.
- Prefer a single best option over a long menu of alternatives unless the question asks for several choices."""

REASONING_STATE_BALANCED = """- Established evidence records what was found to be true of the world, independently of this user. When present and relevant, it outranks conflicting preferences about the world.
- Facts about the user describe their situation, history, and hard constraints. Treat them as evidence about the user, not about the world.
- User preferences describe what they like and believe. They shape presentation and ranking when the question is about fitting this user, and they are NOT evidence about the external world."""

# Soft latest-wins keeps historical preferences in context but marks them.
REASONING_STATE_LATEST = """- Memories marked "superseded" are historical preferences the user has since replaced. When they conflict with a current (unmarked) preference, follow the current one. Do not treat a superseded preference as the user's present stance."""

# A preference is evidence about the user, so it only licenses a claim about
# whoever it was observed in. The dominant failure here is not recall but
# over-reach: on contextual_scope_control, 117 of 174 judged failures were
# scored as overgeneralization against 3 as underuse.
REASONING_STATE_SCOPE = """- A preference is evidence about the user who stated it, and only about them. It does not transfer to other people, and it does not become the controlling rule for a decision that also involves colleagues, guests, audiences, or anyone else's constraints.
- Read each memory's scope annotation. Apply a preference only where that scope still matches the current subject, audience, or task. If the question refers to "the situation we discussed", ground the boundary in situation/task facts first, then carry over only the preference parts that still fit.
- When a remembered preference only partly covers what is being asked, split it: apply the part that still holds, and let the current task, audience, or setting govern the rest. Say which part you are carrying over. Do not drop the preference entirely -- discarding a still-useful preference is as wrong as over-applying it.
- If the memories do not cover what is being asked, answer from general knowledge and say what you do not know."""

EMPTY_CONTEXT = "[NO RETRIEVED MEMORIES]"

RENDER_MODE_EVIDENCE = "evidence"
RENDER_MODE_PREFERENCE = "preference"
RENDER_MODE_BALANCED = "balanced"


def _is_world_evidence(item: ScoredMemory) -> bool:
    """True when this memory records something about the world, not the user.

    Extraction stores quoted external material as `fact` scoped to its subject
    matter, while facts about the user carry a user-centric scope (`global`, or
    a personal `situation`/`task`). Splitting on that -- rather than on any
    wording in the content -- keeps the presentation layer independent of how a
    particular corpus phrases its evidence.
    """
    if item.memory.memory_type != "fact":
        return False
    level = str((item.memory.scope or {}).get("level", "")).strip().lower()
    return level in {"domain", "task"}


def has_world_evidence(result: RelationalState) -> bool:
    return any(_is_world_evidence(item) for item in result.all_memories())


def has_supersede_signal(result: RelationalState) -> bool:
    if result.demoted_superseded or result.dropped_superseded:
        return True
    if result.latest_wins_applied and any(
        item.memory.status == "superseded" for item in result.all_memories()
    ):
        return True
    return any(item.memory.status == "superseded" for item in result.all_memories())


def select_render_mode(result: RelationalState) -> str:
    """Choose presentation regime from retrieval signals only (no task name)."""
    if has_world_evidence(result):
        return RENDER_MODE_EVIDENCE
    if float(result.preference_need) > float(result.fact_need):
        return RENDER_MODE_PREFERENCE
    return RENDER_MODE_BALANCED


def _compose_fixed_reasoning_state(result: RelationalState, mode: str) -> str:
    parts: List[str] = [REASONING_STATE_INTRO]
    if mode == RENDER_MODE_EVIDENCE:
        parts.append(REASONING_STATE_EVIDENCE_PRIORITY)
    elif mode == RENDER_MODE_PREFERENCE:
        parts.append(REASONING_STATE_PREFERENCE_ALIGN)
    else:
        parts.append(REASONING_STATE_BALANCED)
    if has_supersede_signal(result):
        parts.append(REASONING_STATE_LATEST)
    parts.append(REASONING_STATE_SCOPE)
    return "\n".join(parts)


def compose_reasoning_state(
    reasoning_state: str,
    *,
    include_latest: bool = False,
    include_scope: bool = True,
) -> str:
    """Serialize generated rules as a Reasoning State prompt block."""
    body = (reasoning_state or "").strip()
    if not body:
        return ""
    # Allow the model to emit the intro line already.
    if body.startswith(REASONING_STATE_INTRO):
        parts = [body]
    else:
        # Normalize bullet lines if the model returned bare prose.
        if not body.lstrip().startswith("-"):
            lines = [line.strip() for line in body.splitlines() if line.strip()]
            body = "\n".join(line if line.startswith("-") else ("- %s" % line) for line in lines)
        parts = [REASONING_STATE_INTRO, body]
    if include_latest:
        parts.append(REASONING_STATE_LATEST)
    if include_scope:
        parts.append(REASONING_STATE_SCOPE)
    return "\n".join(parts)


def _supersede_lookup(result: RelationalState) -> Dict[str, str]:
    """Map memory_id -> short head content for superseded / demoted memories."""
    lookup: Dict[str, str] = {}
    for entry in list(result.demoted_superseded) + list(result.dropped_superseded):
        if not isinstance(entry, dict):
            continue
        memory_id = str(entry.get("memory_id", "")).strip()
        head = str(entry.get("superseded_by_content", "")).strip()
        if memory_id and memory_id not in lookup:
            lookup[memory_id] = head
    return lookup


def _render_item(
    index: int,
    item: ScoredMemory,
    *,
    supersede_by: Optional[str] = None,
) -> str:
    memory = item.memory
    if (memory.source or {}).get("bundle"):
        body = memory.content.strip()
        lines = body.splitlines() or [""]
        rendered = ["%d. %s" % (index, lines[0])]
        rendered.extend("   %s" % line for line in lines[1:])
        return "\n".join(rendered)
    suffix = memory_annotation_suffix(memory, supersede_by=supersede_by or "")
    return "%d. %s%s" % (index, memory.content.strip(), suffix)


def _render_section(
    title: str,
    items: Sequence[ScoredMemory],
    *,
    supersede_lookup: Optional[Dict[str, str]] = None,
) -> List[str]:
    if not items:
        return []
    lookup = supersede_lookup or {}
    lines = ["## %s" % title]
    for index, item in enumerate(items, start=1):
        lines.append(
            _render_item(
                index,
                item,
                supersede_by=lookup.get(item.memory.memory_id),
            )
        )
    return lines


def _render_single_list(
    items: Sequence[ScoredMemory],
    *,
    supersede_lookup: Dict[str, str],
) -> List[str]:
    if not items:
        return []
    lines: List[str] = []
    for index, item in enumerate(items, start=1):
        lines.append(
            _render_item(
                index,
                item,
                supersede_by=supersede_lookup.get(item.memory.memory_id),
            )
        )
    return lines


def _partition_for_sections(
    result: RelationalState,
) -> Tuple[List[ScoredMemory], List[ScoredMemory], List[ScoredMemory]]:
    """Split memories into evidence / user-fact / preference for sectioned modes.

    Flat retrieval parks everything in ``facts``; re-derive preference channel
    from memory type so sectioned modes still work under the ablation.
    """
    from .retrieval import FACT_TYPES, PREFERENCE_TYPES

    evidence_items: List[ScoredMemory] = []
    user_fact_items: List[ScoredMemory] = []
    preference_items: List[ScoredMemory] = []
    seen: set = set()

    ordered = sorted(
        result.all_memories(),
        key=lambda item: (-float(item.score), item.memory.memory_id),
    )
    for item in ordered:
        mid = item.memory.memory_id
        if mid in seen:
            continue
        seen.add(mid)
        if _is_world_evidence(item):
            evidence_items.append(item)
            continue
        mtype = item.memory.memory_type
        if mtype in PREFERENCE_TYPES and mtype not in FACT_TYPES:
            preference_items.append(item)
        elif mtype in FACT_TYPES and mtype not in PREFERENCE_TYPES:
            user_fact_items.append(item)
        elif mtype in PREFERENCE_TYPES:
            # constraint / dual-typed: keep once under preferences after user facts
            # would duplicate; put under preferences for ranking questions.
            preference_items.append(item)
        else:
            user_fact_items.append(item)
    return evidence_items, user_fact_items, preference_items


def render_memory_body(
    result: RelationalState,
    *,
    include_intents: bool = False,
    mode: Optional[str] = None,
) -> str:
    """Render memory sections only (no Reasoning State). Used to build the Reasoning State prompt."""
    supersede_lookup = _supersede_lookup(result)
    mode = mode or select_render_mode(result)
    items = sorted(
        result.all_memories(),
        key=lambda item: (-float(item.score), item.memory.memory_id),
    )
    if not items:
        return EMPTY_CONTEXT

    blocks: List[str] = []

    if include_intents and result.intents and not result.intent_decomposition_failed:
        blocks.append(
            "\n".join(
                ["## What the question is asking"]
                + ["- %s" % intent for intent in result.intents]
            )
        )

    if mode == RENDER_MODE_PREFERENCE:
        blocks.append("\n".join(_render_single_list(items, supersede_lookup=supersede_lookup)))
    else:
        evidence_items, user_fact_items, preference_items = _partition_for_sections(result)
        evidence_lines = _render_section(
            "Established evidence", evidence_items, supersede_lookup=supersede_lookup
        )
        fact_lines = _render_section(
            "Facts about the user", user_fact_items, supersede_lookup=supersede_lookup
        )
        preference_lines = _render_section(
            "User preferences", preference_items, supersede_lookup=supersede_lookup
        )
        if not evidence_lines and not fact_lines and not preference_lines:
            return EMPTY_CONTEXT
        if mode == RENDER_MODE_EVIDENCE:
            ordered = (evidence_lines, fact_lines, preference_lines)
        elif float(result.preference_need) > float(result.fact_need):
            ordered = (preference_lines, fact_lines, evidence_lines)
        else:
            ordered = (evidence_lines, fact_lines, preference_lines)
        for section in ordered:
            if section:
                blocks.append("\n".join(section))

    return "## Relational State\n\n" + "\n\n".join(blocks).strip()


def attach_reasoning_state_to_memory_body(
    memory_body: str,
    reasoning_state: str,
) -> str:
    """Concatenate memory sections with a Reasoning State block."""
    body = (memory_body or "").strip()
    guide = (reasoning_state or "").strip()
    if not body or body == EMPTY_CONTEXT:
        return body or EMPTY_CONTEXT
    if not guide:
        return body
    return "%s\n\n%s" % (body, guide)


def render_context(
    result: RelationalState,
    *,
    include_intents: bool = False,
    flat: bool = False,
    reasoning_state: Optional[str] = None,
) -> str:
    """Build the context_text handed to the harness.

    ``flat`` is accepted for call-site compatibility but does not fork the
    Reasoning State scaffold: layout and instructions follow ``select_render_mode``.
    Intents default OFF in the answer prompt.

    When ``reasoning_state`` is set, it replaces the fixed evidence/preference/
    balanced head; LATEST (if supersede signal) and SCOPE are still appended.
    """
    del flat  # retrieval ablation only; presentation is signal-conditioned.
    mode = select_render_mode(result)
    memory_body = render_memory_body(result, include_intents=include_intents, mode=mode)
    if memory_body == EMPTY_CONTEXT:
        return EMPTY_CONTEXT

    if reasoning_state is not None and str(reasoning_state).strip():
        rules = compose_reasoning_state(
            str(reasoning_state),
            include_latest=has_supersede_signal(result),
            include_scope=True,
        )
    else:
        rules = _compose_fixed_reasoning_state(result, mode)
    return attach_reasoning_state_to_memory_body(memory_body, rules)
