"""Align dpo-dataset / RL manifests with production ``render_context`` presentation.

Maps flat ``Facts:`` / ``User preference:`` generation manifests onto synthetic
``RelationalState`` objects and renders them with the same ``render_context``
path used at inference (EvoOntoMem / RerankMem after rerank).

Training should consume ``messages`` built here so answer-model RL matches
online memory layout, REASONING_STATE, and supersede annotations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import Memory, ROOT_CONCEPT_ID
from .prompting import (
    REASONING_STATE_EVIDENCE_PRIORITY,
    REASONING_STATE_INTRO,
    REASONING_STATE_LATEST,
    REASONING_STATE_SCOPE,
    has_supersede_signal,
    render_context,
    select_render_mode,
)
from .retrieval import RelationalState, ScoredMemory

VERDICT_SYSTEM_PROMPT = (
    "Evaluate the claim using the memory context below. "
    "Use only information in the memory context for the verdict. "
    "Return exactly this structure:\n"
    "Verdict: SUPPORTED or REFUTED\n"
    "Explanation: One or two sentences."
)

PERSONALIZED_SYSTEM_PROMPT = (
    "Answer as a helpful personalized assistant using the memory context below. "
    "Do not mention the memory system or that preferences were retrieved."
)

LOCOMO_ANSWER_SYSTEM_PROMPT = (
    "You answer questions using the memory context provided below. "
    "Give a clear, direct answer. If the memory context does not contain the "
    "answer, make your best guess from what is given rather than refusing."
)

GENERAL_QA_SYSTEM_PROMPT = (
    "Answer the user's question clearly and directly using general knowledge. "
    "No user-specific memory context is available for this example."
)

MEMSYCO_MEMORY_LABEL = "Retrieved memories from earlier conversation"

MEMSYCO_ANSWER_SYSTEM_BASE = """You are {model_name}, a helpful and natural conversational assistant.
Knowledge cutoff: 2025-06
Current date: {current_date}

Answer in a clear, useful, and natural way."""

WRAP_STYLE_LOCOMO = "locomo"
WRAP_STYLE_MEMSYCO = "memsyco"

UPDATE_SYSTEM_PROMPT = (
    "Answer as a helpful personalized assistant using the memory context below. "
    "Memories may include an update; follow the current preference and do not rely "
    "on a superseded one. Do not mention the memory system."
)


def _memory(
    memory_id: str,
    content: str,
    *,
    memory_type: str,
    scope_level: str = "global",
    scope_value: str = "",
    status: str = "active",
    score: float = 1.0,
    channel: str = "",
) -> ScoredMemory:
    return ScoredMemory(
        memory=Memory(
            memory_id=memory_id,
            content=content.strip(),
            memory_type=memory_type,
            scope={"level": scope_level, "value": scope_value},
            time={},
            source={},
            status=status,
            parent_concept_id=ROOT_CONCEPT_ID,
        ),
        score=score,
        channel=channel or memory_type,
    )


def _compose_verdict_reasoning_state(result: RelationalState) -> str:
    parts = [REASONING_STATE_INTRO, REASONING_STATE_EVIDENCE_PRIORITY]
    if has_supersede_signal(result):
        parts.append(REASONING_STATE_LATEST)
    parts.append(REASONING_STATE_SCOPE)
    return "\n".join(parts)


def _replace_reasoning_state(body: str, reasoning_state: str) -> str:
    if REASONING_STATE_INTRO in body:
        memory_part = body.split(REASONING_STATE_INTRO, 1)[0].rstrip()
        return "%s\n\n%s" % (memory_part, reasoning_state)
    return "%s\n\n%s" % (body.rstrip(), reasoning_state)


def training_dataed_context(
    result: RelationalState,
    *,
    task: str,
    condition: str = "",
    reasoning_state: Optional[str] = None,
) -> Tuple[str, str]:
    """Render memory context; override REASONING_STATE for custom / verdict prefs."""
    mode = select_render_mode(result)
    if reasoning_state is not None and str(reasoning_state).strip():
        body = render_context(
            result,
            include_intents=False,
            flat=False,
            reasoning_state=str(reasoning_state).strip(),
        )
        return body, mode
    body = render_context(result, include_intents=False, flat=False)
    if task == "verdict" and condition == "preferences_only":
        body = _replace_reasoning_state(body, _compose_verdict_reasoning_state(result))
    return body, mode


CLAIM_QA_FACTS_SYSTEM = (
    "Answer the user's question using the memory context below. "
    "Start your answer with Yes or No, then give one or two supporting sentences."
)

CLAIM_QA_PREFERENCES_SYSTEM = (
    "Answer the user's question as a helpful assistant. Use the memory context "
    "together with your own knowledge, and answer truthfully rather than simply "
    "agreeing with the user. Start your answer with Yes or No, then give one or "
    "two supporting sentences."
)


def _parse_bullets_after_header(content: str, header: str) -> List[str]:
    """Parse ``Facts:\\n- ...`` / ``User memories:\\n- ...`` blocks from QA messages."""
    text = (content or "").strip()
    if not text.lower().startswith(header.lower()):
        return []
    body = text[len(header) :].lstrip("\n")
    # Stop at blank line before Question:
    if "\n\nQuestion:" in body:
        body = body.split("\n\nQuestion:", 1)[0]
    elif "\nQuestion:" in body:
        body = body.split("\nQuestion:", 1)[0]
    out: List[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            out.append(stripped[2:].strip())
        elif stripped.startswith("* "):
            out.append(stripped[2:].strip())
    return [item for item in out if item]


def _facts_list_from_qa_row(row: Dict[str, Any]) -> List[str]:
    facts = row.get("facts")
    if isinstance(facts, list) and facts:
        return [str(x).strip() for x in facts if str(x).strip()]
    for message in row.get("messages") or []:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "") != "user":
            continue
        parsed = _parse_bullets_after_header(str(message.get("content") or ""), "Facts:")
        if parsed:
            return parsed
    return []


def relational_state_from_fever_scifact(row: Dict[str, Any]) -> RelationalState:
    condition = str(row.get("condition") or "")
    generation_id = str(
        row.get("generation_id") or row.get("rl_id") or row.get("source_id") or "row"
    )
    if condition in {"facts_only", "facts_qa"}:
        facts = [
            _memory(
                "%s_fact_%d" % (generation_id, index),
                text,
                memory_type="fact",
                scope_level="domain",
                scope_value="external evidence",
                channel="fact",
                score=1.0 - index * 0.01,
            )
            for index, text in enumerate(_facts_list_from_qa_row(row))
        ]
        return RelationalState(
            facts=facts,
            fact_need=0.9,
            preference_need=0.1,
            latest_wins_mode="off",
        )

    preferences = [
        _memory(
            "%s_pref_%d" % (generation_id, index),
            text,
            memory_type="preference",
            scope_level="global",
            channel="preference",
            score=1.0 - index * 0.01,
        )
        for index, text in enumerate(row.get("preference_memories") or [])
    ]
    return RelationalState(
        preferences=preferences,
        fact_need=0.85,
        preference_need=0.15,
        latest_wins_mode="off",
    )


def relational_state_from_personamem(
    row: Dict[str, Any],
    *,
    pairing: Optional[Dict[str, Any]] = None,
    use_pairing_supersede: bool = True,
) -> RelationalState:
    condition = str(row.get("condition") or "")
    generation_id = str(row.get("generation_id") or "row")
    preference = str(row.get("input_preference") or "").strip()

    if condition == "single_no_preference":
        return RelationalState(fact_need=0.5, preference_need=0.5, latest_wins_mode="off")

    if condition == "updated_old_preference_only" and pairing and use_pairing_supersede:
        correct = str(pairing.get("correct_preference") or "").strip()
        incorrect = str(pairing.get("incorrect_preference") or "").strip()
        new_item = _memory(
            "%s_new" % generation_id,
            correct,
            memory_type="preference",
            channel="preference",
            score=1.0,
        )
        old_item = _memory(
            "%s_old" % generation_id,
            incorrect,
            memory_type="preference",
            status="superseded",
            channel="preference",
            score=0.5,
        )
        return RelationalState(
            preferences=[new_item, old_item],
            demoted_superseded=[
                {
                    "memory_id": old_item.memory.memory_id,
                    "content": incorrect,
                    "superseded_by": new_item.memory.memory_id,
                    "superseded_by_content": correct,
                }
            ],
            fact_need=0.2,
            preference_need=0.8,
            latest_wins_applied=True,
            latest_wins_mode="soft",
        )

    if not preference:
        return RelationalState(fact_need=0.5, preference_need=0.5, latest_wins_mode="off")

    item = _memory(
        "%s_pref" % generation_id,
        preference,
        memory_type="preference",
        scope_level="global",
        channel="preference",
    )
    return RelationalState(
        preferences=[item],
        fact_need=0.2,
        preference_need=0.8,
        latest_wins_mode="off",
    )


def infer_task_from_row(row: Dict[str, Any]) -> str:
    source = str(row.get("source") or "")
    condition = str(row.get("condition") or "")
    if source in {"fever", "scifact"} or condition.endswith("_facts") or "preferences" in condition:
        return "verdict"
    return "personalized_qa"


def infer_question_from_row(row: Dict[str, Any]) -> str:
    condition = str(row.get("condition") or "")
    # QA-form FEVER/SciFact: prefer the Yes/No question over the raw claim.
    if condition in {"facts_qa", "preferences_qa"} and row.get("query"):
        return str(row["query"])
    if row.get("query") and not row.get("claim"):
        return str(row["query"])
    if row.get("claim"):
        return str(row["claim"])
    if row.get("query"):
        return str(row["query"])
    return ""


def _locomo_system_prompt(
    *,
    task: str,
    condition: str = "",
    pairing: Optional[Dict[str, Any]] = None,
) -> str:
    if condition == "facts_qa":
        return CLAIM_QA_FACTS_SYSTEM
    if condition == "preferences_qa":
        return CLAIM_QA_PREFERENCES_SYSTEM
    if task == "verdict":
        return VERDICT_SYSTEM_PROMPT
    if condition == "single_no_preference":
        return GENERAL_QA_SYSTEM_PROMPT
    if pairing and pairing.get("pair_type") == "preference_update":
        return UPDATE_SYSTEM_PROMPT
    return PERSONALIZED_SYSTEM_PROMPT


def _memsyco_system_prompt(
    context_text: str,
    *,
    model_name: str = "assistant",
    current_date: str = "2025-06-01",
) -> str:
    base = MEMSYCO_ANSWER_SYSTEM_BASE.format(
        model_name=model_name,
        current_date=current_date,
    )
    return "%s\n\n%s:\n%s" % (base, MEMSYCO_MEMORY_LABEL, context_text)


def wrap_answer_messages(
    context_text: str,
    question: str,
    *,
    task: str,
    condition: str = "",
    pairing: Optional[Dict[str, Any]] = None,
    wrap_style: str = WRAP_STYLE_LOCOMO,
    model_name: str = "assistant",
    current_date: str = "2025-06-01",
) -> List[Dict[str, str]]:
    """Package ``context_text`` like LoCoMo or MemSyco answer calls."""
    style = (wrap_style or WRAP_STYLE_LOCOMO).strip().lower()
    if style == WRAP_STYLE_MEMSYCO:
        system = _memsyco_system_prompt(
            context_text,
            model_name=model_name,
            current_date=current_date,
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": question}]

    system = _locomo_system_prompt(task=task, condition=condition, pairing=pairing)
    use_claim = task == "verdict" and condition not in {"facts_qa", "preferences_qa"}
    if use_claim:
        user = "Memory context:\n%s\n\nClaim: %s" % (context_text, question)
    else:
        user = "Memory context:\n%s\n\nQuestion: %s" % (context_text, question)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def reward_spec_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # Dense QA references win over sparse verdict labels (facts_qa / preferences_qa).
    reference = str(row.get("reference_correct_answer") or "").strip()
    if reference:
        return {
            "type": "token_f1",
            "target": reference,
            "reference_incorrect_answers": list(
                row.get("reference_incorrect_answers") or []
            ),
        }
    task = infer_task_from_row(row)
    if task == "verdict":
        return {
            "type": "verdict_match",
            "target": str(row.get("gold_label") or ""),
        }
    return {
        "type": "token_f1",
        "target": "",
        "reference_incorrect_answers": list(row.get("reference_incorrect_answers") or []),
    }


def build_aligned_training_record(
    row: Dict[str, Any],
    *,
    pairing: Optional[Dict[str, Any]] = None,
    use_pairing_supersede: bool = True,
    wrap_style: str = WRAP_STYLE_LOCOMO,
    model_name: str = "assistant",
    current_date: str = "2025-06-01",
    include_both_wrap_styles: bool = False,
    reasoning_state: Optional[str] = None,
    reasoning_state_source: Optional[str] = None,
) -> Dict[str, Any]:
    """One RL/GRPO-ready record with production-aligned memory rendering."""
    source = str(row.get("source") or "")
    condition = str(row.get("condition") or "")
    task = infer_task_from_row(row)
    question = infer_question_from_row(row)

    if source in {"fever", "scifact"}:
        result = relational_state_from_fever_scifact(row)
    else:
        result = relational_state_from_personamem(
            row,
            pairing=pairing,
            use_pairing_supersede=use_pairing_supersede,
        )

    context_text, render_mode = training_dataed_context(
        result,
        task=task,
        condition=condition,
        reasoning_state=reasoning_state,
    )
    messages = wrap_answer_messages(
        context_text,
        question,
        task=task,
        condition=condition,
        pairing=pairing,
        wrap_style=wrap_style,
        model_name=model_name,
        current_date=current_date,
    )
    record: Dict[str, Any] = {
        "id": str(row.get("generation_id") or row.get("pair_id") or ""),
        "source_group_id": str(row.get("source_group_id") or ""),
        "source": source,
        "condition": condition,
        "task": task,
        "render_mode": render_mode,
        "context_text": context_text,
        "wrap_style": wrap_style,
        "messages": messages,
        "reward_spec": reward_spec_from_row(row),
        "question": question,
    }
    if reasoning_state is not None and str(reasoning_state).strip():
        record["generated_reasoning_state"] = str(reasoning_state).strip()
        record["reasoning_state_source"] = reasoning_state_source or "custom"
    if include_both_wrap_styles:
        record["messages_locomo"] = wrap_answer_messages(
            context_text,
            question,
            task=task,
            condition=condition,
            pairing=pairing,
            wrap_style=WRAP_STYLE_LOCOMO,
            model_name=model_name,
            current_date=current_date,
        )
        record["messages_memsyco"] = wrap_answer_messages(
            context_text,
            question,
            task=task,
            condition=condition,
            pairing=pairing,
            wrap_style=WRAP_STYLE_MEMSYCO,
            model_name=model_name,
            current_date=current_date,
        )
    if row.get("gold_label"):
        record["gold_label"] = row["gold_label"]
    if pairing:
        record["pair_id"] = pairing.get("pair_id")
        record["pair_type"] = pairing.get("pair_type")
    return record


def index_personamem_pairings(pairing_path: Any) -> Dict[str, Dict[str, Any]]:
    """Map generation_id -> pairing_plan row."""
    path = Path(pairing_path)
    by_chosen: Dict[str, Dict[str, Any]] = {}
    by_rejected: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            pairing = json.loads(line)
            by_chosen[str(pairing["chosen_generation_id"])] = pairing
            by_rejected[str(pairing["rejected_generation_id"])] = pairing
    out: Dict[str, Dict[str, Any]] = {}
    out.update(by_chosen)
    out.update(by_rejected)
    return out
