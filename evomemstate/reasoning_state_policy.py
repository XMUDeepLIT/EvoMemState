"""Build Reasoning State policy training records from DPO manifests.

Training records, environment options and output tags use ``reasoning_state``.

The trained model sees ``messages`` (question + memory_body) and emits free-form
REASONING_STATE. Frozen answer scoring uses ``memory_body`` + generated state vs
memory_body alone; see ``evomemstate.reasoning_state_reward``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .prompting import (
    REASONING_STATE_INTRO,
    has_supersede_signal,
    render_memory_body,
    select_render_mode,
)
from .training_data import (
    PERSONALIZED_SYSTEM_PROMPT,
    UPDATE_SYSTEM_PROMPT,
    VERDICT_SYSTEM_PROMPT,
    WRAP_STYLE_LOCOMO,
    WRAP_STYLE_MEMSYCO,
    GENERAL_QA_SYSTEM_PROMPT,
    infer_question_from_row,
    infer_task_from_row,
    relational_state_from_fever_scifact,
    relational_state_from_personamem,
    reward_spec_from_row,
    _locomo_system_prompt,
)

REASONING_STATE_SYSTEM_PROMPT = (
    "You are a memory-routing helper. Given a question and retrieved memories, "
    "write a compact reasoning_state telling an answer model how to use the memories for "
    "this specific question.\n"
    "\n"
    "Infer the question's information need and route the memories accordingly. "
    "Determine whether each relevant memory should serve as primary evidence, "
    "a user preference, a constraint, a condition, historical context, or "
    "supporting context.\n"
    "\n"
    "Prioritize memories by their relevance to the question's requested scope, "
    "time, conditions, and answer type:\n"
    "\n"
    "- When the question asks for a choice or personalized answer, give "
    "applicable user preferences appropriate weight.\n"
    "- When it asks about facts, feasibility, or implementation, prioritize "
    "evidence and constraints over preferences.\n"
    "- When memories conflict or one supersedes another, use the relationship "
    "in the way required by the question. A superseded memory should not "
    "represent the current state, but it may still matter as context, evidence "
    "about an earlier state, or a condition of the question.\n"
    "- Do not combine conflicting memories merely to appear balanced. Identify "
    "which memory should control the answer and what limited role, if any, the "
    "others should play.\n"
    "- Infer chronology, supersession, and scope only when supported by the "
    "retrieved text. Preserve uncertainty when the relationship is unclear.\n"
    "\n"
    "Output 2-4 short, imperative bullets and no more than 80 words total. "
    "Ground the bullets in concrete details from the memories, but include "
    "only details that affect how the question should be answered.\n"
    "\n"
    "Do not answer the question, decide the claim, or explain your reasoning. "
    "Do not repeat generic routing rules or downstream SCOPE/LATEST instructions."
)

# Think-mode: same routing body; only the trailing output contract changes.
# Keep as .replace() of REASONING_STATE_SYSTEM_PROMPT so the validated no-think baseline
# stays byte-identical (see docs/reasoning_state_think_mode.md).
REASONING_STATE_SYSTEM_PROMPT_THINK = REASONING_STATE_SYSTEM_PROMPT.replace(
    "Output 2-4 short, imperative bullets and no more than 80 words total. "
    "Ground the bullets in concrete details from the memories, but include "
    "only details that affect how the question should be answered.\n"
    "\n"
    "Do not answer the question, decide the claim, or explain your reasoning. "
    "Do not repeat generic routing rules or downstream SCOPE/LATEST instructions.",
    "First reason through the routing decision inside <think></think> tags: "
    "work out which memories matter, what role each plays, and how conflicts "
    "or supersession should be handled. This reasoning is about how to use "
    "the memories, not about what the answer is: do not resolve the claim or "
    "draft an answer inside <think>. Then write the final reasoning_state inside "
    "<reasoning_state></reasoning_state> tags: 2-4 short, imperative bullets, no more than 80 words "
    "total, grounded in concrete details from the memories but including only "
    "details that affect how the question should be answered.\n"
    "\n"
    "Output exactly one <think>...</think> block followed by exactly one "
    "<reasoning_state>...</reasoning_state> block, in that order, and nothing else.\n"
    "\n"
    "The <reasoning_state> block must not answer the question, decide the claim, or "
    "explain your reasoning. Do not repeat generic routing rules or "
    "downstream SCOPE/LATEST instructions inside <reasoning_state>.",
)
assert REASONING_STATE_SYSTEM_PROMPT_THINK != REASONING_STATE_SYSTEM_PROMPT, (
    "REASONING_STATE_SYSTEM_PROMPT text changed; update the REASONING_STATE_SYSTEM_PROMPT_THINK "
    ".replace() target in reasoning_state_policy.py"
)

# Memory-only: reasoning_state model never sees the question/claim.
REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY = (
    "You are a memory-routing helper. Given only retrieved memories (no user "
    "question), write a compact reasoning_state telling an answer model how to use those "
    "memories.\n"
    "\n"
    "Organize the memories by the role each should play: primary evidence, "
    "user preference, constraint, condition, historical context, or supporting "
    "context. Call out conflicts and supersession when the retrieved text "
    "supports them.\n"
    "\n"
    "Prioritize clearly:\n"
    "\n"
    "- Separate objective evidence / constraints from user preferences or "
    "beliefs.\n"
    "- When memories conflict or one supersedes another, say which should "
    "control a current-state reading and what limited role, if any, the "
    "others should play.\n"
    "- Do not combine conflicting memories merely to appear balanced.\n"
    "- Infer chronology, supersession, and scope only when supported by the "
    "retrieved text. Preserve uncertainty when the relationship is unclear.\n"
    "\n"
    "Output 2-4 short, imperative bullets and no more than 80 words total. "
    "Ground the bullets in concrete details from the memories.\n"
    "\n"
    "Do not invent a question, answer any question, decide any claim, or "
    "explain your reasoning. Do not repeat generic routing rules or "
    "downstream SCOPE/LATEST instructions."
)

# Memory-only + channel needs: same as above, but also conditioned on
# fact_need / preference_need (still no question).
REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY_NEEDS = (
    "You are a memory-routing helper. Given channel needs (fact_need / "
    "preference_need) and retrieved memories — but no user question — write a "
    "compact reasoning_state telling an answer model how to use those memories.\n"
    "\n"
    "fact_need (0-1) is how much a downstream answer should depend on objective "
    "evidence, history, or hard constraints. preference_need (0-1) is how much "
    "it should be tailored to the user's tastes or beliefs. Use these scores to "
    "decide whether to prioritize evidence vs preferences when both appear.\n"
    "\n"
    "Organize the memories by the role each should play: primary evidence, "
    "user preference, constraint, condition, historical context, or supporting "
    "context. Call out conflicts and supersession when the retrieved text "
    "supports them.\n"
    "\n"
    "Prioritize clearly:\n"
    "\n"
    "- Separate objective evidence / constraints from user preferences or "
    "beliefs.\n"
    "- When memories conflict or one supersedes another, say which should "
    "control a current-state reading and what limited role, if any, the "
    "others should play.\n"
    "- Do not combine conflicting memories merely to appear balanced.\n"
    "- Infer chronology, supersession, and scope only when supported by the "
    "retrieved text. Preserve uncertainty when the relationship is unclear.\n"
    "\n"
    "Output 2-4 short, imperative bullets and no more than 80 words total. "
    "Ground the bullets in concrete details from the memories.\n"
    "\n"
    "Do not invent a question, answer any question, decide any claim, or "
    "explain your reasoning. Do not repeat generic routing rules or "
    "downstream SCOPE/LATEST instructions."
)

_MEMORY_ONLY_THINK_TAIL = (
    "First reason through the routing decision inside <think></think> tags: "
    "work out which memories matter, what role each plays, and how conflicts "
    "or supersession should be handled. This reasoning is about how to use "
    "the memories, not about inventing or answering a question: do not invent "
    "a claim or draft an answer inside <think>. Then write the final reasoning_state "
    "inside <reasoning_state></reasoning_state> tags: 2-4 short, imperative bullets, no more than "
    "80 words total, grounded in concrete details from the memories.\n"
    "\n"
    "Output exactly one <think>...</think> block followed by exactly one "
    "<reasoning_state>...</reasoning_state> block, in that order, and nothing else.\n"
    "\n"
    "The <reasoning_state> block must not invent a question, answer any question, decide "
    "any claim, or explain your reasoning. Do not repeat generic routing "
    "rules or downstream SCOPE/LATEST instructions inside <reasoning_state>."
)
_MEMORY_ONLY_PLAIN_TAIL = (
    "Output 2-4 short, imperative bullets and no more than 80 words total. "
    "Ground the bullets in concrete details from the memories.\n"
    "\n"
    "Do not invent a question, answer any question, decide any claim, or "
    "explain your reasoning. Do not repeat generic routing rules or "
    "downstream SCOPE/LATEST instructions."
)

REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY_THINK = REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY.replace(
    _MEMORY_ONLY_PLAIN_TAIL, _MEMORY_ONLY_THINK_TAIL
)
REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY_NEEDS_THINK = REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY_NEEDS.replace(
    _MEMORY_ONLY_PLAIN_TAIL, _MEMORY_ONLY_THINK_TAIL
)
assert REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY_THINK != REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY
assert REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY_NEEDS_THINK != REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY_NEEDS


def select_reasoning_state_system_prompt(
    *,
    think_mode: bool = False,
    memory_only: bool = False,
    channel_needs: bool = False,
) -> str:
    if memory_only and channel_needs:
        return (
            REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY_NEEDS_THINK
            if think_mode
            else REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY_NEEDS
        )
    if memory_only:
        return (
            REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY_THINK
            if think_mode
            else REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY
        )
    return REASONING_STATE_SYSTEM_PROMPT_THINK if think_mode else REASONING_STATE_SYSTEM_PROMPT


_REASONING_STATE_TAG_RE = re.compile(r"<reasoning_state>(.*?)</reasoning_state>", re.IGNORECASE | re.DOTALL)
_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_UNCLOSED_RE = re.compile(
    r"<think>(?!.*</think>)", re.IGNORECASE | re.DOTALL
)


def extract_reasoning_state_and_thought(
    raw_text: str, *, reasoning_content: str = ""
) -> Tuple[str, str, List[str]]:
    """Split think-mode sampler output into ``(reasoning_state, thought, extraction_notes)``.

    Extraction notes (``think_unclosed``, ``missing_reasoning_state_tag``) are unioned with
    ``reasoning_state_format_violations(reasoning_state)`` at the call site — do not fold them into
    ``reasoning_state_format_violations`` itself (keeps no-think path untouched).
    """
    text = raw_text or ""
    notes: List[str] = []

    think_match = _THINK_TAG_RE.search(text)
    if think_match:
        thought = think_match.group(1).strip()
    elif (reasoning_content or "").strip():
        thought = reasoning_content.strip()
    elif _THINK_OPEN_UNCLOSED_RE.search(text):
        thought = ""
        notes.append("think_unclosed")
    else:
        thought = ""

    reasoning_state_match = _REASONING_STATE_TAG_RE.search(text)
    if reasoning_state_match:
        reasoning_state = reasoning_state_match.group(1).strip()
    else:
        if "think_unclosed" in notes:
            notes.append("missing_reasoning_state_tag")
            reasoning_state = ""
        else:
            reasoning_state = _THINK_TAG_RE.sub("", text)
            reasoning_state = re.sub(r"</think>", "", reasoning_state, flags=re.IGNORECASE).strip()
            if not reasoning_state:
                notes.append("missing_reasoning_state_tag")

    return reasoning_state, thought, notes


def format_think_reasoning_state_assistant(thought: str, reasoning_state: str) -> str:
    """Pack thought + reasoning_state into the think-mode assistant target for SFT."""
    return (
        "<think>\n%s\n</think>\n"
        "<reasoning_state>\n%s\n</reasoning_state>"
        % ((thought or "").strip(), (reasoning_state or "").strip())
    )


def strip_reasoning_state_from_context(context_text: str) -> str:
    """Drop a rendered REASONING_STATE block if present, leave memory sections."""
    body = (context_text or "").strip()
    if REASONING_STATE_INTRO in body:
        return body.split(REASONING_STATE_INTRO, 1)[0].rstrip()
    return body


def format_channel_needs_block(
    fact_need: Optional[float],
    preference_need: Optional[float],
) -> str:
    """Render fact_need / preference_need when ``channel_needs`` is enabled."""
    if fact_need is None and preference_need is None:
        return ""
    fn = 0.5 if fact_need is None else float(fact_need)
    pn = 0.5 if preference_need is None else float(preference_need)
    return (
        "Channel needs:\n"
        "- fact_need: %.2f (how much the answer should rely on objective "
        "evidence / constraints)\n"
        "- preference_need: %.2f (how much it should follow user preferences / "
        "beliefs)\n"
        % (fn, pn)
    )


def reasoning_state_user_prompt(
    question: str,
    memory_body: str,
    *,
    task: str = "",
    condition: str = "",
    memory_only: bool = False,
    channel_needs: bool = False,
    fact_need: Optional[float] = None,
    preference_need: Optional[float] = None,
) -> str:
    if memory_only:
        head = ""
        use_needs = bool(channel_needs)
        if use_needs:
            needs = format_channel_needs_block(fact_need, preference_need)
            if needs:
                head = "%s\n" % needs
        tail = (
            "Use the channel needs to decide whether to prioritize evidence vs "
            "preferences. Do not invent or answer a question."
            if use_needs
            else "Do not invent or answer a question."
        )
        return (
            "%sMemories:\n%s\n\n"
            "Write only 2-4 routing bullets specifying which memories matter, "
            "the role each should play, and how conflicts or updates should be "
            "handled. %s"
            % (head, memory_body.strip(), tail)
        )
    use_claim = task == "verdict" and condition not in {"facts_qa", "preferences_qa"}
    label = "Claim" if use_claim else "Question"
    return (
        "%s:\n%s\n\nMemories:\n%s\n\n"
        "Write only 2-4 routing bullets specifying which memories matter, "
        "the role each should play, and how conflicts or updates should be "
        "handled for this %s. Do not solve the %s."
        % (
            label,
            question.strip(),
            memory_body.strip(),
            label.lower(),
            label.lower(),
        )
    )


def answer_user_template_for_task(task: str, *, condition: str = "") -> str:
    if task == "verdict" and condition not in {"facts_qa", "preferences_qa"}:
        return "Memory context:\n{context}\n\nClaim: {question}"
    return "Memory context:\n{context}\n\nQuestion: {question}"


def build_reasoning_state_training_record(
    row: Dict[str, Any],
    *,
    pairing: Optional[Dict[str, Any]] = None,
    use_pairing_supersede: bool = True,
    wrap_style: str = WRAP_STYLE_LOCOMO,
    model_name: str = "assistant",
    current_date: str = "2025-06-01",
    think_mode: bool = False,
    memory_only: bool = False,
    channel_needs: bool = False,
) -> Dict[str, Any]:
    """One slime-ready reasoning_state-RL record with memory_body + answer templates."""
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

    mode = select_render_mode(result)
    memory_body = render_memory_body(result, include_intents=False, mode=mode)
    has_supersede = has_supersede_signal(result)

    # Prefer explicit row fields (e.g. intent-decomp exports); else RelationalState.
    if row.get("fact_need") is not None:
        fact_need = float(row["fact_need"])
    else:
        fact_need = float(result.fact_need)
    if row.get("preference_need") is not None:
        preference_need = float(row["preference_need"])
    else:
        preference_need = float(result.preference_need)

    use_needs = bool(channel_needs)
    system_prompt = select_reasoning_state_system_prompt(
        think_mode=think_mode,
        memory_only=memory_only,
        channel_needs=use_needs,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": reasoning_state_user_prompt(
                question,
                memory_body,
                task=task,
                condition=condition,
                memory_only=memory_only,
                channel_needs=use_needs,
                fact_need=fact_need if use_needs else None,
                preference_need=preference_need if use_needs else None,
            ),
        },
    ]

    answer_system = _locomo_system_prompt(
        task=task, condition=condition, pairing=pairing
    )
    if wrap_style == WRAP_STYLE_MEMSYCO:
        # MemSyco puts context in system; template still uses {context}.
        answer_system = (
            "You are %s, a helpful and natural conversational assistant.\n"
            "Knowledge cutoff: 2025-06\nCurrent date: %s\n\n"
            "Answer in a clear, useful, and natural way.\n\n"
            "Retrieved memories from earlier conversation:\n{context}"
        ) % (model_name, current_date)
        answer_user_template = "{question}"
    else:
        answer_user_template = answer_user_template_for_task(task, condition=condition)

    reward_spec = reward_spec_from_row(row)
    metadata = {
        "memory_body": memory_body,
        "question": question,
        "task": task,
        "condition": condition,
        "source": source,
        "wrap_style": wrap_style,
        "render_mode": mode,
        "has_supersede": has_supersede,
        "answer_system": answer_system,
        "answer_user_template": answer_user_template,
        "reward_spec": reward_spec,
        "source_group_id": str(row.get("source_group_id") or ""),
        "think_mode": bool(think_mode),
        "memory_only": bool(memory_only),
        "channel_needs": bool(use_needs),
        "fact_need": fact_need,
        "preference_need": preference_need,
    }

    record: Dict[str, Any] = {
        "id": str(row.get("generation_id") or row.get("pair_id") or ""),
        "source_group_id": str(row.get("source_group_id") or ""),
        "source": source,
        "condition": condition,
        "task": task,
        "render_mode": mode,
        "memory_body": memory_body,
        "question": question,
        "wrap_style": wrap_style,
        "has_supersede": has_supersede,
        "think_mode": bool(think_mode),
        "memory_only": bool(memory_only),
        "channel_needs": bool(use_needs),
        "fact_need": fact_need,
        "preference_need": preference_need,
        "messages": messages,
        "reward_spec": reward_spec,
        "answer_system": answer_system,
        "answer_user_template": answer_user_template,
        # Slime Dataset puts this on Sample.metadata for the custom RM.
        "metadata": metadata,
    }
    if row.get("gold_label"):
        record["gold_label"] = row["gold_label"]
    if pairing:
        record["pair_id"] = pairing.get("pair_id")
        record["pair_type"] = pairing.get("pair_type")
    return record
