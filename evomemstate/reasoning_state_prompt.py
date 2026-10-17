"""Inference prompts and output parsing for generated Reasoning State."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .prompting import REASONING_STATE_INTRO

REASONING_STATE_SYSTEM_PROMPT = (
    "You are a memory-routing helper. Given a question and retrieved memories, "
    "write a compact Reasoning State telling an answer model how to use the memories for "
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
# stays byte-identical.
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
    "draft an answer inside <think>. Then write the final Reasoning State inside "
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
    ".replace() target in reasoning_state_prompt.py"
)

# Memory-only: Reasoning State model never sees the question/claim.
REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY = (
    "You are a memory-routing helper. Given only retrieved memories (no user "
    "question), write a compact Reasoning State telling an answer model how to use those "
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
    "compact Reasoning State telling an answer model how to use those memories.\n"
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
    "a claim or draft an answer inside <think>. Then write the final Reasoning State "
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

    Extraction notes record malformed output; the generated rules remain separate
    from any model reasoning text.
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


def format_channel_needs_block(
    fact_need: Optional[float], preference_need: Optional[float]
) -> str:
    """Render retrieval channel needs for memory-only generation."""
    if fact_need is None and preference_need is None:
        return ""
    fact = 0.5 if fact_need is None else float(fact_need)
    preference = 0.5 if preference_need is None else float(preference_need)
    return (
        "Channel needs:\n"
        "- fact_need: %.2f (objective evidence and constraints)\n"
        "- preference_need: %.2f (user preferences and beliefs)\n"
        % (fact, preference)
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
