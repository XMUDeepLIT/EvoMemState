"""Reasoning State format checks (legacy ``reasoning_state`` training interface)."""

from __future__ import annotations

import os
import re
from typing import List, Sequence, Tuple

from .text_clean import strip_generated_state as strip_answer_for_reward

# Tokens that mean the reasoning_state is issuing a SciFact-style verdict.
_VERDICT_TOKEN_RE = re.compile(
    r"\b(SUPPORTED|REFUTED|SUPPORTS|REFUTES|REFUTING)\b",
    re.IGNORECASE,
)
# Phrases that mean the reasoning_state is answering / adjudicating the claim.
_ANSWERS_CLAIM_RE = re.compile(
    r"(supports?\s+the\s+claim|contradicts?\s+the\s+claim|the\s+claim\s+is\b|"
    r"therefore\s+the\s+claim|verdict\s*:|this\s+supports\s+the\s+claim|"
    r"this\s+refutes\s+the\s+claim|directly\s+supporting\s+the\s+claim|"
    r"claim\s+that\s+.+\s+is\s+(true|false|supported|refuted))",
    re.IGNORECASE,
)
# Collapse mode: model copies instruction text instead of writing routing rules.
_ECHO_PHRASE_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"do\s+not\s+answer\s+the\s+(claim|question)",
        r"do\s+not\s+say\s+whether",
        r"do\s+not\s+invent\s+memories",
        r"keep\s+it\s+concise",
        r"under\s*~?\s*500",
        r"no\s+more\s+than\s*80\s+words",
        r"hard\s+rules",
        r"meta-?reasoning_state",
        r"quote\s+a\s+final\s+verdict",
        r"output\s+2-4\s+(short,?\s+)?(imperative\s+)?bullet",
        r"speak\s+only\s+in\s+routing\s+rules",
        r"memory-routing\s+helper",
        r"write\s+(only\s+)?2-4\s+(short\s+|routing\s+)?bullets",
        r"restate\s+these\s+instructions",
        r"downstream\s+scope/?latest",
        r"merely\s+to\s+appear\s+balanced",
    )
]
# Former system-prompt Example (and near-copies) that the policy learned to parrot.
_EXAMPLE_ECHO_RES = [
    re.compile(
        r"prefer\s+established\s+evidence\s+when\s+it\s+conflicts?\s+with\s+user\s+"
        r"(beliefs?|preferences?)",
        re.IGNORECASE,
    ),
    re.compile(r"ignore\s+superseded\s+preferences?", re.IGNORECASE),
    re.compile(
        r"prefer\s+established\s+evidence\s+over\s+user\s+(beliefs?|preferences?)",
        re.IGNORECASE,
    ),
]
# Bare "Scope." / "Scope: global." exploit of the routing-cue regex.
_BARE_SCOPE_LINE_RE = re.compile(
    r"^\s*[-*]?\s*scope(?:\s*:\s*global)?\.?\s*$",
    re.IGNORECASE,
)
# At least one memory-routing cue should appear in a valid reasoning_state.
# Note: bare "scope" alone is not enough (see bare_scope check).
_ROUTING_CUE_RE = re.compile(
    r"\b(evidence|preference|preferences|prefer|supersed(?:e|ed|es)?|"
    r"memor(?:y|ies)|weight|conflict|outrank|established|"
    r"beliefs?|facts?)\b",
    re.IGNORECASE,
)


def _max_chars() -> int:
    return max(80, int(os.environ.get("REASONING_STATE_MAX_CHARS", "600")))


def _max_lines() -> int:
    return max(2, int(os.environ.get("REASONING_STATE_MAX_LINES", "8")))


def _echo_hit_count(text: str) -> int:
    return sum(1 for cre in _ECHO_PHRASE_RES if cre.search(text))


def _example_echo_hits(text: str) -> int:
    return sum(1 for cre in _EXAMPLE_ECHO_RES if cre.search(text))


def _is_bare_scope_reasoning_state(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    return all(_BARE_SCOPE_LINE_RE.match(ln) for ln in lines)


def reasoning_state_format_violations(reasoning_state: str) -> List[str]:
    """Return human-readable violation codes; empty list means OK."""
    text = (strip_answer_for_reward(reasoning_state) or "").strip()
    reasons: List[str] = []
    if not text:
        reasons.append("empty")
        return reasons
    if len(text) > _max_chars():
        reasons.append("too_long")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) > _max_lines():
        reasons.append("too_many_lines")
    if _VERDICT_TOKEN_RE.search(text):
        reasons.append("verdict_token")
    if _ANSWERS_CLAIM_RE.search(text):
        reasons.append("answers_claim")

    if _is_bare_scope_reasoning_state(text):
        reasons.append("bare_scope")

    example_hits = _example_echo_hits(text)
    # One strong Example n-gram is enough: that template was the collapse attractor.
    if example_hits >= 1:
        reasons.append("echo_example")

    echo_hits = _echo_hit_count(text)
    has_routing = bool(_ROUTING_CUE_RE.search(text))
    # Two or more instruction-echo phrases → collapsed to copying the prompt.
    if echo_hits >= 2:
        reasons.append("echo_prompt")
    # No memory-routing content at all (often pure instruction restatement).
    elif not has_routing and "bare_scope" not in reasons:
        reasons.append("no_routing")
    # Single echo phrase dominating a short reasoning_state with little else.
    elif echo_hits >= 1 and len(text) < 220 and not re.search(
        r"\b(prefer|outrank|supersed|weight)\b", text, re.IGNORECASE
    ):
        reasons.append("echo_prompt")

    return reasons


def apply_reasoning_state_format_gate(
    reward: float, reasoning_state: str
) -> Tuple[float, List[str], float]:
    """Keep delta as ``raw_reward``; zero the returned reward on format violation.

    Returns ``(gated_reward, violations, raw_reward)``.
    """
    raw = float(reward)
    violations = reasoning_state_format_violations(reasoning_state)
    if violations:
        return 0.0, violations, raw
    return raw, [], raw


def empty_format_penalty_magnitude() -> float:
    """Absolute penalty when the model emits no usable reasoning_state (``empty``).

    Env ``REASONING_STATE_FORMAT_EMPTY_PENALTY`` (default ``0.05``). Set ``0`` to disable.
    Should be larger than typical early negative deltas so silence is not optimal.
    """
    try:
        return max(0.0, float(os.environ.get("REASONING_STATE_FORMAT_EMPTY_PENALTY", "0.05")))
    except ValueError:
        return 0.05


def format_penalty_for_violations(violations: Sequence[str]) -> float:
    """Return a non-positive format penalty to add on top of gated delta.

    Currently only ``empty`` (no reasoning_state after strip) gets a fixed penalty; other
    format failures still zero delta but do not add an extra penalty.
    """
    if "empty" in violations:
        mag = empty_format_penalty_magnitude()
        return -mag if mag > 0.0 else 0.0
    return 0.0
