"""LLM judge: does a generated Reasoning State leak answer statements?

``reasoning_state`` remains the identifier in the reward and judge interfaces.

Uses the same frozen answer OpenAI API as reasoning_state RL (default Qwen3-8B).
Required output format (short, no chain-of-thought)::

    <answer>yes</answer>   # or no

Only the text inside ``<answer>...</answer>`` is used for the decision.
``yes`` = reasoning_state contains answer-like statements → penalize.
``no``  = high-level routing / usage instructions only → OK.

Few-shot chat turns are prepended so the model stays on-format even with
``REASONING_STATE_ANSWER_JUDGE_MAX_TOKENS`` as low as 32.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_ANSWER_TAG_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)

JUDGE_SYSTEM_PROMPT = """\
You are a format auditor for memory-routing reasoning_states. Decide whether the reasoning_state \
already contains answer content for the user's question, or whether it is only \
high-level routing / usage instructions.

Say YES (contains answer) if the reasoning_state does any of the following:
- States a final conclusion (yes/no, true/false, supported/refuted, correct/incorrect)
- Gives factual claims that could serve as the final answer (names, dates, causes, \
  outcomes, definitions about the question topic)
- Encodes the answer polarity by declaring a user belief true/false or "the \
  question/claim is true/false"
- Instructs what the final answer wording should be (e.g. "say yes", "answer is no")
- Paraphrases of the above (e.g. "I think X is true", "this seems correct")

Say NO (routing only) if the reasoning_state ONLY:
- Points to which memories to use / ignore and their roles
- Describes how to resolve conflicts (evidence vs preference, supersede, etc.)
- Gives process rules without stating the factual conclusion
  (e.g. "treat preference as belief, not evidence" WITHOUT saying the belief is wrong \
  as a fact about the world / WITHOUT giving the answer)

Ignore whether the answer content is correct. Correct and incorrect answer \
statements both count as YES.

Do NOT solve the question. Do NOT explain. Do NOT output <think> or any prose.
Output EXACTLY one of these two lines (nothing else):

<answer>yes</answer>

or

<answer>no</answer>
"""

# Short in-context examples: (question, reasoning_state, "yes"|"no")
_JUDGE_FEWSHOT: List[Tuple[str, str, str]] = [
    (
        "Was Splash released in the 1980s?",
        "- Prefer memory 1 as primary evidence for the release year.\n"
        "- Treat memory 2 as supporting context only.\n"
        "- Ignore superseded preferences when they conflict with evidence.",
        "no",
    ),
    (
        "Was Splash released in the 1980s?",
        "- Use memory 1 as primary evidence.\n"
        "- The film was released in 1984, so answer yes.",
        "yes",
    ),
    (
        "Does the user prefer window or aisle seats?",
        "- Use the latest preference memory as the controlling preference.\n"
        "- If an older preference is marked superseded, do not let it override.",
        "no",
    ),
    (
        "Does the user prefer window or aisle seats?",
        "- Prefer the seat preference memory.\n"
        "- Answer aisle; ignore the older window preference.",
        "yes",
    ),
]


def judge_enabled() -> bool:
    raw = (os.environ.get("REASONING_STATE_ANSWER_JUDGE") or "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def parse_judge_answer(text: str) -> Tuple[Optional[bool], str]:
    """Parse ``<answer>yes|no</answer>``.

    Returns ``(contains_answer, raw_tag_text)``.
    ``contains_answer`` is True/False when parsed, or None on parse failure.
    Uses the **last** ``<answer>`` tag if several appear.
    """
    matches = list(_ANSWER_TAG_RE.finditer(text or ""))
    if not matches:
        return None, ""
    raw = (matches[-1].group(1) or "").strip().lower()
    # Keep first token; allow "yes." / "no," etc.
    token = re.split(r"[\s,.;:!?]+", raw, maxsplit=1)[0] if raw else ""
    if token in {"yes", "y", "true", "1"}:
        return True, raw
    if token in {"no", "n", "false", "0"}:
        return False, raw
    return None, raw


def _audit_user_content(question: str, reasoning_state: str) -> str:
    return (
        "Question:\n%s\n\n"
        "Reasoning State to audit:\n%s\n\n"
        "Does this reasoning_state contain answer statements (not only routing instructions)? "
        "Reply with ONLY <answer>yes</answer> or <answer>no</answer>."
        % ((question or "").strip(), (reasoning_state or "").strip())
    )


def build_judge_messages(question: str, reasoning_state: str) -> list:
    """System + few-shot turns + the real audit request."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
    ]
    for ex_q, ex_reasoning_state, label in _JUDGE_FEWSHOT:
        messages.append({"role": "user", "content": _audit_user_content(ex_q, ex_reasoning_state)})
        messages.append({"role": "assistant", "content": "<answer>%s</answer>" % label})
    messages.append({"role": "user", "content": _audit_user_content(question, reasoning_state)})
    return messages


def _judge_max_tokens() -> int:
    # Floor at 16 so env=32 actually applies (old floor of 64 blocked short budgets).
    return max(16, int(os.environ.get("REASONING_STATE_ANSWER_JUDGE_MAX_TOKENS") or "32"))


def _judge_extra_body() -> Dict[str, Any]:
    """Disable Qwen3 thinking tokens so max_tokens is not eaten by CoT."""
    return {"chat_template_kwargs": {"enable_thinking": False}}


def interpret_judge_result(
    raw_response: str,
) -> Dict[str, Any]:
    """Turn raw judge text into a structured decision dict."""
    contains, tag = parse_judge_answer(raw_response)
    return {
        "raw": raw_response,
        "answer_tag": tag,
        "contains_answer": contains,
        "parse_ok": contains is not None,
        # Fail-open on parse errors: do not zero reward just because format broke.
        "penalize": bool(contains is True),
    }


async def judge_reasoning_state_contains_answer_async(
    question: str,
    reasoning_state: str,
    *,
    call_fn=None,
) -> Dict[str, Any]:
    """Call frozen 8B judge; return interpret_judge_result dict."""
    from .reasoning_state_reward import call_frozen_answer_async

    messages = build_judge_messages(question, reasoning_state)
    fn = call_fn or call_frozen_answer_async
    try:
        raw = await fn(
            messages,
            temperature=0.0,
            max_tokens=_judge_max_tokens(),
            extra_body=_judge_extra_body(),
        )
    except TypeError:
        # Tests / older stubs may not accept extra_body.
        try:
            raw = await fn(messages, temperature=0.0, max_tokens=_judge_max_tokens())
        except Exception as exc:
            logger.warning("reasoning_state_judge failed: %s", exc)
            return {
                "raw": "",
                "answer_tag": "",
                "contains_answer": None,
                "parse_ok": False,
                "penalize": False,
                "error": str(exc),
            }
    except Exception as exc:
        logger.warning("reasoning_state_judge failed: %s", exc)
        return {
            "raw": "",
            "answer_tag": "",
            "contains_answer": None,
            "parse_ok": False,
            "penalize": False,
            "error": str(exc),
        }
    result = interpret_judge_result(raw or "")
    if not result["parse_ok"]:
        logger.info(
            "reasoning_state_judge parse_fail tag=%r raw_head=%r",
            result["answer_tag"],
            (raw or "")[:160],
        )
    return result


def judge_reasoning_state_contains_answer(
    question: str,
    reasoning_state: str,
    *,
    call_fn=None,
) -> Dict[str, Any]:
    """Sync wrapper for tests / offline."""
    from .reasoning_state_reward import call_frozen_answer

    messages = build_judge_messages(question, reasoning_state)
    fn = call_fn or call_frozen_answer
    try:
        raw = fn(
            messages,
            temperature=0.0,
            max_tokens=_judge_max_tokens(),
            extra_body=_judge_extra_body(),
        )
    except TypeError:
        try:
            raw = fn(messages, temperature=0.0, max_tokens=_judge_max_tokens())
        except Exception as exc:
            logger.warning("reasoning_state_judge failed: %s", exc)
            return {
                "raw": "",
                "answer_tag": "",
                "contains_answer": None,
                "parse_ok": False,
                "penalize": False,
                "error": str(exc),
            }
    except Exception as exc:
        logger.warning("reasoning_state_judge failed: %s", exc)
        return {
            "raw": "",
            "answer_tag": "",
            "contains_answer": None,
            "parse_ok": False,
            "penalize": False,
            "error": str(exc),
        }
    return interpret_judge_result(raw or "")
