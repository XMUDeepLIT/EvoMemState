"""Lightweight cleanup for generated Reasoning State text."""

from __future__ import annotations

import re

_THINKING_BLOCK_RE = re.compile(
    r"<think>.*?</think>|<thinking>.*?</thinking>",
    re.IGNORECASE | re.DOTALL,
)
_THINKING_CLOSE_RE = re.compile(
    r"</think>|</thinking>",
    re.IGNORECASE,
)
_CHAT_END_RE = re.compile(r"<\|im_end\|>\s*$")


def strip_generated_state(text: str) -> str:
    """Drop chain-of-thought / chat suffixes before using a Reasoning State."""
    if not text:
        return ""
    body = str(text).strip()
    lower = body.lower()
    if "<think>" in lower and "</think>" not in lower:
        return ""
    if "<thinking>" in lower and "</thinking>" not in lower:
        return ""
    body = _THINKING_BLOCK_RE.sub("", body).strip()
    match = _THINKING_CLOSE_RE.search(body)
    if match is not None:
        body = body[match.end() :].strip()
    body = _CHAT_END_RE.sub("", body).strip()
    return body
