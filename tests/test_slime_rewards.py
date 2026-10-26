"""Tests for evomemstate.slime_rewards."""

from __future__ import annotations

import pytest

from evomemstate.slime_rewards import (
    parse_verdict,
    reward_from_spec,
    token_f1_reward,
    verdict_match_reward,
)

try:
    from slime.utils.types import Sample
except ImportError:
    Sample = None  # type: ignore[misc, assignment]


def test_parse_verdict():
    assert parse_verdict("Verdict: SUPPORTED\nExplanation: ok") == "SUPPORTED"
    assert parse_verdict("verdict: refuted") == "REFUTED"
    assert parse_verdict("no verdict here") is None


def test_verdict_match_reward():
    assert verdict_match_reward("Verdict: REFUTED\nExplanation: x", "REFUTED") == 1.0
    assert verdict_match_reward("Verdict: SUPPORTED", "REFUTED") == 0.0
    assert verdict_match_reward("unknown", "REFUTED") == 0.0


def test_token_f1_reward():
    score = token_f1_reward("The IMO targets 2030 emissions.", "IMO 2030 emission targets")
    assert 0.0 < score < 1.0


def test_strip_thinking_before_f1():
    thinking = "Okay, Arctic documentaries...\n\nBased on your interest, try Chasing Ice."
    target = "Chasing Ice captures polar landscapes"
    full_f1 = token_f1_reward(thinking, target)
    plain_f1 = token_f1_reward("Try Chasing Ice for polar landscapes.", target)
    assert plain_f1 >= full_f1


def test_verdict_after_thinking():
    response = "reasoning\n\n\nVerdict: REFUTED\nExplanation: no."
    assert verdict_match_reward(response, "REFUTED") == 1.0


def test_reward_from_spec_dispatch_sync():
    if Sample is None:
        pytest.skip("slime not installed")

    import asyncio

    verdict_sample = Sample(
        prompt="p",
        response="Verdict: REFUTED\nExplanation: born 1971.",
        label={"type": "verdict_match", "target": "REFUTED"},
    )
    assert asyncio.run(reward_from_spec(None, verdict_sample)) == 1.0

    qa_sample = Sample(
        prompt="p",
        response="Try the night market for food festivals.",
        label={
            "type": "token_f1",
            "target": "Try the night market.",
        },
    )
    assert asyncio.run(reward_from_spec(None, qa_sample)) > 0.5


def test_reward_from_spec_dispatch():
    if Sample is None:
        pytest.skip("slime not installed")

    import asyncio

    verdict_sample = Sample(
        prompt="p",
        response="Verdict: REFUTED\nExplanation: born 1971.",
        label={"type": "verdict_match", "target": "REFUTED"},
    )
    assert asyncio.run(reward_from_spec(None, verdict_sample)) == 1.0

    qa_sample = Sample(
        prompt="p",
        response="Try the night market for food festivals.",
        label={
            "type": "token_f1",
            "target": "Try the night market.",
        },
    )
    assert asyncio.run(reward_from_spec(None, qa_sample)) > 0.5
