"""Tests for reasoning_state-generator RL helpers."""

from __future__ import annotations

from types import SimpleNamespace

from evomemstate.reasoning_state_policy import (
    REASONING_STATE_SYSTEM_PROMPT,
    build_reasoning_state_training_record,
    strip_reasoning_state_from_context,
)
from evomemstate.reasoning_state_reward import (
    build_answer_messages,
    context_with_reasoning_state,
    score_pair,
)
from evomemstate.prompting import (
    REASONING_STATE_INTRO,
    EMPTY_CONTEXT,
    compose_reasoning_state,
    render_context,
    render_memory_body,
)
from evomemstate.training_data import relational_state_from_fever_scifact


def test_qa_scifact_row_uses_token_f1_and_yes_no_prompt():
    row = {
        "rl_id": "scifact_2_preferences_qa_belief",
        "condition": "preferences_qa",
        "source": "scifact",
        "source_group_id": "scifact_2",
        "claim": "1 in 5 million in UK have abnormal PrP positivity.",
        "query": "Is it true that 1 in 5 million in UK have abnormal PrP positivity?",
        "gold_label": "REFUTED",
        "preference_memories": [
            "The user firmly believes the following statement is true: "
            '"1 in 5 million in UK have abnormal PrP positivity."',
            "The user prefers tea without sugar.",
        ],
        "reference_correct_answer": (
            "No. A large-scale UK survey found abnormal PrP in 16 of 32,441 "
            "appendix samples, far above 1 in 5 million."
        ),
        "reference_incorrect_answers": ["Yes, that's right."],
        "messages": [
            {
                "role": "user",
                "content": (
                    "User memories:\n- x\n\nQuestion: Is it true that ...?\n\n"
                    "Answer in one or two sentences, starting with Yes or No."
                ),
            }
        ],
    }
    rec = build_reasoning_state_training_record(row)
    assert rec["reward_spec"]["type"] == "token_f1"
    assert rec["reward_spec"]["target"].startswith("No.")
    assert "Is it true that" in rec["question"]
    assert "Start your answer with Yes or No" in rec["answer_system"]
    assert "Claim:" not in rec["answer_user_template"]
    assert "Question:" in rec["answer_user_template"]
    assert "## User preferences" in rec["memory_body"] or "preference" in rec["memory_body"].lower()


def test_qa_facts_row_parses_facts_from_messages():
    row = {
        "rl_id": "scifact_2_facts_qa",
        "condition": "facts_qa",
        "source": "scifact",
        "source_group_id": "scifact_2",
        "claim": "1 in 5 million in UK have abnormal PrP positivity.",
        "query": "Is it true that 1 in 5 million in UK have abnormal PrP positivity?",
        "gold_label": "REFUTED",
        "reference_correct_answer": "No. Prevalence is about 493 per million.",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Facts:\n"
                    "- [Paper: survey] RESULTS Of the 32,441 appendix samples 16 were positive.\n\n"
                    "Question: Is it true that ...?\n\n"
                    "Answer in one or two sentences, starting with Yes or No."
                ),
            }
        ],
    }
    result = relational_state_from_fever_scifact(row)
    assert result.facts
    assert "32,441" in result.facts[0].memory.content
    rec = build_reasoning_state_training_record(row)
    assert rec["reward_spec"]["type"] == "token_f1"
    assert "## Established evidence" in rec["memory_body"]


def test_render_memory_body_has_no_reasoning_state():
    row = {
        "generation_id": "t1",
        "condition": "facts_only",
        "source": "scifact",
        "facts": ["Paper says X is true."],
        "claim": "X is true.",
        "gold_label": "SUPPORTED",
    }
    result = relational_state_from_fever_scifact(row)
    body = render_memory_body(result)
    full = render_context(result)
    assert REASONING_STATE_INTRO not in body
    assert "## Established evidence" in body
    assert REASONING_STATE_INTRO in full
    assert body in full


def test_reasoning_state_in_render_context():
    row = {
        "generation_id": "t2",
        "condition": "facts_only",
        "source": "scifact",
        "facts": ["Evidence line."],
        "claim": "Claim.",
        "gold_label": "REFUTED",
    }
    result = relational_state_from_fever_scifact(row)
    custom = "- Prefer established evidence over user beliefs for this claim."
    out = render_context(result, reasoning_state=custom)
    assert "Prefer established evidence" in out
    # Fixed evidence-priority boilerplate should be replaced by custom text.
    assert "It outranks everything else here" not in out


def test_build_aligned_record_reasoning_state():
    from evomemstate.training_data import build_aligned_training_record

    row = {
        "generation_id": "t2b",
        "condition": "facts_only",
        "source": "scifact",
        "facts": ["Evidence line."],
        "claim": "Claim.",
        "gold_label": "REFUTED",
    }
    custom = "- Weight the Evidence line memory for this claim."
    rec = build_aligned_training_record(
        row, reasoning_state=custom, reasoning_state_source="qwen3-8b"
    )
    assert rec["reasoning_state_source"] == "qwen3-8b"
    assert rec["generated_reasoning_state"] == custom
    assert "Weight the Evidence line" in rec["context_text"]
    assert "It outranks everything else here" not in rec["context_text"]
    assert any("Weight the Evidence line" in (m.get("content") or "") for m in rec["messages"])


def test_compose_reasoning_state_bullets():
    text = compose_reasoning_state("Prefer facts\nIgnore preferences about the world")
    assert text.startswith(REASONING_STATE_INTRO)
    assert "- Prefer facts" in text


def test_build_reasoning_state_training_record():
    row = {
        "generation_id": "scifact_1_facts",
        "condition": "facts_only",
        "source": "scifact",
        "source_group_id": "g1",
        "facts": ["Born in 1971."],
        "claim": "Born in the 80s.",
        "gold_label": "REFUTED",
    }
    record = build_reasoning_state_training_record(row, wrap_style="locomo")
    assert record["messages"][0]["content"] == REASONING_STATE_SYSTEM_PROMPT
    assert "Memories:" in record["messages"][1]["content"]
    assert REASONING_STATE_INTRO not in record["memory_body"]
    assert record["reward_spec"]["type"] == "verdict_match"
    assert record["metadata"]["memory_body"] == record["memory_body"]
    assert "{context}" in record["answer_user_template"]


def test_score_pair_delta_with_mock_answer():
    sample = SimpleNamespace(
        label={"type": "verdict_match", "target": "REFUTED"},
        metadata={
            "memory_body": "## Established evidence\n1. Born in 1971.",
            "question": "Born in the 80s.",
            "answer_system": "Evaluate. Return Verdict: SUPPORTED or REFUTED",
            "answer_user_template": "Memory context:\n{context}\n\nClaim: {question}",
            "has_supersede": False,
            "reward_spec": {"type": "verdict_match", "target": "REFUTED"},
        },
        response="- Prefer established evidence over user beliefs.",
        reward=None,
    )

    def fake_answer(messages):
        joined = " ".join(m["content"] for m in messages)
        if "Reasoning State" in joined:
            return "Verdict: REFUTED\nExplanation: 1971."
        return "Verdict: SUPPORTED\nExplanation: guess."

    reward, sw, so, debug = score_pair(sample, sample.response, answer_fn=fake_answer)
    assert sw == 1.0
    assert so == 0.0
    assert debug["new_acc"] == 1.0
    assert debug["delta_acc"] == 1.0
    assert reward == 1.0  # delta only
    assert "REFUTED" in debug["answer_with"]


def test_reward_from_reasoning_state_batch_uses_async_mock(monkeypatch):
    import asyncio
    import os

    from evomemstate import reasoning_state_reward as mod

    calls = {"n": 0}
    monkeypatch.setenv("REASONING_STATE_ANSWER_JUDGE", "0")

    async def fake_async(messages, **kwargs):
        calls["n"] += 1
        await asyncio.sleep(0)
        joined = " ".join(m["content"] for m in messages)
        if "Reasoning State" in joined:
            return "Verdict: REFUTED"
        return "Verdict: SUPPORTED"

    monkeypatch.setattr(mod, "call_frozen_answer_async", fake_async)

    def make_sample(i):
        return SimpleNamespace(
            index=i,
            label={"type": "verdict_match", "target": "REFUTED"},
            metadata={
                "memory_body": "## Established evidence\n1. Born in 1971.",
                "question": "Born in the 80s.",
                "answer_system": "SYS",
                "answer_user_template": "Memory context:\n{context}\n\nClaim: {question}",
                "has_supersede": False,
            },
            response="- Weight the 1971 birth record over conflicting preferences %s" % i,
            reward=None,
        )

    samples = [make_sample(i) for i in range(4)]
    rewards = asyncio.run(mod.reward_from_reasoning_state(None, samples))
    assert rewards == [1.0, 1.0, 1.0, 1.0]
    assert calls["n"] == 8


def test_reward_from_reasoning_state_judge_yes_applies_penalty(monkeypatch):
    import asyncio

    from evomemstate import reasoning_state_reward as mod

    monkeypatch.setenv("REASONING_STATE_ANSWER_JUDGE", "1")
    monkeypatch.setenv("REASONING_STATE_ANSWER_JUDGE_PENALTY", "0.1")

    async def fake_async(messages, **kwargs):
        await asyncio.sleep(0)
        joined = " ".join(m["content"] for m in messages)
        if "Reasoning State to audit" in joined or "format auditor" in joined.lower():
            return "<answer>yes</answer>"
        if any("Reasoning State to audit" in (m.get("content") or "") for m in messages):
            return "<answer>yes</answer>"
        if "Reasoning State" in joined:
            return "Verdict: REFUTED"
        return "Verdict: SUPPORTED"

    monkeypatch.setattr(mod, "call_frozen_answer_async", fake_async)

    sample = SimpleNamespace(
        index=0,
        label={"type": "verdict_match", "target": "REFUTED"},
        metadata={
            "memory_body": "## Established evidence\n1. Born in 1971.",
            "question": "Born in the 80s.",
            "answer_system": "SYS",
            "answer_user_template": "Memory context:\n{context}\n\nClaim: {question}",
            "has_supersede": False,
        },
        # Soft leak that current regex format-gate may miss; LLM judge still flags it.
        response=(
            "- Prefer memory 1 as primary evidence over preferences.\n"
            "- I think the statement about the 80s birth year is true based on context."
        ),
        reward=None,
    )
    reward = asyncio.run(mod.reward_from_reasoning_state(None, sample))
    # delta=+1.0 (with correct, without wrong), penalty=-0.1
    assert abs(reward - 0.9) < 1e-9
    debug = sample.metadata["reasoning_state_reward_debug"]
    assert debug["raw_delta_acc"] == 1.0
    assert debug["delta_acc"] == 1.0
    assert debug["penalty_acc"] == -0.1
    assert debug["judge_penalize"] is True
    assert debug["judge_contains_answer"] is True
    assert "answer_in_reasoning_state" in debug["format_violations"]


def test_post_process_reasoning_state_rewards_separate_whitening(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv("REASONING_STATE_ADVANTAGE", raising=False)
    from evomemstate.reasoning_state_reward import post_process_reasoning_state_rewards

    def make(delta_acc, penalty_acc=0.0, index=0):
        return SimpleNamespace(
            index=index,
            reward=delta_acc + penalty_acc,
            metadata={
                "reasoning_state_reward_debug": {
                    "new_acc": 0.5,
                    "delta_acc": delta_acc,
                    "penalty_acc": penalty_acc,
                    "reward": delta_acc + penalty_acc,
                }
            },
        )

    # Group A: deltas [1, 0], penalties [0, -0.1]
    # Group B: deltas [0.5, -0.5], penalties [0, 0]
    samples = [
        make(1.0, 0.0, 0),
        make(0.0, -0.1, 1),
        make(0.5, 0.0, 2),
        make(-0.5, 0.0, 3),
    ]
    args = SimpleNamespace(
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        grpo_std_normalization=True,
    )
    raw, adv = post_process_reasoning_state_rewards(args, samples)
    assert raw == [1.0, -0.1, 0.5, -0.5]

    # Match implementation: centered / (std + 1e-6) with unbiased std.
    # Group of 2 with values differing by d → |centered|=d/2, std=d/sqrt(2).
    def pair_scale(spread: float) -> float:
        # values [a, a-spread] → mean mid, centered [spread/2, -spread/2]
        # std = sqrt(2*(spread/2)^2 / 1) = spread/sqrt(2)
        std = spread / (2.0**0.5)
        return (spread / 2.0) / (std + 1e-6)

    sd = pair_scale(1.0)
    sp = pair_scale(0.1)
    assert abs(adv[0] - (sd + sp)) < 1e-9
    assert abs(adv[1] - (-sd - sp)) < 1e-9
    # Group1: penalty all 0 → whiten 0
    assert abs(adv[2] - sd) < 1e-9
    assert abs(adv[3] + sd) < 1e-9


def test_post_process_reasoning_state_rewards_baseline_anchored(monkeypatch):
    """Δ/std without subtracting the group mean, so S_0 is not cancelled."""
    from types import SimpleNamespace

    monkeypatch.setenv("REASONING_STATE_ADVANTAGE", "baseline_anchored")
    from evomemstate.reasoning_state_reward import post_process_reasoning_state_rewards

    def make(delta_acc, penalty_acc=0.0, index=0):
        return SimpleNamespace(
            index=index,
            reward=delta_acc + penalty_acc,
            metadata={
                "reasoning_state_reward_debug": {
                    "new_acc": 0.5,
                    "delta_acc": delta_acc,
                    "penalty_acc": penalty_acc,
                    "reward": delta_acc + penalty_acc,
                }
            },
        )

    # Group A: deltas [1, 0] — the Δ=0 reasoning_state matches S_0 and must stay adv 0,
    # not be pushed negative by centering. Penalties [0, -0.1].
    # Group B: both reasoning_states beat S_0 by the same amount, same penalty → std 0
    # → raw values pass through (still positive vs baseline).
    samples = [
        make(1.0, 0.0, 0),
        make(0.0, -0.1, 1),
        make(0.5, -0.05, 2),
        make(0.5, -0.05, 3),
    ]
    args = SimpleNamespace(
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        grpo_std_normalization=True,
    )
    raw, adv = post_process_reasoning_state_rewards(args, samples)
    assert raw == [1.0, -0.1, 0.45, 0.45]

    # Group of 2, spread d: unbiased std = d/sqrt(2). No centering.
    std_d = 1.0 / (2.0 ** 0.5)
    std_p = 0.1 / (2.0 ** 0.5)
    assert abs(adv[0] - (1.0 / (std_d + 1e-6))) < 1e-9
    assert abs(adv[1] - (0.0 + (-0.1) / (std_p + 1e-6))) < 1e-9
    # Constant delta 0.5 and constant penalty -0.05 both pass through.
    assert abs(adv[2] - (0.5 - 0.05)) < 1e-9
    assert abs(adv[3] - (0.5 - 0.05)) < 1e-9
    # Mean advantage stays positive: baseline was not subtracted away.
    assert sum(adv) / len(adv) > 0.0


def test_singleflight_dedupes_identical_answer_calls(monkeypatch, tmp_path):
    import asyncio
    from evomemstate import reasoning_state_reward as mod

    monkeypatch.setenv("REASONING_STATE_ANSWER_CACHE_DIR", str(tmp_path))
    mod._MEMORY_CACHE.clear()
    mod._INFLIGHT.clear()
    mod._SEM = None
    mod._ASYNC_CLIENT = None

    started = asyncio.Event()
    release = asyncio.Event()
    calls = {"n": 0}

    class _Msg:
        content = "Verdict: REFUTED"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        async def create(self, **kwargs):
            calls["n"] += 1
            started.set()
            await release.wait()
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    async def fake_client():
        return _Client()

    monkeypatch.setattr(mod, "_get_async_client", fake_client)

    msgs = [{"role": "user", "content": "same"}]

    async def run():
        t1 = asyncio.create_task(mod.call_frozen_answer_async(msgs))
        await started.wait()
        t2 = asyncio.create_task(mod.call_frozen_answer_async(msgs))
        await asyncio.sleep(0)
        release.set()
        out = await asyncio.gather(t1, t2)
        return out

    a, b = asyncio.run(run())
    assert a == b == "Verdict: REFUTED"
    assert calls["n"] == 1


def test_strip_reasoning_state_helper():
    text = "## Facts\n1. x\n\nReasoning State:\n- y"
    assert strip_reasoning_state_from_context(text) == "## Facts\n1. x"


def test_build_answer_messages_locomo():
    msgs = build_answer_messages(
        context="MEM",
        question="Q?",
        answer_system="SYS",
        answer_user_template="Memory context:\n{context}\n\nQuestion: {question}",
    )
    assert msgs[0]["role"] == "system"
    assert "MEM" in msgs[1]["content"]
    assert "Q?" in msgs[1]["content"]


def test_context_with_reasoning_state_attaches_reasoning_state():
    out = context_with_reasoning_state("## Facts\n1. a", "- Prefer evidence", has_supersede=False)
    assert out.startswith("## Facts")
    assert REASONING_STATE_INTRO in out
    assert "Prefer evidence" in out


def test_empty_memory_body():
    assert EMPTY_CONTEXT


def test_reasoning_state_format_gate_zeros_verdict_tokens():
    from evomemstate.reasoning_state_format import apply_reasoning_state_format_gate, reasoning_state_format_violations

    bad = "- The evidence SUPPORTS the claim about WBRT."
    assert "verdict_token" in reasoning_state_format_violations(bad) or "answers_claim" in reasoning_state_format_violations(bad)
    gated, viol, raw = apply_reasoning_state_format_gate(0.5, bad)
    assert gated == 0.0 and raw == 0.5 and viol

    good = (
        "- Weight the 1971 birth record over conflicting decade claims.\n"
        "- Drop superseded preference entries about birth year."
    )
    assert reasoning_state_format_violations(good) == []
    gated2, viol2, raw2 = apply_reasoning_state_format_gate(-0.2, good)
    assert gated2 == -0.2 and not viol2 and raw2 == -0.2


def test_reasoning_state_format_echo_prompt_collapsed():
    from evomemstate.reasoning_state_format import reasoning_state_format_violations

    echo = (
        "- Do not answer the claim/question yourself.\n"
        "- Do not say whether a claim is true/false.\n"
        "- Do not invent memories or quote a final verdict.\n"
        "- Keep it concise (under ~500 characters)."
    )
    viol = reasoning_state_format_violations(echo)
    assert "echo_prompt" in viol


def test_reasoning_state_format_echo_example_and_bare_scope():
    from evomemstate.reasoning_state_format import reasoning_state_format_violations

    example = (
        "- Prefer established evidence when it conflicts with user beliefs.\n"
        "- Ignore superseded preferences."
    )
    assert "echo_example" in reasoning_state_format_violations(example)
    assert "echo_example" in reasoning_state_format_violations(
        "- Prefer established evidence over user preferences."
    )
    assert "bare_scope" in reasoning_state_format_violations("- Scope.")
    assert "bare_scope" in reasoning_state_format_violations("- Scope: global.")


def test_reasoning_state_format_too_long():
    from evomemstate.reasoning_state_format import reasoning_state_format_violations
    import os

    os.environ["REASONING_STATE_MAX_CHARS"] = "50"
    try:
        assert "too_long" in reasoning_state_format_violations("- Prefer evidence " + ("x" * 80))
    finally:
        os.environ.pop("REASONING_STATE_MAX_CHARS", None)


def test_empty_reasoning_state_format_penalty():
    from evomemstate.reasoning_state_format import (
        format_penalty_for_violations,
        reasoning_state_format_violations,
    )
    import os

    os.environ["REASONING_STATE_FORMAT_EMPTY_PENALTY"] = "0.05"
    try:
        assert "empty" in reasoning_state_format_violations("")
        assert "empty" in reasoning_state_format_violations("<|im_end|>")
        assert "empty" in reasoning_state_format_violations("<think>\nonly think\n")
        assert format_penalty_for_violations(["empty"]) == -0.05
        assert format_penalty_for_violations(["verdict_token"]) == 0.0
        assert format_penalty_for_violations([]) == 0.0
    finally:
        os.environ.pop("REASONING_STATE_FORMAT_EMPTY_PENALTY", None)


def test_reasoning_state_system_prompt_not_bullet_rules():
    from evomemstate.reasoning_state_policy import REASONING_STATE_SYSTEM_PROMPT

    # Avoid bullet-style hard rules / verbatim Example that the 0.6B model parroted.
    assert "Hard rules:" not in REASONING_STATE_SYSTEM_PROMPT
    assert "- Do NOT answer" not in REASONING_STATE_SYSTEM_PROMPT
    assert "Example:" not in REASONING_STATE_SYSTEM_PROMPT
    assert "Prefer established evidence when it conflicts with user beliefs" not in REASONING_STATE_SYSTEM_PROMPT
    assert "no more than 80 words" in REASONING_STATE_SYSTEM_PROMPT
    assert "Do not combine conflicting memories merely to appear balanced" in REASONING_STATE_SYSTEM_PROMPT


def test_reasoning_state_system_prompt_think_replace():
    from evomemstate.reasoning_state_policy import REASONING_STATE_SYSTEM_PROMPT, REASONING_STATE_SYSTEM_PROMPT_THINK

    assert REASONING_STATE_SYSTEM_PROMPT_THINK != REASONING_STATE_SYSTEM_PROMPT
    assert "<think>" in REASONING_STATE_SYSTEM_PROMPT_THINK
    assert "<reasoning_state>" in REASONING_STATE_SYSTEM_PROMPT_THINK
    assert "Output exactly one <think>" in REASONING_STATE_SYSTEM_PROMPT_THINK
    # Shared routing body preserved.
    assert "Do not combine conflicting memories merely to appear balanced" in REASONING_STATE_SYSTEM_PROMPT_THINK


def test_reasoning_state_memory_only_prompts():
    from evomemstate.reasoning_state_policy import (
        REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY,
        REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY_NEEDS,
        REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY_THINK,
        reasoning_state_user_prompt,
        select_reasoning_state_system_prompt,
    )

    assert "Given only retrieved memories" in REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY
    assert "fact_need" not in REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY
    assert "fact_need" in REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY_NEEDS
    assert "preference_need" in REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY_NEEDS
    assert "<think>" in REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY_THINK
    assert select_reasoning_state_system_prompt(think_mode=False, memory_only=True) == (
        REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY
    )
    assert select_reasoning_state_system_prompt(
        think_mode=False, memory_only=True, channel_needs=True
    ) == REASONING_STATE_SYSTEM_PROMPT_MEMORY_ONLY_NEEDS

    user = reasoning_state_user_prompt(
        "What color is the sky?",
        "## facts\n- sky is blue",
        memory_only=True,
    )
    assert user.startswith("Memories:")
    assert "Channel needs:" not in user
    assert "What color is the sky?" not in user

    user_needs = reasoning_state_user_prompt(
        "What color is the sky?",
        "## facts\n- sky is blue",
        memory_only=True,
        channel_needs=True,
        fact_need=0.9,
        preference_need=0.1,
    )
    assert "Channel needs:" in user_needs
    assert "fact_need: 0.90" in user_needs
    assert "preference_need: 0.10" in user_needs
    assert "What color is the sky?" not in user_needs

    user_q = reasoning_state_user_prompt(
        "What color is the sky?",
        "## facts\n- sky is blue",
        memory_only=False,
    )
    assert "Question:" in user_q
    assert "What color is the sky?" in user_q
    assert "Channel needs:" not in user_q


def test_extract_reasoning_state_and_thought():
    from evomemstate.reasoning_state_policy import extract_reasoning_state_and_thought, format_think_reasoning_state_assistant

    raw = (
        "<think>\nPrefer evidence over belief.\n</think>\n"
        "<reasoning_state>\n- Use memory 1 as evidence.\n</reasoning_state>"
    )
    reasoning_state, thought, notes = extract_reasoning_state_and_thought(raw)
    assert notes == []
    assert thought == "Prefer evidence over belief."
    assert reasoning_state == "- Use memory 1 as evidence."

    # Fallback: closed think, missing reasoning_state wrapper → remainder after strip.
    raw2 = "<think>\nplan\n</think>\n- Use memory 2 as constraint."
    reasoning_state2, thought2, notes2 = extract_reasoning_state_and_thought(raw2)
    assert thought2 == "plan"
    assert "memory 2" in reasoning_state2
    assert "missing_reasoning_state_tag" not in notes2

    # Unclosed think → empty reasoning_state + notes.
    reasoning_state3, thought3, notes3 = extract_reasoning_state_and_thought("<think>\nnever closes")
    assert reasoning_state3 == ""
    assert "think_unclosed" in notes3
    assert "missing_reasoning_state_tag" in notes3

    # reasoning_content fallback when no inline think tags.
    reasoning_state4, thought4, notes4 = extract_reasoning_state_and_thought(
        "<reasoning_state>\n- Prefer evidence.\n</reasoning_state>",
        reasoning_content="route from evidence",
    )
    assert thought4 == "route from evidence"
    assert reasoning_state4 == "- Prefer evidence."
    assert notes4 == []

    packed = format_think_reasoning_state_assistant("reason", "- Prefer evidence.")
    assert packed.startswith("<think>\nreason\n</think>\n")
    assert "<reasoning_state>\n- Prefer evidence.\n</reasoning_state>" in packed


def test_parse_judge_answer_yes_no():
    from evomemstate.reasoning_state_judge import interpret_judge_result, parse_judge_answer

    yes, raw = parse_judge_answer("<answer>yes</answer>")
    assert yes is True and raw == "yes"
    no, raw = parse_judge_answer("<answer> no </answer>")
    assert no is False and "no" in raw
    # Still tolerant of leftover think + last tag wins
    last, _ = parse_judge_answer(
        "<answer>no</answer>\n<think>x</think>\n<answer>YES</answer>"
    )
    assert last is True
    missing, _ = parse_judge_answer("I think yes without tags")
    assert missing is None

    decision = interpret_judge_result("<answer>yes</answer>")
    assert decision["penalize"] is True
    decision_no = interpret_judge_result("<answer>no</answer>")
    assert decision_no["penalize"] is False
    # Fail-open on parse error
    decision_bad = interpret_judge_result("sure, contains answer")
    assert decision_bad["parse_ok"] is False
    assert decision_bad["penalize"] is False


def test_judge_messages_include_fewshot_and_query():
    from evomemstate.reasoning_state_judge import (
        JUDGE_SYSTEM_PROMPT,
        _JUDGE_FEWSHOT,
        build_judge_messages,
    )

    msgs = build_judge_messages(
        "Is Splash a film?",
        "- Use memory 1 as primary evidence.",
    )
    assert msgs[0]["role"] == "system"
    assert "Do NOT output <think>" in JUDGE_SYSTEM_PROMPT
    assert "<answer>yes</answer>" in JUDGE_SYSTEM_PROMPT
    # system must not require a think block in the output schema
    assert "</think>" not in JUDGE_SYSTEM_PROMPT
    assert "brief reason" not in JUDGE_SYSTEM_PROMPT
    # system + 4 few-shot (user,assistant)*2 pairs wait: 4 examples → 8 turns + 1 query
    assert len(msgs) == 1 + 2 * len(_JUDGE_FEWSHOT) + 1
    assert all(m["role"] == "assistant" and m["content"] in {"<answer>yes</answer>", "<answer>no</answer>"} for m in msgs if m["role"] == "assistant")
    assert msgs[-1]["role"] == "user"
    assert "Is Splash a film?" in msgs[-1]["content"]
    assert "Use memory 1" in msgs[-1]["content"]
    assert "ONLY <answer>yes</answer>" in msgs[-1]["content"]
