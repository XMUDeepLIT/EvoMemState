"""Tests for dpo-manifest → render_context alignment."""

from __future__ import annotations

import json
from pathlib import Path

from evomemstate.training_data import (
    build_aligned_training_record,
    index_personamem_pairings,
    relational_state_from_fever_scifact,
)
from evomemstate.prompting import REASONING_STATE_EVIDENCE_PRIORITY, REASONING_STATE_PREFERENCE_ALIGN, render_context


def test_fever_facts_uses_evidence_sections():
    row = {
        "generation_id": "fever_test_facts",
        "condition": "facts_only",
        "source": "fever",
        "facts": ["Born in 1971."],
        "claim": "Born in the 80s.",
        "gold_label": "REFUTED",
        "source_group_id": "fever_test",
    }
    record = build_aligned_training_record(row)
    assert "## Established evidence" in record["context_text"]
    assert REASONING_STATE_EVIDENCE_PRIORITY.splitlines()[0] in record["context_text"]
    assert record["reward_spec"]["type"] == "verdict_match"
    assert "Facts:" not in record["messages"][1]["content"]
    assert "Memory context:" in record["messages"][1]["content"]


def test_fever_preferences_forces_evidence_reasoning_state():
    row = {
        "generation_id": "fever_test_prefs",
        "condition": "preferences_only",
        "source": "fever",
        "preference_memories": ["User believes the claim is true."],
        "claim": "Test claim.",
        "gold_label": "REFUTED",
        "source_group_id": "fever_test",
    }
    result = relational_state_from_fever_scifact(row)
    plain = render_context(result)
    assert "## User preferences" in plain or "## Retrieved memories" in plain

    record = build_aligned_training_record(row)
    assert REASONING_STATE_EVIDENCE_PRIORITY.splitlines()[0] in record["context_text"]
    assert REASONING_STATE_PREFERENCE_ALIGN.splitlines()[0] not in record["context_text"]


def test_memsyco_wrap_puts_context_in_system():
    row = {
        "generation_id": "pm_test",
        "condition": "single_preference_only",
        "source": "personamem_v2",
        "input_preference": "Enjoys food festivals",
        "query": "Weekend ideas?",
        "source_group_id": "persona_1",
        "reference_correct_answer": "Try the night market.",
    }
    from evomemstate.training_data import WRAP_STYLE_MEMSYCO, build_aligned_training_record

    record = build_aligned_training_record(row, wrap_style=WRAP_STYLE_MEMSYCO)
    system = record["messages"][0]["content"]
    user = record["messages"][1]["content"]
    assert "Retrieved memories from earlier conversation" in system
    assert record["context_text"] in system
    assert "Memory context:" not in user
    assert user == "Weekend ideas?"


def test_personamem_update_old_includes_supersede():
    root = Path(__file__).resolve().parents[1] / "dpo-dataset/data"
    pairing_path = root / "personamem/generation/pairing_plan.jsonl"
    if not pairing_path.exists():
        return
    pairings = index_personamem_pairings(pairing_path)
    sample = next(iter(pairings.values()))
    rejected_id = sample["rejected_generation_id"]
    manifest = root / "personamem/generation/updated_old_preference_only.jsonl"
    row = None
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["generation_id"] == rejected_id:
                row = item
                break
    if row is None:
        return
    record = build_aligned_training_record(row, pairing=sample)
    assert "superseded by:" in record["context_text"].lower() or "superseded" in record["context_text"].lower()
