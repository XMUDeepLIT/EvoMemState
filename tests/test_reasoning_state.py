"""Inference contracts for the generated Reasoning State."""

import unittest
from unittest import mock

from evomemstate.models import Memory, ROOT_CONCEPT_ID
from evomemstate.reasoning_state import render_with_reasoning_state
from evomemstate.reasoning_state_prompt import (
    extract_reasoning_state_and_thought, reasoning_state_user_prompt,
)
from evomemstate.retrieval import RelationalState, ScoredMemory


class ReasoningStateInferenceTest(unittest.TestCase):
    def test_tagged_rules_are_extracted_without_thought(self):
        raw = "<think>private routing analysis</think><reasoning_state>- Use the current preference.</reasoning_state>"
        rules, thought, notes = extract_reasoning_state_and_thought(raw)
        self.assertEqual("- Use the current preference.", rules)
        self.assertEqual("private routing analysis", thought)
        self.assertEqual([], notes)

    def test_memory_only_prompt_includes_channel_needs(self):
        prompt = reasoning_state_user_prompt(
            "Where should we meet?", "The user prefers quiet cafes.",
            memory_only=True, channel_needs=True,
            fact_need=0.25, preference_need=0.75,
        )
        self.assertIn("fact_need: 0.25", prompt)
        self.assertIn("preference_need: 0.75", prompt)
        self.assertNotIn("Where should we meet?", prompt)

    def test_relational_state_and_generated_rules_reach_answer_context(self):
        memory = Memory(
            memory_id="memory_1",
            content="The user now prefers quiet cafes.",
            memory_type="preference",
            scope={"level": "global", "value": ""},
            time={},
            source={},
            status="active",
            parent_concept_id=ROOT_CONCEPT_ID,
        )
        state = RelationalState(preferences=[ScoredMemory(memory, 1.0, "preference")])
        generated = ("- Prioritize the user's current quiet cafe preference.", {"reasoning_state_ok": True})
        with mock.patch("evomemstate.reasoning_state.generate_reasoning_state", return_value=generated):
            context, meta = render_with_reasoning_state(state, "Where should we meet?", force=True)
        self.assertIn("The user now prefers quiet cafes.", context)
        self.assertIn("Reasoning State:", context)
        self.assertIn(generated[0], context)
        self.assertTrue(meta["reasoning_state_ok"])


if __name__ == "__main__":
    unittest.main()
