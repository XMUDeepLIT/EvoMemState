"""Tests for Phase 1 static harness (hook order, arm flags, behavior parity)."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from evomemstate.harness import HarnessContext, load_profile, run_context_pipeline
from evomemstate.harness.profile import ArmFlags, HarnessProfile
from evomemstate.harness.rules import RuleRegistry
from evomemstate.io import normalize_session
from evomemstate.models import OntologyState
from evomemstate.ontology import OntologyStore
from evomemstate.prompting import render_context
from evomemstate.retrieval import retrieve


def _empty_store() -> OntologyStore:
    store = OntologyStore(OntologyState.empty("u1"))
    session = normalize_session(
        {
            "user_id": "u1",
            "session_id": "s1",
            "messages": [{"role": "user", "content": "I prefer quiet cafes."}],
        },
        index=1,
    )
    store.apply_delta(
        {
            "concept_edits": [],
            "memories": [
                {
                    "temp_id": "m1",
                    "content": "The user prefers quiet cafes.",
                    "type": "preference",
                    "scope": {"level": "domain", "value": "dining"},
                    "source": {"turn_ids": ["turn_0001"], "evidence_quote": "quiet cafes"},
                    "parent_ref": "concept_root",
                    "confidence": 0.9,
                }
            ],
            "relations": [],
        },
        session,
    )
    return store


class _EvalConfig:
    method = "EvoMemState"
    top_k = 4


class HarnessProfileTest(unittest.TestCase):
    def test_default_profile_loads_four_hooks(self):
        profile = load_profile("default")
        self.assertEqual("1.0.0", profile.version)
        self.assertIn("record_graph_stats", profile.hook_rules("on_after_build"))
        self.assertIn("resolve_arm_flags", profile.hook_rules("on_before_retrieve"))
        self.assertIn("compute_generation_signals", profile.hook_rules("on_after_retrieve"))
        self.assertIn("signal_conditioned_render", profile.hook_rules("on_before_generation"))



class HarnessPipelineTest(unittest.TestCase):
    def test_hook_order_and_audit_trail(self):
        store = _empty_store()
        ctx = HarnessContext(
            prior_dialogue="",
            user_question="Where should we eat?",
            eval_config=_EvalConfig(),
            method="EvoMemState",
            store=store,
            build_settings={"concept_top_k": 8},
        )
        run_context_pipeline(ctx)

        self.assertTrue(ctx.applied_rules[0].startswith("on_after_build:"))
        self.assertTrue(any(r.startswith("on_before_retrieve:") for r in ctx.applied_rules))
        self.assertTrue(any(r.startswith("on_after_retrieve:") for r in ctx.applied_rules))
        last = ctx.applied_rules[-1]
        self.assertTrue(last.startswith("on_before_generation:"))

        self.assertIn(ctx.render_mode, {"preference", "balanced"})
        self.assertIn("quiet cafes", ctx.context_text)
        self.assertEqual(1, ctx.graph_stats["n_memories_total"])

    def test_decompose_follows_env_when_arm_says_env(self):
        profile = load_profile("default")
        ctx = HarnessContext(
            prior_dialogue="",
            user_question="test",
            eval_config=_EvalConfig(),
            method="EvoMemState",
            store=_empty_store(),
        )
        with mock.patch.dict(os.environ, {"EVOMEMSTATE_DECOMPOSE": "1"}, clear=False):
            RuleRegistry.get("resolve_arm_flags")(ctx, profile)
        self.assertTrue(ctx.decompose_enabled)


if __name__ == "__main__":
    unittest.main()
