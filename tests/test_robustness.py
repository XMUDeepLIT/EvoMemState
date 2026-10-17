import asyncio
import unittest

from evomemstate.builder import AsyncMemoryBuilder
from evomemstate.extractor import AsyncOntologyGuidedExtractor
from evomemstate.io import normalize_session
from evomemstate.llm import _extract_json
from evomemstate.models import OntologyState
from evomemstate.ontology import OntologyStore
from evomemstate.retrieval import (
    allocate_channel_budget,
    decompose_intent,
    estimate_channel_needs,
    _parse_channel_needs,
)


class RobustnessTest(unittest.TestCase):
    def test_extract_json_selects_schema_object_from_extra_text(self):
        parsed = _extract_json(
            'analysis {"status":"draft"}\n'
            '```json\n{"concept_edits":[],"memories":[],"relations":[]}\n``` trailing'
        )
        self.assertEqual([], parsed["concept_edits"])
        self.assertEqual([], parsed["memories"])
        self.assertEqual([], parsed["relations"])

    def test_invalid_turn_id_is_recovered_from_evidence(self):
        session = normalize_session(
            {
                "user_id": "u1",
                "session_id": "s1",
                "messages": [{
                    "role": "user",
                    "content": "I prefer quiet hotels near public transportation.",
                }],
            },
            index=1,
        )
        store = OntologyStore(OntologyState.empty("u1"))
        result = store.apply_delta(
            {
                "concept_edits": [],
                "memories": [{
                    "temp_id": "m1",
                    "content": "The user prefers quiet hotels near public transportation.",
                    "type": "preference",
                    "scope": {"level": "domain", "value": "travel accommodation"},
                    "source": {
                        "turn_ids": ["turn_9999"],
                        "evidence_quote": "prefer quiet hotels near public transportation",
                    },
                    "parent_ref": "concept_root",
                    "confidence": 0.95,
                }],
                "relations": [],
            },
            session,
        )
        memory = store.state.memories[result.created_memory_ids[0]]
        self.assertEqual(["turn_0001"], memory.source["turn_ids"])

    def test_new_memory_can_supersede_another_memory_from_same_session(self):
        session = normalize_session(
            {
                "user_id": "u1",
                "session_id": "s1",
                "messages": [{
                    "role": "user",
                    "content": "I first wanted strict adaptations, but now I prefer creative adaptations.",
                }],
            },
            index=1,
        )
        store = OntologyStore(OntologyState.empty("u1"))
        result = store.apply_delta(
            {
                "concept_edits": [],
                "memories": [
                    {
                        "temp_id": "m_old",
                        "content": "The user preferred strict adaptations.",
                        "type": "preference",
                        "scope": {"level": "domain", "value": "film adaptations"},
                        "source": {
                            "turn_ids": ["turn_0001"],
                            "evidence_quote": "first wanted strict adaptations",
                        },
                        "parent_ref": "concept_root",
                        "confidence": 0.9,
                    },
                    {
                        "temp_id": "m_new",
                        "content": "The user now prefers creative adaptations.",
                        "type": "preference",
                        "scope": {"level": "domain", "value": "film adaptations"},
                        "source": {
                            "turn_ids": ["turn_0001"],
                            "evidence_quote": "now I prefer creative adaptations",
                        },
                        "parent_ref": "concept_root",
                        "confidence": 0.95,
                    },
                ],
                "relations": [{
                    "type": "SUPERSEDES",
                    "source_ref": "m_new",
                    "target_ref": "m_old",
                    "reason": "The preference changed within the current conversation.",
                }],
            },
            session,
        )
        old_id, new_id = result.created_memory_ids
        self.assertEqual("superseded", store.state.memories[old_id].status)
        self.assertEqual("active", store.state.memories[new_id].status)
        self.assertTrue(any(
            edge.edge_type == "SUPERSEDES"
            and edge.source_id == new_id
            and edge.target_id == old_id
            for edge in store.state.edges.values()
        ))

    def test_parent_outside_retrieval_is_remapped_to_a_candidate(self):
        store = OntologyStore(OntologyState.empty("u1"))
        setup = normalize_session(
            {
                "user_id": "u1",
                "session_id": "setup",
                "messages": [{"role": "user", "content": "I discuss travel and work."}],
            },
            index=1,
        )
        setup_result = store.apply_delta(
            {
                "concept_edits": [
                    {
                        "op": "CREATE",
                        "temp_id": "travel",
                        "name": "Travel Accommodation",
                        "description": "Hotel and lodging preferences for travel.",
                        "parent_ref": "concept_root",
                    },
                    {
                        "op": "CREATE",
                        "temp_id": "work",
                        "name": "Work Tools",
                        "description": "Tools used for work.",
                        "parent_ref": "concept_root",
                    },
                ],
                "memories": [],
                "relations": [],
            },
            setup,
        )
        travel_id, work_id = setup_result.created_concept_ids
        session = normalize_session(
            {
                "user_id": "u1",
                "session_id": "s2",
                "messages": [{
                    "role": "user",
                    "content": "I prefer quiet hotels when traveling.",
                }],
            },
            index=2,
        )
        result = store.apply_delta(
            {
                "concept_edits": [],
                "memories": [{
                    "temp_id": "m1",
                    "content": "The user prefers quiet hotels when traveling.",
                    "type": "preference",
                    "scope": {"level": "domain", "value": "travel accommodation"},
                    "source": {
                        "turn_ids": ["turn_0001"],
                        "evidence_quote": "prefer quiet hotels when traveling",
                    },
                    "parent_ref": work_id,
                    "confidence": 0.95,
                }],
                "relations": [],
                "_retrieval": {
                    "concept_ids": ["concept_root", travel_id],
                    "memory_ids": [],
                },
            },
            session,
        )
        memory = store.state.memories[result.created_memory_ids[0]]
        self.assertEqual(travel_id, memory.parent_concept_id)
        self.assertTrue(any("Remapped memory" in warning for warning in result.warnings))

    def test_independent_users_await_llm_concurrently(self):
        class FakeAsyncClient:
            def __init__(self):
                self.active = 0
                self.max_active = 0

            async def complete_json(self, messages):
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                await asyncio.sleep(0.01)
                self.active -= 1
                return {"concept_edits": [], "memories": [], "relations": []}

        async def run():
            client = FakeAsyncClient()

            async def build(user_id):
                store = OntologyStore(OntologyState.empty(user_id))
                builder = AsyncMemoryBuilder(store, AsyncOntologyGuidedExtractor(client))
                session = normalize_session(
                    {
                        "user_id": user_id,
                        "session_id": "s1",
                        "messages": [{"role": "user", "content": "I prefer concise answers."}],
                    },
                    index=1,
                )
                await builder.build_session(session)
                return store

            stores = await asyncio.gather(build("u1"), build("u2"))
            self.assertGreaterEqual(client.max_active, 2)
            self.assertTrue(all("s1" in store.state.processed_sessions for store in stores))

        asyncio.run(run())

    def test_allocate_channel_budget_follows_needs_with_floor(self):
        fact_k, pref_k = allocate_channel_budget(10, 0.9, 0.1)
        self.assertEqual(10, fact_k + pref_k)
        self.assertGreaterEqual(fact_k, 2)
        self.assertGreaterEqual(pref_k, 2)
        self.assertGreater(fact_k, pref_k)

        fact_k, pref_k = allocate_channel_budget(10, 0.1, 0.9)
        self.assertEqual(10, fact_k + pref_k)
        self.assertGreater(pref_k, fact_k)

        fact_k, pref_k = allocate_channel_budget(10, 0.5, 0.5)
        self.assertEqual((5, 5), (fact_k, pref_k))

    def test_parse_channel_needs_accepts_floats_and_legacy_bools(self):
        self.assertEqual((0.7, 0.3), _parse_channel_needs({"fact_need": 0.7, "preference_need": 0.3}))
        self.assertEqual(
            (0.8, 0.2),
            _parse_channel_needs({"needs_facts": True, "needs_preferences": False}),
        )

    def test_invalid_memory_fields_are_skipped_with_warnings(self):
        session = normalize_session(
            {
                "user_id": "u1",
                "session_id": "s1",
                "messages": [{
                    "role": "user",
                    "content": "I prefer quiet hotels and I used to like loud parties.",
                }],
            },
            index=1,
        )
        store = OntologyStore(OntologyState.empty("u1"))
        result = store.apply_delta(
            {
                "concept_edits": [],
                "memories": [
                    {
                        "temp_id": "m_aliased",
                        "content": "The user prefers quiet hotels.",
                        "type": "negative_preference",
                        "source": {
                            "turn_ids": ["turn_0001"],
                            "evidence_quote": "prefer quiet hotels",
                        },
                        "parent_ref": "concept_root",
                        "confidence": 0.9,
                    },
                    {
                        "temp_id": "m_bad_type",
                        "content": "The user likes loud parties.",
                        "type": "vibes",
                        "source": {
                            "turn_ids": ["turn_0001"],
                            "evidence_quote": "like loud parties",
                        },
                        "parent_ref": "concept_root",
                        "confidence": 0.9,
                    },
                    {
                        "temp_id": "m_empty",
                        "content": "   ",
                        "type": "fact",
                        "source": {
                            "turn_ids": ["turn_0001"],
                            "evidence_quote": "prefer quiet hotels",
                        },
                        "parent_ref": "concept_root",
                        "confidence": 0.9,
                    },
                ],
                "relations": [],
            },
            session,
        )
        self.assertEqual(1, len(result.created_memory_ids))
        memory = store.state.memories[result.created_memory_ids[0]]
        self.assertEqual("preference", memory.memory_type)
        self.assertTrue(any("type" in warning for warning in result.warnings))
        self.assertTrue(any("empty" in warning for warning in result.warnings))
        self.assertIn("s1", store.state.processed_sessions)

    def test_decompose_intent_without_client_uses_lexical_prior(self):
        intents, terms, failed, fact_need, preference_need = decompose_intent(
            "When did Caroline go to the support group?"
        )
        self.assertTrue(failed)
        self.assertEqual(["When did Caroline go to the support group?"], intents)
        self.assertEqual([], terms)
        self.assertGreater(fact_need, preference_need)

        _, _, _, fact_need, preference_need = decompose_intent(
            "What laptop should I buy given what I prefer?"
        )
        self.assertGreater(preference_need, fact_need)

    def test_decompose_intent_reads_fact_need_from_client(self):
        class FakeClient:
            def complete_json(self, messages):
                return {
                    "intents": ["find objective evidence"],
                    "topic_terms": ["atlas"],
                    "fact_need": 0.85,
                    "preference_need": 0.15,
                }

        intents, terms, failed, fact_need, preference_need = decompose_intent(
            "Which model is better according to the report?",
            client=FakeClient(),
        )
        self.assertFalse(failed)
        self.assertEqual(["find objective evidence"], intents)
        self.assertEqual(["atlas"], terms)
        self.assertEqual(0.85, fact_need)
        self.assertEqual(0.15, preference_need)


if __name__ == "__main__":
    unittest.main()
