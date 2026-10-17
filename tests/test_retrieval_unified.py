"""Tests for soft latest-wins, signal-conditioned render, and need weighting."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from evomemstate.io import normalize_session
from evomemstate.models import Memory, OntologyState
from evomemstate.ontology import OntologyStore
from evomemstate.prompting import (
    has_world_evidence,
    render_context,
    select_render_mode,
)
from evomemstate.retrieval import (
    RelationalState,
    ScoredMemory,
    _global_reserve_slots,
    _question_has_anaphora,
    retrieve,
)


def _session(text: str, session_id: str = "s1"):
    return normalize_session(
        {
            "user_id": "u1",
            "session_id": session_id,
            "messages": [{"role": "user", "content": text}],
        },
        index=1,
    )


def _store_with_supersede() -> OntologyStore:
    store = OntologyStore(OntologyState.empty("u1"))
    session = _session(
        "I used to like mainstream films, but now I prefer foreign documentaries."
    )
    store.apply_delta(
        {
            "concept_edits": [{
                "op": "CREATE",
                "temp_id": "c_film",
                "name": "Film taste",
                "description": "Movie preferences",
                "parent_ref": "concept_root",
            }],
            "memories": [
                {
                    "temp_id": "m_old",
                    "content": "The user used to like mainstream films.",
                    "type": "preference",
                    "scope": {"level": "domain", "value": "film"},
                    "source": {
                        "turn_ids": ["turn_0001"],
                        "evidence_quote": "used to like mainstream films",
                    },
                    "parent_ref": "c_film",
                    "confidence": 0.9,
                },
                {
                    "temp_id": "m_new",
                    "content": "The user now prefers foreign documentaries.",
                    "type": "preference",
                    "scope": {"level": "domain", "value": "film"},
                    "source": {
                        "turn_ids": ["turn_0001"],
                        "evidence_quote": "now I prefer foreign documentaries",
                    },
                    "parent_ref": "c_film",
                    "confidence": 0.95,
                },
                {
                    "temp_id": "m_fact",
                    "content": "The user is choosing a film for a weekend screening.",
                    "type": "fact",
                    "scope": {"level": "situation", "value": "weekend film choice"},
                    "source": {
                        "turn_ids": ["turn_0001"],
                        "evidence_quote": "prefer foreign documentaries",
                    },
                    "parent_ref": "concept_root",
                    "confidence": 0.8,
                },
            ],
            "relations": [{
                "type": "SUPERSEDES",
                "source_ref": "m_new",
                "target_ref": "m_old",
                "reason": "Preference updated from mainstream to foreign documentaries.",
            }],
        },
        session,
    )
    return store


def _pref_memory() -> Memory:
    return Memory(
        memory_id="m_pref",
        content="The user prefers quiet cafes.",
        memory_type="preference",
        scope={"level": "domain", "value": "dining"},
        time={},
        source={},
        status="active",
        parent_concept_id="concept_root",
        confidence=0.9,
    )


def _user_fact_memory() -> Memory:
    return Memory(
        memory_id="m_user_fact",
        content="The user has a date this weekend.",
        memory_type="experience",
        scope={"level": "situation", "value": "weekend date"},
        time={},
        source={},
        status="active",
        parent_concept_id="concept_root",
        confidence=0.9,
    )


def _world_evidence_memory() -> Memory:
    return Memory(
        memory_id="m_evidence",
        content="A report says noisy cafes score higher on throughput.",
        memory_type="fact",
        scope={"level": "domain", "value": "dining"},
        time={},
        source={},
        status="active",
        parent_concept_id="concept_root",
        confidence=0.9,
    )


class UnifiedRetrievalTest(unittest.TestCase):
    def test_global_reserve_slots(self):
        self.assertEqual(0, _global_reserve_slots(1))
        self.assertEqual(2, _global_reserve_slots(5))
        self.assertEqual(5, _global_reserve_slots(10))
        self.assertEqual(10, _global_reserve_slots(20))

    def test_anaphora_cue_detection(self):
        self.assertTrue(
            _question_has_anaphora(
                "In the situation we discussed, what should they do?"
            )
        )
        self.assertFalse(_question_has_anaphora("Which movie should I watch?"))

    def test_similarity_hit_is_replaced_by_relation_complete_memory(self):
        store = _store_with_supersede()
        result = retrieve(
            store,
            "What film should I watch this weekend?",
            fact_top_n=1,
            preference_top_n=1,
            latest_wins=True,
        )
        self.assertEqual("emb_seed", result.candidate_pool_mode)
        self.assertEqual("off", result.latest_wins_mode)
        self.assertFalse(result.demoted_superseded)
        self.assertFalse(result.dropped_superseded)
        bundled = [
            item.memory
            for item in result.all_memories()
            if (item.memory.source or {}).get("bundle")
        ]
        self.assertEqual(1, len(bundled))
        text = bundled[0].content
        self.assertIn("mainstream", text)
        self.assertIn("foreign documentaries", text)
        self.assertIn("scope:", text)
        self.assertIn("superseded", text)
        self.assertIn("SUPERSEDES", text)
        self.assertNotIn("Evidence unit:", text)
        member_ids = set(bundled[0].source["member_ids"])
        old_id = next(
            mid
            for mid, mem in store.state.memories.items()
            if "mainstream" in mem.content
        )
        new_id = next(
            mid
            for mid, mem in store.state.memories.items()
            if "foreign documentaries" in mem.content
        )
        self.assertEqual({old_id, new_id}, member_ids)
        # The two endpoints are one memory, not two scored rows.
        self.assertEqual(1, sum("mainstream" in item.memory.content for item in result.all_memories()))

    def test_similarity_topk_keeps_the_unselected_relation_endpoint(self):
        """A seed outside the other endpoint's rank still carries that endpoint."""
        store = _store_with_supersede()
        result = retrieve(
            store,
            "foreign documentaries",
            fact_top_n=1,
            preference_top_n=0,
        )
        self.assertEqual(1, result.n_embedding_seeds)
        self.assertEqual(1, len(result.all_memories()))
        text = result.all_memories()[0].memory.content
        self.assertIn("foreign documentaries", text)
        self.assertIn("mainstream", text)
        self.assertIn("SUPERSEDES", text)

    def test_hard_latest_wins_does_not_drop_superseded_endpoint(self):
        store = _store_with_supersede()
        with mock.patch.dict(os.environ, {"EVOMEMSTATE_LATEST_WINS_HARD": "1"}, clear=False):
            result = retrieve(
                store,
                "What film should I watch this weekend?",
                fact_top_n=2,
                preference_top_n=2,
                latest_wins=True,
            )
        self.assertEqual("off", result.latest_wins_mode)
        self.assertFalse(result.dropped_superseded)
        bundled = next(
            item.memory
            for item in result.all_memories()
            if (item.memory.source or {}).get("bundle")
        )
        self.assertIn("mainstream", bundled.content)
        self.assertIn("foreign documentaries", bundled.content)

    def test_unrelated_memory_stays_a_single_memory(self):
        store = _store_with_supersede()
        result = retrieve(
            store,
            "What film should I watch this weekend?",
            fact_top_n=5,
            preference_top_n=5,
            latest_wins=True,
        )
        fact = next(
            item.memory
            for item in result.all_memories()
            if "weekend screening" in item.memory.content
        )
        self.assertNotIn("Relation:", fact.content)
        self.assertFalse((fact.source or {}).get("bundle"))

    def test_conflict_endpoints_share_one_memory(self):
        store = OntologyStore(OntologyState.empty("u1"))
        session = _session(
            "The user is allergic to milk. The user is not allergic to milk."
        )
        store.apply_delta(
            {
                "concept_edits": [{
                    "op": "CREATE",
                    "temp_id": "c_food",
                    "name": "Food",
                    "description": "Diet facts",
                    "parent_ref": "concept_root",
                }],
                "memories": [
                    {
                        "temp_id": "m_yes",
                        "content": "The user is allergic to milk.",
                        "type": "fact",
                        "scope": {"level": "global", "value": "diet"},
                        "source": {
                            "turn_ids": ["turn_0001"],
                            "evidence_quote": "allergic to milk",
                        },
                        "parent_ref": "c_food",
                        "confidence": 0.8,
                    },
                    {
                        "temp_id": "m_no",
                        "content": "The user is not allergic to milk.",
                        "type": "fact",
                        "scope": {"level": "global", "value": "diet"},
                        "source": {
                            "turn_ids": ["turn_0001"],
                            "evidence_quote": "not allergic to milk",
                        },
                        "parent_ref": "c_food",
                        "confidence": 0.8,
                    },
                ],
                "relations": [{
                    "type": "CONFLICTS_WITH",
                    "source_ref": "m_yes",
                    "target_ref": "m_no",
                    "reason": "The two statements contradict each other.",
                }],
            },
            session,
        )
        result = retrieve(
            store,
            "Is the user allergic to milk?",
            fact_top_n=1,
            preference_top_n=0,
        )
        bundled = [
            item.memory
            for item in result.all_memories()
            if (item.memory.source or {}).get("bundle")
        ]
        self.assertEqual(1, len(bundled))
        self.assertIn("allergic to milk", bundled[0].content)
        self.assertIn("not allergic to milk", bundled[0].content)
        self.assertIn("CONFLICTS_WITH", bundled[0].content)
        self.assertEqual(1, len(result.all_memories()))

    def test_similarity_selection_respects_total_k(self):
        store = _store_with_supersede()
        result = retrieve(
            store,
            "Recommend a movie for me",
            fact_top_n=3,
            preference_top_n=3,
            latest_wins=True,
        )
        self.assertLessEqual(len(result.all_memories()), 6)
        self.assertEqual("off", result.latest_wins_mode)

    def test_render_includes_scope_and_superseded_label(self):
        store = _store_with_supersede()
        result = retrieve(
            store,
            "What should I watch?",
            fact_top_n=5,
            preference_top_n=5,
            latest_wins=True,
        )
        old = next(m for m in store.state.memories.values() if "mainstream" in m.content)
        result.preferences.append(
            ScoredMemory(memory=old, score=0.01, channel="preference")
        )
        result.demoted_superseded.append(
            {
                "memory_id": old.memory_id,
                "content": old.content,
                "superseded_by": "memory_x",
                "superseded_by_content": "The user now prefers foreign documentaries.",
            }
        )
        text = render_context(result, flat=False)
        self.assertIn("scope:", text)
        self.assertIn("superseded by:", text)
        self.assertIn("historical preferences", text)

        flat_text = render_context(result, flat=True)
        self.assertIn("scope:", flat_text)

    def test_preference_mode_single_list_without_evidence_priority(self):
        result = RelationalState(
            facts=[
                ScoredMemory(memory=_user_fact_memory(), score=0.4, channel="fact")
            ],
            preferences=[
                ScoredMemory(memory=_pref_memory(), score=0.9, channel="preference")
            ],
            fact_need=0.2,
            preference_need=0.9,
            latest_wins_applied=False,
        )
        self.assertFalse(has_world_evidence(result))
        self.assertEqual("preference", select_render_mode(result))
        text = render_context(result, flat=False, include_intents=False)
        self.assertIn("## Relational State", text)
        self.assertNotIn("## Established evidence", text)
        self.assertNotIn("## User preferences", text)
        self.assertIn("one concrete recommendation", text)
        self.assertNotIn("letting it change the recommendation is not", text)
        self.assertNotIn("What the question is asking", text)
        # Distinctive evidence-priority clause must stay off.
        self.assertNotIn(
            "even if the user has consistently favoured the opposite", text
        )

    def test_evidence_mode_sections_and_priority_guidance(self):
        result = RelationalState(
            facts=[
                ScoredMemory(
                    memory=_world_evidence_memory(), score=0.8, channel="fact"
                )
            ],
            preferences=[
                ScoredMemory(memory=_pref_memory(), score=0.9, channel="preference")
            ],
            fact_need=0.7,
            preference_need=0.6,
            latest_wins_applied=False,
        )
        self.assertTrue(has_world_evidence(result))
        self.assertEqual("evidence", select_render_mode(result))
        text = render_context(result, flat=False, include_intents=False)
        self.assertIn("## Established evidence", text)
        self.assertIn("## User preferences", text)
        self.assertIn("letting it change the recommendation is not", text)
        self.assertLess(
            text.find("Established evidence"), text.find("User preferences")
        )
        self.assertIn(
            "even if the user has consistently favoured the opposite", text
        )
        # Preference-align block should not appear in evidence mode.
        self.assertNotIn("Prefer a single best option over a long menu", text)

    def test_flat_flag_uses_same_signal_render(self):
        """flat=True must not force evidence-priority when signals say preference."""
        result = RelationalState(
            facts=[
                ScoredMemory(memory=_pref_memory(), score=0.9, channel="memory"),
                ScoredMemory(memory=_user_fact_memory(), score=0.5, channel="memory"),
            ],
            preferences=[],
            fact_need=0.2,
            preference_need=0.9,
            latest_wins_applied=False,
        )
        self.assertEqual("preference", select_render_mode(result))
        text = render_context(result, flat=True)
        self.assertIn("## Relational State", text)
        self.assertIn("one concrete recommendation", text)
        self.assertNotIn("letting it change the recommendation is not", text)

    def test_return_all_scored_skips_topk_cut_but_keeps_same_pool_features(self):
        store = _store_with_supersede()
        # Non-None client enables intent-decomp path (concept localize + backfill).
        client = object()
        kwargs = dict(
            client=client,
            fact_top_n=1,
            preference_top_n=1,
            latest_wins=True,
            concept_top_k=8,
        )
        with mock.patch(
            "evomemstate.retrieval.decompose_intent",
            return_value=(["film"], ["weekend"], False, 0.5, 0.5),
        ), mock.patch.dict(
            os.environ, {"EVOMEMSTATE_DISABLE_INTENT_DECOMP": "0"}, clear=False
        ):
            os.environ.pop("EVOMEMSTATE_DISABLE_INTENT_DECOMP", None)
            cut = retrieve(
                store,
                "What film should I watch this weekend?",
                return_all_scored=False,
                **kwargs,
            )
            pool = retrieve(
                store,
                "What film should I watch this weekend?",
                return_all_scored=True,
                **kwargs,
            )
        self.assertLessEqual(len(cut.all_memories()), 2)
        self.assertGreaterEqual(len(pool.all_memories()), len(cut.all_memories()))
        self.assertEqual(cut.selected_concept_ids, pool.selected_concept_ids)
        self.assertEqual(cut.latest_wins_mode, pool.latest_wins_mode)
        cut_ids = {item.memory.memory_id for item in cut.all_memories()}
        pool_ids = {item.memory.memory_id for item in pool.all_memories()}
        self.assertTrue(cut_ids.issubset(pool_ids))
        self.assertEqual(pool.candidate_pool_mode, "need_scored")
        self.assertTrue(pool.intent_decomposition_used)

    def test_struct_pool_flag_skips_need_rescore_and_caps_semantic(self):
        """STRUCT: concept ∪ semantic top-K; no need re-score / no latest-wins."""
        store = _store_with_many_off_concept()
        question = "What film should I watch this weekend?"
        total_k = 2
        client = object()
        kwargs = dict(
            client=client,
            fact_top_n=1,
            preference_top_n=1,
            latest_wins=True,
            concept_top_k=2,
            return_all_scored=True,
        )
        with mock.patch(
            "evomemstate.retrieval.decompose_intent",
            return_value=(["film"], ["weekend"], False, 0.5, 0.5),
        ), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EVOMEMSTATE_STRUCT_CANDIDATE_POOL", None)
            os.environ.pop("EVOMEMSTATE_DISABLE_INTENT_DECOMP", None)
            legacy = retrieve(store, question, **kwargs)

        with mock.patch(
            "evomemstate.retrieval.decompose_intent",
            return_value=(["film"], ["weekend"], False, 0.5, 0.5),
        ), mock.patch.dict(
            os.environ,
            {"EVOMEMSTATE_STRUCT_CANDIDATE_POOL": "1"},
            clear=False,
        ):
            struct = retrieve(store, question, **kwargs)

        self.assertEqual(legacy.candidate_pool_mode, "need_scored")
        self.assertEqual(struct.candidate_pool_mode, "struct")
        self.assertEqual(struct.latest_wins_mode, "off")
        self.assertFalse(struct.latest_wins_applied)
        self.assertFalse(struct.demoted_superseded)
        self.assertFalse(struct.dropped_superseded)
        self.assertLessEqual(struct.n_semantic_backfill, total_k)
        # Struct pool size ≈ concept + semantic (no SUPERSEDES head inject).
        n_struct = len(struct.all_memories())
        self.assertEqual(
            n_struct,
            struct.n_concept_memories + struct.n_semantic_backfill,
        )
        # Legacy backfill budget is larger (max(total_k, concept_top_k)).
        self.assertGreaterEqual(
            legacy.n_semantic_backfill, struct.n_semantic_backfill
        )

        # Need re-score must not change relative order vs pure similarity on
        # the same eligible set: mock extreme needs and ensure struct ranks
        # ignore them (fact_need/preference_need still recorded for rerank).
        with mock.patch(
            "evomemstate.retrieval.decompose_intent",
            return_value=(["film"], ["weekend"], False, 0.05, 0.99),
        ), mock.patch.dict(
            os.environ,
            {"EVOMEMSTATE_STRUCT_CANDIDATE_POOL": "1"},
            clear=False,
        ):
            high_pref = retrieve(store, question, **kwargs)
        with mock.patch(
            "evomemstate.retrieval.decompose_intent",
            return_value=(["film"], ["weekend"], False, 0.99, 0.05),
        ), mock.patch.dict(
            os.environ,
            {"EVOMEMSTATE_STRUCT_CANDIDATE_POOL": "1"},
            clear=False,
        ):
            high_fact = retrieve(store, question, **kwargs)
        self.assertEqual(
            [m.memory.memory_id for m in high_pref.all_memories()],
            [m.memory.memory_id for m in high_fact.all_memories()],
        )
        self.assertAlmostEqual(high_pref.preference_need, 0.99)
        self.assertAlmostEqual(high_fact.fact_need, 0.99)

    def test_struct_pool_ignores_hard_latest_wins(self):
        store = _store_with_supersede()
        with mock.patch.dict(
            os.environ,
            {
                "EVOMEMSTATE_STRUCT_CANDIDATE_POOL": "1",
                "EVOMEMSTATE_LATEST_WINS_HARD": "1",
            },
            clear=False,
        ):
            result = retrieve(
                store,
                "What film should I watch this weekend?",
                fact_top_n=5,
                preference_top_n=5,
                latest_wins=True,
                return_all_scored=True,
            )
        self.assertEqual(result.candidate_pool_mode, "struct")
        self.assertEqual(result.latest_wins_mode, "off")
        old_id = next(
            mid
            for mid, mem in store.state.memories.items()
            if "mainstream" in mem.content
        )
        retrieved_ids = {item.memory.memory_id for item in result.all_memories()}
        self.assertIn(old_id, retrieved_ids)


def _store_with_many_off_concept() -> OntologyStore:
    """Film concept + many unrelated root memories (forces semantic backfill)."""
    store = OntologyStore(OntologyState.empty("u1"))
    session = _session("Film taste and many other topics.")
    memories = [
        {
            "temp_id": "m_old",
            "content": "The user used to like mainstream films.",
            "type": "preference",
            "scope": {"level": "domain", "value": "film"},
            "source": {
                "turn_ids": ["turn_0001"],
                "evidence_quote": "used to like mainstream films",
            },
            "parent_ref": "c_film",
            "confidence": 0.9,
        },
        {
            "temp_id": "m_new",
            "content": "The user now prefers foreign documentaries.",
            "type": "preference",
            "scope": {"level": "domain", "value": "film"},
            "source": {
                "turn_ids": ["turn_0001"],
                "evidence_quote": "now I prefer foreign documentaries",
            },
            "parent_ref": "c_film",
            "confidence": 0.95,
        },
    ]
    for i in range(12):
        memories.append(
            {
                "temp_id": f"m_noise_{i}",
                "content": f"Unrelated note about cooking recipe number {i}.",
                "type": "fact",
                "scope": {"level": "domain", "value": "cooking"},
                "source": {
                    "turn_ids": ["turn_0001"],
                    "evidence_quote": f"recipe {i}",
                },
                "parent_ref": "concept_root",
                "confidence": 0.5,
            }
        )
    store.apply_delta(
        {
            "concept_edits": [{
                "op": "CREATE",
                "temp_id": "c_film",
                "name": "Film taste",
                "description": "Movie preferences",
                "parent_ref": "concept_root",
            }],
            "memories": memories,
            "relations": [{
                "type": "SUPERSEDES",
                "source_ref": "m_new",
                "target_ref": "m_old",
                "reason": "Preference updated.",
            }],
        },
        session,
    )
    return store


if __name__ == "__main__":
    unittest.main()
