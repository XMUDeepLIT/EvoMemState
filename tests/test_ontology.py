import json
import tempfile
import unittest
from pathlib import Path

from evomemstate.io import normalize_session
from evomemstate.models import OntologyState
from evomemstate.ontology import OntologyStore


class OntologyStoreTest(unittest.TestCase):
    def setUp(self):
        self.store = OntologyStore(OntologyState.empty("user-1"))

    @staticmethod
    def session(session_id, timestamp, text):
        return normalize_session(
            {
                "user_id": "user-1",
                "session_id": session_id,
                "timestamp": timestamp,
                "messages": [{"role": "user", "content": text}],
            },
            index=1,
        )

    def test_incremental_memory_supersedes_old_leaf(self):
        first = self.session("s1", "2026-07-01T00:00:00Z", "I prefer Windows laptops for work.")
        first_delta = {
            "concept_edits": [
                {
                    "op": "CREATE",
                    "temp_id": "c_device",
                    "name": "Work Device Preference",
                    "description": "Preferences that apply to work devices.",
                    "parent_ref": "concept_root",
                }
            ],
            "memories": [
                {
                    "temp_id": "m_windows",
                    "content": "The user prefers Windows laptops for work.",
                    "type": "preference",
                    "scope": {"level": "domain", "value": "work laptop selection"},
                    "source": {"turn_ids": ["turn_0001"], "evidence_quote": "prefer Windows laptops"},
                    "parent_ref": "c_device",
                    "confidence": 0.98,
                }
            ],
            "relations": [],
        }
        first_result = self.store.apply_delta(first_delta, first)
        old_id = first_result.created_memory_ids[0]
        concept_id = first_result.created_concept_ids[0]

        second = self.session("s2", "2026-07-20T00:00:00Z", "I now prefer macOS for my work laptop.")
        second_delta = {
            "concept_edits": [],
            "memories": [
                {
                    "temp_id": "m_macos",
                    "content": "The user prefers macOS for work laptops.",
                    "type": "preference",
                    "scope": {"level": "domain", "value": "work laptop selection"},
                    "source": {"turn_ids": ["turn_0001"], "evidence_quote": "now prefer macOS"},
                    "parent_ref": concept_id,
                    "confidence": 0.99,
                }
            ],
            "relations": [
                {
                    "type": "SUPERSEDES",
                    "source_ref": "m_macos",
                    "target_ref": old_id,
                    "reason": "The user explicitly changed the preference.",
                }
            ],
        }
        second_result = self.store.apply_delta(second_delta, second)
        new_id = second_result.created_memory_ids[0]

        self.assertEqual("superseded", self.store.state.memories[old_id].status)
        self.assertEqual("active", self.store.state.memories[new_id].status)
        self.assertEqual(concept_id, self.store.state.memories[new_id].parent_concept_id)
        self.assertTrue(
            any(
                edge.edge_type == "SUPERSEDES"
                and edge.source_id == new_id
                and edge.target_id == old_id
                for edge in self.store.state.edges.values()
            )
        )

    def test_conflict_marks_both_active_memories_disputed(self):
        first = self.session("s1", "2026-07-01T00:00:00Z", "I believe remote work improves focus.")
        first_result = self.store.apply_delta(
            {
                "concept_edits": [],
                "memories": [{
                    "temp_id": "m_old",
                    "content": "The user believes remote work improves focus.",
                    "type": "belief",
                    "scope": {"level": "domain", "value": "work arrangement"},
                    "source": {"turn_ids": ["turn_0001"], "evidence_quote": "remote work improves focus"},
                    "parent_ref": "concept_root",
                    "confidence": 0.9,
                }],
                "relations": [],
            },
            first,
        )
        old_id = first_result.created_memory_ids[0]
        second = self.session("s2", "2026-07-02T00:00:00Z", "I also believe remote work can reduce my focus.")
        second_result = self.store.apply_delta(
            {
                "concept_edits": [],
                "memories": [{
                    "temp_id": "m_new",
                    "content": "The user believes remote work can reduce focus.",
                    "type": "belief",
                    "scope": {"level": "domain", "value": "work arrangement"},
                    "source": {"turn_ids": ["turn_0001"], "evidence_quote": "remote work can reduce my focus"},
                    "parent_ref": "concept_root",
                    "confidence": 0.8,
                }],
                "relations": [{
                    "type": "CONFLICTS_WITH",
                    "source_ref": "m_new",
                    "target_ref": old_id,
                    "reason": "Both beliefs are currently stated without a resolution.",
                }],
            },
            second,
        )
        new_id = second_result.created_memory_ids[0]
        self.assertEqual("disputed", self.store.state.memories[old_id].status)
        self.assertEqual("disputed", self.store.state.memories[new_id].status)

    def test_merge_rejects_ancestor_into_descendant(self):
        first = self.session("s1", "2026-07-01T00:00:00Z", "I discuss my work devices.")
        result = self.store.apply_delta(
            {
                "concept_edits": [
                    {"op": "CREATE", "temp_id": "parent", "name": "Devices", "parent_ref": "concept_root"},
                    {"op": "CREATE", "temp_id": "child", "name": "Work Devices", "parent_ref": "parent"},
                ],
                "memories": [],
                "relations": [],
            },
            first,
        )
        parent_id, child_id = result.created_concept_ids
        second = self.session("s2", "2026-07-02T00:00:00Z", "The categories are duplicates.")
        with self.assertRaises(ValueError):
            self.store.apply_delta(
                {
                    "concept_edits": [{
                        "op": "MERGE",
                        "source_refs": [parent_id],
                        "target_ref": child_id,
                    }],
                    "memories": [],
                    "relations": [],
                },
                second,
            )
        self.assertNotIn("s2", self.store.state.processed_sessions)
        self.assertEqual(parent_id, self.store.state.concepts[child_id].parent_id)

    def test_minor_quote_rewrite_is_grounded_back_to_exact_source(self):
        session = self.session(
            "s1",
            "2026-07-01T00:00:00Z",
            "I am trying to make better recommendations for myself instead of picking whatever is most popular.",
        )
        result = self.store.apply_delta(
            {
                "concept_edits": [],
                "memories": [{
                    "temp_id": "m1",
                    "content": "The user prefers personalized recommendations over popular choices.",
                    "type": "preference",
                    "scope": {"level": "domain", "value": "recommendation strategy"},
                    "source": {
                        "turn_ids": ["turn_0001"],
                        "evidence_quote": "making better recommendations for myself instead of picking whatever is most popular.",
                    },
                    "parent_ref": "concept_root",
                    "confidence": 0.95,
                }],
                "relations": [],
            },
            session,
        )
        memory = self.store.state.memories[result.created_memory_ids[0]]
        self.assertEqual(
            "make better recommendations for myself instead of picking whatever is most popular",
            memory.source["evidence_quote"],
        )

    def test_unrelated_quote_is_skipped_without_losing_the_session(self):
        session = self.session("s1", "2026-07-01T00:00:00Z", "I dislike long reports.")
        result = self.store.apply_delta(
            {
                "concept_edits": [],
                "memories": [{
                    "temp_id": "m1",
                    "content": "The user prefers macOS laptops.",
                    "type": "preference",
                    "scope": {"level": "domain", "value": "devices"},
                    "source": {
                        "turn_ids": ["turn_0001"],
                        "evidence_quote": "I strongly prefer macOS laptops for work",
                    },
                    "parent_ref": "concept_root",
                    "confidence": 0.9,
                }],
                "relations": [],
            },
            session,
        )
        self.assertEqual({}, self.store.state.memories)
        self.assertTrue(any("could not be grounded" in warning for warning in result.warnings))
        self.assertIn("s1", self.store.state.processed_sessions)

    def test_persistence_and_idempotent_session(self):
        session = self.session("s1", "2026-07-01T00:00:00Z", "I dislike long reports.")
        delta = {
            "concept_edits": [],
            "memories": [
                {
                    "temp_id": "m1",
                    "content": "The user dislikes long reports.",
                    "type": "preference",
                    "scope": {"level": "task", "value": "report writing"},
                    "source": {
                        "turn_ids": ["turn_0001"],
                        "evidence_quote": "dislike long reports"
                    },
                    "parent_ref": "concept_root",
                    "confidence": 0.9,
                }
            ],
            "relations": [],
        }
        self.store.apply_delta(delta, session)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            self.store.save(path)
            loaded = OntologyStore.load(path, "user-1")
            skipped = loaded.apply_delta(delta, session)
            self.assertTrue(skipped.skipped)
            self.assertEqual(1, len(loaded.state.memories))
            json.loads(path.read_text(encoding="utf-8"))


class RetrievedEvidenceGroundingTest(unittest.TestCase):
    """Evidence delivered inside <information> blocks must be storable as fact.

    MemSyco's memory_evidence_conflict puts 100% of its decisive evidence in
    such blocks on assistant turns. Grounding used to accept only user/system
    turns, so those memories were silently dropped and the evidence reached
    just 2% of graphs.
    """

    @staticmethod
    def session(messages):
        return normalize_session(
            {
                "user_id": "user-1",
                "session_id": "s1",
                "timestamp": "2026-07-01T00:00:00Z",
                "messages": messages,
            },
            index=1,
        )

    @staticmethod
    def delta(quote, turn_id="turn_0002"):
        return {
            "concept_edits": [],
            "memories": [{
                "temp_id": "m_evidence",
                "content": "Model Boreal preserved figures more reliably than Atlas.",
                "type": "fact",
                "scope": {"level": "domain", "value": "text summarization systems"},
                "source": {"turn_ids": [turn_id], "evidence_quote": quote},
                "parent_ref": "concept_root",
                "confidence": 0.9,
            }],
            "relations": [],
        }

    def test_evidence_inside_retrieved_block_is_grounded(self):
        session = self.session([
            {"role": "user", "content": "I am leaning towards Model Atlas."},
            {
                "role": "assistant",
                "content": (
                    "Here is what I found.\n"
                    "<information>An independent evaluation found that Boreal "
                    "preserved figures and named entities much more reliably "
                    "than Atlas on finance-heavy documents.</information>"
                ),
            },
        ])
        store = OntologyStore(OntologyState.empty("user-1"))
        result = store.apply_delta(
            self.delta("Boreal preserved figures and named entities much more reliably than Atlas"),
            session,
        )
        self.assertEqual(1, len(result.created_memory_ids))
        memory = store.state.memories[result.created_memory_ids[0]]
        self.assertEqual("fact", memory.memory_type)

    def test_grounding_does_not_depend_on_tag_vocabulary(self):
        """Detection is structural: an unseen tag name must work identically.

        The whole point of not keeping a whitelist is that the next corpus can
        invent its own tag and still get its evidence stored.
        """
        for wrapper in ("retrieved_docs", "corpus-snippet", "ctx-3"):
            with self.subTest(wrapper=wrapper):
                session = self.session([
                    {"role": "user", "content": "I am leaning towards Model Atlas."},
                    {
                        "role": "assistant",
                        "content": (
                            f"<{wrapper}>An independent evaluation found that Boreal "
                            f"preserved figures and named entities much more reliably "
                            f"than Atlas on finance-heavy documents.</{wrapper}>"
                        ),
                    },
                ])
                store = OntologyStore(OntologyState.empty("user-1"))
                result = store.apply_delta(
                    self.delta("Boreal preserved figures and named entities much more reliably than Atlas"),
                    session,
                )
                self.assertEqual(1, len(result.created_memory_ids))

    def test_blockquoted_evidence_is_grounded(self):
        session = self.session([
            {"role": "user", "content": "I am leaning towards Model Atlas."},
            {
                "role": "assistant",
                "content": (
                    "From the procurement summary:\n\n"
                    "> An independent evaluation found that Boreal preserved figures\n"
                    "> and named entities much more reliably than Atlas.\n"
                ),
            },
        ])
        store = OntologyStore(OntologyState.empty("user-1"))
        result = store.apply_delta(
            self.delta("Boreal preserved figures and named entities much more reliably than Atlas"),
            session,
        )
        self.assertEqual(1, len(result.created_memory_ids))

    def test_inline_markup_is_not_a_quoted_block(self):
        """<b>...</b> inside the assistant's sentence is still the assistant talking."""
        session = self.session([
            {"role": "user", "content": "I am leaning towards Model Atlas."},
            {
                "role": "assistant",
                "content": (
                    "Honestly, <b>Model Atlas is clearly the superior choice for "
                    "your finance reports</b>, and I would not overthink it."
                ),
            },
        ])
        store = OntologyStore(OntologyState.empty("user-1"))
        result = store.apply_delta(
            self.delta("Model Atlas is clearly the superior choice"),
            session,
        )
        self.assertEqual([], result.created_memory_ids)

    def test_assistant_prose_outside_block_stays_ungrounded(self):
        """The assistant agreeing with the user must never become a stored fact.

        This is the sycophancy the benchmark exists to measure; promoting it to
        a grounded memory would defeat the task.
        """
        session = self.session([
            {"role": "user", "content": "I am leaning towards Model Atlas."},
            {
                "role": "assistant",
                "content": "You are right, Model Atlas is clearly the superior choice here.",
            },
        ])
        store = OntologyStore(OntologyState.empty("user-1"))
        result = store.apply_delta(
            self.delta("Model Atlas is clearly the superior choice"),
            session,
        )
        self.assertEqual([], result.created_memory_ids)
        self.assertTrue(any("could not be grounded" in w for w in result.warnings))


if __name__ == "__main__":
    unittest.main()
