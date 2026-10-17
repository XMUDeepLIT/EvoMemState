from __future__ import annotations

import json
from typing import Any, Dict, Protocol

from .ontology import OntologyStore


class JSONClient(Protocol):
    def complete_json(self, messages: list[dict[str, str]]) -> Dict[str, Any]:
        ...


class AsyncJSONClient(Protocol):
    async def complete_json(self, messages: list[dict[str, str]]) -> Dict[str, Any]:
        ...


EXTRACTION_PROMPT = """You construct one user's self-evolving ontological memory.
The input contains a new dialogue and a small branch retrieved from the user's existing ontology.
In ONE response, extract atomic memories and propose the minimal ontology evolution required to organize them.

This memory stores two different things, and both matter:
- what is true of the USER (preferences, beliefs, constraints, experiences)
- what is true of the WORLD, as established by evidence surfaced in the dialogue (facts)
Keep the second kind independent of the first. Evidence does not become less true because
the user prefers otherwise, and it must survive in the memory on its own terms.

Fixed meta-ontology:
- memory type: preference | fact | belief | experience | constraint
- scope level: global | domain | task | situation
- memory status is managed by the system; every new memory starts active
- memory relation: SUPERSEDES | CONFLICTS_WITH
- concept edits are limited to CREATE | MERGE | SPLIT

Return exactly one JSON object with this shape:
{
  "concept_edits": [
    {
      "op": "CREATE",
      "temp_id": "c_new_1",
      "name": "concise concept name",
      "description": "what belongs under this concept",
      "aliases": [],
      "parent_ref": "an existing concept_id or an earlier temp_id"
    },
    {
      "op": "MERGE",
      "source_refs": ["existing concept_id"],
      "target_ref": "existing concept_id or temp_id",
      "aliases": []
    },
    {
      "op": "SPLIT",
      "source_ref": "existing concept_id",
      "children": [
        {"temp_id": "c_child_1", "name": "...", "description": "...", "aliases": []}
      ]
    }
  ],
  "memories": [
    {
      "temp_id": "m_new_1",
      "content": "one self-contained, time-aware atomic proposition",
      "type": "preference",
      "scope": {"level": "domain", "value": "work laptop selection"},
      "time": {"observed_at": "", "valid_from": "", "valid_to": ""},
      "source": {"turn_ids": ["turn_0001"], "evidence_quote": "exact short quote"},
      "parent_ref": "existing concept_id or temp_id",
      "confidence": 0.0
    }
  ],
  "relations": [
    {
      "type": "SUPERSEDES",
      "source_ref": "m_new_2",
      "target_ref": "an existing memory_id, or an earlier temp_id from this same response",
      "reason": "brief evidence-grounded reason"
    }
  ]
}

Rules:
1. Extract anything that could answer a later factual or temporal question about either speaker, not only durable preferences. This includes one-off events, dates, quantities, named entities, past actions ("went to", "signed up for", "started", "bought"), and biographical facts (how long, since when, how many years). The bar is "could someone be asked about this later?", not "is this a stable trait?". Skip only pure greetings, backchannel ("yeah", "haha", "nice"), and meta-chatter with no propositional content.
2. A leaf is atomic: do not combine independent propositions. Preserve qualifiers, negation, scope, and time.
3. Reuse the narrowest suitable existing concept. CREATE only when no supplied concept fits. Prefer concept reuse over synonyms.
4. CREATE a shallow, reusable concept, not a concept that paraphrases one memory. Do not create a new root or fixed domain taxonomy.
5. MERGE only clear duplicate concepts. SPLIT only a genuinely overloaded concept and provide reusable children.
6. Compare each new memory both with the supplied branch memories and with the other memories you emit in this same response. Use SUPERSEDES when a newer state replaces an older state of the same kind; use CONFLICTS_WITH for an unresolved contradiction. Do not relation-link mere topical similarity.
7. A SUPERSEDES pair must share the same memory type, but it need NOT share the same concept, scope, or wording. When a user moves from one object of preference to another, the two memories will naturally sit under different concepts -- emit the relation anyway. Judge replacement by whether the newer statement makes the older one no longer describe the user's current state.
8. When a newer preference replaces an older one, emit BOTH memories plus a SUPERSEDES from the newer to the older; never delete, merge, or rewrite the older memory. This applies even when both memories come from the same dialogue: extract the earlier preference as its own memory first, then the later one, then link them. Signals of replacement include "used to", "no longer", "these days", "switched to", "now I prefer", and any explicit statement that an earlier interest has been satisfied, completed, or abandoned.
9. Scope controls applicability. Use global only when the statement truly applies broadly.
10. Cite an exact evidence quote copied verbatim from a provided turn, and cite only that turn's turn_id. You may cite user turns, system turns, and material that a turn visibly quotes from elsewhere -- text set off from the speaker's own words by a delimiter such as a tagged block, a fenced block, or a blockquote. Never cite a speaker's own unquoted prose from an assistant turn: their opinions, agreement, and recommendations are not evidence. Confidence must be between 0 and 1.
11. Quoted material is EXTERNAL EVIDENCE about the world, not a statement about the user. Record it as type `fact`, with scope describing the subject matter rather than the user. Record it EVEN WHEN IT CONTRADICTS what the user prefers or believes, and record the side that contradicts them just as fully as the side that supports them. Keep the specific discriminating detail -- named entities, measured outcomes, comparative findings, and the conditions under which each holds. Never fold evidence into the user's viewpoint: "user still prefers X after reading the report" is a `preference` and does NOT preserve what the report actually said, so emit the underlying finding as its own separate `fact` memory as well.
12. Never let the user's stated preference decide which evidence is worth keeping. When evidence favours an option the user did not choose, that fact is MORE useful to store, not less.
13. Extract from EVERY speaker in the dialogue, not just one. A long-running conversation has two participants; memories about the other speaker ("Melanie has been painting for seven years") answer just as many later questions as memories about the primary user. Do not collapse or skip a turn because it is short, because it answers a question, or because it is the non-primary speaker.
14. Preserve temporal anchors verbatim. When an utterance states or implies a date, duration, or point in time ("yesterday", "last May", "seven years now", "since 2016", "two days before July 12"), capture that exact figure in the memory content AND set the `time` field: use `observed_at` for the session date (already given as session_time), `valid_from` for when the event/began, and `valid_to` for when it ended. Resolve relative dates ("yesterday" relative to the session date) to an absolute date when the session date is known. A memory whose content drops the date is useless for any "when" question.
15. Return up to 32 high-value atomic memories. Favor specific events, dated experiences, biographical facts, external evidence, preferences, constraints, and explicit changes. It is better to extract a concrete dated fact than to omit it for being "low-value" -- completeness of recoverable facts matters more than brevity.
16. If there is genuinely no propositional content, return empty arrays. Output JSON only.
"""


def _build_extraction_request(
    session: Dict[str, Any],
    store: OntologyStore,
    *,
    top_k: int,
    memory_top_l: int,
) -> tuple[list[dict[str, str]], Dict[str, Any]]:
    dialogue_text = "\n".join(
        "%s: %s" % (item["role"], item["content"])
        for item in session.get("messages", [])
    )
    branch = store.local_branch(dialogue_text, top_k=top_k, memory_top_l=memory_top_l)
    payload = {
        "user_id": session["user_id"],
        "session_id": session["session_id"],
        "session_time": session.get("timestamp", ""),
        "dialogue": session.get("messages", []),
        "relevant_ontology_branch": branch,
    }
    return [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ], branch


def _finalize_extraction(result: Dict[str, Any], branch: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("concept_edits", "memories", "relations"):
        if key not in result:
            result[key] = []
        if not isinstance(result[key], list):
            raise ValueError("Extractor field %s must be a list" % key)
    result["_retrieval"] = {
        "concept_ids": [item["concept_id"] for item in branch["concepts"]],
        "memory_ids": [item["memory_id"] for item in branch["memories"]],
    }
    return result


class OntologyGuidedExtractor:
    def __init__(
        self,
        client: JSONClient,
        top_k: int = 5,
        memory_top_l: int = 40,
    ) -> None:
        self.client = client
        self.top_k = max(1, top_k)
        self.memory_top_l = max(1, memory_top_l)

    def extract(self, session: Dict[str, Any], store: OntologyStore) -> Dict[str, Any]:
        messages, branch = _build_extraction_request(
            session,
            store,
            top_k=self.top_k,
            memory_top_l=self.memory_top_l,
        )
        return _finalize_extraction(self.client.complete_json(messages), branch)


class AsyncOntologyGuidedExtractor:
    def __init__(
        self,
        client: AsyncJSONClient,
        top_k: int = 5,
        memory_top_l: int = 40,
    ) -> None:
        self.client = client
        self.top_k = max(1, top_k)
        self.memory_top_l = max(1, memory_top_l)

    async def extract(self, session: Dict[str, Any], store: OntologyStore) -> Dict[str, Any]:
        messages, branch = _build_extraction_request(
            session,
            store,
            top_k=self.top_k,
            memory_top_l=self.memory_top_l,
        )
        result = await self.client.complete_json(messages)
        return _finalize_extraction(result, branch)
