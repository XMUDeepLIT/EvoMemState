from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple

from .ontology import ApplyResult, OntologyStore


class Extractor(Protocol):
    def extract(self, session: Dict[str, Any], store: OntologyStore) -> Dict[str, Any]:
        ...


class AsyncExtractor(Protocol):
    async def extract(self, session: Dict[str, Any], store: OntologyStore) -> Dict[str, Any]:
        ...


class MemoryBuilder:
    """Runs ontology-guided extraction and commits its delta as one transaction."""

    def __init__(self, store: OntologyStore, extractor: Extractor) -> None:
        self.store = store
        self.extractor = extractor

    def build_session(
        self, session: Dict[str, Any]
    ) -> Tuple[ApplyResult, Dict[str, Any]]:
        session_id = str(session["session_id"])
        if session_id in self.store.state.processed_sessions:
            return ApplyResult(session_id=session_id, skipped=True), {
                "concept_edits": [],
                "memories": [],
                "relations": [],
            }
        delta = self.extractor.extract(session, self.store)
        result = self.store.apply_delta(delta, session)
        return result, delta


class AsyncMemoryBuilder:
    """Keeps sessions sequential per user while awaiting EVA without blocking other users."""

    def __init__(self, store: OntologyStore, extractor: AsyncExtractor) -> None:
        self.store = store
        self.extractor = extractor

    async def build_session(
        self,
        session: Dict[str, Any],
    ) -> Tuple[ApplyResult, Dict[str, Any]]:
        session_id = str(session["session_id"])
        if session_id in self.store.state.processed_sessions:
            return ApplyResult(session_id=session_id, skipped=True), {
                "concept_edits": [],
                "memories": [],
                "relations": [],
            }
        delta = await self.extractor.extract(session, self.store)
        result = self.store.apply_delta(delta, session)
        return result, delta
