from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .models import (
    EDGE_TYPES,
    MEMORY_STATUSES,
    MEMORY_TYPES,
    ROOT_CONCEPT_ID,
    SCOPE_LEVELS,
    Concept,
    Edge,
    Memory,
    OntologyState,
    utc_now,
)


@dataclass
class ApplyResult:
    session_id: str
    skipped: bool = False
    created_concept_ids: List[str] = field(default_factory=list)
    created_memory_ids: List[str] = field(default_factory=list)
    created_edge_ids: List[str] = field(default_factory=list)
    reused_memory_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "skipped": self.skipped,
            "created_concept_ids": self.created_concept_ids,
            "created_memory_ids": self.created_memory_ids,
            "created_edge_ids": self.created_edge_ids,
            "reused_memory_ids": self.reused_memory_ids,
            "warnings": self.warnings,
        }


def _normalized_text(value: str) -> str:
    return " ".join(str(value).casefold().split())


_MEMORY_TYPE_ALIASES = {
    "preferences": "preference",
    "pref": "preference",
    "prefs": "preference",
    "negative_preference": "preference",
    "positive_preference": "preference",
    "dislike": "preference",
    "like": "preference",
    "opinion": "belief",
    "opinions": "belief",
    "facts": "fact",
    "event": "experience",
    "events": "experience",
    "experiences": "experience",
    "constraints": "constraint",
    "rule": "constraint",
    "rules": "constraint",
}


def _coerce_memory_content(raw: Any) -> str:
    if isinstance(raw, list):
        return " ".join(str(item).strip() for item in raw if str(item).strip()).strip()
    if isinstance(raw, dict):
        for key in ("text", "content", "value", "summary"):
            value = raw.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""
    return str(raw or "").strip()


def _normalize_memory_type(raw: Any) -> str:
    memory_type = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if memory_type in MEMORY_TYPES:
        return memory_type
    return _MEMORY_TYPE_ALIASES.get(memory_type, memory_type)


def _ground_evidence_quote(quote: str, source: str, threshold: float = 0.84) -> Optional[str]:
    """Return an exact source span for an exact or minimally rewritten quote."""
    quote = str(quote or "").strip()
    source = str(source or "")
    if not quote or not source:
        return None
    if quote in source:
        return quote

    quote_words = [match.group(0).casefold() for match in re.finditer(r"\w+(?:['’-]\w+)*", quote)]
    source_matches = list(re.finditer(r"\w+(?:['’-]\w+)*", source))
    source_words = [match.group(0).casefold() for match in source_matches]
    if len(quote_words) < 4 or len(source_words) < 4:
        return None

    best_score = 0.0
    best_bounds: Optional[Tuple[int, int]] = None
    minimum = max(3, len(quote_words) - 2)
    maximum = min(len(source_words), len(quote_words) + 2)
    for window_size in range(minimum, maximum + 1):
        for start in range(0, len(source_words) - window_size + 1):
            end = start + window_size
            ratio = SequenceMatcher(None, quote_words, source_words[start:end]).ratio()
            coverage = min(len(quote_words), window_size) / max(len(quote_words), window_size)
            score = ratio * coverage
            if score > best_score:
                best_score = score
                best_bounds = (start, end)
    if best_score < threshold or best_bounds is None:
        return None

    start, end = best_bounds
    return source[source_matches[start].start() : source_matches[end - 1].end()]


def _terms(value: str) -> Set[str]:
    normalized = _normalized_text(value)
    words = set(re.findall(r"[\w]+", normalized, flags=re.UNICODE))
    compact = re.sub(r"\s+", "", normalized)
    if compact:
        words.update(compact[index : index + 2] for index in range(max(0, len(compact) - 1)))
    return {item for item in words if item}


def _similarity(left: str, right: str) -> float:
    a, b = _terms(left), _terms(right)
    if not a or not b:
        return 0.0
    overlap = len(a & b) / max(1, len(a | b))
    substring = 0.25 if _normalized_text(left) in _normalized_text(right) else 0.0
    return overlap + substring


def _evidence_terms(value: str) -> Set[str]:
    stopwords = {
        "the", "a", "an", "and", "or", "but", "i", "me", "my", "we", "our",
        "user", "to", "of", "for", "in", "on", "at", "is", "am", "are", "was",
        "were", "be", "been", "being", "it", "this", "that", "with", "from",
    }
    return {
        match.group(0).casefold()
        for match in re.finditer(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", str(value))
        if len(match.group(0)) > 1 and match.group(0).casefold() not in stopwords
    }


# Material an assistant turn is *carrying* rather than asserting.
#
# The judgement is STRUCTURAL, not lexical: we do not keep a list of blessed tag
# names. Anything the author set off from the surrounding prose with a matched
# delimiter counts. A tag whitelist would silently fail closed on the next
# corpus that writes <retrieved_docs> or uses blockquotes, and "which corpus is
# this" is exactly the kind of knowledge that should not be compiled into the
# memory layer.
#
# What matters is the contrast: delimited text was pasted in from somewhere,
# while the assistant's own prose is unmarked. That distinction is what keeps
# sycophantic agreement ("You're right, Atlas is clearly better") out of the
# graph, and it survives a change of tag vocabulary.
_DELIMITED_BLOCK_PATTERNS = (
    # Matched XML/HTML-style tags of any name: <information>, <retrieved_docs>,
    # <search_result>, <ctx-3>, ... Group 2 is the payload.
    re.compile(r"<([A-Za-z][\w:-]*)\b[^>]*>(.*?)</\1\s*>", re.DOTALL),
    # Fenced code/quote blocks: ```...``` or ~~~...~~~
    re.compile(r"(`{3,}|~{3,})[^\n]*\n(.*?)\1", re.DOTALL),
)

# Markdown blockquotes: one or more consecutive lines starting with ">".
_BLOCKQUOTE_RE = re.compile(r"(?:^[ \t]*>[ \t]?.*(?:\n|$))+", re.MULTILINE)

# Tags that mark up prose rather than set off quoted material. A <b>bold</b>
# span inside the assistant's own sentence is still the assistant talking.
_INLINE_MARKUP_TAGS = frozenset(
    {"b", "i", "u", "em", "strong", "code", "span", "a", "br", "small", "sub", "sup"}
)

# Below this length a "block" is a formatting artifact, not a quoted document.
_MIN_BLOCK_CHARS = 40


def _delimited_blocks(content: str) -> List[str]:
    """Return text the author explicitly set off from their own prose."""
    blocks: List[str] = []
    for pattern in _DELIMITED_BLOCK_PATTERNS:
        for match in pattern.finditer(content):
            if pattern is _DELIMITED_BLOCK_PATTERNS[0]:
                if match.group(1).lower() in _INLINE_MARKUP_TAGS:
                    continue
            payload = match.group(2).strip()
            if len(payload) >= _MIN_BLOCK_CHARS:
                blocks.append(payload)
    for match in _BLOCKQUOTE_RE.finditer(content):
        payload = "\n".join(
            line.lstrip().lstrip(">").strip() for line in match.group(0).splitlines()
        ).strip()
        if len(payload) >= _MIN_BLOCK_CHARS:
            blocks.append(payload)
    return blocks


def _retrieved_evidence_turns(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Assistant turns carrying quoted material, reduced to just that material.

    `_resolve_memory_evidence` considers only user/system turns, so a memory
    quoting evidence that arrived through the assistant failed to ground and was
    dropped. On MemSyco's memory_evidence_conflict -- where 100% of the decisive
    evidence is delivered this way -- that put the evidence in just 2% of graphs.

    Only the text INSIDE the delimiters is exposed. The assistant's surrounding
    prose stays unciteable on purpose: it is frequently the sycophantic
    agreement this benchmark exists to catch, and promoting it to a grounded
    `fact` would defeat the task.
    """
    turns: List[Dict[str, Any]] = []
    for item in session.get("messages", []):
        if str(item.get("role", "")).lower() != "assistant":
            continue
        content = str(item.get("content", ""))
        blocks = _delimited_blocks(content)
        if not blocks:
            continue
        turns.append(
            {
                "turn_id": item.get("turn_id", ""),
                "role": "assistant",
                "content": "\n\n".join(blocks),
            }
        )
    return turns


def _resolve_memory_evidence(
    session: Dict[str, Any],
    requested_turn_ids: List[str],
    quote: str,
    memory_content: str,
) -> Optional[Tuple[List[str], str]]:
    source_turns = [
        item
        for item in session.get("messages", [])
        if str(item.get("role", "")).lower() in {"user", "system"}
        and str(item.get("content", "")).strip()
    ]
    source_turns.extend(_retrieved_evidence_turns(session))
    requested = set(requested_turn_ids)
    cited = [item for item in source_turns if str(item.get("turn_id", "")) in requested]
    search_groups = [cited, source_turns] if cited else [source_turns]
    for threshold in (0.84, 0.68):
        for turns in search_groups:
            for item in turns:
                matched = _ground_evidence_quote(
                    quote,
                    str(item.get("content", "")),
                    threshold=threshold,
                )
                if matched is not None:
                    return [str(item.get("turn_id", ""))], matched

    query_terms = _evidence_terms(quote + " " + memory_content)
    best: Optional[Tuple[int, float, Dict[str, Any], str]] = None
    for item in (cited or source_turns):
        source = str(item.get("content", ""))
        spans = [
            match.group(0).strip()
            for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", source)
            if match.group(0).strip()
        ] or [source]
        for span in spans:
            span_terms = _evidence_terms(span)
            shared = len(query_terms & span_terms)
            coverage = shared / max(1, min(len(query_terms), len(span_terms)))
            candidate = (shared, coverage, item, span)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is not None and best[0] >= 3 and best[1] >= 0.20:
        return [str(best[2].get("turn_id", ""))], best[3]
    return None


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


class OntologyStore:
    """Persistent user ontology and the atomic merge logic for one extraction delta."""

    def __init__(self, state: OntologyState, *, slot_policy: str = "strict") -> None:
        if slot_policy not in {"strict", "compatible"}:
            raise ValueError("Unsupported slot_policy: %r" % (slot_policy,))
        self.state = state
        # Controls how tightly a SUPERSEDES/CONFLICTS_WITH pair must match.
        # "strict"     -- endpoints share parent concept, type, and scope.
        # "compatible" -- endpoints share only memory type, so a preference that
        #                 migrates to a different topic can still be linked.
        self.slot_policy = slot_policy

    def _shares_slot(self, source: Memory, target: Memory) -> bool:
        if source.memory_type != target.memory_type:
            return False
        if self.slot_policy == "compatible":
            return True
        return (
            source.parent_concept_id == target.parent_concept_id
            and source.scope == target.scope
        )

    @classmethod
    def load(cls, path: Path, user_id: str, *, slot_policy: str = "strict") -> "OntologyStore":
        if path.exists():
            state = OntologyState.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if state.user_id != user_id:
                raise ValueError("Ontology user_id mismatch: %s != %s" % (state.user_id, user_id))
            cls._repair_counters(state)
            cls._validate_state(state)
            return cls(state, slot_policy=slot_policy)
        return cls(OntologyState.empty(user_id), slot_policy=slot_policy)

    @staticmethod
    def _repair_counters(state: OntologyState) -> None:
        collections = {
            "concept": state.concepts,
            "memory": state.memories,
            "edge": state.edges,
        }
        for kind, values in collections.items():
            maximum = int(state.counters.get(kind, 0))
            for value in values:
                match = re.fullmatch(r"%s_(\d+)" % kind, value)
                if match:
                    maximum = max(maximum, int(match.group(1)))
            state.counters[kind] = maximum

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _concept_text(self, concept_id: str) -> str:
        concept = self.state.concepts[concept_id]
        parts = [concept.name, concept.description] + concept.aliases
        representatives = [
            memory
            for memory in self.state.memories.values()
            if memory.parent_concept_id == concept_id and memory.status == "active"
        ]
        representatives.sort(key=lambda item: (item.updated_at, item.memory_id), reverse=True)
        parts.extend(memory.content for memory in representatives[:20])
        return " ".join(parts)

    def select_relevant_concepts(self, dialogue_text: str, top_k: int = 5) -> List[str]:
        candidates: List[Tuple[float, str]] = []
        for concept_id, concept in self.state.concepts.items():
            if concept_id == ROOT_CONCEPT_ID or concept.status != "active":
                continue
            candidates.append((_similarity(dialogue_text, self._concept_text(concept_id)), concept_id))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        positive = [concept_id for score, concept_id in candidates if score > 0]
        if positive:
            return positive[: max(1, top_k)]
        return [concept_id for _, concept_id in candidates[: max(1, top_k)]] or [ROOT_CONCEPT_ID]

    def _ancestor_ids(self, concept_id: str) -> List[str]:
        result: List[str] = []
        seen: Set[str] = set()
        current = concept_id
        while current and current not in seen and current in self.state.concepts:
            seen.add(current)
            result.append(current)
            current = self.state.concepts[current].parent_id
        if ROOT_CONCEPT_ID not in result:
            result.append(ROOT_CONCEPT_ID)
        return result

    def local_branch(
        self,
        dialogue_text: str,
        top_k: int = 5,
        memory_top_l: int = 40,
    ) -> Dict[str, Any]:
        selected = self.select_relevant_concepts(dialogue_text, top_k=top_k)
        concept_ids: Set[str] = set()
        for concept_id in selected:
            concept_ids.update(self._ancestor_ids(concept_id))
            concept_ids.update(
                child_id
                for child_id, child in self.state.concepts.items()
                if child.parent_id == concept_id and child.status == "active"
            )
        candidates = [
            memory
            for memory in self.state.memories.values()
            if memory.parent_concept_id in concept_ids
        ]
        status_bonus = {"active": 0.30, "disputed": 0.15, "superseded": 0.05}
        candidates.sort(
            key=lambda item: (
                _similarity(dialogue_text, item.content) + status_bonus.get(item.status, 0.0),
                item.updated_at,
                item.memory_id,
            ),
            reverse=True,
        )
        limit = max(1, int(memory_top_l))
        memories = candidates[:limit]
        memory_ids = {memory.memory_id for memory in memories}
        for edge in self.state.edges.values():
            if edge.edge_type not in {"SUPERSEDES", "CONFLICTS_WITH"}:
                continue
            if edge.source_id in memory_ids and edge.target_id in self.state.memories:
                memory_ids.add(edge.target_id)
            if edge.target_id in memory_ids and edge.source_id in self.state.memories:
                memory_ids.add(edge.source_id)
        memories = [self.state.memories[memory_id] for memory_id in sorted(memory_ids)]
        for memory in memories:
            concept_ids.update(self._ancestor_ids(memory.parent_concept_id))
        edges = []
        for edge in self.state.edges.values():
            if edge.edge_type == "PARENT_OF":
                include = edge.source_id in concept_ids and edge.target_id in concept_ids
            elif edge.edge_type == "CONTAINS":
                include = edge.source_id in concept_ids and edge.target_id in memory_ids
            else:
                include = edge.source_id in memory_ids and edge.target_id in memory_ids
            if include:
                edges.append(edge)
        return {
            "selected_concept_ids": selected,
            "concepts": [
                {
                    "concept_id": concept.concept_id,
                    "name": concept.name,
                    "description": concept.description,
                    "aliases": concept.aliases,
                    "parent_id": concept.parent_id,
                    "status": concept.status,
                }
                for concept in self.state.concepts.values()
                if concept.concept_id in concept_ids
            ],
            "memories": [
                {
                    "memory_id": item.memory_id,
                    "content": item.content,
                    "memory_type": item.memory_type,
                    "scope": item.scope,
                    "time": item.time,
                    "status": item.status,
                    "parent_concept_id": item.parent_concept_id,
                    "confidence": item.confidence,
                }
                for item in memories
            ],
            "edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "edge_type": edge.edge_type,
                }
                for edge in edges
            ],
        }

    @staticmethod
    def _find_concept_by_name(state: OntologyState, name: str) -> Optional[str]:
        wanted = _normalized_text(name)
        for concept_id, concept in state.concepts.items():
            names = [concept.name] + concept.aliases
            if concept.status == "active" and any(_normalized_text(item) == wanted for item in names):
                return concept_id
        return None

    @staticmethod
    def _resolve_concept(
        state: OntologyState,
        reference: Any,
        temporary_ids: Dict[str, str],
    ) -> Optional[str]:
        value = str(reference or "").strip()
        if not value:
            return None
        if value in temporary_ids:
            return temporary_ids[value]
        if value in state.concepts:
            return value
        return OntologyStore._find_concept_by_name(state, value)

    @staticmethod
    def _resolve_memory(
        state: OntologyState,
        reference: Any,
        temporary_ids: Dict[str, str],
    ) -> Optional[str]:
        value = str(reference or "").strip()
        if value in temporary_ids:
            return temporary_ids[value]
        if value in state.memories:
            return value
        return None

    @staticmethod
    def _add_edge(
        state: OntologyState,
        source_id: str,
        target_id: str,
        edge_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if edge_type not in EDGE_TYPES or not source_id or not target_id or source_id == target_id:
            return None
        if edge_type == "CONFLICTS_WITH" and source_id > target_id:
            source_id, target_id = target_id, source_id
        for edge in state.edges.values():
            if (
                edge.source_id == source_id
                and edge.target_id == target_id
                and edge.edge_type == edge_type
            ):
                return edge.edge_id
        edge_id = state.next_id("edge")
        state.edges[edge_id] = Edge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            metadata=metadata or {},
        )
        return edge_id

    @staticmethod
    def _remove_edge(state: OntologyState, source_id: str, target_id: str, edge_type: str) -> None:
        for edge_id, edge in list(state.edges.items()):
            if (
                edge.source_id == source_id
                and edge.target_id == target_id
                and edge.edge_type == edge_type
            ):
                del state.edges[edge_id]

    @staticmethod
    def _is_ancestor(state: OntologyState, ancestor_id: str, descendant_id: str) -> bool:
        seen: Set[str] = set()
        current = descendant_id
        while current and current not in seen and current in state.concepts:
            if current == ancestor_id:
                return True
            seen.add(current)
            current = state.concepts[current].parent_id
        return False

    @staticmethod
    def _has_supersedes_path(state: OntologyState, start_id: str, target_id: str) -> bool:
        adjacency: Dict[str, List[str]] = {}
        for edge in state.edges.values():
            if edge.edge_type == "SUPERSEDES":
                adjacency.setdefault(edge.source_id, []).append(edge.target_id)
        pending = [start_id]
        seen: Set[str] = set()
        while pending:
            current = pending.pop()
            if current == target_id:
                return True
            if current in seen:
                continue
            seen.add(current)
            pending.extend(adjacency.get(current, []))
        return False

    @staticmethod
    def _validate_state(state: OntologyState) -> None:
        root = state.concepts.get(ROOT_CONCEPT_ID)
        if root is None or root.parent_id:
            raise ValueError("Ontology must contain a parentless root concept")
        for concept_id, concept in state.concepts.items():
            if concept_id == ROOT_CONCEPT_ID:
                continue
            if concept.parent_id not in state.concepts:
                raise ValueError("Concept %s has an unknown parent" % concept_id)
            if concept.status == "active" and state.concepts[concept.parent_id].status != "active":
                raise ValueError("Active concept %s has a non-active parent" % concept_id)
            seen: Set[str] = set()
            current = concept_id
            while current:
                if current in seen:
                    raise ValueError("Concept hierarchy contains a cycle at %s" % current)
                seen.add(current)
                current = state.concepts[current].parent_id
        for memory_id, memory in state.memories.items():
            if memory.parent_concept_id not in state.concepts:
                raise ValueError("Memory %s has an unknown parent concept" % memory_id)
            if memory.memory_type not in MEMORY_TYPES or memory.status not in MEMORY_STATUSES:
                raise ValueError("Memory %s has an invalid type or status" % memory_id)
        for edge in state.edges.values():
            if edge.edge_type == "PARENT_OF":
                if edge.source_id not in state.concepts or edge.target_id not in state.concepts:
                    raise ValueError("PARENT_OF edge %s has an unknown endpoint" % edge.edge_id)
            elif edge.edge_type == "CONTAINS":
                if edge.source_id not in state.concepts or edge.target_id not in state.memories:
                    raise ValueError("CONTAINS edge %s has an unknown endpoint" % edge.edge_id)
            elif edge.source_id not in state.memories or edge.target_id not in state.memories:
                raise ValueError("Memory edge %s has an unknown endpoint" % edge.edge_id)
            if edge.edge_type == "SUPERSEDES" and OntologyStore._has_supersedes_path(
                state, edge.target_id, edge.source_id
            ):
                raise ValueError("SUPERSEDES graph contains a cycle")
        for concept_id, concept in state.concepts.items():
            if concept_id == ROOT_CONCEPT_ID or concept.status != "active":
                continue
            parents = [
                edge.source_id
                for edge in state.edges.values()
                if edge.edge_type == "PARENT_OF" and edge.target_id == concept_id
            ]
            if parents != [concept.parent_id]:
                raise ValueError("Concept %s does not have exactly one matching PARENT_OF edge" % concept_id)
        for memory_id, memory in state.memories.items():
            parents = [
                edge.source_id
                for edge in state.edges.values()
                if edge.edge_type == "CONTAINS" and edge.target_id == memory_id
            ]
            if parents != [memory.parent_concept_id]:
                raise ValueError("Memory %s does not have exactly one matching CONTAINS edge" % memory_id)

    @staticmethod
    def _scope(value: Any) -> Dict[str, str]:
        if isinstance(value, str):
            text = value.strip()
            return {"level": "global" if text.casefold() == "global" else "domain", "value": text}
        if not isinstance(value, dict):
            return {"level": "global", "value": "global"}
        level = str(value.get("level", "global")).strip().lower()
        if level not in SCOPE_LEVELS:
            level = "global"
        text = str(value.get("value", value.get("description", level))).strip() or level
        return {"level": level, "value": text}

    @staticmethod
    def _time(value: Any, session_time: str) -> Dict[str, str]:
        result = dict(value) if isinstance(value, dict) else {}
        observed_at = str(result.get("observed_at", session_time)).strip() or session_time
        return {
            "observed_at": observed_at,
            "valid_from": str(result.get("valid_from", "")),
            "valid_to": str(result.get("valid_to", "")),
        }

    @staticmethod
    def _create_concept(
        state: OntologyState,
        raw: Dict[str, Any],
        temporary_ids: Dict[str, str],
        result: ApplyResult,
        *,
        forced_parent: Optional[str] = None,
    ) -> Optional[str]:
        name = str(raw.get("name", "")).strip()
        temp_id = str(raw.get("temp_id", raw.get("concept_ref", ""))).strip()
        if not name:
            result.warnings.append("Ignored CREATE concept without a name")
            return None
        duplicate = OntologyStore._find_concept_by_name(state, name)
        if duplicate:
            if temp_id:
                temporary_ids[temp_id] = duplicate
            concept = state.concepts[duplicate]
            for alias in _string_list(raw.get("aliases", [])):
                if _normalized_text(alias) not in {_normalized_text(item) for item in concept.aliases}:
                    concept.aliases.append(alias)
            return duplicate
        parent_reference = raw.get("parent_id", raw.get("parent_ref", ROOT_CONCEPT_ID))
        parent_id = forced_parent or OntologyStore._resolve_concept(
            state, parent_reference, temporary_ids
        )
        if not parent_id or state.concepts[parent_id].status != "active":
            raise ValueError("CREATE concept %s has an invalid parent_ref" % name)
        concept_id = state.next_id("concept")
        state.concepts[concept_id] = Concept(
            concept_id=concept_id,
            name=name,
            description=str(raw.get("description", "")).strip(),
            aliases=_string_list(raw.get("aliases", [])),
            parent_id=parent_id,
        )
        if temp_id:
            temporary_ids[temp_id] = concept_id
        edge_id = OntologyStore._add_edge(state, parent_id, concept_id, "PARENT_OF")
        result.created_concept_ids.append(concept_id)
        if edge_id:
            result.created_edge_ids.append(edge_id)
        return concept_id

    @staticmethod
    def _merge_concepts(
        state: OntologyState,
        raw: Dict[str, Any],
        temporary_ids: Dict[str, str],
        result: ApplyResult,
    ) -> None:
        target_id = OntologyStore._resolve_concept(
            state, raw.get("target_id", raw.get("target_ref")), temporary_ids
        )
        source_refs = raw.get("source_ids", raw.get("source_refs", []))
        if (
            not target_id
            or target_id == ROOT_CONCEPT_ID
            or state.concepts[target_id].status != "active"
            or not isinstance(source_refs, list)
            or not source_refs
        ):
            raise ValueError("MERGE has an invalid target or empty source_refs")
        for reference in source_refs:
            source_id = OntologyStore._resolve_concept(state, reference, temporary_ids)
            if (
                not source_id
                or source_id in {target_id, ROOT_CONCEPT_ID}
                or state.concepts[source_id].status != "active"
            ):
                raise ValueError("MERGE has an invalid source_ref")
            if OntologyStore._is_ancestor(state, source_id, target_id):
                raise ValueError("MERGE cannot move an ancestor into its descendant")
            source = state.concepts[source_id]
            target = state.concepts[target_id]
            for alias in [source.name] + source.aliases + _string_list(raw.get("aliases", [])):
                if _normalized_text(alias) != _normalized_text(target.name) and alias not in target.aliases:
                    target.aliases.append(alias)
            source.status = "merged"
            source.updated_at = utc_now()
            for concept in state.concepts.values():
                if concept.parent_id == source_id:
                    OntologyStore._remove_edge(state, source_id, concept.concept_id, "PARENT_OF")
                    concept.parent_id = target_id
                    OntologyStore._add_edge(state, target_id, concept.concept_id, "PARENT_OF")
            for memory in state.memories.values():
                if memory.parent_concept_id == source_id:
                    OntologyStore._remove_edge(state, source_id, memory.memory_id, "CONTAINS")
                    memory.parent_concept_id = target_id
                    OntologyStore._add_edge(state, target_id, memory.memory_id, "CONTAINS")
            for temp_id, resolved in list(temporary_ids.items()):
                if resolved == source_id:
                    temporary_ids[temp_id] = target_id

    def apply_delta(self, delta: Dict[str, Any], session: Dict[str, Any]) -> ApplyResult:
        session_id = str(session["session_id"])
        if session_id in self.state.processed_sessions:
            return ApplyResult(session_id=session_id, skipped=True)
        if not isinstance(delta, dict):
            raise ValueError("Extraction delta must be an object")
        for key in ("concept_edits", "memories", "relations"):
            if key not in delta or not isinstance(delta[key], list):
                raise ValueError("Extraction delta field %s must be a list" % key)
            if any(not isinstance(item, dict) for item in delta[key]):
                raise ValueError("Extraction delta field %s contains a non-object" % key)
        if len(delta["concept_edits"]) > 30 or len(delta["memories"]) > 100 or len(delta["relations"]) > 100:
            raise ValueError("Extraction delta exceeds the configured size limits")

        original_memory_ids = set(self.state.memories)
        retrieval = delta.get("_retrieval") if isinstance(delta.get("_retrieval"), dict) else None
        allowed_concept_ids = (
            set(_string_list(retrieval.get("concept_ids", []))) if retrieval is not None else None
        )
        allowed_memory_ids = (
            set(_string_list(retrieval.get("memory_ids", []))) if retrieval is not None else None
        )
        state = copy.deepcopy(self.state)
        result = ApplyResult(session_id=session_id)
        concept_refs: Dict[str, str] = {}
        memory_refs: Dict[str, str] = {}

        for raw in delta["concept_edits"]:
            operation = str(raw.get("op", "")).strip().upper()
            if operation == "CREATE":
                temp_id = str(raw.get("temp_id", "")).strip()
                if not temp_id or temp_id in concept_refs:
                    raise ValueError("CREATE concept temp_id must be non-empty and unique")
                self._create_concept(state, raw, concept_refs, result)
            elif operation == "MERGE":
                self._merge_concepts(state, raw, concept_refs, result)
            elif operation == "SPLIT":
                source_id = self._resolve_concept(
                    state, raw.get("source_id", raw.get("source_ref")), concept_refs
                )
                children = raw.get("children", [])
                if (
                    not source_id
                    or source_id == ROOT_CONCEPT_ID
                    or state.concepts[source_id].status != "active"
                    or not isinstance(children, list)
                    or not children
                    or any(not isinstance(child, dict) for child in children)
                ):
                    raise ValueError("SPLIT has an invalid source or children")
                for child in children:
                    temp_id = str(child.get("temp_id", "")).strip()
                    if not temp_id or temp_id in concept_refs:
                        raise ValueError("SPLIT child temp_id must be non-empty and unique")
                    self._create_concept(
                        state, child, concept_refs, result, forced_parent=source_id
                    )
            else:
                raise ValueError("Unsupported concept edit: %s" % operation)

        for raw in delta["memories"]:
            content = _coerce_memory_content(raw.get("content", ""))
            memory_type = _normalize_memory_type(raw.get("type", raw.get("memory_type", "")))
            temp_id = str(raw.get("temp_id", raw.get("memory_ref", ""))).strip()
            if not temp_id or temp_id in memory_refs:
                result.warnings.append(
                    "Ignored memory with empty or duplicate temp_id: %s" % (temp_id or "<empty>")
                )
                continue
            if not content:
                result.warnings.append("Ignored memory %s because content is empty" % temp_id)
                continue
            if memory_type not in MEMORY_TYPES:
                result.warnings.append(
                    "Ignored memory %s because type %r is invalid" % (temp_id, memory_type)
                )
                continue
            if len(content) > 4000:
                result.warnings.append(
                    "Truncated oversized content for memory %s from %d to 4000 chars"
                    % (temp_id, len(content))
                )
                content = content[:4000]
            parent_reference = raw.get("parent_concept_id", raw.get("parent_ref", ROOT_CONCEPT_ID))
            parent_id = self._resolve_concept(state, parent_reference, concept_refs)
            if not parent_id or state.concepts[parent_id].status != "active":
                result.warnings.append(
                    "Remapped memory %s parent from invalid %r to %s"
                    % (temp_id, parent_reference, ROOT_CONCEPT_ID)
                )
                parent_id = ROOT_CONCEPT_ID
            if (
                allowed_concept_ids is not None
                and parent_id in self.state.concepts
                and parent_id not in allowed_concept_ids
            ):
                query = "%s %s" % (content, json.dumps(raw.get("scope", {}), ensure_ascii=False))
                candidates: List[Tuple[float, str]] = []
                for candidate_id in allowed_concept_ids:
                    concept = state.concepts.get(candidate_id)
                    if concept is None or concept.status != "active" or candidate_id == ROOT_CONCEPT_ID:
                        continue
                    concept_text = " ".join(
                        [concept.name, concept.description] + list(concept.aliases)
                    )
                    candidates.append((_similarity(query, concept_text), candidate_id))
                candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
                replacement_parent = (
                    candidates[0][1] if candidates and candidates[0][0] > 0 else ROOT_CONCEPT_ID
                )
                result.warnings.append(
                    "Remapped memory %s parent from unavailable %s to %s"
                    % (temp_id, parent_id, replacement_parent)
                )
                parent_id = replacement_parent
            duplicate_id = next(
                (
                    item.memory_id
                    for item in state.memories.values()
                    if item.status == "active"
                    and item.parent_concept_id == parent_id
                    and _normalized_text(item.content) == _normalized_text(content)
                ),
                None,
            )
            if duplicate_id:
                if temp_id:
                    memory_refs[temp_id] = duplicate_id
                result.reused_memory_ids.append(duplicate_id)
                continue
            raw_source = raw.get("source", {}) if isinstance(raw.get("source"), dict) else {}
            requested_turn_ids = _string_list(
                raw_source.get("turn_ids", raw.get("evidence_turn_ids", []))
            )
            evidence_quote = str(
                raw_source.get("evidence_quote", raw.get("evidence_quote", ""))
            ).strip()
            grounded = _resolve_memory_evidence(
                session,
                requested_turn_ids,
                evidence_quote,
                content,
            )
            if grounded is None:
                result.warnings.append(
                    "Ignored memory %s because its evidence could not be grounded" % temp_id
                )
                continue
            cited_turn_ids, evidence_quote = grounded
            try:
                confidence = float(raw.get("confidence"))
            except (TypeError, ValueError):
                result.warnings.append(
                    "Ignored memory %s because confidence is invalid" % temp_id
                )
                continue
            if not 0.0 <= confidence <= 1.0:
                result.warnings.append(
                    "Ignored memory %s because confidence is out of range" % temp_id
                )
                continue
            memory_id = state.next_id("memory")
            memory = Memory(
                memory_id=memory_id,
                content=content,
                memory_type=memory_type,
                scope=self._scope(raw.get("scope", {})),
                time=self._time(raw.get("time", {}), str(session.get("timestamp", ""))),
                source={
                    "session_id": session_id,
                    "turn_ids": cited_turn_ids,
                    "evidence_quote": evidence_quote,
                },
                status="active",
                parent_concept_id=parent_id,
                confidence=confidence,
            )
            state.memories[memory_id] = memory
            if temp_id:
                memory_refs[temp_id] = memory_id
            edge_id = self._add_edge(state, parent_id, memory_id, "CONTAINS")
            result.created_memory_ids.append(memory_id)
            if edge_id:
                result.created_edge_ids.append(edge_id)

        for raw in delta["relations"]:
            relation_type = str(raw.get("type", raw.get("edge_type", ""))).strip().upper()
            if relation_type not in {"SUPERSEDES", "CONFLICTS_WITH"}:
                result.warnings.append("Ignored unsupported memory relation: %s" % relation_type)
                continue
            source_id = self._resolve_memory(
                state, raw.get("source_id", raw.get("source_ref")), memory_refs
            )
            target_id = self._resolve_memory(
                state, raw.get("target_id", raw.get("target_ref")), memory_refs
            )
            target_is_retrieved_old = (
                target_id in original_memory_ids
                and (allowed_memory_ids is None or target_id in allowed_memory_ids)
            )
            target_is_new = target_id in result.created_memory_ids
            if (
                not source_id
                or not target_id
                or source_id == target_id
                or source_id not in result.created_memory_ids
                or not (target_is_retrieved_old or target_is_new)
            ):
                result.warnings.append("Ignored memory relation with unresolved or unavailable endpoints")
                continue
            source = state.memories[source_id]
            target = state.memories[target_id]
            if not self._shares_slot(source, target):
                result.warnings.append("Ignored memory relation whose endpoints do not share one slot")
                continue
            if relation_type == "SUPERSEDES":
                if source.status != "active" or target.status != "active":
                    result.warnings.append("Ignored SUPERSEDES with a non-active endpoint")
                    continue
                if self._has_supersedes_path(state, target_id, source_id):
                    result.warnings.append("Ignored SUPERSEDES that would create a cycle")
                    continue
            reason = str(raw.get("reason", "")).strip()
            if not reason:
                result.warnings.append("Ignored memory relation without a reason")
                continue
            edge_id = self._add_edge(
                state,
                source_id,
                target_id,
                relation_type,
                {"reason": reason},
            )
            if edge_id:
                result.created_edge_ids.append(edge_id)
            if relation_type == "SUPERSEDES":
                target.status = "superseded"
                target.updated_at = utc_now()
            else:
                for memory_id in (source_id, target_id):
                    if state.memories[memory_id].status == "active":
                        state.memories[memory_id].status = "disputed"
                        state.memories[memory_id].updated_at = utc_now()

        self._validate_state(state)
        state.processed_sessions[session_id] = {
            "processed_at": utc_now(),
            "timestamp": str(session.get("timestamp", "")),
            "created_concepts": len(result.created_concept_ids),
            "created_memories": len(result.created_memory_ids),
            "created_edges": len(result.created_edge_ids),
        }
        state.updated_at = utc_now()
        self.state = state
        return result
