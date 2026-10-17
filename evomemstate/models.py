from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


MEMORY_TYPES = {"preference", "fact", "belief", "experience", "constraint"}
MEMORY_STATUSES = {"active", "superseded", "disputed"}
SCOPE_LEVELS = {"global", "domain", "task", "situation"}
EDGE_TYPES = {"PARENT_OF", "CONTAINS", "SUPERSEDES", "CONFLICTS_WITH"}
ROOT_CONCEPT_ID = "concept_root"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Concept:
    concept_id: str
    name: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    parent_id: str = ROOT_CONCEPT_ID
    status: str = "active"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Concept":
        return cls(
            concept_id=str(value["concept_id"]),
            name=str(value["name"]),
            description=str(value.get("description", "")),
            aliases=[str(item) for item in value.get("aliases", [])],
            parent_id=str(value.get("parent_id", ROOT_CONCEPT_ID)),
            status=str(value.get("status", "active")),
            created_at=str(value.get("created_at", utc_now())),
            updated_at=str(value.get("updated_at", utc_now())),
        )


@dataclass
class Memory:
    memory_id: str
    content: str
    memory_type: str
    scope: Dict[str, str]
    time: Dict[str, str]
    source: Dict[str, Any]
    status: str
    parent_concept_id: str
    confidence: float = 1.0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Memory":
        return cls(
            memory_id=str(value["memory_id"]),
            content=str(value["content"]),
            memory_type=str(value["memory_type"]),
            scope=dict(value.get("scope", {})),
            time=dict(value.get("time", {})),
            source=dict(value.get("source", {})),
            status=str(value.get("status", "active")),
            parent_concept_id=str(value.get("parent_concept_id", ROOT_CONCEPT_ID)),
            confidence=float(value.get("confidence", 1.0)),
            created_at=str(value.get("created_at", utc_now())),
            updated_at=str(value.get("updated_at", utc_now())),
        )


@dataclass
class Edge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Edge":
        return cls(
            edge_id=str(value["edge_id"]),
            source_id=str(value["source_id"]),
            target_id=str(value["target_id"]),
            edge_type=str(value["edge_type"]),
            metadata=dict(value.get("metadata", {})),
            created_at=str(value.get("created_at", utc_now())),
        )


@dataclass
class OntologyState:
    user_id: str
    concepts: Dict[str, Concept]
    memories: Dict[str, Memory]
    edges: Dict[str, Edge]
    processed_sessions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    counters: Dict[str, int] = field(
        default_factory=lambda: {"concept": 0, "memory": 0, "edge": 0}
    )
    schema_version: str = "1.0"
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def empty(cls, user_id: str) -> "OntologyState":
        root = Concept(
            concept_id=ROOT_CONCEPT_ID,
            name="User Memory",
            description="Root of the user's dynamic instance ontology.",
            parent_id="",
        )
        return cls(user_id=user_id, concepts={ROOT_CONCEPT_ID: root}, memories={}, edges={})

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "OntologyState":
        concepts = {
            key: Concept.from_dict(item)
            for key, item in dict(value.get("concepts", {})).items()
        }
        if ROOT_CONCEPT_ID not in concepts:
            root = Concept(
                concept_id=ROOT_CONCEPT_ID,
                name="User Memory",
                description="Root of the user's dynamic instance ontology.",
                parent_id="",
            )
            concepts[ROOT_CONCEPT_ID] = root
        return cls(
            user_id=str(value["user_id"]),
            concepts=concepts,
            memories={
                key: Memory.from_dict(item)
                for key, item in dict(value.get("memories", {})).items()
            },
            edges={
                key: Edge.from_dict(item)
                for key, item in dict(value.get("edges", {})).items()
            },
            processed_sessions=dict(value.get("processed_sessions", {})),
            counters={
                "concept": int(value.get("counters", {}).get("concept", 0)),
                "memory": int(value.get("counters", {}).get("memory", 0)),
                "edge": int(value.get("counters", {}).get("edge", 0)),
            },
            schema_version=str(value.get("schema_version", "1.0")),
            updated_at=str(value.get("updated_at", utc_now())),
        )

    def next_id(self, kind: str) -> str:
        if kind not in self.counters:
            raise ValueError("Unknown id kind: %s" % kind)
        self.counters[kind] += 1
        return "%s_%06d" % (kind, self.counters[kind])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "user_id": self.user_id,
            "updated_at": self.updated_at,
            "counters": dict(self.counters),
            "processed_sessions": dict(self.processed_sessions),
            "concepts": {key: asdict(item) for key, item in self.concepts.items()},
            "memories": {key: asdict(item) for key, item in self.memories.items()},
            "edges": {key: asdict(item) for key, item in self.edges.items()},
        }
