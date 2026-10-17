from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


@dataclass
class BenchmarkCase:
    dataset: str
    case_id: str
    user_id: str
    sessions: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def manifest_entry(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "case_id": self.case_id,
            "user_id": self.user_id,
            "session_count": len(self.sessions),
            "message_count": sum(len(item["messages"]) for item in self.sessions),
            "metadata": self.metadata,
        }


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256((str(seed) + ":" + value).encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("%s line %d is not a JSON object" % (path, line_number))
            yield value


def _chunk_messages(
    messages: Sequence[Dict[str, Any]],
    *,
    session_prefix: str,
    max_conversation_messages: int,
    chunk_mode: str = "pairs",
) -> List[Dict[str, Any]]:
    """Split a message list into extraction sessions.

    ``chunk_mode="pairs"`` slices the conversation into fixed windows of
    ``max_conversation_messages``. ``chunk_mode="whole_dialogue"`` emits a
    single session holding the entire conversation, so that a preference and
    the later preference that replaces it are visible in one extraction call.
    """
    if chunk_mode not in {"pairs", "whole_dialogue"}:
        raise ValueError("Unsupported chunk_mode: %r" % (chunk_mode,))
    normalized: List[Dict[str, str]] = []
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role in {"system", "user", "assistant"} and content:
            normalized.append(
                {
                    "turn_id": "ctx_%06d" % index,
                    "role": role,
                    "content": content,
                }
            )
    system_messages = [item for item in normalized if item["role"] == "system"]
    conversation = [item for item in normalized if item["role"] != "system"]
    if chunk_mode == "whole_dialogue":
        chunks = [conversation] if conversation else []
    else:
        size = max(2, int(max_conversation_messages))
        if size % 2:
            size += 1
        chunks = [conversation[index : index + size] for index in range(0, len(conversation), size)]
    sessions: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        if index == 1:
            chunk = system_messages + chunk
        if not chunk:
            continue
        sessions.append(
            {
                "session_id": "%s_chunk_%04d" % (session_prefix, index),
                "timestamp": "sequence_%04d" % index,
                "messages": chunk,
            }
        )
    return sessions


def sample_memsyco(
    root: Path,
    *,
    per_task: int = 5,
    seed: int = 20260724,
    chunk_mode: str = "whole_dialogue",
    tasks: Sequence[str] | None = None,
) -> List[BenchmarkCase]:
    data_dir = root / "data"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    cases: List[BenchmarkCase] = []
    for task, config in manifest["tasks"].items():
        if tasks is not None and task not in tasks:
            continue
        rows = list(_read_jsonl(data_dir / str(config["file"])))
        rows.sort(key=lambda item: (_rank(seed, str(item["id"])), str(item["id"])))
        if len(rows) < per_task:
            raise ValueError("MemSyco task %s has fewer than %d samples" % (task, per_task))
        for row in rows[:per_task]:
            case_id = str(row["id"])
            sessions = _chunk_messages(
                row["dialogue"],
                session_prefix=case_id,
                max_conversation_messages=2,
                chunk_mode=chunk_mode,
            )
            user_id = "memsyco:%s:%s" % (task, case_id)
            for session in sessions:
                session["user_id"] = user_id
            cases.append(
                BenchmarkCase(
                    dataset="memsyco",
                    case_id=case_id,
                    user_id=user_id,
                    sessions=sessions,
                    metadata={
                        "task": task,
                        "subtype": row.get("metadata", {}).get("subtype", ""),
                        "topic": row.get("metadata", {}).get("topic", ""),
                        "source_file": str(config["file"]),
                        "dialogue_turns": len(row["dialogue"]),
                        "chunk_mode": chunk_mode,
                        # Gold annotations. Never fed to extraction or to the
                        # answer model -- carried so that downstream scoring and
                        # gold-pair linkage checks can run against the artifacts.
                        "question": row.get("question", ""),
                        "gold_memory": row.get("memory", {}),
                        "evaluation": row.get("evaluation", {}),
                    },
                )
            )
    return cases


def _select_persona_rows(
    questions_path: Path,
    *,
    count: int,
    seed: int,
) -> List[Dict[str, str]]:
    with questions_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    best_by_context: Dict[str, Dict[str, str]] = {}
    for row in rows:
        context_id = row["shared_context_id"]
        current = best_by_context.get(context_id)
        endpoint = int(float(row["end_index_in_shared_context"]))
        current_endpoint = int(float(current["end_index_in_shared_context"])) if current else -1
        if current is None or endpoint > current_endpoint:
            best_by_context[context_id] = row
        elif endpoint == current_endpoint and str(row["question_id"]) < str(current["question_id"]):
            best_by_context[context_id] = row
    candidates = list(best_by_context.values())
    candidates.sort(
        key=lambda item: (
            _rank(seed, item["shared_context_id"]),
            item["shared_context_id"],
        )
    )
    selected: List[Dict[str, str]] = []
    seen_personas = set()
    for row in candidates:
        if row["persona_id"] in seen_personas:
            continue
        selected.append(row)
        seen_personas.add(row["persona_id"])
        if len(selected) == count:
            return selected
    for row in candidates:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) == count:
            return selected
    raise ValueError("PersonaMem contains fewer than %d unique 128k contexts" % count)


def sample_personamem_128k(
    root: Path,
    *,
    count: int = 10,
    seed: int = 20260724,
    max_conversation_messages: int = 80,
) -> List[BenchmarkCase]:
    selected = _select_persona_rows(root / "questions_128k.csv", count=count, seed=seed)
    selected_by_context = {row["shared_context_id"]: row for row in selected}
    contexts: Dict[str, List[Dict[str, Any]]] = {}
    for record in _read_jsonl(root / "shared_contexts_128k.jsonl"):
        for context_id, messages in record.items():
            if context_id in selected_by_context:
                if not isinstance(messages, list):
                    raise ValueError("PersonaMem context %s is not a message list" % context_id)
                contexts[context_id] = messages
    missing = set(selected_by_context) - set(contexts)
    if missing:
        raise ValueError("Missing PersonaMem contexts: %s" % sorted(missing))

    cases: List[BenchmarkCase] = []
    for row in selected:
        context_id = row["shared_context_id"]
        endpoint = min(int(float(row["end_index_in_shared_context"])), len(contexts[context_id]))
        history = contexts[context_id][:endpoint]
        case_id = "persona_%s_%s" % (row["persona_id"], context_id[:12])
        user_id = "personamem:%s:%s" % (row["persona_id"], context_id[:12])
        sessions = _chunk_messages(
            history,
            session_prefix=case_id,
            max_conversation_messages=max_conversation_messages,
        )
        for session in sessions:
            session["user_id"] = user_id
        cases.append(
            BenchmarkCase(
                dataset="personamem_128k",
                case_id=case_id,
                user_id=user_id,
                sessions=sessions,
                metadata={
                    "persona_id": row["persona_id"],
                    "question_id": row["question_id"],
                    "question_type": row["question_type"],
                    "topic": row["topic"],
                    "shared_context_id": context_id,
                    "end_index_in_shared_context": endpoint,
                    "context_length_in_tokens": int(float(row["context_length_in_tokens"])),
                    "context_length_in_letters": int(float(row["context_length_in_letters"])),
                },
            )
        )
    return cases
