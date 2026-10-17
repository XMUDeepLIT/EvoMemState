from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


ROLE_MAP = {
    "user": "user",
    "human": "user",
    "用户": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "bot": "assistant",
    "助手": "assistant",
    "system": "system",
}


def _messages_from_record(record: Dict[str, Any]) -> List[Dict[str, str]]:
    raw_messages = (
        record.get("messages")
        or record.get("dialogue_turns")
        or record.get("conversation")
        or record.get("turns")
        or []
    )
    messages: List[Dict[str, str]] = []
    for index, item in enumerate(raw_messages):
        if not isinstance(item, dict):
            continue
        raw_role = item.get("role", item.get("speaker", ""))
        role = ROLE_MAP.get(str(raw_role).strip().lower())
        content = str(item.get("content", item.get("text", ""))).strip()
        if role and content:
            messages.append(
                {
                    "turn_id": str(item.get("turn_id", "turn_%04d" % (index + 1))),
                    "role": role,
                    "content": content,
                }
            )
    return messages


def normalize_session(
    record: Any,
    *,
    index: int,
    default_user_id: Optional[str] = None,
) -> Dict[str, Any]:
    if isinstance(record, list):
        record = {"messages": record}
    if not isinstance(record, dict):
        raise ValueError("Session %d must be a JSON object or a message list" % index)
    user_id = str(
        record.get("user_id")
        or record.get("uid")
        or record.get("profile_id")
        or default_user_id
        or ""
    ).strip()
    if not user_id:
        raise ValueError("Session %d has no user_id; pass --user-id" % index)
    session_id = str(
        record.get("session_id")
        or record.get("session_uuid")
        or record.get("query_id")
        or "session_%06d" % index
    ).strip()
    messages = _messages_from_record(record)
    if not any(item["role"] == "user" for item in messages):
        raise ValueError("Session %s contains no user turn" % session_id)
    return {
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": str(
            record.get("timestamp")
            or record.get("session_time")
            or record.get("created_at")
            or ""
        ),
        "messages": messages,
    }


def _read_records(path: Path) -> Iterable[Any]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError("Invalid JSONL at line %d" % line_number) from exc
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for item in payload:
            yield item
    elif isinstance(payload, dict) and isinstance(payload.get("sessions"), list):
        for item in payload["sessions"]:
            yield item
    else:
        yield payload


def iter_sessions(path: Path, default_user_id: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    for index, record in enumerate(_read_records(path), start=1):
        yield normalize_session(record, index=index, default_user_id=default_user_id)
