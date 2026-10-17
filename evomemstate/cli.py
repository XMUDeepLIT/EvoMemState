from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from .builder import MemoryBuilder
from .extractor import OntologyGuidedExtractor
from .io import iter_sessions
from .llm import OpenAIJSONClient
from .ontology import OntologyStore


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "user"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return "%s_%s" % (cleaned[:80], digest)


def _append_event(path: Path, event: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evomemstate",
        description="Build a self-evolving ontological memory from dialogue sessions.",
    )
    parser.add_argument("--input", required=True, type=Path, help="JSON or JSONL dialogue file")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--user-id", default=None, help="Fallback user id when input omits it")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument(
        "--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Name of the environment variable containing the API key",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise RuntimeError("Environment variable %s is not set" % args.api_key_env)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runtimes: Dict[str, Tuple[OntologyStore, MemoryBuilder, Path]] = {}
    processed = skipped = failed = 0
    for index, session in enumerate(iter_sessions(args.input, args.user_id), start=1):
        if args.limit > 0 and index > args.limit:
            break
        user_id = session["user_id"]
        if user_id not in runtimes:
            user_name = _safe_name(user_id)
            state_path = args.output_dir / "users" / (user_name + ".json")
            store = OntologyStore.load(state_path, user_id)
            client = OpenAIJSONClient(
                model=args.model,
                api_key=api_key,
                base_url=args.base_url,
                cache_dir=args.output_dir / "cache" / user_name,
                temperature=args.temperature,
            )
            extractor = OntologyGuidedExtractor(client, top_k=args.top_k)
            runtimes[user_id] = (store, MemoryBuilder(store, extractor), state_path)
        store, builder, state_path = runtimes[user_id]
        try:
            result, delta = builder.build_session(session)
            if result.skipped:
                skipped += 1
            else:
                store.save(state_path)
                processed += 1
            _append_event(
                args.output_dir / "events.jsonl",
                {
                    "user_id": user_id,
                    "session_id": session["session_id"],
                    "result": result.to_dict(),
                    "delta": delta,
                },
            )
            print(
                "[%s] concepts +%d, memories +%d, edges +%d%s"
                % (
                    session["session_id"],
                    len(result.created_concept_ids),
                    len(result.created_memory_ids),
                    len(result.created_edge_ids),
                    " (skipped)" if result.skipped else "",
                )
            )
        except Exception as exc:
            failed += 1
            _append_event(
                args.output_dir / "errors.jsonl",
                {
                    "user_id": user_id,
                    "session_id": session["session_id"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            print("[%s] ERROR: %s" % (session["session_id"], exc))

    print("Done: processed=%d skipped=%d failed=%d" % (processed, skipped, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
