from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .builder import AsyncMemoryBuilder, MemoryBuilder
from .datasets import BenchmarkCase, sample_memsyco, sample_personamem_128k
from .extractor import AsyncOntologyGuidedExtractor, OntologyGuidedExtractor
from .llm import AsyncEvaJSONClient, EvaJSONClient
from .ontology import OntologyStore


_PRINT_LOCK = threading.Lock()


def _safe_name(value: str) -> str:
    prefix = "".join(char if char.isalnum() or char in "_.-" else "_" for char in value)
    prefix = prefix.strip("._")[:100] or "case"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return "%s_%s" % (prefix, digest)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _run_case(
    case: BenchmarkCase,
    *,
    output_dir: Path,
    client: EvaJSONClient,
    concept_top_k: int,
    memory_top_l: int,
) -> Dict[str, Any]:
    case_dir = output_dir / case.dataset / "cases" / _safe_name(case.case_id)
    state_path = case_dir / "ontology.json"
    _atomic_json(case_dir / "selection.json", case.manifest_entry())
    store = OntologyStore.load(state_path, case.user_id)
    builder = MemoryBuilder(
        store,
        OntologyGuidedExtractor(
            client,
            top_k=concept_top_k,
            memory_top_l=memory_top_l,
        ),
    )
    processed = skipped = 0
    error: Optional[Dict[str, str]] = None
    for session in case.sessions:
        try:
            result, delta = builder.build_session(session)
            if result.skipped:
                skipped += 1
            else:
                processed += 1
                store.save(state_path)
                _append_jsonl(
                    case_dir / "events.jsonl",
                    {
                        "session_id": session["session_id"],
                        "result": result.to_dict(),
                        "delta": delta,
                    },
                )
        except Exception as exc:
            error = {
                "session_id": str(session["session_id"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _append_jsonl(case_dir / "errors.jsonl", error)
            break
    if error is None:
        error_path = case_dir / "errors.jsonl"
        if error_path.exists():
            error_path.unlink()
    state = store.state
    summary = {
        "dataset": case.dataset,
        "case_id": case.case_id,
        "user_id": case.user_id,
        "sessions_total": len(case.sessions),
        "sessions_processed": processed,
        "sessions_skipped": skipped,
        "sessions_committed": len(state.processed_sessions),
        "concepts": len(state.concepts),
        "memories": len(state.memories),
        "edges": len(state.edges),
        "memory_statuses": {
            status: sum(1 for item in state.memories.values() if item.status == status)
            for status in ("active", "superseded", "disputed")
        },
        "error": error,
        "metadata": case.metadata,
    }
    _atomic_json(case_dir / "summary.json", summary)
    with _PRINT_LOCK:
        print(
            "[%s/%s] sessions=%d/%d concepts=%d memories=%d edges=%d%s"
            % (
                case.dataset,
                case.case_id,
                len(state.processed_sessions),
                len(case.sessions),
                len(state.concepts),
                len(state.memories),
                len(state.edges),
                " ERROR=" + error["error"] if error else "",
            ),
            flush=True,
        )
    return summary


async def _run_case_async(
    case: BenchmarkCase,
    *,
    output_dir: Path,
    client: AsyncEvaJSONClient,
    concept_top_k: int,
    memory_top_l: int,
) -> Dict[str, Any]:
    case_dir = output_dir / case.dataset / "cases" / _safe_name(case.case_id)
    state_path = case_dir / "ontology.json"
    _atomic_json(case_dir / "selection.json", case.manifest_entry())
    store = OntologyStore.load(state_path, case.user_id)
    builder = AsyncMemoryBuilder(
        store,
        AsyncOntologyGuidedExtractor(
            client,
            top_k=concept_top_k,
            memory_top_l=memory_top_l,
        ),
    )
    processed = skipped = 0
    error: Optional[Dict[str, str]] = None
    for session in case.sessions:
        try:
            result, delta = await builder.build_session(session)
            if result.skipped:
                skipped += 1
            else:
                processed += 1
                store.save(state_path)
                _append_jsonl(
                    case_dir / "events.jsonl",
                    {
                        "session_id": session["session_id"],
                        "result": result.to_dict(),
                        "delta": delta,
                    },
                )
        except Exception as exc:
            error = {
                "session_id": str(session["session_id"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _append_jsonl(case_dir / "errors.jsonl", error)
            break
    if error is None:
        error_path = case_dir / "errors.jsonl"
        if error_path.exists():
            error_path.unlink()
    state = store.state
    summary = {
        "dataset": case.dataset,
        "case_id": case.case_id,
        "user_id": case.user_id,
        "sessions_total": len(case.sessions),
        "sessions_processed": processed,
        "sessions_skipped": skipped,
        "sessions_committed": len(state.processed_sessions),
        "concepts": len(state.concepts),
        "memories": len(state.memories),
        "edges": len(state.edges),
        "memory_statuses": {
            status: sum(1 for item in state.memories.values() if item.status == status)
            for status in ("active", "superseded", "disputed")
        },
        "error": error,
        "metadata": case.metadata,
    }
    _atomic_json(case_dir / "summary.json", summary)
    print(
        "[%s/%s] sessions=%d/%d concepts=%d memories=%d edges=%d%s"
        % (
            case.dataset,
            case.case_id,
            len(state.processed_sessions),
            len(case.sessions),
            len(state.concepts),
            len(state.memories),
            len(state.edges),
            " ERROR=" + error["error"] if error else "",
        ),
        flush=True,
    )
    return summary


async def _run_cases_async(
    cases: List[BenchmarkCase],
    *,
    output_dir: Path,
    client: AsyncEvaJSONClient,
    concept_top_k: int,
    memory_top_l: int,
) -> List[Dict[str, Any]]:
    tasks = [
        asyncio.create_task(
            _run_case_async(
                case,
                output_dir=output_dir,
                client=client,
                concept_top_k=concept_top_k,
                memory_top_l=memory_top_l,
            )
        )
        for case in cases
    ]
    return list(await asyncio.gather(*tasks))


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build ontology memories for sampled benchmarks on EVA.")
    parser.add_argument(
        "--memsyco-root",
        type=Path,
        default=project_root / "datasets" / "MemSyco-Bench",
    )
    parser.add_argument(
        "--personamem-root",
        type=Path,
        default=project_root / "data" / "PersonaMem-v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "benchmark_memory_build",
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--memsyco-per-task", type=int, default=5)
    parser.add_argument("--persona-count", type=int, default=10)
    parser.add_argument("--persona-chunk-messages", type=int, default=32)
    parser.add_argument("--concept-top-k", type=int, default=8)
    parser.add_argument("--memory-top-l", type=int, default=40)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Maximum concurrent EVA requests across independent cases",
    )
    parser.add_argument("--sync", action="store_true", help="Use the legacy thread-based runner")
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    memsyco = sample_memsyco(
        args.memsyco_root,
        per_task=args.memsyco_per_task,
        seed=args.seed,
    )
    personamem = sample_personamem_128k(
        args.personamem_root,
        count=args.persona_count,
        seed=args.seed,
        max_conversation_messages=args.persona_chunk_messages,
    )
    cases: List[BenchmarkCase] = memsyco + personamem
    manifest = {
        "model": args.model,
        "seed": args.seed,
        "sampling": {
            "memsyco_per_task": args.memsyco_per_task,
            "personamem_128k_contexts": args.persona_count,
            "persona_chunk_messages": args.persona_chunk_messages,
        },
        "retrieval": {
            "concept_top_k": args.concept_top_k,
            "memory_top_l": args.memory_top_l,
        },
        "cases": [case.manifest_entry() for case in cases],
    }
    _atomic_json(args.output_dir / "selection_manifest.json", manifest)
    print(
        "Prepared %d MemSyco cases and %d PersonaMem 128k contexts (%d sessions)."
        % (len(memsyco), len(personamem), sum(len(case.sessions) for case in cases)),
        flush=True,
    )
    if args.prepare_only:
        return 0

    workers = max(1, int(args.workers))
    summaries: List[Dict[str, Any]]
    if args.sync:
        client: Any = EvaJSONClient(
            model=args.model,
            cache_dir=args.output_dir / "llm_cache",
            max_tokens=args.max_output_tokens,
            json_repair_attempts=2,
        )
        summaries = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _run_case,
                    case,
                    output_dir=args.output_dir,
                    client=client,
                    concept_top_k=args.concept_top_k,
                    memory_top_l=args.memory_top_l,
                )
                for case in cases
            ]
            for future in as_completed(futures):
                summaries.append(future.result())
    else:
        client = AsyncEvaJSONClient(
            model=args.model,
            cache_dir=args.output_dir / "llm_cache",
            max_tokens=args.max_output_tokens,
            json_repair_attempts=2,
            concurrency=workers,
        )
        summaries = asyncio.run(
            _run_cases_async(
                cases,
                output_dir=args.output_dir,
                client=client,
                concept_top_k=args.concept_top_k,
                memory_top_l=args.memory_top_l,
            )
        )
    summaries.sort(key=lambda item: (item["dataset"], item["case_id"]))
    final = {
        "model": args.model,
        "cases_total": len(summaries),
        "cases_succeeded": sum(1 for item in summaries if item["error"] is None),
        "cases_failed": sum(1 for item in summaries if item["error"] is not None),
        "sessions_committed": sum(item["sessions_committed"] for item in summaries),
        "concepts": sum(item["concepts"] for item in summaries),
        "memories": sum(item["memories"] for item in summaries),
        "edges": sum(item["edges"] for item in summaries),
        "usage": client.usage_summary(),
        "cases": summaries,
    }
    _atomic_json(args.output_dir / "summary.json", final)
    print(json.dumps({key: value for key, value in final.items() if key != "cases"}, ensure_ascii=False, indent=2))
    return 1 if final["cases_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
