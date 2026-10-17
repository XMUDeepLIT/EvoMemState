"""Build EvoMemState memory graphs from LoCoMo-Refined conversations and answer
every question in questions.jsonl, writing predictions.jsonl in the exact
format LoCoMo-Refined's own src/evaluate.py expects.

The runner uses ontology-guided retrieval and a request-specific Reasoning
State from the online policy. Set ``EVOMEMSTATE_REASONING_STATE_ENABLED=1`` and ``REASONING_STATE_*``.

LoCoMo-Refined's scorer (evaluate.py) is untouched -- this script only
produces {"qa_id", "predicted_answer"} records for it to consume. Usage:

    python -m evomemstate.run_locomo \
        --questions-path <LOCOMO>/data/public/questions.jsonl \
        --conversations-path <LOCOMO>/data/public/conversations.jsonl \
        --output-path <LOCOMO>/outputs/predictions.jsonl


"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .builder import MemoryBuilder
from .embedding import embedding_client_from_env
from .extractor import OntologyGuidedExtractor
from .llm import OpenAIJSONClient
from .models import OntologyState
from .ontology import OntologyStore
from .prompting import render_context
from .retrieval import retrieve

# Bump when a change would invalidate previously built graphs.
# v2: extraction prompt rewritten for higher recall (both speakers, dated facts,
# 32-cap) and render_context now emits memory.time. Both change graph contents
# and the rendered context, so v1 graphs must be rebuilt.
GRAPH_VERSION = "v2"

_PRINT_LOCK = threading.Lock()

ANSWER_SYSTEM_PROMPT = """You answer questions about a conversation between two \
people, using only the memory context provided below. Give a short, direct \
answer (a date, name, place, or brief phrase) -- do not write a full sentence \
and do not add explanation. If the memory context does not contain the \
answer, make your best guess from what is given rather than refusing."""


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _build_settings(model: str, concept_top_k: int, memory_top_l: int) -> Dict[str, Any]:
    return {
        "model": model,
        "slot_policy": os.environ.get("EVOMEMSTATE_SLOT_POLICY", "compatible"),
        "concept_top_k": concept_top_k,
        "memory_top_l": memory_top_l,
        "version": GRAPH_VERSION,
    }


def _conversation_sessions(conversation: Dict[str, Any], sample_id: str) -> List[Dict[str, Any]]:
    user_id = "locomo:%s" % sample_id
    sessions: List[Dict[str, Any]] = []
    for session in conversation.get("sessions", []):
        session_index = session.get("session_index")
        messages = []
        for message in session.get("messages", []):
            text = (message.get("text") or "").strip()
            if not text:
                continue
            speaker = message.get("speaker") or "unknown"
            dia_id = message.get("dia_id") or "m%d" % len(messages)
            messages.append(
                {
                    "turn_id": "%s_%s" % (sample_id, dia_id),
                    "role": "user",
                    "content": "%s: %s" % (speaker, text),
                }
            )
        if not messages:
            continue
        sessions.append(
            {
                "session_id": "%s_s%03d" % (sample_id, int(session_index or 0)),
                "timestamp": session.get("date_time") or "",
                "user_id": user_id,
                "messages": messages,
            }
        )
    sessions.sort(key=lambda item: item["session_id"])
    return sessions


def _graph_key(sessions: List[Dict[str, Any]], settings: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for session in sessions:
        digest.update(session["session_id"].encode("utf-8"))
        digest.update(b"\0")
        for message in session["messages"]:
            digest.update(" ".join(message["content"].split()).encode("utf-8"))
            digest.update(b"\0")
    digest.update(json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()[:24]


def _graph_root(override: Optional[str]) -> Path:
    if override:
        return Path(override)
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "outputs" / "locomo_graphs"


def _build_or_load_graph(
    sample_id: str,
    conversation: Optional[Dict[str, Any]] = None,
    *,
    client: OpenAIJSONClient,
    settings: Dict[str, Any],
    graph_root: Path,
    sessions: Optional[List[Dict[str, Any]]] = None,
    user_id: Optional[str] = None,
) -> OntologyStore:
    if sessions is None:
        if conversation is None:
            raise ValueError("Provide conversation or sessions for graph build")
        sessions = _conversation_sessions(conversation, sample_id)
    key = _graph_key(sessions, settings)
    graph_dir = graph_root / sample_id
    state_path = graph_dir / "ontology.json"
    user_id = user_id or ("locomo:%s" % sample_id)

    store = OntologyStore.load(state_path, user_id, slot_policy=settings["slot_policy"])
    if state_path.exists():
        # Invalidate the cache when settings (model, retrieval knobs, or GRAPH_VERSION)
        # change: a v1 graph built under a different extraction prompt or render path is
        # not comparable to a v2 run, so a stale state file must be discarded rather than
        # reused. build_settings.json records the key the on-disk graph was built under.
        settings_path = graph_dir / "build_settings.json"
        expected_key = key
        if settings_path.exists():
            try:
                prior = json.loads(settings_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - corrupt settings file -> force rebuild
                prior = {}
            if prior.get("graph_key") != expected_key or prior.get("version") != settings["version"]:
                with _PRINT_LOCK:
                    print(
                        "[graph] %s cache stale (key/version changed), rebuilding" % sample_id,
                        flush=True,
                    )
                store = OntologyStore(
                    OntologyState.empty(user_id), slot_policy=settings["slot_policy"]
                )
            else:
                return store
        else:
            return store

    builder = MemoryBuilder(
        store,
        OntologyGuidedExtractor(
            client,
            top_k=settings["concept_top_k"],
            memory_top_l=settings["memory_top_l"],
        ),
    )
    error: Optional[Dict[str, str]] = None
    for index, session in enumerate(sessions, start=1):
        with _PRINT_LOCK:
            print(
                "[graph] %s building session %d/%d (%s)..." % (sample_id, index, len(sessions), session["session_id"]),
                flush=True,
            )
        try:
            builder.build_session(session)
        except Exception as exc:  # noqa: BLE001 - malformed extraction output, skip and continue
            error = {
                "session_id": str(session["session_id"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            with _PRINT_LOCK:
                print(
                    "[graph] %s WARNING: session %s failed (%s: %s), continuing with remaining sessions"
                    % (sample_id, session["session_id"], error["error_type"], error["error"]),
                    flush=True,
                )
            continue
        store.save(state_path)
    if error is not None:
        _atomic_json(graph_dir / "errors.json", error)
    _atomic_json(graph_dir / "build_settings.json", {**settings, "graph_key": key})
    with _PRINT_LOCK:
        print(
            "[graph] %s sessions=%d/%d concepts=%d memories=%d edges=%d"
            % (
                sample_id,
                len(store.state.processed_sessions),
                len(sessions),
                len(store.state.concepts),
                len(store.state.memories),
                len(store.state.edges),
            ),
            flush=True,
        )
    return store


class _AnswerClient:
    """Plain-text completion client with disk caching, mirroring OpenAIJSONClient's cache."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        cache_dir: Optional[Path] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.system_prompt = system_prompt or ANSWER_SYSTEM_PROMPT
        self.client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, messages: List[Dict[str, str]]) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(
            json.dumps({"model": self.model, "messages": messages}, ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()
        return self.cache_dir / (key + ".txt")

    @staticmethod
    def _strip_thinking(text: str) -> str:
        # Qwen3 may emit <think>...</think> even when we ask to disable thinking.
        cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
        if cleaned:
            return cleaned
        return (text or "").strip()

    def answer(self, question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "Memory context:\n%s\n\nQuestion: %s" % (context, question)},
        ]
        cache_path = self._cache_path(messages)
        if cache_path is not None and cache_path.exists():
            return self._strip_thinking(cache_path.read_text(encoding="utf-8"))

        last_error: Optional[Exception] = None
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        # Local Qwen3-vLLM: disable chain-of-thought so LoCoMo answers stay short.
        if "qwen3" in self.model.lower() or "qwen/qwen3" in self.model.lower():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(**kwargs)
                text = self._strip_thinking(response.choices[0].message.content or "")
                if cache_path is not None:
                    cache_path.write_text(text, encoding="utf-8")
                return text
            except Exception as exc:  # noqa: BLE001 - network/provider path
                last_error = exc
        raise RuntimeError("Answer generation failed after retries") from last_error


def build_question_context(
    question: Dict[str, Any],
    *,
    store: OntologyStore,
    extraction_client: OpenAIJSONClient,
    embedding_client: Any,
    concept_top_k: int,
    top_k: int,
    reasoning_state_cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Retrieve relational evidence and generate the request's Reasoning State."""
    half = max(1, top_k // 2)
    result = retrieve(
        store,
        question["question"],
        client=extraction_client,
        embedding_client=embedding_client,
        concept_top_k=concept_top_k,
        fact_top_n=top_k - half,
        preference_top_n=half,
        flat=False,
        latest_wins=True,
    )
    from .reasoning_state import render_with_reasoning_state

    context, reasoning_state_meta = render_with_reasoning_state(
        result,
        question["question"],
        cache_dir=reasoning_state_cache_dir,
    )
    return {
        "context_text": context,
        "method": "EvoMemState",
        "n_retrieved": len(result.all_memories()),
        "n_facts": len(result.facts),
        "n_preferences": len(result.preferences),
        "fact_need": float(result.fact_need),
        "preference_need": float(result.preference_need),
        "embedding_used": bool(result.embedding_used),
        "latest_wins_mode": getattr(result, "latest_wins_mode", "soft"),
        "online_reasoning_state": bool(reasoning_state_meta.get("online_reasoning_state")),
        "online_reasoning_state_ok": bool(reasoning_state_meta.get("reasoning_state_ok")),
    }


def _answer_question(
    question: Dict[str, Any],
    *,
    store: OntologyStore,
    extraction_client: OpenAIJSONClient,
    answer_client: _AnswerClient,
    embedding_client: Any,
    concept_top_k: int,
    top_k: int,
    audit_path: Optional[Path] = None,
    reasoning_state_cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Answer one LoCoMo question (context build + answer generation)."""
    built = build_question_context(
        question,
        store=store,
        extraction_client=extraction_client,
        embedding_client=embedding_client,
        concept_top_k=concept_top_k,
        top_k=top_k,
        reasoning_state_cache_dir=reasoning_state_cache_dir,
    )
    if audit_path is not None:
        try:
            row = {
                "qa_id": question.get("qa_id"),
                "sample_id": question.get("sample_id"),
                "question": question.get("question"),
                **{k: v for k, v in built.items() if k != "context_text"},
            }
            with _PRINT_LOCK:
                with audit_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass

    answer = answer_client.answer(question["question"], built["context_text"])
    return {"qa_id": question["qa_id"], "predicted_answer": answer}


def build_parser() -> argparse.ArgumentParser:
    locomo_root_default = Path(os.environ.get("LOCOMO_BENCHMARK_ROOT", "")) if os.environ.get(
        "LOCOMO_BENCHMARK_ROOT"
    ) else None
    parser = argparse.ArgumentParser(description="Build LoCoMo-Refined predictions.jsonl using EvoMemState.")
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=(locomo_root_default / "data" / "public" / "questions.jsonl") if locomo_root_default else None,
        required=locomo_root_default is None,
    )
    parser.add_argument(
        "--conversations-path",
        type=Path,
        default=(locomo_root_default / "data" / "public" / "conversations.jsonl") if locomo_root_default else None,
        required=locomo_root_default is None,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=(locomo_root_default / "outputs" / "predictions.jsonl") if locomo_root_default else None,
        required=locomo_root_default is None,
    )
    parser.add_argument("--graph-root", default=os.environ.get("EVOMEMSTATE_GRAPH_ROOT_LOCOMO"))
    parser.add_argument(
        "--model",
        default=os.environ.get("MEMORY_LLM_MODEL", "deepseek-v4-flash"),
        help="Build / decomposition model (MEMORY_LLM_MODEL).",
    )
    parser.add_argument(
        "--answer-model",
        default=None,
        help="Answer-only model (default: LOCOMO_ANSWER_MODEL / ANSWER_MODEL, else --model).",
    )
    parser.add_argument(
        "--answer-base-url",
        default=None,
        help="Answer API base URL (default: LOCOMO_ANSWER_BASE_URL / GEN_B_BASE_URL / MEMORY_BASE_URL)",
    )
    parser.add_argument(
        "--answer-api-key",
        default=None,
        help="Answer API key (default: LOCOMO_ANSWER_API_KEY / GEN_B_API_KEY / MEMORY_API_KEY)",
    )
    parser.add_argument("--concept-top-k", type=int, default=int(os.environ.get("EVOMEMSTATE_CONCEPT_TOP_K", "8")))
    parser.add_argument("--memory-top-l", type=int, default=int(os.environ.get("EVOMEMSTATE_MEMORY_TOP_L", "40")))
    parser.add_argument("--retrieval-top-k", type=int, default=10, help="Total facts+preferences retrieved per question")
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=None,
        help="Optional JSONL audit of retrieval and Reasoning State decisions",
    )
    parser.add_argument("--sample-ids", default=None, help="Comma-separated conversation sample_ids to restrict to")
    parser.add_argument("--limit", type=int, default=None, help="Only answer the first N questions (after filtering)")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent question-answering workers")
    parser.add_argument(
        "--prepare-workers",
        type=int,
        default=int(os.environ.get("LOCOMO_PREPARE_WORKERS") or os.environ.get("PREBUILD_WORKERS") or "4"),
        help="Concurrent conversation graph builds (DeepSeek extraction). Sessions within a conversation stay serial.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Only build graphs, skip answering questions")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    api_key = os.environ.get("MEMORY_API_KEY", "")
    base_url = os.environ.get("MEMORY_BASE_URL", "")
    if not api_key or not base_url:
        raise RuntimeError("Set MEMORY_API_KEY and MEMORY_BASE_URL (see .env).")

    questions = _read_jsonl(args.questions_path)
    conversations = {record["sample_id"]: record for record in _read_jsonl(args.conversations_path)}

    if args.sample_ids:
        wanted = {value.strip() for value in args.sample_ids.split(",") if value.strip()}
        questions = [item for item in questions if item["sample_id"] in wanted]
        conversations = {key: value for key, value in conversations.items() if key in wanted}

    if args.limit is not None:
        questions = questions[: args.limit]

    settings = _build_settings(args.model, args.concept_top_k, args.memory_top_l)
    graph_root = _graph_root(args.graph_root)

    # Answer model can differ from the build and decomposition model.
    # Do NOT auto-pick GEN_B_MODEL from .env — that is always set for MemSyco and
    # would silently change LoCoMo answer quality. Use --answer-model or LOCOMO_ANSWER_*.
    answer_model = (
        args.answer_model
        or os.environ.get("LOCOMO_ANSWER_MODEL")
        or os.environ.get("ANSWER_MODEL")
        or args.model
    )
    answer_base_url = (
        args.answer_base_url
        or os.environ.get("LOCOMO_ANSWER_BASE_URL")
        or os.environ.get("ANSWER_BASE_URL")
        or os.environ.get("GEN_B_BASE_URL")
        or base_url
    )
    answer_api_key = (
        args.answer_api_key
        or os.environ.get("LOCOMO_ANSWER_API_KEY")
        or os.environ.get("ANSWER_API_KEY")
        or os.environ.get("GEN_B_API_KEY")
        or api_key
    )
    if not answer_api_key or not answer_base_url:
        raise RuntimeError("Answer client needs LOCOMO_ANSWER_* / GEN_B_* or MEMORY_* credentials.")

    extraction_client = OpenAIJSONClient(
        model=args.model,
        api_key=api_key,
        base_url=base_url,
        cache_dir=graph_root / "llm_cache",
    )
    method_slug = "EvoMemState"
    answer_model_slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in answer_model)
    answer_cache_name = "answer_cache_%s_%s" % (method_slug, answer_model_slug)
    answer_client = _AnswerClient(
        model=answer_model,
        api_key=answer_api_key,
        base_url=answer_base_url,
        cache_dir=graph_root / answer_cache_name,
    )
    with _PRINT_LOCK:
        print(
            "[config] build/decomp=%s  answer=%s (%s)"
            % (args.model, answer_model, answer_base_url),
            flush=True,
        )
    embedding_client = embedding_client_from_env(cache_dir=graph_root / "embedding_cache")
    if args.audit_path is not None:
        args.audit_path.parent.mkdir(parents=True, exist_ok=True)
        # Fresh audit file per run so re-runs do not append stale rows.
        args.audit_path.write_text("", encoding="utf-8")

    needed_sample_ids = sorted({item["sample_id"] for item in questions})
    stores: Dict[str, OntologyStore] = {}
    prepare_workers = max(1, int(args.prepare_workers))

    def _prepare_one(sample_id: str) -> tuple[str, Optional[OntologyStore]]:
        conversation = conversations.get(sample_id)
        if conversation is None:
            with _PRINT_LOCK:
                print("[graph] WARNING: no conversation record for %s, skipping" % sample_id, flush=True)
            return sample_id, None
        store = _build_or_load_graph(
            sample_id,
            conversation,
            client=extraction_client,
            settings=settings,
            graph_root=graph_root,
        )
        return sample_id, store

    if prepare_workers <= 1 or len(needed_sample_ids) <= 1:
        for sample_id in needed_sample_ids:
            sid, store = _prepare_one(sample_id)
            if store is not None:
                stores[sid] = store
    else:
        with _PRINT_LOCK:
            print(
                "[graph] preparing %d conversations with %d workers..."
                % (len(needed_sample_ids), prepare_workers),
                flush=True,
            )
        with ThreadPoolExecutor(max_workers=prepare_workers) as pool:
            futures = [pool.submit(_prepare_one, sample_id) for sample_id in needed_sample_ids]
            for future in as_completed(futures):
                sid, store = future.result()
                if store is not None:
                    stores[sid] = store

    if args.prepare_only:
        return 0

    answerable = [item for item in questions if item["sample_id"] in stores]
    predictions: List[Optional[Dict[str, Any]]] = [None] * len(answerable)

    def _work(index: int, question: Dict[str, Any]) -> None:
        store = stores[question["sample_id"]]
        prediction = _answer_question(
            question,
            store=store,
            extraction_client=extraction_client,
            answer_client=answer_client,
            embedding_client=embedding_client,
            concept_top_k=args.concept_top_k,
            top_k=args.retrieval_top_k,
            audit_path=args.audit_path,
            reasoning_state_cache_dir=graph_root / "reasoning_state_cache",
        )
        predictions[index] = prediction
        with _PRINT_LOCK:
            print(
                "[answer:%s] %s -> %r"
                % (method_slug, prediction["qa_id"], prediction["predicted_answer"][:80]),
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(_work, index, question) for index, question in enumerate(answerable)]
        for future in as_completed(futures):
            future.result()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            if prediction is None:
                continue
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    print(
        "Wrote %d predictions to %s (%d questions skipped: missing conversation)"
        % (sum(1 for item in predictions if item is not None), args.output_path, len(questions) - len(answerable)),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
