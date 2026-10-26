#!/usr/bin/env python3
"""Convert dpo manifests to reasoning_state-generator RL JSONL.

Default: SciFact **QA-form** (dense ``token_f1`` references from
``dpo-dataset/rl/data/scifact_train_refined``) + PersonaMem (locomo wrap).

Usage::

    python scripts/prepare_reasoning_state_rl_jsonl.py \\
        --output dpo-dataset/rl_reasoning_state/slime_reasoning_state_locomo_v3.jsonl

    # Legacy sparse verdict_match SciFact manifests
    python scripts/prepare_reasoning_state_rl_jsonl.py --legacy-scifact-verdict \\
        --output dpo-dataset/rl_reasoning_state/slime_reasoning_state_locomo_verdict.jsonl

    # SciFact QA only
    python scripts/prepare_reasoning_state_rl_jsonl.py --scifact-only \\
        --output dpo-dataset/rl_reasoning_state/slime_reasoning_state_scifact_qa.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evomemstate.reasoning_state_policy import build_reasoning_state_training_record
from evomemstate.training_data import index_personamem_pairings

DEFAULT_MANIFESTS_SCIFACT_LEGACY = [
    "scifact/generation/train/facts_only.jsonl",
    "scifact/generation/train/preferences_only.jsonl",
]

DEFAULT_MANIFESTS_PERSONAMEM = [
    "personamem/generation/single_preference_only.jsonl",
    "personamem/generation/single_no_preference.jsonl",
    "personamem/generation/updated_correct_preference_only.jsonl",
    "personamem/generation/updated_old_preference_only.jsonl",
]

SCIFACT_EVAL_MANIFESTS_LEGACY = [
    "scifact/generation/dev/facts_only.jsonl",
    "scifact/generation/dev/preferences_only.jsonl",
]

DEFAULT_SCIFACT_QA_DIR = Path("dpo-dataset/rl/data/scifact_train_refined")
DEFAULT_SCIFACT_QA_EVAL_DIR = Path("dpo-dataset/rl/data/scifact_dev_refined")


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _load_claims_index(claims_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if claims_path is None or not claims_path.exists():
        return {}
    index: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(claims_path):
        gid = str(row.get("source_group_id") or row.get("claim_id") or "")
        if gid:
            index[gid] = row
    return index


def _enrich_qa_row(
    row: Dict[str, Any], claims_by_group: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Attach facts / references from claims_qa when facts_qa omitted them."""
    out = dict(row)
    gid = str(out.get("source_group_id") or "")
    claim = claims_by_group.get(gid) or {}
    if claim:
        if not out.get("facts") and claim.get("facts"):
            out["facts"] = list(claim["facts"])
        if not str(out.get("reference_correct_answer") or "").strip():
            ref = str(claim.get("reference_correct_answer") or "").strip()
            if ref:
                out["reference_correct_answer"] = ref
                out["reference_incorrect_answers"] = list(
                    claim.get("reference_incorrect_answers") or []
                )
                out["answer_source"] = claim.get("answer_source") or out.get(
                    "answer_source"
                )
        if not out.get("query") and claim.get("query"):
            out["query"] = claim["query"]
        if not out.get("gold_label") and claim.get("gold_label"):
            out["gold_label"] = claim["gold_label"]
    return out


def _qa_manifest_paths(qa_dir: Path) -> List[Path]:
    facts = qa_dir / "facts_qa_refined.jsonl"
    prefs = qa_dir / "preferences_qa_refined.jsonl"
    # Fall back to unreined names if refined missing.
    if not facts.exists():
        facts = qa_dir / "facts_qa.jsonl"
    if not prefs.exists():
        prefs = qa_dir / "preferences_qa.jsonl"
    return [facts, prefs]


def _claims_path(qa_dir: Path) -> Path:
    refined = qa_dir / "claims_qa_refined.jsonl"
    if refined.exists():
        return refined
    return qa_dir / "claims_qa.jsonl"


def _build_records(
    *,
    root: Path,
    manifests: List[str],
    qa_paths: List[Path],
    claims_by_group: Dict[str, Dict[str, Any]],
    sources: Set[str],
    wrap_style: str,
    use_pairing_supersede: bool,
    limit: int,
    pairings: Dict[str, Dict[str, Any]],
    think_mode: bool = False,
    memory_only: bool = False,
    channel_needs: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    records: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}

    def _consume(row: Dict[str, Any]) -> bool:
        nonlocal records, counts
        src = str(row.get("source") or "")
        if sources and src not in sources:
            return False
        pairing: Optional[Dict[str, Any]] = None
        gid = str(row.get("generation_id") or "")
        if gid and gid in pairings:
            pairing = pairings[gid]
        record = build_reasoning_state_training_record(
            row,
            pairing=pairing,
            use_pairing_supersede=use_pairing_supersede,
            wrap_style=wrap_style,
            think_mode=think_mode,
            memory_only=memory_only,
            channel_needs=channel_needs,
        )
        if not record.get("memory_body") or record["memory_body"] == "[NO RETRIEVED MEMORIES]":
            if str(row.get("condition") or "") == "single_no_preference":
                return False
        # Skip empty SciFact facts_qa that failed to resolve evidence.
        if str(row.get("condition") or "") == "facts_qa" and "## Established evidence" not in (
            record.get("memory_body") or ""
        ):
            return False
        records.append(record)
        key = "%s:%s" % (record["source"], record["condition"])
        counts[key] = counts.get(key, 0) + 1
        return True

    for path in qa_paths:
        if not path.exists():
            raise FileNotFoundError("QA manifest not found: %s" % path)
        for row in _read_jsonl(path):
            enriched = _enrich_qa_row(row, claims_by_group)
            _consume(enriched)
            if limit and len(records) >= limit:
                return records, counts

    for rel in manifests:
        path = root / rel
        if not path.exists():
            raise FileNotFoundError("Manifest not found: %s" % path)
        for row in _read_jsonl(path):
            _consume(row)
            if limit and len(records) >= limit:
                return records, counts
    return records, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reasoning_state-generator RL JSONL")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dpo-dataset/data"),
        help="Root for PersonaMem / legacy SciFact generation manifests",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dpo-dataset/rl_reasoning_state/slime_reasoning_state_locomo_v3.jsonl"),
    )
    parser.add_argument("--manifest", action="append", default=None)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Keep rows with this source (repeatable). Default: scifact + personamem_v2",
    )
    parser.add_argument(
        "--scifact-only",
        action="store_true",
        help="Only SciFact manifests/sources",
    )
    parser.add_argument(
        "--personamem-only",
        action="store_true",
        help="Only PersonaMem manifests/sources",
    )
    parser.add_argument(
        "--scifact-qa-dir",
        type=Path,
        default=DEFAULT_SCIFACT_QA_DIR,
        help="Dir with facts/preferences/claims *_qa*_refined.jsonl (token_f1)",
    )
    parser.add_argument(
        "--legacy-scifact-verdict",
        action="store_true",
        help="Use old verdict_match SciFact generation manifests instead of QA",
    )
    parser.add_argument(
        "--fever-qa-dir",
        type=Path,
        default=None,
        help="Optional FEVER QA dir (same layout as scifact_train_refined)",
    )
    parser.add_argument(
        "--wrap-style",
        choices=["locomo", "memsyco"],
        default="locomo",
    )
    parser.add_argument("--no-pairing-supersede", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="If set, shuffle records before --limit (for PersonaMem eval holdout)",
    )
    parser.add_argument(
        "--think-mode",
        action="store_true",
        help="Use REASONING_STATE_SYSTEM_PROMPT_THINK (<think>/<reasoning_state> contract) in messages",
    )
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="ReasoningState prompt omits question/claim (memories only)",
    )
    parser.add_argument(
        "--channel-needs",
        action="store_true",
        help="Inject fact_need/preference_need into reasoning_state prompt (use with --memory-only)",
    )
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    qa_paths: List[Path] = []
    claims_by_group: Dict[str, Dict[str, Any]] = {}
    manifests: List[str] = []

    use_qa = not args.legacy_scifact_verdict and not args.personamem_only
    if args.manifest:
        manifests = list(args.manifest)
        use_qa = False
    elif args.personamem_only:
        manifests = list(DEFAULT_MANIFESTS_PERSONAMEM)
    elif args.legacy_scifact_verdict:
        manifests = list(DEFAULT_MANIFESTS_SCIFACT_LEGACY)
        if not args.scifact_only:
            manifests = manifests + list(DEFAULT_MANIFESTS_PERSONAMEM)
    else:
        # Default: SciFact QA (+ optional FEVER QA) + PersonaMem
        qa_dir = args.scifact_qa_dir
        if not qa_dir.is_absolute():
            qa_dir = (_REPO / qa_dir).resolve()
        qa_paths.extend(_qa_manifest_paths(qa_dir))
        claims_by_group.update(_load_claims_index(_claims_path(qa_dir)))
        if args.fever_qa_dir is not None:
            fever_dir = args.fever_qa_dir
            if not fever_dir.is_absolute():
                fever_dir = (_REPO / fever_dir).resolve()
            qa_paths.extend(_qa_manifest_paths(fever_dir))
            claims_by_group.update(_load_claims_index(_claims_path(fever_dir)))
        if not args.scifact_only:
            manifests = list(DEFAULT_MANIFESTS_PERSONAMEM)

    if args.source:
        sources: Set[str] = set(args.source)
    elif args.scifact_only:
        sources = {"scifact", "fever"}
    elif args.personamem_only:
        sources = {"personamem_v2"}
    else:
        sources = {"scifact", "fever", "personamem_v2"}

    pairing_path = root / "personamem/generation/pairing_plan.jsonl"
    pairings: Dict[str, Dict[str, Any]] = {}
    if pairing_path.exists():
        pairings = index_personamem_pairings(pairing_path)

    limit_for_build = 0 if args.shuffle_seed is not None else args.limit
    records, counts = _build_records(
        root=root,
        manifests=manifests,
        qa_paths=qa_paths,
        claims_by_group=claims_by_group,
        sources=sources,
        wrap_style=args.wrap_style,
        use_pairing_supersede=not args.no_pairing_supersede,
        limit=limit_for_build,
        pairings=pairings,
        think_mode=bool(args.think_mode),
        memory_only=bool(args.memory_only),
        channel_needs=bool(args.channel_needs),
    )
    if args.shuffle_seed is not None:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(records)
        if args.limit:
            records = records[: args.limit]
        counts = {}
        for record in records:
            key = "%s:%s" % (record["source"], record["condition"])
            counts[key] = counts.get(key, 0) + 1

    reward_types: Dict[str, int] = {}
    for record in records:
        rtype = str((record.get("reward_spec") or {}).get("type") or "")
        reward_types[rtype] = reward_types.get(rtype, 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "output": str(args.output.resolve()),
        "total_records": len(records),
        "by_source_condition": counts,
        "by_reward_spec_type": reward_types,
        "wrap_style": args.wrap_style,
        "think_mode": bool(args.think_mode),
        "memory_only": bool(args.memory_only),
        "channel_needs": bool(args.channel_needs),
        "sources": sorted(sources),
        "scifact_mode": "qa_token_f1" if use_qa else "legacy_or_custom",
        "qa_paths": [str(p) for p in qa_paths],
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
