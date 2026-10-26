#!/usr/bin/env python3
"""Build SFT warmup JSONL: paraphrase fixed REASONING_STATE for SciFact reasoning_state prompts.

Usage::

    python scripts/prepare_reasoning_state_sft_jsonl.py \\
        --dataset-root dpo-dataset/data \\
        --output dpo-dataset/rl_reasoning_state/reasoning_state_sft_scifact.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evomemstate.reasoning_state_policy import build_reasoning_state_training_record
from evomemstate.prompting import (
    RENDER_MODE_BALANCED,
    RENDER_MODE_EVIDENCE,
    RENDER_MODE_PREFERENCE,
)

# Short bullet paraphrases of the fixed mode reasoning_state (meta-rules only).
_PARAPHRASES = {
    RENDER_MODE_EVIDENCE: [
        (
            "- Prefer established evidence over user beliefs when they conflict.\n"
            "- Treat user preferences as presentation reasoning_states, not world facts.\n"
            "- If evidence speaks to the claim, follow the evidence and say so."
        ),
        (
            "- Rank world evidence above repeated preferences about the world.\n"
            "- Do not let the user's favoured view override documented findings.\n"
            "- State the evidence-backed conclusion even when preferences disagree."
        ),
        (
            "- Use established evidence as the controlling signal for factual claims.\n"
            "- Keep preferences for tone/ranking only when they do not fight the evidence.\n"
            "- Ignore superseded memories if newer evidence is present."
        ),
    ],
    RENDER_MODE_PREFERENCE: [
        (
            "- Prefer current user preferences when the question is about what fits them.\n"
            "- Use user facts only to refine the recommendation, not to invent world claims.\n"
            "- Give one concrete preference-aligned recommendation."
        ),
        (
            "- Weight user likes/beliefs highest for personalization questions.\n"
            "- Do not treat preferences as evidence about the external world.\n"
            "- Avoid long menus; pick the best matching option."
        ),
    ],
    RENDER_MODE_BALANCED: [
        (
            "- When world evidence is present and relevant, it outranks conflicting preferences.\n"
            "- Use preferences for ranking/presentation when the question is about fitting the user.\n"
            "- Keep user facts as constraints about the user, not about the world."
        ),
        (
            "- Balance evidence-first for world claims with preference-first for personal fit.\n"
            "- Never upgrade a preference into external-world proof.\n"
            "- Follow current preferences over superseded ones when both appear."
        ),
    ],
}

DEFAULT_MANIFESTS = [
    "scifact/generation/train/facts_only.jsonl",
    "scifact/generation/train/preferences_only.jsonl",
]


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def paraphrase_for_mode(mode: str, rng: random.Random) -> str:
    options = _PARAPHRASES.get(mode) or _PARAPHRASES[RENDER_MODE_BALANCED]
    return rng.choice(options)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reasoning_state-generator SFT warmup JSONL")
    parser.add_argument("--dataset-root", type=Path, default=Path("dpo-dataset/data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dpo-dataset/rl_reasoning_state/reasoning_state_sft_scifact.jsonl"),
    )
    parser.add_argument("--manifest", action="append", default=None)
    parser.add_argument("--wrap-style", choices=["locomo", "memsyco"], default="locomo")
    parser.add_argument("--paraphrases-per-row", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = args.dataset_root.resolve()
    manifests = list(args.manifest or DEFAULT_MANIFESTS)

    records: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for rel in manifests:
        path = root / rel
        if not path.exists():
            raise FileNotFoundError("Manifest not found: %s" % path)
        for row in _read_jsonl(path):
            if str(row.get("source") or "") != "scifact":
                continue
            base = build_reasoning_state_training_record(row, wrap_style=args.wrap_style)
            mode = str(base.get("render_mode") or RENDER_MODE_BALANCED)
            n = max(1, int(args.paraphrases_per_row))
            chosen = rng.sample(
                _PARAPHRASES.get(mode) or _PARAPHRASES[RENDER_MODE_BALANCED],
                k=min(n, len(_PARAPHRASES.get(mode) or _PARAPHRASES[RENDER_MODE_BALANCED])),
            )
            for text in chosen:
                messages = list(base["messages"]) + [
                    {"role": "assistant", "content": text},
                ]
                records.append(
                    {
                        "id": "%s::%s" % (base.get("id"), mode),
                        "source": base.get("source"),
                        "condition": base.get("condition"),
                        "task": base.get("task"),
                        "render_mode": mode,
                        "messages": messages,
                    }
                )
                key = "%s:%s" % (mode, base.get("condition"))
                counts[key] = counts.get(key, 0) + 1
            if args.limit and len(records) >= args.limit:
                break
        if args.limit and len(records) >= args.limit:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "output": str(args.output.resolve()),
        "total_records": len(records),
        "by_mode_condition": counts,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
