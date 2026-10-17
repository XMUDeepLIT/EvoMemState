"""Load static rule profiles (SKILLOPT-style external state, no optimizer)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping


DEFAULT_PROFILE_NAME = "default"
HOOK_NAMES = (
    "on_after_build",
    "on_before_retrieve",
    "on_after_retrieve",
    "on_before_generation",
)


@dataclass(frozen=True)
class ArmFlags:
    """Method switches for retrieval and rendering."""

    flat: bool = False
    latest_wins: bool = True
    decompose: bool | str = "env"  # True/False or "env" -> EVOMEMSTATE_DECOMPOSE


@dataclass(frozen=True)
class HarnessProfile:
    version: str
    profile: str
    arms: Mapping[str, ArmFlags]
    hooks: Mapping[str, tuple[str, ...]]

    def hook_rules(self, hook: str) -> tuple[str, ...]:
        return self.hooks.get(hook, ())

    def arm_for(self, method: str) -> ArmFlags:
        return self.arms.get(method, self.arms.get("EvoMemState", ArmFlags()))


def _rules_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "rules"


def _parse_arm(raw: Dict[str, Any]) -> ArmFlags:
    decompose = raw.get("decompose", "env")
    if isinstance(decompose, str) and decompose not in {"env"}:
        decompose = decompose.lower() in {"1", "true", "yes"}
    return ArmFlags(
        flat=bool(raw.get("flat", False)),
        latest_wins=bool(raw.get("latest_wins", True)),
        decompose=decompose,
    )


def _parse_profile(data: Dict[str, Any]) -> HarnessProfile:
    arms = {
        name: _parse_arm(spec)
        for name, spec in (data.get("arms") or {}).items()
    }
    hooks = {
        hook: tuple(data.get("hooks", {}).get(hook, []))
        for hook in HOOK_NAMES
    }
    return HarnessProfile(
        version=str(data.get("version", "0")),
        profile=str(data.get("profile", DEFAULT_PROFILE_NAME)),
        arms=arms,
        hooks=hooks,
    )


def load_profile(name: str | None = None) -> HarnessProfile:
    """Load ``evomemstate/rules/<name>.json`` (default: env or ``default``)."""
    chosen = (
        name
        or os.environ.get("EVOMEMSTATE_RULES_PROFILE", "").strip()
        or DEFAULT_PROFILE_NAME
    )
    path = _rules_dir() / ("%s.json" % chosen)
    if not path.exists():
        raise FileNotFoundError("harness profile not found: %s" % path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return _parse_profile(data)
