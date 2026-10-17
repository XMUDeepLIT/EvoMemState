"""Static lifecycle harness for EvoMemState.

Inspired by Harness-R1 hook overlays and SKILLOPT skill documents:
rules are versioned, auditable units registered at fixed lifecycle points.
No optimizer / RL engineer is wired yet — profiles are static JSON manifests.
"""

from .context import HarnessContext
from .pipeline import run_context_pipeline
from .profile import HarnessProfile, load_profile
from .rules import RuleRegistry, register_rule

__all__ = [
    "HarnessContext",
    "HarnessProfile",
    "RuleRegistry",
    "load_profile",
    "register_rule",
    "run_context_pipeline",
]
