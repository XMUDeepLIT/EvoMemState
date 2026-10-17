"""Run ordered rule chains at each lifecycle hook."""

from __future__ import annotations

from .builtin import run_retrieval  # noqa: F401 — registers built-in rules
from .context import HarnessContext
from .profile import HarnessProfile
from .rules import RuleRegistry


def _run_hook(hook: str, ctx: HarnessContext, profile: HarnessProfile) -> None:
    for rule_id in profile.hook_rules(hook):
        RuleRegistry.get(rule_id)(ctx, profile)
        ctx.applied_rules.append("%s:%s" % (hook, rule_id))


def run_context_pipeline(
    ctx: HarnessContext,
    profile: HarnessProfile | None = None,
) -> HarnessContext:
    """Execute build → retrieve → generate lifecycle with static rules."""
    from .profile import load_profile

    if profile is None:
        profile = load_profile()
    ctx.rules_version = profile.version
    ctx.rules_profile = profile.profile

    _run_hook("on_after_build", ctx, profile)
    _run_hook("on_before_retrieve", ctx, profile)

    run_retrieval(ctx, profile)

    _run_hook("on_after_retrieve", ctx, profile)
    _run_hook("on_before_generation", ctx, profile)
    return ctx
