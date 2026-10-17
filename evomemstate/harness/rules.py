"""Rule registry — executable patches at lifecycle hooks (Harness-R1 style)."""

from __future__ import annotations

from typing import Callable, Dict, List, Protocol

from .context import HarnessContext
from .profile import HarnessProfile

RuleFn = Callable[[HarnessContext, HarnessProfile], None]


class RuleRegistry:
    """Maps rule ids (manifest strings) to Python callables."""

    _rules: Dict[str, RuleFn] = {}

    @classmethod
    def register(cls, rule_id: str, fn: RuleFn) -> RuleFn:
        if rule_id in cls._rules:
            raise ValueError("duplicate rule id: %s" % rule_id)
        cls._rules[rule_id] = fn
        return fn

    @classmethod
    def get(cls, rule_id: str) -> RuleFn:
        try:
            return cls._rules[rule_id]
        except KeyError as exc:
            known = ", ".join(sorted(cls._rules))
            raise KeyError(
                "unknown rule %r; registered: [%s]" % (rule_id, known)
            ) from exc

    @classmethod
    def known(cls) -> List[str]:
        return sorted(cls._rules)


def register_rule(rule_id: str) -> Callable[[RuleFn], RuleFn]:
    """Decorator to register a built-in rule implementation."""

    def decorator(fn: RuleFn) -> RuleFn:
        return RuleRegistry.register(rule_id, fn)

    return decorator


class HookRunner(Protocol):
    def __call__(
        self,
        hook: str,
        ctx: HarnessContext,
        profile: HarnessProfile,
    ) -> None: ...
