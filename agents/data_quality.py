"""Data Quality agent: plans the cleaning and repairs it when validation fails.

Deterministic on purpose. Which repair answers which validation error is an
explicit, auditable table, not a free-form LLM decision.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ml_engine.preprocessing import build_plan

# validation error code -> (plan flag to switch on, description for the log)
REPAIRS: dict[str, tuple[str, str]] = {
    "inf_values": ("replace_inf", "replace infinite values with missing values"),
}


def plan_cleaning(profile: dict[str, Any], target: str, task: str) -> dict[str, Any]:
    return build_plan(profile, target, task)


def apply_repairs(
    plan: dict[str, Any], errors: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    """Return (new_plan, descriptions of repairs applied). Empty list = nothing to try."""
    new_plan = deepcopy(plan)
    applied: list[str] = []
    for err in errors:
        repair = REPAIRS.get(err["code"])
        if err.get("recoverable") and repair and not new_plan.get(repair[0]):
            new_plan[repair[0]] = True
            applied.append(repair[1])
    return new_plan, applied
