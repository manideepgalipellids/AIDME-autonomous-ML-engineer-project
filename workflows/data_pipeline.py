"""Data-preparation workflow: Analyst -> Quality -> Preprocess -> Validate.

    START -> analyst -> quality -> preprocess -> validate -> END
                          ^                          |
                          +------- retry (repair) ---+

Every node is a plain function ``state -> partial state update``. The flow is
declared once in ``FLOW`` and executed either by LangGraph (default) or by a
tiny pure-Python loop (``engine="python"``, used by fast unit tests). Both
engines run the exact same nodes and routers.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
from typing_extensions import TypedDict

from agents.data_analyst import choose_target
from agents.data_quality import apply_repairs, plan_cleaning
from ml_engine.preprocessing import apply_plan
from ml_engine.profiling import profile_dataframe
from ml_engine.validation import validate_prepared

MAX_RETRIES_CAP = 3


class PipelineState(TypedDict, total=False):
    # inputs
    df_raw: pd.DataFrame
    instruction: str | None
    user_target: str | None
    user_task: str | None
    max_retries: int
    # produced by the nodes
    profile: dict[str, Any]
    analyst: dict[str, Any]
    target: str
    task: str
    plan: dict[str, Any]
    attempt: int
    df_clean: pd.DataFrame
    changes: list[str]
    validation: dict[str, Any]
    status: str  # running | succeeded | failed
    errors: list[str]
    log: list[str]


def _fail(state: PipelineState, message: str, **extra: Any) -> PipelineState:
    return {
        "status": "failed",
        "errors": [*state.get("errors", []), message],
        "log": [*state.get("log", []), f"FAILED: {message}"],
        **extra,
    }


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def analyst_node(state: PipelineState) -> PipelineState:
    try:
        profile = profile_dataframe(state["df_raw"])
    except ValueError as exc:
        return _fail(state, f"Profiling failed: {exc}")
    try:
        decision = choose_target(
            profile,
            instruction=state.get("instruction"),
            user_target=state.get("user_target"),
            user_task=state.get("user_task"),
        )
    except ValueError as exc:
        return _fail(state, str(exc), profile=profile)
    return {
        "profile": profile,
        "analyst": decision,
        "target": decision["target"],
        "task": decision["task"],
        "attempt": 0,
        "status": "running",
        "log": [
            *state.get("log", []),
            f"Analyst: target '{decision['target']}' ({decision['task']}), "
            f"chosen by {decision['source']}.",
        ],
    }


def quality_node(state: PipelineState) -> PipelineState:
    log = list(state.get("log", []))
    if state.get("plan") is None:
        plan = plan_cleaning(state["profile"], state["target"], state["task"])
        return {"plan": plan, "log": [*log, "Quality: built the cleaning plan."]}

    # A previous validation failed with recoverable errors: try to repair.
    attempt = state.get("attempt", 0) + 1
    plan, applied = apply_repairs(state["plan"], state["validation"]["errors"])
    if not applied:
        return _fail(state, "No automatic repair is available for the validation errors.")
    return {
        "plan": plan,
        "attempt": attempt,
        "log": [*log, f"Quality: retry {attempt}: " + "; ".join(applied) + "."],
    }


def preprocess_node(state: PipelineState) -> PipelineState:
    try:
        clean, changes = apply_plan(state["df_raw"], state["plan"])
    except Exception as exc:  # noqa: BLE001 - a crash must become a clean failure
        return _fail(state, f"Cleaning crashed: {exc}")
    return {
        "df_clean": clean,
        "changes": changes,
        "log": [
            *state.get("log", []),
            f"Preprocess: {len(state['df_raw'])} -> {len(clean)} rows, "
            f"{state['df_raw'].shape[1]} -> {clean.shape[1]} columns.",
        ],
    }


def validate_node(state: PipelineState) -> PipelineState:
    report = validate_prepared(state["df_clean"], state["target"], state["task"])
    log = [*state.get("log", [])]
    if report["passed"]:
        return {
            "validation": report,
            "status": "succeeded",
            "log": [*log, "Validate: all checks passed."],
        }

    messages = [e["message"] for e in report["errors"]]
    can_retry = all(e["recoverable"] for e in report["errors"])
    if not can_retry:
        return {
            "validation": report,
            "status": "failed",
            "errors": [*state.get("errors", []), *messages],
            "log": [*log, f"Validate: failed ({len(messages)} error(s)); not recoverable."],
        }
    if state.get("attempt", 0) >= state.get("max_retries", 0):
        return {
            "validation": report,
            "status": "failed",
            "errors": [*state.get("errors", []), *messages, "Retry limit reached."],
            "log": [*log, "Validate: failed and the retry limit is reached."],
        }
    return {
        "validation": report,
        "status": "running",
        "log": [*log, f"Validate: {len(messages)} recoverable error(s); retrying."],
    }


# --------------------------------------------------------------------------- #
# Routers and flow (single source of truth for both engines)
# --------------------------------------------------------------------------- #
def _continue_or_end(state: PipelineState) -> str:
    return "end" if state.get("status") == "failed" else "continue"


def _retry_or_end(state: PipelineState) -> str:
    return "retry" if state.get("status") == "running" else "end"


# node name -> (function, router, {router label: next node or None for the end})
FLOW: dict[str, tuple[Callable, Callable, dict[str, str | None]]] = {
    "analyst": (analyst_node, _continue_or_end, {"continue": "quality", "end": None}),
    "quality": (quality_node, _continue_or_end, {"continue": "preprocess", "end": None}),
    "preprocess": (preprocess_node, _continue_or_end, {"continue": "validate", "end": None}),
    "validate": (validate_node, _retry_or_end, {"retry": "quality", "end": None}),
}
START_NODE = "analyst"


def build_graph():
    """Compile the LangGraph graph described by FLOW."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "langgraph is not installed. Run `pip install langgraph` "
            "(or use engine='python')."
        ) from exc

    builder = StateGraph(PipelineState)
    builder.add_edge(START, START_NODE)
    for name, (fn, router, targets) in FLOW.items():
        builder.add_node(name, fn)
        builder.add_conditional_edges(
            name, router, {label: (dest or END) for label, dest in targets.items()}
        )
    return builder.compile()


def _run_python(state: PipelineState) -> PipelineState:
    state = dict(state)  # type: ignore[assignment]
    current: str | None = START_NODE
    for _ in range(10 + 4 * MAX_RETRIES_CAP):  # guard against accidental loops
        fn, router, targets = FLOW[current]  # type: ignore[index]
        state.update(fn(state))  # type: ignore[arg-type]
        current = targets[router(state)]
        if current is None:
            return state  # type: ignore[return-value]
    raise RuntimeError("Pipeline did not terminate.")


def run_pipeline(
    df: pd.DataFrame,
    *,
    instruction: str | None = None,
    target: str | None = None,
    task: str | None = None,
    max_retries: int = 2,
    engine: str = "langgraph",
) -> PipelineState:
    """Run the data-preparation workflow and return the final state."""
    initial: PipelineState = {
        "df_raw": df,
        "instruction": instruction,
        "user_target": target,
        "user_task": task,
        "max_retries": max(0, min(int(max_retries), MAX_RETRIES_CAP)),
        "status": "running",
        "errors": [],
        "log": [],
    }
    if engine == "python":
        return _run_python(initial)
    if engine == "langgraph":
        return build_graph().invoke(initial, {"recursion_limit": 50})
    raise ValueError(f"Unknown engine '{engine}'.")


def result_report(final: PipelineState) -> dict[str, Any]:
    """JSON-safe summary of a finished run (no DataFrames)."""
    raw = final["df_raw"]
    clean = final.get("df_clean")
    return {
        "status": final["status"],
        "target": final.get("target"),
        "task": final.get("task"),
        "analyst": final.get("analyst"),
        "plan": final.get("plan"),
        "changes": final.get("changes", []),
        "validation": final.get("validation"),
        "errors": final.get("errors", []),
        "log": final.get("log", []),
        "attempts": final.get("attempt", 0),
        "rows_before": int(len(raw)),
        "cols_before": int(raw.shape[1]),
        "rows_after": int(len(clean)) if clean is not None else None,
        "cols_after": int(clean.shape[1]) if clean is not None else None,
    }
