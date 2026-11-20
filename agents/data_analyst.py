"""Data Analyst agent: decides which column to predict and which ML task it is.

The LLM (optional) may only *choose among existing columns*. Its answer is
validated against the dataset, and the task type is always derived
deterministically from the column profile. If the LLM is unavailable or answers
badly, the heuristic from the profiler is used.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("aidme.analyst")

CLASSIFICATION_KINDS = {"binary", "boolean", "categorical"}
NUMERIC_KINDS = {"numeric", "numeric_as_text"}
MAX_CLASSES_FOR_NUMERIC_CLASSIFICATION = 10

SYSTEM_PROMPT = (
    "You help choose the prediction target for a machine-learning project. "
    "You get the user's instruction and a list of dataset columns. Pick the one "
    "column the user wants to predict. Reply with JSON only: "
    '{"target": "<exact column name or null>", "reason": "<one short sentence>"}. '
    "Use null if the instruction does not make the target clear."
)


def infer_task(col: dict[str, Any]) -> str:
    """Deterministic task type for a column profile."""
    kind = col["inferred_type"]
    if kind in CLASSIFICATION_KINDS:
        return "classification"
    if kind in NUMERIC_KINDS:
        few = col["n_unique"] <= MAX_CLASSES_FOR_NUMERIC_CLASSIFICATION
        return "classification" if few else "regression"
    raise ValueError(
        f"Column '{col['name']}' ({kind.replace('_', ' ')}) cannot be used as a prediction target."
    )


def _resolve_column(name: str, columns: list[dict[str, Any]]) -> dict[str, Any]:
    for col in columns:
        if col["name"] == name:
            return col
    wanted = name.strip().lower()
    for col in columns:
        if col["name"].strip().lower() == wanted:
            return col
    available = ", ".join(c["name"] for c in columns[:15])
    more = "..." if len(columns) > 15 else ""
    raise ValueError(f"Column '{name}' was not found. Available columns: {available}{more}")


def _llm_choose(profile: dict[str, Any], instruction: str) -> dict[str, Any] | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        payload = {
            "instruction": instruction,
            "columns": [
                {
                    "name": c["name"],
                    "type": c["inferred_type"],
                    "n_unique": c["n_unique"],
                    "missing_pct": c["missing_pct"],
                }
                for c in profile["columns"]
            ],
        }
        client = OpenAI(api_key=api_key, timeout=30)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content or "{}")
    except Exception:  # noqa: BLE001 - never fail the pipeline because of the LLM
        logger.exception("LLM target selection failed; using heuristic")
        return None


def choose_target(
    profile: dict[str, Any],
    instruction: str | None = None,
    user_target: str | None = None,
    user_task: str | None = None,
) -> dict[str, Any]:
    """Return {"target", "task", "source", "reason"} or raise ValueError."""
    columns = profile["columns"]
    chosen: dict[str, Any] | None = None

    if user_target:
        col = _resolve_column(user_target, columns)
        chosen = {
            "target": col["name"],
            "task": infer_task(col),
            "source": "user",
            "reason": "Target specified by the user.",
        }

    if chosen is None and instruction and instruction.strip():
        answer = _llm_choose(profile, instruction.strip())
        name = (answer or {}).get("target")
        if isinstance(name, str):
            match = next((c for c in columns if c["name"] == name), None)
            if match is not None:
                try:
                    chosen = {
                        "target": match["name"],
                        "task": infer_task(match),
                        "source": "llm",
                        "reason": str(answer.get("reason", ""))[:300]
                        or "Chosen by the language model from the instruction.",
                    }
                except ValueError:
                    chosen = None  # LLM picked an unusable column; fall through

    if chosen is None:
        candidates = profile["candidate_targets"]
        if not candidates:
            raise ValueError(
                "Could not identify a target column automatically. Please choose one."
            )
        top = candidates[0]
        chosen = {
            "target": top["column"],
            "task": top["task"],
            "source": "heuristic",
            "reason": top["reason"],
        }

    if user_task and user_task != chosen["task"]:
        if user_task not in {"classification", "regression"}:
            raise ValueError(f"Unknown task '{user_task}'. Use classification or regression.")
        col = _resolve_column(chosen["target"], columns)
        if user_task == "regression" and col["inferred_type"] not in NUMERIC_KINDS:
            raise ValueError(f"Column '{col['name']}' is not numeric, so it cannot be a regression target.")
        chosen["task"] = user_task
        chosen["reason"] += f" Task set to {user_task} by the user."
    return chosen
