"""Plain-English summary of a dataset profile.

Uses an LLM when OPENAI_API_KEY is set, otherwise (or on any failure) a
deterministic rule-based summary. Only aggregate statistics are ever sent to
the LLM: raw cell values are stripped first (see strip_raw_values).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from ml_engine.profiling import strip_raw_values

logger = logging.getLogger("aidme.summary")

SYSTEM_PROMPT = (
    "You are a senior data scientist. You receive a statistical profile of a "
    "tabular dataset as JSON. Write a concise summary (max 180 words) for a "
    "business user: what the data looks like, the most important data-quality "
    "problems, which column looks like the prediction target, and the next "
    "cleaning steps you recommend. Use only facts present in the JSON. Do not "
    "invent numbers."
)


def rule_based_summary(profile: dict[str, Any]) -> str:
    o = profile["overview"]
    parts = [
        f"The dataset has {o['n_rows']:,} rows and {o['n_cols']} columns "
        f"({o['missing_cell_pct']}% of cells missing, "
        f"{o['duplicate_rows']} duplicate rows)."
    ]
    issues = profile["issues"]
    serious = [i for i in issues if i["severity"] in {"high", "medium"}]
    if serious:
        bullets = []
        for i in serious[:5]:
            where = f" in '{i['column']}'" if i["column"] else ""
            bullets.append(f"{i['code'].replace('_', ' ')}{where}")
        parts.append("Main data-quality problems: " + "; ".join(bullets) + ".")
    elif issues:
        parts.append("Only minor data-quality issues were found.")
    else:
        parts.append("No data-quality issues were detected.")

    targets = profile["candidate_targets"]
    if targets:
        t = targets[0]
        parts.append(
            f"Most likely prediction target: '{t['column']}' ({t['task']})."
        )
    else:
        parts.append("No obvious prediction target was found; please specify one.")
    return " ".join(parts)


def _llm_summary(profile: dict[str, Any]) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=30)
        payload = json.dumps(strip_raw_values(profile))
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            temperature=0.2,
        )
        text = (response.choices[0].message.content or "").strip()
        return text or None
    except Exception:  # noqa: BLE001 - never fail an upload because of the LLM
        logger.exception("LLM summary failed; falling back to rule-based summary")
        return None


def summarize_profile(profile: dict[str, Any]) -> tuple[str, str]:
    """Return (summary_text, source) where source is 'llm' or 'rules'."""
    text = _llm_summary(profile)
    if text:
        return text, "llm"
    return rule_based_summary(profile), "rules"
