"""Deterministic dataset profiling.

This module has NO dependency on any LLM. It turns a DataFrame into a plain,
JSON-serialisable dict that later agents (and the LLM summary) can consume.

Design rule for the whole project: reliable, testable Python does the data
work; LLM agents only decide what to call and explain the results.
"""

from __future__ import annotations

import io
import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROFILE_VERSION = 1

# Keys that contain raw cell values. They are removed before a profile is sent
# to an external LLM (data minimisation / GDPR).
RAW_VALUE_KEYS = {"sample_values", "top_values", "label_variants"}

MISSING_MEDIUM_PCT = 5.0
MISSING_HIGH_PCT = 40.0
NUMERIC_AS_TEXT_SHARE = 0.9
OUTLIER_MEDIUM_PCT = 5.0
OUTLIER_LOW_PCT = 1.0
HIGH_CARDINALITY_UNIQUE = 50
CATEGORICAL_MAX_UNIQUE = 20
TARGET_NAME_HINT = re.compile(
    r"(churn|target|label|class|outcome|fraud|default|subscri|survived|"
    r"price|sales|revenue|amount|score)",
    re.IGNORECASE,
)

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_dataframe(content: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded bytes into a DataFrame. Raises ValueError on bad input."""
    ext = Path(filename).suffix.lower()
    if not content:
        raise ValueError("The uploaded file is empty.")

    if ext == ".xlsx":
        try:
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        except Exception as exc:  # noqa: BLE001 - surface a clean message
            raise ValueError(f"Could not read the Excel file: {exc}") from exc
    elif ext in {".csv", ".tsv", ".txt"}:
        df = _read_delimited(content)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .csv, .tsv or .xlsx.")

    if df.shape[1] == 0 or df.shape[0] == 0:
        raise ValueError("The dataset has no rows or no columns.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_delimited(content: bytes) -> pd.DataFrame:
    text = None
    for encoding in ("utf-8-sig", "latin-1"):  # latin-1 never fails
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    assert text is not None

    # sep=None sniffs the delimiter, which matters for German ';' CSV exports.
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        if df.shape[1] > 1:
            return df
    except Exception:  # noqa: BLE001 - fall back to the plain parser
        pass
    try:
        return pd.read_csv(io.StringIO(text))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not parse the file as CSV: {exc}") from exc


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clean(value: Any) -> Any:
    """Make a scalar JSON-safe (no numpy types, no NaN/inf)."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _numeric_stats(num: pd.Series) -> dict[str, Any]:
    num = num.replace([np.inf, -np.inf], np.nan).dropna()
    if num.empty:
        return {}
    q1, med, q3 = (float(x) for x in num.quantile([0.25, 0.5, 0.75]))
    iqr = q3 - q1
    outliers = 0
    if iqr > 0:
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = int(((num < lo) | (num > hi)).sum())
    return {
        "mean": _clean(num.mean()),
        "std": _clean(num.std()),
        "min": _clean(num.min()),
        "q25": _clean(q1),
        "median": _clean(med),
        "q75": _clean(q3),
        "max": _clean(num.max()),
        "outlier_count": outliers,
        "outlier_pct": round(100 * outliers / len(num), 2),
    }


def _label_variants(raw: pd.Series) -> list[list[str]]:
    """Groups of labels that are identical once case/whitespace are ignored."""
    canon = raw.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    frame = pd.DataFrame({"raw": raw, "canon": canon}).drop_duplicates()
    groups = frame.groupby("canon")["raw"].apply(list)
    return [sorted(g)[:5] for g in groups if len(g) > 1][:3]


def _looks_like_datetime(values: pd.Series) -> bool:
    sample = values.head(200)
    if sample.empty or sample.str.len().mean() < 6:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.9)


# --------------------------------------------------------------------------- #
# Column profiling
# --------------------------------------------------------------------------- #
def _profile_column(series: pd.Series, n_rows: int) -> dict[str, Any]:
    name = str(series.name)
    info: dict[str, Any] = {"name": name, "dtype": str(series.dtype)}
    numeric_for_stats: pd.Series | None = None

    if pd.api.types.is_bool_dtype(series):
        non_null = series.dropna()
        info["inferred_type"] = "boolean"
        n_missing = n_rows - len(non_null)
        values = non_null
    elif pd.api.types.is_datetime64_any_dtype(series):
        non_null = series.dropna()
        info["inferred_type"] = "datetime"
        n_missing = n_rows - len(non_null)
        values = non_null
    elif pd.api.types.is_numeric_dtype(series):
        non_null = series.replace([np.inf, -np.inf], np.nan).dropna()
        n_missing = n_rows - len(non_null)
        values = non_null
        numeric_for_stats = non_null
        info["inferred_type"] = "binary" if non_null.nunique() == 2 else "numeric"
    else:
        raw = series.dropna().astype(str)
        stripped = raw.str.strip()
        values = stripped[stripped != ""]
        n_missing = n_rows - len(values)  # blanks count as missing
        n_unique = int(values.nunique())

        converted = pd.to_numeric(values, errors="coerce")
        numeric_share = float(converted.notna().mean()) if len(values) else 0.0

        if len(values) and numeric_share >= NUMERIC_AS_TEXT_SHARE:
            info["inferred_type"] = "numeric_as_text"
            numeric_for_stats = converted.dropna()
            info["non_numeric_values"] = int(converted.isna().sum())
        elif len(values) and _looks_like_datetime(values):
            info["inferred_type"] = "datetime_as_text"
        elif n_unique == 2:
            info["inferred_type"] = "binary"
        elif n_unique <= CATEGORICAL_MAX_UNIQUE:
            info["inferred_type"] = "categorical"
        elif len(values) and n_unique / len(values) > 0.95:
            info["inferred_type"] = "id_or_text"
        else:
            info["inferred_type"] = "high_cardinality_text"

        if 1 < raw.nunique() <= 200:
            variants = _label_variants(raw)
            if variants:
                info["label_variants"] = variants

    n_unique = int(values.nunique()) if len(values) else 0
    info["missing_count"] = int(n_missing)
    info["missing_pct"] = round(100 * n_missing / n_rows, 2) if n_rows else 0.0
    info["n_unique"] = n_unique
    info["unique_ratio"] = round(n_unique / len(values), 4) if len(values) else 0.0

    if numeric_for_stats is not None:
        info["numeric"] = _numeric_stats(numeric_for_stats)

    if info["inferred_type"] in {"categorical", "binary", "boolean"} and len(values):
        top = values.astype(str).value_counts().head(5)
        info["top_values"] = {str(k): int(v) for k, v in top.items()}
    info["sample_values"] = [_clean(v) for v in values.head(3).tolist()]
    return info


# --------------------------------------------------------------------------- #
# Dataset-level analysis
# --------------------------------------------------------------------------- #
def _issues(overview: dict[str, Any], columns: list[dict[str, Any]]) -> list[dict]:
    out: list[dict[str, Any]] = []

    def add(severity: str, code: str, message: str, column: str | None = None) -> None:
        out.append(
            {"severity": severity, "code": code, "column": column, "message": message}
        )

    dup_pct = overview["duplicate_pct"]
    if overview["duplicate_rows"] > 0:
        add(
            "high" if dup_pct > 10 else "medium",
            "duplicate_rows",
            f"{overview['duplicate_rows']} fully duplicated rows ({dup_pct}%).",
        )

    for col in columns:
        name, kind = col["name"], col["inferred_type"]
        miss = col["missing_pct"]
        if col["missing_count"] == overview["n_rows"]:
            add("high", "all_missing", "Column has no values.", name)
            continue
        if miss >= MISSING_HIGH_PCT:
            add("high", "missing_high", f"{miss}% of values are missing.", name)
        elif miss >= MISSING_MEDIUM_PCT:
            add("medium", "missing_medium", f"{miss}% of values are missing.", name)
        elif col["missing_count"] > 0:
            add(
                "low",
                "missing_low",
                f"{col['missing_count']} missing values ({miss}%).",
                name,
            )

        if col["n_unique"] <= 1:
            add("medium", "constant_column", "Column holds a single value.", name)
        if kind == "numeric_as_text":
            bad = col.get("non_numeric_values", 0)
            add(
                "medium",
                "numeric_as_text",
                "Numeric data is stored as text; convert it to a number type"
                + (f" ({bad} non-numeric entries)." if bad else "."),
                name,
            )
        if kind == "datetime_as_text":
            add("low", "datetime_as_text", "Dates are stored as text.", name)
        if col.get("label_variants"):
            add(
                "medium",
                "inconsistent_labels",
                "Some category labels differ only by case or spacing.",
                name,
            )
        if kind in {"id_or_text"} or (
            kind == "numeric"
            and col["unique_ratio"] == 1.0
            and col["numeric"].get("min", 0) is not None
            and overview["n_rows"] > 1
            and float(col["numeric"].get("max", 0) or 0).is_integer()
        ):
            add(
                "low",
                "likely_identifier",
                "Every value is unique; probably an ID. Exclude it from model features.",
                name,
            )
        if kind == "high_cardinality_text" and col["n_unique"] > HIGH_CARDINALITY_UNIQUE:
            add(
                "low",
                "high_cardinality",
                f"{col['n_unique']} distinct categories; needs special encoding.",
                name,
            )
        num = col.get("numeric") or {}
        pct = num.get("outlier_pct", 0)
        if pct >= OUTLIER_MEDIUM_PCT:
            add("medium", "outliers", f"{pct}% of values are IQR outliers.", name)
        elif pct >= OUTLIER_LOW_PCT:
            add("low", "outliers", f"{pct}% of values are IQR outliers.", name)

    out.sort(key=lambda i: (_SEVERITY_ORDER[i["severity"]], i["column"] or ""))
    return out


def _candidate_targets(df: pd.DataFrame, columns: list[dict[str, Any]]) -> list[dict]:
    candidates: list[dict[str, Any]] = []
    for col in columns:
        kind, name = col["inferred_type"], col["name"]
        if col["n_unique"] <= 1 or kind in {"id_or_text", "high_cardinality_text"}:
            continue
        if col["missing_pct"] >= MISSING_HIGH_PCT:
            continue
        hinted = bool(TARGET_NAME_HINT.search(name))
        score, task, reason = 0, "unknown", ""
        if kind in {"binary", "boolean"}:
            task, score = "classification", 2
            reason = "Two distinct values."
            counts = df[col["name"]].astype(str).str.strip().value_counts(normalize=True)
            col_minority = round(100 * float(counts.min()), 2) if len(counts) else None
        elif kind == "categorical" and col["n_unique"] <= 10 and hinted:
            task, score, reason = "classification", 1, "Few classes and a target-like name."
            col_minority = None
        elif kind in {"numeric", "numeric_as_text"} and col["n_unique"] > 10 and hinted:
            task, score, reason = "regression", 1, "Continuous column with a target-like name."
            col_minority = None
        else:
            continue
        if hinted:
            score += 3
            reason += " Name suggests a target."
        entry: dict[str, Any] = {
            "column": name,
            "task": task,
            "reason": reason.strip(),
            "score": score,
        }
        if col_minority is not None:
            entry["minority_class_pct"] = col_minority
        candidates.append(entry)
    candidates.sort(key=lambda c: -c["score"])
    return candidates[:5]


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Return a JSON-serialisable profile of ``df``."""
    if df.empty:
        raise ValueError("Cannot profile an empty DataFrame.")
    n_rows, n_cols = df.shape
    dup = int(df.duplicated().sum())
    columns = [_profile_column(df[c], n_rows) for c in df.columns]

    overview = {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "duplicate_rows": dup,
        "duplicate_pct": round(100 * dup / n_rows, 2),
        "total_missing_cells": int(sum(c["missing_count"] for c in columns)),
        "missing_cell_pct": round(
            100 * sum(c["missing_count"] for c in columns) / (n_rows * n_cols), 2
        ),
        "type_counts": {},
    }
    for c in columns:
        t = c["inferred_type"]
        overview["type_counts"][t] = overview["type_counts"].get(t, 0) + 1

    return {
        "profile_version": PROFILE_VERSION,
        "overview": overview,
        "columns": columns,
        "issues": _issues(overview, columns),
        "candidate_targets": _candidate_targets(df, columns),
    }


def strip_raw_values(profile: dict[str, Any]) -> dict[str, Any]:
    """Copy of the profile without any raw cell values (safe to send to an LLM)."""
    cleaned = {k: v for k, v in profile.items() if k != "columns"}
    cleaned["columns"] = [
        {k: v for k, v in col.items() if k not in RAW_VALUE_KEYS}
        for col in profile["columns"]
    ]
    return cleaned
