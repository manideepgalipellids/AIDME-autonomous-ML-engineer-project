"""Deterministic data cleaning and the model-ready preprocessor.

Two different kinds of preparation live here, on purpose:

1. ``apply_plan``: stateless, row/column-level cleaning (fix types, normalise
   labels, drop duplicates, drop useless columns). Nothing is *learned* from the
   data, so it is safe to run on the whole dataset before splitting it.

2. ``build_preprocessor``: an sklearn transformer that imputes, scales and
   one-hot encodes. It LEARNS statistics (medians, categories), so in Phase 3 it
   must be fitted on training folds only, inside a Pipeline, to avoid leakage.
   Phase 2 only uses it as a smoke test.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# "customer_id", "ID", "uuid", "row_index" ... but not "paid" or "valid".
ID_NAME = re.compile(r"(^|[_\s-])(id|uuid|guid|index|idx|key)($|[_\s-])", re.IGNORECASE)
MISSING_CATEGORY = "__missing__"


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #
def build_plan(profile: dict[str, Any], target: str, task: str) -> dict[str, Any]:
    """Derive a cleaning plan from the dataset profile. Pure and deterministic."""
    columns = profile["columns"]

    def with_kind(*kinds: str) -> list[str]:
        return [c["name"] for c in columns if c["inferred_type"] in kinds]

    identifier_cols = []
    for issue in profile["issues"]:
        if issue["code"] != "likely_identifier" or not issue["column"]:
            continue
        col = next(c for c in columns if c["name"] == issue["column"])
        # Free text / ID strings are always useless as features. Unique numbers
        # are only treated as IDs when the name says so (a continuous integer
        # column such as a salary can also be fully unique).
        if col["inferred_type"] == "id_or_text" or ID_NAME.search(col["name"]):
            identifier_cols.append(col["name"])

    return {
        "target": target,
        "task": task,
        "numeric_as_text": with_kind("numeric_as_text"),
        "normalize_labels": [c["name"] for c in columns if c.get("label_variants")],
        "datetime_as_text": [c for c in with_kind("datetime_as_text") if c != target],
        "identifier_columns": [c for c in identifier_cols if c != target],
        "replace_inf": False,
        "drop_duplicates": True,
        "drop_rows_missing_target": True,
    }


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def _text_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if not (
            pd.api.types.is_numeric_dtype(df[c])
            or pd.api.types.is_bool_dtype(df[c])
            or pd.api.types.is_datetime64_any_dtype(df[c])
        )
    ]


def apply_plan(df: pd.DataFrame, plan: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Apply ``plan`` to a copy of ``df``. Returns (clean_df, human-readable changes)."""
    out = df.copy()
    target = plan["target"]
    changes: list[str] = []

    # 1. Trim whitespace; blank strings become missing values.
    blanks_total = 0
    for c in _text_columns(out):
        s = out[c].astype("string").str.strip()
        blank = (s == "").fillna(False)
        blanks_total += int(blank.sum())
        out[c] = s.mask(blank)
    if blanks_total:
        changes.append(f"Converted {blanks_total} blank text entries to missing values.")

    # 2. Infinite values -> missing (only when a validation retry asked for it).
    if plan.get("replace_inf"):
        n_inf = 0
        for c in out.columns:
            if pd.api.types.is_numeric_dtype(out[c]) and not pd.api.types.is_bool_dtype(out[c]):
                mask = out[c].isin([np.inf, -np.inf])
                n_inf += int(mask.sum())
                out[c] = out[c].mask(mask)
        changes.append(f"Replaced {n_inf} infinite values with missing values.")

    # 3. Numbers stored as text.
    for c in plan["numeric_as_text"]:
        if c not in out:
            continue
        before_missing = int(out[c].isna().sum())
        converted = pd.to_numeric(out[c].astype("string").str.strip(), errors="coerce")
        out[c] = converted.astype("float64")
        bad = int(out[c].isna().sum()) - before_missing
        note = f" ({bad} unparseable entries set to missing)" if bad > 0 else ""
        changes.append(f"Converted '{c}' from text to a number{note}.")

    # 4. Labels that differ only by case/spacing -> most frequent spelling.
    for c in plan["normalize_labels"]:
        if c not in out:
            continue
        s = out[c].astype("string")
        canon = s.str.lower().str.replace(r"\s+", " ", regex=True)
        frame = pd.DataFrame({"raw": s, "canon": canon}).dropna()
        if frame.empty:
            continue
        spelling = frame.groupby("canon")["raw"].agg(lambda x: x.value_counts().idxmax())
        mapped = canon.map(spelling.to_dict())
        n_changed = int(((mapped != s) & s.notna()).fillna(False).sum())
        out[c] = mapped
        changes.append(f"Unified inconsistent labels in '{c}' ({n_changed} values changed).")

    # 5. Dates stored as text -> year / month / weekday features.
    for c in plan["datetime_as_text"]:
        if c not in out:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(out[c], errors="coerce")
        out[f"{c}_year"] = parsed.dt.year.astype("float64")
        out[f"{c}_month"] = parsed.dt.month.astype("float64")
        out[f"{c}_weekday"] = parsed.dt.day_of_week.astype("float64")
        out = out.drop(columns=[c])
        changes.append(f"Split date column '{c}' into year, month and weekday features.")

    # 6. Duplicate rows (after cleaning, so newly identical rows are caught).
    if plan["drop_duplicates"]:
        n = len(out)
        out = out.drop_duplicates()
        if len(out) < n:
            changes.append(f"Removed {n - len(out)} duplicate rows.")

    # 7. Rows without a target value cannot be used for training.
    if plan["drop_rows_missing_target"] and target in out:
        missing = out[target].isna()
        if missing.any():
            out = out.loc[~missing]
            changes.append(f"Dropped {int(missing.sum())} rows with a missing target value.")

    # 8. Useless columns: identifiers, constants, empty columns (target is kept).
    to_drop: list[str] = []
    for c in out.columns:
        if c == target:
            continue
        n_unique = out[c].nunique(dropna=True)
        if c in plan["identifier_columns"]:
            to_drop.append(c)
            changes.append(f"Dropped identifier column '{c}'.")
        elif n_unique == 0:
            to_drop.append(c)
            changes.append(f"Dropped empty column '{c}'.")
        elif n_unique == 1:
            to_drop.append(c)
            changes.append(f"Dropped constant column '{c}'.")
    out = out.drop(columns=to_drop)

    # 9. Normalise dtypes for sklearn: plain object text with NaN, bools as 0/1.
    out = out.reset_index(drop=True)
    for c in out.columns:
        if isinstance(out[c].dtype, pd.StringDtype):
            out[c] = out[c].astype(object).where(out[c].notna(), np.nan)
        elif c != target and pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype("int64")
    return out, changes


# --------------------------------------------------------------------------- #
# Model-ready preprocessor (fitted later, on training data only)
# --------------------------------------------------------------------------- #
def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = list(X.select_dtypes(include=["number"]).columns)
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Unfitted sklearn preprocessor for the feature columns of ``X``.

    Missing values are imputed *and flagged*: numeric columns get a
    missing-indicator column, categorical columns get an explicit
    ``__missing__`` category. That keeps "informative missingness" (for example
    an option that simply has no wifi feature) available to the model.
    """
    numeric, categorical = split_feature_types(X)
    if not numeric and not categorical:
        raise ValueError("No feature columns to preprocess.")

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")
