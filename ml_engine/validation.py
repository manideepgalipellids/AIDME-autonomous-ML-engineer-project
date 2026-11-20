"""Checks that decide whether a prepared dataset is safe to train on.

Errors stop the pipeline (or trigger a repair + retry when ``recoverable``).
Warnings are reported but never block.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_engine.preprocessing import build_preprocessor

MIN_ROWS = 30
MIN_CLASS_COUNT = 5  # enough for 5-fold stratified cross-validation
IMBALANCE_WARN_SHARE = 0.10
HIGH_MISSING_WARN_PCT = 40.0
HIGH_CARDINALITY = 50
SMOKE_TEST_MAX_ROWS = 5000


def _names(cols: list[str], limit: int = 5) -> str:
    shown = ", ".join(f"'{c}'" for c in cols[:limit])
    return shown + (f" and {len(cols) - limit} more" if len(cols) > limit else "")


def validate_prepared(
    df: pd.DataFrame,
    target: str,
    task: str,
    *,
    min_rows: int = MIN_ROWS,
    min_class_count: int = MIN_CLASS_COUNT,
    smoke_test: bool = True,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def error(code: str, message: str, recoverable: bool = False) -> None:
        errors.append({"code": code, "message": message, "recoverable": recoverable})

    def warn(code: str, message: str) -> None:
        warnings.append({"code": code, "message": message})

    def report() -> dict[str, Any]:
        return {
            "passed": not errors,
            "errors": errors,
            "warnings": warnings,
            "n_rows": int(len(df)),
            "n_features": int(max(df.shape[1] - (1 if target in df else 0), 0)),
        }

    if target not in df.columns:
        error("target_missing", f"Target column '{target}' is not in the prepared data.")
        return report()

    features = [c for c in df.columns if c != target]
    if not features:
        error("no_features", "No feature columns are left after cleaning.")
    if len(df) < min_rows:
        error("too_few_rows", f"Only {len(df)} rows remain; at least {min_rows} are needed.")

    y = df[target]
    if task == "classification":
        counts = y.value_counts()
        if len(counts) < 2:
            error("target_single_class", "The target has fewer than two distinct classes.")
        elif int(counts.min()) < min_class_count:
            rare = [str(k) for k, v in counts.items() if v < min_class_count]
            error(
                "rare_class",
                f"Classes with fewer than {min_class_count} rows: {_names(rare)}. "
                "Merge or remove them, or choose another target.",
            )
        elif counts.min() / counts.sum() < IMBALANCE_WARN_SHARE:
            share = round(100 * counts.min() / counts.sum(), 1)
            warn("class_imbalance", f"Smallest class is only {share}% of rows; use F1/ROC-AUC.")
    elif task == "regression":
        if not pd.api.types.is_numeric_dtype(y):
            error("target_not_numeric", f"Regression target '{target}' is not numeric.")
        elif y.nunique() < 2:
            error("target_constant", "The regression target has a single value.")
    else:
        error("unknown_task", f"Unknown task '{task}'.")

    # Infinite values break sklearn; the Quality agent can repair this.
    inf_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
        and df[c].isin([np.inf, -np.inf]).any()
    ]
    if inf_cols:
        error(
            "inf_values",
            f"Infinite values found in {_names(inf_cols)}.",
            recoverable=True,
        )

    for c in features:
        miss = 100 * float(df[c].isna().mean()) if len(df) else 0.0
        if miss >= HIGH_MISSING_WARN_PCT:
            warn(
                "high_missing_feature",
                f"'{c}' is {miss:.1f}% missing; it will be imputed and flagged.",
            )
        if df[c].dtype == object and df[c].nunique() > HIGH_CARDINALITY:
            warn(
                "high_cardinality_feature",
                f"'{c}' has {df[c].nunique()} categories; one-hot encoding will be wide.",
            )

    # Validation by execution: can the real preprocessor fit on this data?
    if smoke_test and not errors:
        X = df[features]
        if len(X) > SMOKE_TEST_MAX_ROWS:
            X = X.sample(SMOKE_TEST_MAX_ROWS, random_state=0)
        try:
            transformed = np.asarray(build_preprocessor(X).fit_transform(X), dtype=float)
            if np.isnan(transformed).any():
                error("preprocessor_nan_output", "The preprocessor still produces missing values.")
        except Exception as exc:  # noqa: BLE001 - report instead of crashing the pipeline
            error("preprocessor_failed", f"Preprocessor could not be fitted: {str(exc)[:200]}")

    return report()
