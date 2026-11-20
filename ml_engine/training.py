"""Model training with cross-validation (Phase 3).

Leak-free by construction: the preprocessor (impute, scale, one-hot) sits INSIDE
an sklearn Pipeline, so each CV fold fits it on its own training part only.

Every candidate is compared against a trivial baseline (always predict the
majority class / the mean). A model that cannot beat the baseline is flagged.
XGBoost and MLflow are optional: they are used when installed, skipped otherwise.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from ml_engine.preprocessing import build_preprocessor

logger = logging.getLogger("aidme.training")

MIN_ROWS_FOR_TRAINING = 30
BASELINE = "baseline"


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 - any import problem means "not available"
        return False


def candidate_models(task: str, seed: int) -> dict[str, Any]:
    """Name -> unfitted estimator. The baseline always comes first."""
    if task == "classification":
        models: dict[str, Any] = {
            BASELINE: DummyClassifier(strategy="prior"),
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(
                n_estimators=200, random_state=seed, n_jobs=-1
            ),
        }
        if _xgboost_available():
            from xgboost import XGBClassifier

            models["xgboost"] = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                random_state=seed, n_jobs=1, verbosity=0,
            )
    else:
        models = {
            BASELINE: DummyRegressor(strategy="mean"),
            "ridge": Ridge(),
            "random_forest": RandomForestRegressor(
                n_estimators=200, random_state=seed, n_jobs=-1
            ),
        }
        if _xgboost_available():
            from xgboost import XGBRegressor

            models["xgboost"] = XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                random_state=seed, n_jobs=1, verbosity=0,
            )
    return models


def _scoring(task: str, n_classes: int) -> tuple[dict[str, str], str, bool]:
    """(scoring dict, primary metric, higher_is_better)."""
    if task == "classification":
        if n_classes == 2:
            scoring = {
                "accuracy": "accuracy", "f1": "f1", "precision": "precision",
                "recall": "recall", "roc_auc": "roc_auc",
            }
            return scoring, "f1", True
        scoring = {
            "accuracy": "accuracy", "f1": "f1_macro", "precision": "precision_macro",
            "recall": "recall_macro", "roc_auc": "roc_auc_ovr_weighted",
        }
        return scoring, "f1", True
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    return scoring, "rmse", False


def _summarise(cv: dict[str, np.ndarray], scoring: dict[str, str], seconds: float) -> dict:
    metrics: dict[str, float | None] = {}
    for name in scoring:
        values = cv[f"test_{name}"]
        if name in {"mae", "rmse"}:  # sklearn returns these negated
            values = -values
        mean = float(np.nanmean(values)) if np.isfinite(values).any() else None
        std = float(np.nanstd(values)) if np.isfinite(values).any() else None
        metrics[name] = None if mean is None else round(mean, 4)
        metrics[f"{name}_std"] = None if std is None else round(std, 4)
    metrics["fit_seconds"] = round(seconds, 2)
    return metrics


def train_models(
    df: pd.DataFrame,
    target: str,
    task: str,
    *,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Cross-validate all candidates. Returns a JSON-safe report plus the refitted best model.

    Raises ValueError for data that cannot be trained on.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is not in the data.")
    if len(df) < MIN_ROWS_FOR_TRAINING:
        raise ValueError(f"Need at least {MIN_ROWS_FOR_TRAINING} rows, got {len(df)}.")
    if task not in {"classification", "regression"}:
        raise ValueError(f"Unknown task '{task}'.")

    X = df.drop(columns=[target])
    y_raw = df[target]
    classes: list[str] = []
    if task == "classification":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw.astype(str))
        classes = [str(c) for c in encoder.classes_]
        counts = np.bincount(y)
        folds = min(cv_folds, int(counts.min()))
        if len(classes) < 2 or folds < 2:
            raise ValueError("Each class needs at least 2 rows to cross-validate.")
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    else:
        y = pd.to_numeric(y_raw, errors="raise").to_numpy(dtype=float)
        folds = min(cv_folds, len(df))
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)

    scoring, primary, higher_is_better = _scoring(task, len(classes))
    results: list[dict[str, Any]] = []
    for name, estimator in candidate_models(task, seed).items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("model", estimator)])
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                # the baseline never predicts the minority class: precision is undefined (=0)
                warnings.simplefilter("ignore", UndefinedMetricWarning)
                cv = cross_validate(
                    pipe, X, y, cv=splitter, scoring=scoring, error_score="raise", n_jobs=1
                )
            metrics = _summarise(cv, scoring, time.perf_counter() - started)
            results.append({"model": name, "status": "ok", "metrics": metrics})
        except Exception as exc:  # noqa: BLE001 - one bad model must not stop the rest
            logger.warning("Model %s failed: %s", name, exc)
            results.append({"model": name, "status": "failed", "error": str(exc)[:300]})

    ok = [r for r in results if r["status"] == "ok" and r["metrics"].get(primary) is not None]
    contenders = [r for r in ok if r["model"] != BASELINE]
    if not contenders:
        raise ValueError("No model could be trained on this data.")
    contenders.sort(key=lambda r: r["metrics"][primary], reverse=higher_is_better)
    for rank, r in enumerate(contenders, start=1):
        r["rank"] = rank

    baseline = next((r for r in ok if r["model"] == BASELINE), None)
    best = contenders[0]
    notes: list[str] = []
    beats_baseline: bool | None = None
    if baseline is not None:
        b, m = baseline["metrics"][primary], best["metrics"][primary]
        beats_baseline = bool(m > b if higher_is_better else m < b)
        if not beats_baseline:
            notes.append(
                "The best model does not beat the trivial baseline; "
                "the features may not predict the target."
            )
    if not _xgboost_available():
        notes.append("XGBoost is not installed, so it was skipped.")

    best_name = best["model"]
    final = Pipeline(
        [("prep", build_preprocessor(X)), ("model", candidate_models(task, seed)[best_name])]
    ).fit(X, y)

    report = {
        "task": task,
        "target": target,
        "primary_metric": primary,
        "cv_folds": folds,
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "classes": classes,
        "best_model": best_name,
        "beats_baseline": beats_baseline,
        "results": sorted(results, key=lambda r: r.get("rank", 99)),
        "notes": notes,
        "note_on_metrics": (
            "Metrics are cross-validated estimates (mean over folds), "
            "not performance on a separate test set."
        ),
    }
    return {"report": report, "model": final, "feature_names": list(X.columns), "classes": classes}


# --------------------------------------------------------------------------- #
# Persistence and tracking
# --------------------------------------------------------------------------- #
def save_model(result: dict[str, Any], path: Path) -> None:
    joblib.dump(
        {
            "model": result["model"],
            "feature_names": result["feature_names"],
            "classes": result["classes"],
            "report": result["report"],
        },
        path,
    )


def log_to_mlflow(report: dict[str, Any], model_path: Path | None, run_label: str) -> str | None:
    """Log one parent run plus one child per model. Never raises; returns the run id."""
    try:
        import mlflow
    except Exception:  # noqa: BLE001
        return None
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "aidme"))
        with mlflow.start_run(run_name=run_label) as parent:
            mlflow.log_params(
                {
                    "task": report["task"], "target": report["target"],
                    "cv_folds": report["cv_folds"], "n_rows": report["n_rows"],
                    "n_features": report["n_features"], "best_model": report["best_model"],
                }
            )
            for r in report["results"]:
                with mlflow.start_run(run_name=r["model"], nested=True):
                    mlflow.log_param("model", r["model"])
                    if r["status"] == "ok":
                        mlflow.log_metrics(
                            {k: v for k, v in r["metrics"].items() if v is not None}
                        )
            best = next(r for r in report["results"] if r["model"] == report["best_model"])
            mlflow.log_metrics(
                {f"best_{k}": v for k, v in best["metrics"].items() if v is not None}
            )
            if model_path is not None and model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path="model")
            return parent.info.run_id
    except Exception:  # noqa: BLE001 - tracking problems must not break training
        logger.exception("MLflow logging failed")
        return None


def load_model(path: Path) -> dict[str, Any]:
    return joblib.load(path)


def predict_dataframe(bundle: dict[str, Any], df: pd.DataFrame) -> list[Any]:
    """Predict with a saved bundle; returns original class labels for classification."""
    missing = [c for c in bundle["feature_names"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")
    pred = bundle["model"].predict(df[bundle["feature_names"]])
    if bundle["classes"]:
        return [bundle["classes"][int(i)] for i in pred]
    return [float(p) for p in pred]
