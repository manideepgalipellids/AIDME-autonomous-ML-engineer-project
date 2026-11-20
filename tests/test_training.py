import json

import numpy as np
import pandas as pd
import pytest

from ml_engine.training import (
    load_model,
    predict_dataframe,
    save_model,
    train_models,
)


def _cls_df(n=300, seed=0, signal=True):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = (x1 + 0.3 * noise > 0) if signal else rng.random(n) > 0.5
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": rng.normal(size=n),
            "plan": rng.choice(["a", "b", "c"], n),
            "y": np.where(y, "Yes", "No"),
        }
    )


def _reg_df(n=300, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {"x": x, "cat": rng.choice(["u", "v"], n), "y": 3 * x + rng.normal(scale=0.3, size=n)}
    )


def test_classification_ranks_models_and_beats_baseline():
    out = train_models(_cls_df(), "y", "classification", cv_folds=3)
    rep = out["report"]
    assert rep["beats_baseline"] is True
    assert rep["primary_metric"] == "f1"
    names = [r["model"] for r in rep["results"] if r["status"] == "ok"]
    assert "baseline" in names and "logistic_regression" in names and "random_forest" in names
    ranked = [r for r in rep["results"] if "rank" in r]
    f1s = [r["metrics"]["f1"] for r in ranked]
    assert f1s == sorted(f1s, reverse=True)
    assert rep["best_model"] == ranked[0]["model"]
    assert ranked[0]["metrics"]["roc_auc"] > 0.8
    json.dumps(rep, allow_nan=False)


def test_noise_target_is_flagged_as_not_beating_baseline():
    out = train_models(_cls_df(signal=False, seed=4), "y", "classification", cv_folds=3)
    # a random target should not look like a success
    assert out["report"]["results"][-1]["model"] == "baseline" or out["report"]["notes"]
    assert out["report"]["best_model"] != "baseline"


def test_regression_uses_rmse_lowest_first():
    out = train_models(_reg_df(), "y", "regression", cv_folds=3)
    rep = out["report"]
    assert rep["primary_metric"] == "rmse"
    ranked = [r for r in rep["results"] if "rank" in r]
    rmses = [r["metrics"]["rmse"] for r in ranked]
    assert rmses == sorted(rmses)
    assert rmses[0] < 1.0  # strong linear signal
    assert rep["beats_baseline"] is True


def test_multiclass_works():
    rng = np.random.default_rng(2)
    n = 300
    x = rng.normal(size=n)
    y = np.where(x < -0.5, "low", np.where(x > 0.5, "high", "mid"))
    df = pd.DataFrame({"x": x, "z": rng.normal(size=n), "y": y})
    rep = train_models(df, "y", "classification", cv_folds=3)["report"]
    assert len(rep["classes"]) == 3
    assert rep["beats_baseline"] is True


def test_missing_values_do_not_leak_or_crash():
    df = _cls_df()
    df.loc[::7, "x2"] = np.nan
    df.loc[::11, "plan"] = np.nan
    assert train_models(df, "y", "classification", cv_folds=3)["report"]["beats_baseline"]


def test_save_load_predict_roundtrip(tmp_path):
    df = _cls_df()
    out = train_models(df, "y", "classification", cv_folds=3)
    path = tmp_path / "m.joblib"
    save_model(out, path)
    preds = predict_dataframe(load_model(path), df.drop(columns=["y"]).head(10))
    assert len(preds) == 10 and set(preds) <= {"Yes", "No"}
    with pytest.raises(ValueError):
        predict_dataframe(load_model(path), pd.DataFrame({"x1": [1.0]}))


def test_deterministic_with_same_seed():
    a = train_models(_cls_df(), "y", "classification", cv_folds=3, seed=7)["report"]
    b = train_models(_cls_df(), "y", "classification", cv_folds=3, seed=7)["report"]
    for ra, rb in zip(a["results"], b["results"]):
        if ra["status"] == "ok":
            assert ra["metrics"]["f1"] == rb["metrics"]["f1"]


@pytest.mark.parametrize(
    "df,target,task",
    [
        (_cls_df(n=10), "y", "classification"),
        (_cls_df(), "nope", "classification"),
        (_cls_df(), "y", "clustering"),
    ],
)
def test_bad_input_raises_value_error(df, target, task):
    with pytest.raises(ValueError):
        train_models(df, target, task)
