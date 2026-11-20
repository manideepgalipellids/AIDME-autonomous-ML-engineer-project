import json

import numpy as np
import pandas as pd
import pytest

from agents import data_analyst
from agents.data_analyst import choose_target, infer_task
from ml_engine.profiling import profile_dataframe
from workflows.data_pipeline import result_report, run_pipeline


def _churn_df(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(n)],
            "tenure": rng.integers(1, 72, n),
            "charges": rng.normal(60, 10, n).round(2),
            "total": [str(x) for x in rng.normal(500, 50, n).round(2)],
            "paperless": rng.choice(["Yes", "No"], n),
            "country": "DE",
            "churn": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        }
    )
    df.loc[0:4, "total"] = " "
    df.loc[30:39, "paperless"] = "yes "
    return df


def _inf_df() -> pd.DataFrame:
    x = np.arange(80, dtype=float)
    x[::4] = np.inf
    return pd.DataFrame({"x": x, "z": np.arange(80) % 7, "y": ["a", "b"] * 40})


# ----------------------------- python engine ------------------------------- #
def test_pipeline_succeeds_on_messy_data():
    final = run_pipeline(_churn_df(), engine="python")
    assert final["status"] == "succeeded"
    assert final["target"] == "churn" and final["task"] == "classification"
    assert final["attempt"] == 0
    assert "customer_id" not in final["df_clean"].columns
    assert final["validation"]["passed"]


def test_pipeline_retries_and_repairs_inf():
    final = run_pipeline(_inf_df(), target="y", engine="python")
    assert final["status"] == "succeeded"
    assert final["attempt"] == 1
    assert final["plan"]["replace_inf"] is True
    assert any("retry 1" in line for line in final["log"])


def test_pipeline_respects_retry_limit():
    final = run_pipeline(_inf_df(), target="y", max_retries=0, engine="python")
    assert final["status"] == "failed"
    assert any("Retry limit" in e for e in final["errors"])


def test_pipeline_stops_on_unknown_target_without_retry():
    final = run_pipeline(_churn_df(), target="does_not_exist", engine="python")
    assert final["status"] == "failed"
    assert final.get("attempt", 0) == 0
    assert "was not found" in final["errors"][0]
    assert "df_clean" not in final


def test_pipeline_stops_on_rare_class_without_retry():
    df = pd.DataFrame({"x": np.arange(80.0), "y": ["a"] * 78 + ["b"] * 2})
    final = run_pipeline(df, target="y", engine="python")
    assert final["status"] == "failed"
    assert final["attempt"] == 0
    assert any("fewer than" in e for e in final["errors"])


def test_pipeline_fails_cleanly_when_no_target_found():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
    final = run_pipeline(df, engine="python")
    assert final["status"] == "failed"
    assert "target" in final["errors"][0].lower()


def test_result_report_is_strict_json():
    for df, kwargs in [(_churn_df(), {}), (_inf_df(), {"target": "y"})]:
        report = result_report(run_pipeline(df, engine="python", **kwargs))
        json.dumps(report, allow_nan=False)
        assert report["rows_before"] == len(df)


def test_input_dataframe_is_not_modified():
    df = _churn_df()
    before = df.copy()
    run_pipeline(df, engine="python")
    pd.testing.assert_frame_equal(df, before)


# ------------------------------- analyst ----------------------------------- #
def test_choose_target_user_choice_is_case_insensitive():
    profile = profile_dataframe(_churn_df())
    decision = choose_target(profile, user_target="  CHURN ")
    assert decision["target"] == "churn"
    assert decision["source"] == "user"


def test_choose_target_task_override_requires_numeric_for_regression():
    profile = profile_dataframe(_churn_df())
    with pytest.raises(ValueError):
        choose_target(profile, user_target="churn", user_task="regression")


def test_infer_task_rejects_text_targets():
    profile = profile_dataframe(_churn_df())
    col = next(c for c in profile["columns"] if c["name"] == "customer_id")
    with pytest.raises(ValueError):
        infer_task(col)


def test_llm_answer_for_unknown_column_falls_back_to_heuristic(monkeypatch):
    monkeypatch.setattr(
        data_analyst, "_llm_choose", lambda p, i: {"target": "invented", "reason": "x"}
    )
    profile = profile_dataframe(_churn_df())
    decision = choose_target(profile, instruction="predict churn")
    assert decision["source"] == "heuristic"
    assert decision["target"] == "churn"


def test_valid_llm_answer_is_used(monkeypatch):
    monkeypatch.setattr(
        data_analyst, "_llm_choose", lambda p, i: {"target": "paperless", "reason": "asked"}
    )
    profile = profile_dataframe(_churn_df())
    decision = choose_target(profile, instruction="predict paperless billing")
    assert decision["source"] == "llm"
    assert decision["target"] == "paperless"


# ------------------------------ langgraph ---------------------------------- #
def test_langgraph_engine_matches_python_engine():
    pytest.importorskip("langgraph")
    for df, kwargs in [(_churn_df(), {}), (_inf_df(), {"target": "y"})]:
        a = result_report(run_pipeline(df, engine="python", **kwargs))
        b = result_report(run_pipeline(df, engine="langgraph", **kwargs))
        assert a == b


def test_langgraph_engine_handles_failure():
    pytest.importorskip("langgraph")
    final = run_pipeline(_churn_df(), target="nope", engine="langgraph")
    assert final["status"] == "failed"
