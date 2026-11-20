import numpy as np
import pandas as pd
import pytest

from ml_engine.preprocessing import apply_plan, build_plan, build_preprocessor
from ml_engine.profiling import profile_dataframe
from ml_engine.validation import validate_prepared


@pytest.fixture()
def messy_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(n)],
            "tenure": rng.integers(1, 72, n),
            "charges": rng.normal(60, 10, n).round(2),
            "total": [str(x) for x in rng.normal(500, 50, n).round(2)],
            "plan": rng.choice(["basic", "pro", "team"], n),
            "paperless": rng.choice(["Yes", "No"], n),
            "country": "DE",
            "signup": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
            "churn": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        }
    )
    df.loc[0:4, "total"] = " "
    df.loc[10:14, "charges"] = np.nan
    df.loc[30:39, "paperless"] = "yes "
    df.loc[50:52, "churn"] = np.nan
    return pd.concat([df, df.iloc[60:66]], ignore_index=True)  # 6 duplicates


def _inf_df() -> pd.DataFrame:
    x = np.arange(60, dtype=float)
    x[::3] = np.inf
    return pd.DataFrame({"x": x, "y": ["a", "b"] * 30})


def _prepare(df, target="churn", task="classification"):
    plan = build_plan(profile_dataframe(df), target, task)
    return plan, *apply_plan(df, plan)


def test_plan_contents(messy_df):
    plan, _, _ = _prepare(messy_df)
    assert plan["numeric_as_text"] == ["total"]
    assert plan["normalize_labels"] == ["paperless"]
    assert plan["datetime_as_text"] == ["signup"]
    assert "customer_id" in plan["identifier_columns"]


def test_apply_plan_cleans_everything(messy_df):
    _, clean, changes = _prepare(messy_df)
    assert pd.api.types.is_float_dtype(clean["total"])
    assert set(clean["paperless"].dropna().unique()) == {"Yes", "No"}
    assert clean["churn"].notna().all()
    assert not clean.duplicated().any()
    for dropped in ("customer_id", "country", "signup"):
        assert dropped not in clean.columns
    # all dates are in 2024, so the derived year is constant and dropped
    assert {"signup_month", "signup_weekday"} <= set(clean.columns)
    assert "signup_year" not in clean.columns
    assert any("duplicate" in c for c in changes)
    assert any("missing target" in c for c in changes)


def test_apply_plan_does_not_mutate_input(messy_df):
    before = messy_df.copy()
    _prepare(messy_df)
    pd.testing.assert_frame_equal(messy_df, before)


def test_numeric_id_kept_unless_name_says_id():
    rng = np.random.default_rng(2)
    n = 100
    df = pd.DataFrame(
        {
            "salary": rng.permutation(np.arange(30000, 30000 + n)),  # unique ints, not an ID
            "row_id": np.arange(n),
            "y": rng.choice(["a", "b"], n),
        }
    )
    plan = build_plan(profile_dataframe(df), "y", "classification")
    assert "row_id" in plan["identifier_columns"]
    assert "salary" not in plan["identifier_columns"]


def test_preprocessor_fits_cleaned_data(messy_df):
    _, clean, _ = _prepare(messy_df)
    X = clean.drop(columns=["churn"])
    Xt = build_preprocessor(X).fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert not np.isnan(Xt).any()


def test_validation_passes_on_clean_data(messy_df):
    _, clean, _ = _prepare(messy_df)
    report = validate_prepared(clean, "churn", "classification")
    assert report["passed"], report["errors"]


def test_validation_flags_inf_as_recoverable():
    df = _inf_df()
    report = validate_prepared(df, "y", "classification")
    assert not report["passed"]
    assert report["errors"][0]["code"] == "inf_values"
    assert report["errors"][0]["recoverable"] is True


def test_replace_inf_repair_works():
    df = _inf_df()
    plan = build_plan(profile_dataframe(df), "y", "classification")
    plan["replace_inf"] = True
    clean, _ = apply_plan(df, plan)
    assert validate_prepared(clean, "y", "classification")["passed"]


def test_validation_rare_class_is_not_recoverable():
    df = pd.DataFrame(
        {"x": np.arange(60.0), "y": ["a"] * 58 + ["b"] * 2}
    )
    report = validate_prepared(df, "y", "classification")
    err = next(e for e in report["errors"] if e["code"] == "rare_class")
    assert err["recoverable"] is False


def test_validation_missing_target_column():
    report = validate_prepared(pd.DataFrame({"x": [1, 2, 3]}), "y", "classification")
    assert report["errors"][0]["code"] == "target_missing"


def test_validation_regression_needs_numeric_target():
    df = pd.DataFrame({"x": np.arange(40.0), "y": ["a", "b"] * 20})
    report = validate_prepared(df, "y", "regression")
    assert any(e["code"] == "target_not_numeric" for e in report["errors"])
