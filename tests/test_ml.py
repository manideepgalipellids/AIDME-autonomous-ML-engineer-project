import json

import numpy as np
import pandas as pd
import pytest

from ml_engine.profiling import (
    load_dataframe,
    profile_dataframe,
    strip_raw_values,
)


@pytest.fixture()
def messy_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(n)],
            "tenure": rng.integers(1, 72, n),
            "charges": rng.normal(60, 10, n).round(2),
            "total": [str(x) for x in rng.normal(500, 50, n).round(2)],
            "plan": rng.choice(["basic", "pro", "team"], n),
            "paperless": rng.choice(["Yes", "No"], n),
            "country": "DE",
            "churn": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        }
    )
    df.loc[0:4, "total"] = " "  # blanks in a numeric-as-text column
    df.loc[10:14, "charges"] = np.nan
    df.loc[20, "charges"] = 5000.0  # outlier
    df.loc[30:39, "paperless"] = "yes "  # inconsistent label
    return pd.concat([df, df.iloc[:6]], ignore_index=True)  # 6 duplicates


def _codes(profile, column=None):
    return {
        i["code"]
        for i in profile["issues"]
        if column is None or i["column"] == column
    }


def test_overview_counts(messy_df):
    p = profile_dataframe(messy_df)
    assert p["overview"]["n_rows"] == 206
    assert p["overview"]["n_cols"] == 8
    assert p["overview"]["duplicate_rows"] == 6


def test_detects_expected_issues(messy_df):
    p = profile_dataframe(messy_df)
    assert "duplicate_rows" in _codes(p)
    assert "numeric_as_text" in _codes(p, "total")
    assert "constant_column" in _codes(p, "country")
    assert "inconsistent_labels" in _codes(p, "paperless")
    assert "likely_identifier" in _codes(p, "customer_id")
    assert "missing_low" in _codes(p, "charges")
    assert "outliers" in _codes(p, "charges")


def test_numeric_as_text_blanks_count_as_missing(messy_df):
    p = profile_dataframe(messy_df)
    total = next(c for c in p["columns"] if c["name"] == "total")
    assert total["inferred_type"] == "numeric_as_text"
    assert total["missing_count"] >= 5
    assert "numeric" in total  # stats are computed on the converted numbers


def test_candidate_target_is_churn(messy_df):
    p = profile_dataframe(messy_df)
    top = p["candidate_targets"][0]
    assert top["column"] == "churn"
    assert top["task"] == "classification"


def test_profile_is_strict_json(messy_df):
    p = profile_dataframe(messy_df)
    json.dumps(p, allow_nan=False)  # raises on NaN/inf/numpy types


def test_strip_raw_values_removes_cell_values(messy_df):
    clean = strip_raw_values(profile_dataframe(messy_df))
    for col in clean["columns"]:
        assert "sample_values" not in col
        assert "top_values" not in col
        assert "label_variants" not in col


def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        profile_dataframe(pd.DataFrame())


def test_load_csv_semicolon_and_german_encoding():
    content = "name;stadt;umsatz\nA;Köln;10\nB;Düsseldorf;20\n".encode("latin-1")
    df = load_dataframe(content, "daten.csv")
    assert list(df.columns) == ["name", "stadt", "umsatz"]
    assert len(df) == 2


def test_load_rejects_bad_input():
    with pytest.raises(ValueError):
        load_dataframe(b"", "x.csv")
    with pytest.raises(ValueError):
        load_dataframe(b"a,b\n1,2\n", "x.exe")


def test_load_xlsx_roundtrip(tmp_path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "t.xlsx"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_excel(path, index=False)
    df = load_dataframe(path.read_bytes(), "t.xlsx")
    assert df.shape == (3, 2)
