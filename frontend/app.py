"""AIDME Streamlit frontend (Phase 1): upload a dataset and view its profile."""

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="AIDME", page_icon="🧠", layout="wide")
st.title("AIDME: Autonomous AI Data & ML Engineer")
st.caption("Upload a dataset, review its profile, then prepare it for machine learning.")


def api(method: str, path: str, **kwargs):
    return requests.request(method, f"{API_URL}{path}", timeout=120, **kwargs)


def show_dataset(ds: dict) -> None:
    profile = ds["profile"]
    o = profile["overview"]

    st.subheader(ds["filename"])
    source = "AI-written" if ds["summary_source"] == "llm" else "rule-based"
    st.info(f"**Summary ({source})**\n\n{ds['summary']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{o['n_rows']:,}")
    c2.metric("Columns", o["n_cols"])
    c3.metric("Missing cells", f"{o['missing_cell_pct']}%")
    c4.metric("Duplicate rows", o["duplicate_rows"])

    st.markdown("#### Data-quality issues")
    if profile["issues"]:
        st.dataframe(pd.DataFrame(profile["issues"]), hide_index=True)
    else:
        st.success("No issues detected.")

    st.markdown("#### Likely prediction targets")
    if profile["candidate_targets"]:
        st.dataframe(pd.DataFrame(profile["candidate_targets"]), hide_index=True)
    else:
        st.write("No obvious target column found.")

    st.markdown("#### Columns")
    rows = []
    for col in profile["columns"]:
        num = col.get("numeric") or {}
        rows.append(
            {
                "column": col["name"],
                "type": col["inferred_type"],
                "dtype": col["dtype"],
                "missing %": col["missing_pct"],
                "unique": col["n_unique"],
                "mean": num.get("mean"),
                "min": num.get("min"),
                "max": num.get("max"),
                "outlier %": num.get("outlier_pct"),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def show_run(run: dict) -> None:
    report = run["report"]
    if run["status"] == "succeeded":
        st.success("Data preparation succeeded.")
    else:
        st.error("Data preparation failed.")
        for message in report["errors"]:
            st.write(f"- {message}")

    analyst = report.get("analyst")
    if analyst:
        st.write(
            f"**Target:** `{analyst['target']}` ({analyst['task']}), chosen by "
            f"{analyst['source']}. {analyst['reason']}"
        )

    if report.get("rows_after") is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{report['rows_after']:,}", f"{report['rows_after'] - report['rows_before']:,}")
        c2.metric(
            "Columns", report["cols_after"], report["cols_after"] - report["cols_before"]
        )
        c3.metric("Repair retries", report["attempts"])

    if report["changes"]:
        st.markdown("**What was changed**")
        for change in report["changes"]:
            st.write(f"- {change}")

    validation = report.get("validation") or {}
    for warning in validation.get("warnings", []):
        st.warning(warning["message"])

    with st.expander("Pipeline log"):
        for line in report["log"]:
            st.write(f"- {line}")

    if run["has_output"]:
        resp = api("GET", f"/prepared/{run['id']}/download")
        if resp.status_code == 200:
            st.download_button(
                "Download prepared CSV",
                data=resp.content,
                file_name=f"prepared_{run['id'][:8]}.csv",
                mime="text/csv",
                key=f"dl_{run['id']}",
            )


def prepare_section(ds: dict) -> None:
    st.divider()
    st.subheader("Prepare data for machine learning")
    st.caption(
        "Runs the Analyst, Quality, Preprocess and Validate steps. "
        "Leave the target on auto-detect or choose the column you want to predict."
    )
    columns = [c["name"] for c in ds["profile"]["columns"]]
    c1, c2 = st.columns(2)
    target = c1.selectbox("Target column", ["(auto-detect)", *columns], key=f"target_{ds['id']}")
    task = c2.selectbox(
        "Task", ["auto", "classification", "regression"], key=f"task_{ds['id']}"
    )
    instruction = st.text_input(
        "Instruction (optional)",
        placeholder="e.g. Predict which customers will churn",
        key=f"instr_{ds['id']}",
    )
    if st.button("Run preparation", type="primary", key=f"prep_{ds['id']}"):
        body = {
            "target": None if target == "(auto-detect)" else target,
            "task": None if task == "auto" else task,
            "instruction": instruction or None,
        }
        with st.spinner("Preparing data..."):
            try:
                resp = api("POST", f"/datasets/{ds['id']}/prepare", json=body)
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
            else:
                if resp.status_code == 200:
                    st.session_state[f"run_{ds['id']}"] = resp.json()
                else:
                    st.error(resp.json().get("detail", resp.text))

    run = st.session_state.get(f"run_{ds['id']}")
    if run:
        show_run(run)


with st.sidebar:
    st.header("Datasets")
    try:
        history = api("GET", "/datasets").json()
    except requests.RequestException:
        history = []
        st.error(f"Cannot reach the API at {API_URL}. Is the backend running?")
    for item in history:
        if st.button(f"{item['filename']} ({item['n_rows']:,} rows)", key=item["id"]):
            st.session_state["selected"] = item["id"]

uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "tsv", "txt", "xlsx"])
if uploaded and st.button("Analyze dataset", type="primary"):
    with st.spinner("Profiling dataset..."):
        try:
            resp = api("POST", "/datasets", files={"file": (uploaded.name, uploaded.getvalue())})
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
        else:
            if resp.status_code == 201:
                st.session_state["selected"] = resp.json()["id"]
                st.session_state["fresh"] = resp.json()
            else:
                st.error(resp.json().get("detail", resp.text))

selected = st.session_state.get("selected")
if selected:
    fresh = st.session_state.get("fresh")
    if fresh and fresh["id"] == selected:
        dataset = fresh
    else:
        r = api("GET", f"/datasets/{selected}")
        dataset = r.json() if r.status_code == 200 else None
    if dataset:
        show_dataset(dataset)
        prepare_section(dataset)
        st.divider()
        if st.button("Delete this dataset"):
            api("DELETE", f"/datasets/{selected}")
            st.session_state.pop("selected", None)
            st.session_state.pop("fresh", None)
            st.rerun()
    else:
        st.warning("That dataset no longer exists.")
        st.session_state.pop("selected", None)
