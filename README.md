# AIDME: Autonomous AI Data & ML Engineer

Upload a dataset, describe a business problem, get a validated, deployable,
monitored ML model. Built as a multi-agent system (LangGraph) on top of
deterministic, tested Python pipelines.

**Status: Phase 2 (data preparation) complete.** Upload a CSV/Excel file, get
an automatic data profile and summary, then run a LangGraph pipeline that picks
the target, cleans the data, validates it and returns a model-ready CSV.

## Design rule

Reliable Python does the data and ML work. LLM agents only plan, choose tools
and explain. Agents never execute arbitrary generated code.

## Quick start (no Docker)

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # optional: add OPENAI_API_KEY

# terminal 1: API  (SQLite file, no database setup needed)
uvicorn backend.main:app --reload

# terminal 2: UI
streamlit run frontend/app.py
```

Open http://localhost:8501 and upload `datasets/sample/synthetic_dirty_churn.csv`.
API docs: http://localhost:8000/docs

## Quick start (Docker Compose, PostgreSQL)

```bash
cp .env.example .env                 # set POSTGRES_PASSWORD
docker compose up --build
```

UI on http://localhost:8501, API on http://localhost:8000.

## Data preparation pipeline (Phase 2)

```
START -> analyst -> quality -> preprocess -> validate -> END
                      ^                          |
                      +------ retry (repair) ----+
```

| Step | What it does |
| --- | --- |
| Analyst | Picks the target column and task. Your choice wins; otherwise an optional LLM may choose among existing columns; otherwise a heuristic. The task type is always derived deterministically. |
| Quality | Builds a cleaning plan from the profile. On a recoverable validation error it applies a repair from an explicit table (`agents/data_quality.py`) and retries. |
| Preprocess | Fixes numbers stored as text, unifies labels (`Yes` / `yes `), splits date text into features, removes duplicate rows, rows without a target, identifier, constant and empty columns. |
| Validate | Checks target, class counts, row count, infinite values, and fits the real sklearn preprocessor as a smoke test. Unrecoverable errors stop the run with a clear reason. |

Design notes:

- **Cleaning vs. learning.** `apply_plan` learns nothing from the data, so it
  is safe before splitting. `build_preprocessor` (impute, scale, one-hot)
  learns statistics, so Phase 3 must fit it on training folds only, inside a
  Pipeline. Phase 2 only uses it as a smoke test.
- **Missing values are kept as signal.** Numeric columns get a
  missing-indicator; categorical columns get an explicit `__missing__`
  category. A column that is missing because "this option has no such feature"
  is information, not noise.
- **Failures are results, not crashes.** The API returns HTTP 200 with
  `status: "failed"` and the reasons.
- `engine="python"` runs the same nodes without LangGraph (used for fast unit
  tests); `PIPELINE_ENGINE` selects the engine for the API.

API: `POST /datasets/{id}/prepare`, `GET /datasets/{id}/prepared`,
`GET /prepared/{run_id}`, `GET /prepared/{run_id}/download`.

## Tests

```bash
pytest
```

`test_ml.py` covers profiling, `test_preprocessing.py` cleaning and validation,
`test_pipeline.py` the agents and workflow (including retry and failure paths,
and a LangGraph-vs-Python parity check), `test_api.py` the HTTP endpoints. The
API and LangGraph tests skip themselves if those packages are not installed.

## Sample data

- `python scripts/make_synthetic_data.py` creates a deliberately messy churn
  dataset (missing values, duplicates, numbers stored as text, outliers, a
  constant column, inconsistent labels).
- Download the real Telco churn dataset from Kaggle
  (https://www.kaggle.com/datasets/blastchar/telco-customer-churn) into
  `datasets/sample/`. Its `TotalCharges` column is text with blank entries,
  which the profiler detects.

## Privacy (GDPR)

- When an OpenAI key is set, only aggregate statistics are sent to the LLM.
  Raw cell values (samples, top values, label variants) are stripped first.
  Column names are still sent.
- Uploaded files are stored on disk under random IDs. `DELETE /datasets/{id}`
  removes the file and its metadata.
- Before real users: add authentication, a retention policy, and a data
  processing agreement with your LLM provider.

## Roadmap

- [x] Phase 1: upload, profiling, summary, API, UI, Docker
- [x] Phase 2: LangGraph graph (Analyst, Quality, Preprocess, Validate) with retry
- [ ] Phase 3: model training, MLflow, evaluation agents
- [ ] Phase 4: SHAP, human approval, retries, RAG, memory
- [ ] Phase 5: deployment packages, Evidently drift monitoring
- [ ] Phase 6: benchmark, auth, public deployment, demo video

## Project layout

```
frontend/     Streamlit UI
backend/      FastAPI app and services
ml_engine/    Deterministic data/ML code (profiling, preprocessing, validation)
agents/       Analyst and Quality agents
workflows/    LangGraph data-preparation pipeline
rag/          Retrieval (Phase 4)
monitoring/   Drift detection (Phase 5)
database/     SQLAlchemy models and connection
tests/        pytest suite
scripts/      Helper scripts
docker/       Dockerfile
```
