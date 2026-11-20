# AIDME: Autonomous AI Data & ML Engineer

Upload a dataset, describe a business problem, get a validated, deployable,
monitored ML model. Built as a multi-agent system (LangGraph) on top of
deterministic, tested Python pipelines.

**Status: Phase 1 (foundation) complete.** Upload a CSV/Excel file and get an
automatic data profile, data-quality issues, likely prediction targets and a
plain-English summary.

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

## Tests

```bash
pytest
```

`tests/test_ml.py` covers the profiling logic. `tests/test_api.py` covers the
HTTP endpoints (it skips itself if the API packages are not installed).

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
- [ ] Phase 2: LangGraph graph (Analyst, Quality, Preprocess) with validation
- [ ] Phase 3: model training, MLflow, evaluation agents
- [ ] Phase 4: SHAP, human approval, retries, RAG, memory
- [ ] Phase 5: deployment packages, Evidently drift monitoring
- [ ] Phase 6: benchmark, auth, public deployment, demo video

## Project layout

```
frontend/     Streamlit UI
backend/      FastAPI app and services
ml_engine/    Deterministic data/ML code (profiling.py so far)
agents/       LangGraph agents (Phase 2+)
workflows/    Pipelines (Phase 2+)
rag/          Retrieval (Phase 4)
monitoring/   Drift detection (Phase 5)
database/     SQLAlchemy models and connection
tests/        pytest suite
scripts/      Helper scripts
docker/       Dockerfile
```
