"""AIDME API (Phase 1): upload a dataset, get a profile and summary."""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.services.summary import summarize_profile
from database.connection import get_db, init_db
from database.models import Dataset, PreparedRun
from ml_engine.profiling import load_dataframe, profile_dataframe
from workflows.data_pipeline import result_report, run_pipeline

logger = logging.getLogger("aidme.api")

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
UPLOAD_DIR = DATA_DIR / "uploads"
PREPARED_DIR = DATA_DIR / "prepared"
PIPELINE_ENGINE = os.getenv("PIPELINE_ENGINE", "langgraph")  # or "python"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    yield


app = FastAPI(title="AIDME API", version="0.1.0", lifespan=lifespan)


def _summary_row(ds: Dataset) -> dict:
    return {
        "id": ds.id,
        "filename": ds.filename,
        "n_rows": ds.n_rows,
        "n_cols": ds.n_cols,
        "created_at": ds.created_at.isoformat() if ds.created_at else None,
    }


def _detail(ds: Dataset) -> dict:
    return {
        **_summary_row(ds),
        "summary": ds.summary,
        "summary_source": ds.summary_source,
        "profile": ds.profile,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Sync (def) endpoints on purpose: pandas work is CPU-bound and FastAPI runs
# sync endpoints in a thread pool instead of blocking the event loop.
@app.post("/datasets", status_code=201)
def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict:
    filename = Path(file.filename or "upload").name
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            415, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (limit {MAX_UPLOAD_BYTES // 2**20} MB).")

    try:
        df = load_dataframe(content, filename)
        profile = profile_dataframe(df)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    summary, source = summarize_profile(profile)

    dataset_id = str(uuid.uuid4())
    stored_path = UPLOAD_DIR / f"{dataset_id}{ext}"  # never trust client filenames
    stored_path.write_bytes(content)

    record = Dataset(
        id=dataset_id,
        filename=filename,
        stored_path=str(stored_path),
        n_rows=profile["overview"]["n_rows"],
        n_cols=profile["overview"]["n_cols"],
        profile=profile,
        summary=summary,
        summary_source=source,
    )
    db.add(record)
    db.commit()
    return _detail(record)


@app.get("/datasets")
def list_datasets(db: Session = Depends(get_db)) -> list[dict]:
    rows = db.scalars(select(Dataset).order_by(Dataset.created_at.desc()).limit(100))
    return [_summary_row(r) for r in rows]


@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, db: Session = Depends(get_db)) -> dict:
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(404, "Dataset not found.")
    return _detail(ds)


@app.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)) -> None:
    """Remove the stored file and its metadata (data minimisation / GDPR)."""
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(404, "Dataset not found.")
    runs = db.scalars(select(PreparedRun).where(PreparedRun.dataset_id == dataset_id)).all()
    for run in runs:
        if run.output_path:
            Path(run.output_path).unlink(missing_ok=True)
        db.delete(run)
    db.flush()  # delete child rows before the parent (PostgreSQL enforces the foreign key)
    Path(ds.stored_path).unlink(missing_ok=True)
    db.delete(ds)
    db.commit()


# --------------------------------------------------------------------------- #
# Data preparation (Phase 2)
# --------------------------------------------------------------------------- #
class PrepareRequest(BaseModel):
    target: str | None = Field(default=None, max_length=255)
    task: Literal["classification", "regression"] | None = None
    instruction: str | None = Field(default=None, max_length=2000)
    max_retries: int = Field(default=2, ge=0, le=3)


def _run_summary(run: PreparedRun) -> dict:
    return {
        "id": run.id,
        "dataset_id": run.dataset_id,
        "status": run.status,
        "target": run.target,
        "task": run.task,
        "has_output": bool(run.output_path),
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


def _run_detail(run: PreparedRun) -> dict:
    return {**_run_summary(run), "instruction": run.instruction, "report": run.report}


@app.post("/datasets/{dataset_id}/prepare")
def prepare_dataset(
    dataset_id: str, req: PrepareRequest, db: Session = Depends(get_db)
) -> dict:
    """Run Analyst -> Quality -> Preprocess -> Validate on a stored dataset.

    A pipeline that cannot produce valid data still returns HTTP 200 with
    status "failed" and the reasons, because the request itself was processed.
    """
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(404, "Dataset not found.")
    path = Path(ds.stored_path)
    if not path.exists():
        raise HTTPException(410, "The stored file is gone; please upload the dataset again.")

    try:
        df = load_dataframe(path.read_bytes(), ds.filename)
        final = run_pipeline(
            df,
            instruction=req.instruction,
            target=req.target,
            task=req.task,
            max_retries=req.max_retries,
            engine=PIPELINE_ENGINE,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(503, str(exc)) from exc

    run_id = str(uuid.uuid4())
    output_path = None
    if final["status"] == "succeeded":
        out = PREPARED_DIR / f"{run_id}.csv"
        final["df_clean"].to_csv(out, index=False)
        output_path = str(out)

    run = PreparedRun(
        id=run_id,
        dataset_id=dataset_id,
        status=final["status"],
        target=final.get("target"),
        task=final.get("task"),
        instruction=req.instruction,
        report=result_report(final),
        output_path=output_path,
    )
    db.add(run)
    db.commit()
    return _run_detail(run)


@app.get("/datasets/{dataset_id}/prepared")
def list_prepared_runs(dataset_id: str, db: Session = Depends(get_db)) -> list[dict]:
    if db.get(Dataset, dataset_id) is None:
        raise HTTPException(404, "Dataset not found.")
    runs = db.scalars(
        select(PreparedRun)
        .where(PreparedRun.dataset_id == dataset_id)
        .order_by(PreparedRun.created_at.desc())
        .limit(50)
    )
    return [_run_summary(r) for r in runs]


@app.get("/prepared/{run_id}")
def get_prepared_run(run_id: str, db: Session = Depends(get_db)) -> dict:
    run = db.get(PreparedRun, run_id)
    if run is None:
        raise HTTPException(404, "Run not found.")
    return _run_detail(run)


@app.get("/prepared/{run_id}/download")
def download_prepared(run_id: str, db: Session = Depends(get_db)) -> FileResponse:
    run = db.get(PreparedRun, run_id)
    if run is None or not run.output_path or not Path(run.output_path).exists():
        raise HTTPException(404, "No prepared file for this run.")
    return FileResponse(
        run.output_path, media_type="text/csv", filename=f"prepared_{run_id[:8]}.csv"
    )
