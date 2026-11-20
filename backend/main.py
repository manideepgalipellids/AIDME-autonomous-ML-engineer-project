"""AIDME API (Phase 1): upload a dataset, get a profile and summary."""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.services.summary import summarize_profile
from database.connection import get_db, init_db
from database.models import Dataset
from ml_engine.profiling import load_dataframe, profile_dataframe

logger = logging.getLogger("aidme.api")

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
UPLOAD_DIR = DATA_DIR / "uploads"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
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
    Path(ds.stored_path).unlink(missing_ok=True)
    db.delete(ds)
    db.commit()
