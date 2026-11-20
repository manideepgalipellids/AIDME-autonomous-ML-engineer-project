import os
import tempfile

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("sqlalchemy")
pytest.importorskip("multipart")

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["DATA_DIR"] = _tmp
os.environ.pop("OPENAI_API_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402

CSV = b"id,tenure,churn\n1,5,Yes\n2,8,No\n3,12,No\n4,3,Yes\n5,40,No\n"


@pytest.fixture()
def client():
    with TestClient(app) as c:  # context manager runs the lifespan (creates tables)
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_profile_list_get_delete(client):
    resp = client.post("/datasets", files={"file": ("tiny.csv", CSV, "text/csv")})
    assert resp.status_code == 201
    body = resp.json()
    assert body["n_rows"] == 5
    assert body["summary_source"] == "rules"
    assert body["profile"]["candidate_targets"][0]["column"] == "churn"

    ds_id = body["id"]
    assert any(d["id"] == ds_id for d in client.get("/datasets").json())
    assert client.get(f"/datasets/{ds_id}").status_code == 200

    assert client.delete(f"/datasets/{ds_id}").status_code == 204
    assert client.get(f"/datasets/{ds_id}").status_code == 404


def test_rejects_unsupported_type(client):
    resp = client.post("/datasets", files={"file": ("x.exe", b"abc", "application/octet-stream")})
    assert resp.status_code == 415


def test_rejects_empty_file(client):
    resp = client.post("/datasets", files={"file": ("empty.csv", b"", "text/csv")})
    assert resp.status_code == 422
