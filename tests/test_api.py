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
try:  # use the real LangGraph engine whenever it is installed
    import langgraph  # noqa: F401

    os.environ["PIPELINE_ENGINE"] = "langgraph"
except ImportError:
    os.environ["PIPELINE_ENGINE"] = "python"

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


def _prep_csv(n: int = 120) -> bytes:
    lines = ["customer_id,tenure,charges,plan,churn"]
    for i in range(n):
        plan = ["basic", "pro", "team"][i % 3]
        churn = "Yes" if (i * 7) % 10 < 3 else "No"
        lines.append(f"C{i},{i % 60 + 1},{50 + (i * 13) % 40}.5,{plan},{churn}")
    return ("\n".join(lines) + "\n").encode()


def _upload(client, content: bytes = None) -> str:
    resp = client.post(
        "/datasets", files={"file": ("prep.csv", content or _prep_csv(), "text/csv")}
    )
    assert resp.status_code == 201
    return resp.json()["id"]


def test_prepare_success_download_and_cleanup(client):
    ds_id = _upload(client)
    resp = client.post(f"/datasets/{ds_id}/prepare", json={})
    assert resp.status_code == 200
    run = resp.json()
    assert run["status"] == "succeeded"
    assert run["target"] == "churn" and run["task"] == "classification"
    assert run["has_output"] is True
    assert run["report"]["validation"]["passed"] is True

    csv = client.get(f"/prepared/{run['id']}/download")
    assert csv.status_code == 200
    header = csv.text.splitlines()[0]
    assert "churn" in header and "customer_id" not in header  # identifier dropped

    listed = client.get(f"/datasets/{ds_id}/prepared").json()
    assert [r["id"] for r in listed] == [run["id"]]

    assert client.delete(f"/datasets/{ds_id}").status_code == 204
    assert client.get(f"/prepared/{run['id']}").status_code == 404


def test_prepare_unknown_target_reports_failure(client):
    ds_id = _upload(client)
    resp = client.post(f"/datasets/{ds_id}/prepare", json={"target": "nope"})
    assert resp.status_code == 200
    run = resp.json()
    assert run["status"] == "failed"
    assert run["has_output"] is False
    assert "was not found" in run["report"]["errors"][0]
    assert client.get(f"/prepared/{run['id']}/download").status_code == 404


def test_prepare_unknown_dataset_is_404(client):
    assert client.post("/datasets/does-not-exist/prepare", json={}).status_code == 404


def test_prepare_rejects_invalid_task(client):
    ds_id = _upload(client)
    resp = client.post(f"/datasets/{ds_id}/prepare", json={"task": "clustering"})
    assert resp.status_code == 422


def test_train_download_and_guard(client):
    ds_id = _upload(client, _prep_csv(240))
    run = client.post(f"/datasets/{ds_id}/prepare", json={}).json()
    assert run["status"] == "succeeded"

    resp = client.post(f"/prepared/{run['id']}/train", json={"cv_folds": 3})
    assert resp.status_code == 200
    tr = resp.json()
    assert tr["status"] == "succeeded"
    assert tr["report"]["best_model"] and tr["has_model"]
    assert client.get(f"/training/{tr['id']}/model").status_code == 200
    assert len(client.get(f"/prepared/{run['id']}/training").json()) == 1

    # a failed preparation cannot be trained on
    bad = client.post(f"/datasets/{ds_id}/prepare", json={"target": "nope"}).json()
    assert client.post(f"/prepared/{bad['id']}/train", json={}).status_code == 409

    assert client.delete(f"/datasets/{ds_id}").status_code == 204
    assert client.get(f"/training/{tr['id']}/model").status_code == 404
