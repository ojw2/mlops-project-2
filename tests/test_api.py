"""API tests using FastAPI's TestClient.

Requires a trained model in app/model/ — the CI workflow (and conftest-less
local runs) should execute `python scripts/train.py` first. The Makefile
and workflow both handle this.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # TestClient as a context manager triggers the lifespan (model load)
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200


def test_readyz_after_startup(client):
    r = client.get("/readyz")
    assert r.status_code == 200


def test_metadata(client):
    r = client.get("/metadata")
    assert r.status_code == 200
    assert "model_version" in r.json()


def test_predict_happy_path(client):
    r = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in ("setosa", "versicolor", "virginica")
    assert pytest.approx(sum(body["probabilities"].values()), abs=0.01) == 1.0


def test_predict_rejects_missing_field(client):
    r = client.post("/predict", json={"sepal_length": 5.1})
    assert r.status_code == 422


def test_predict_rejects_out_of_range(client):
    r = client.post(
        "/predict",
        json={
            "sepal_length": 999,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    assert r.status_code == 422
