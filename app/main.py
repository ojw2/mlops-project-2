"""FastAPI inference service for the Iris classifier.

Endpoints:
    GET  /healthz   - liveness probe (is the process up?)
    GET  /readyz    - readiness probe (is the model loaded?)
    GET  /metadata  - model version + training info
    POST /predict   - run inference
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(__file__).resolve().parent / "model"

# Loaded at startup, held in app state
state: dict = {"model": None, "metadata": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup instead of per-request."""
    state["model"] = joblib.load(MODEL_DIR / "model.joblib")
    with open(MODEL_DIR / "metadata.json") as f:
        state["metadata"] = json.load(f)
    yield
    state["model"] = None
    state["metadata"] = None


app = FastAPI(title="iris-classifier", lifespan=lifespan)


class PredictRequest(BaseModel):
    sepal_length: float = Field(..., ge=0, le=20, examples=[5.1])
    sepal_width: float = Field(..., ge=0, le=20, examples=[3.5])
    petal_length: float = Field(..., ge=0, le=20, examples=[1.4])
    petal_width: float = Field(..., ge=0, le=20, examples=[0.2])


class PredictResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]
    model_version: str


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.get("/metadata")
def metadata():
    return state["metadata"]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    features = [[req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]]
    model = state["model"]
    meta = state["metadata"]

    pred_idx = int(model.predict(features)[0])
    probs = model.predict_proba(features)[0]

    return PredictResponse(
        prediction=meta["target_names"][pred_idx],
        probabilities={
            name: round(float(p), 4)
            for name, p in zip(meta["target_names"], probs)
        },
        model_version=meta["model_version"],
    )
