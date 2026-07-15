"""Tests for the training pipeline.

These run train.main() against a temp directory and assert on the
artifacts it produces — the contract the serving app depends on.
"""

import json

import joblib
import pytest

from scripts import train


@pytest.fixture()
def trained_model_dir(tmp_path, monkeypatch):
    """Run training with MODEL_DIR pointed at a temp directory."""
    model_dir = tmp_path / "model"
    monkeypatch.setattr(train, "MODEL_DIR", model_dir)
    train.main()
    return model_dir


def test_training_produces_artifacts(trained_model_dir):
    assert (trained_model_dir / "model.joblib").exists()
    assert (trained_model_dir / "metadata.json").exists()


def test_metadata_contract(trained_model_dir):
    """The serving app depends on these exact keys."""
    with open(trained_model_dir / "metadata.json") as f:
        meta = json.load(f)

    assert set(meta["target_names"]) == {"setosa", "versicolor", "virginica"}
    assert len(meta["feature_names"]) == 4
    assert meta["model_version"]


def test_model_meets_accuracy_floor(trained_model_dir):
    """Quality gate: block deploys of models that regressed badly."""
    with open(trained_model_dir / "metadata.json") as f:
        meta = json.load(f)
    assert meta["test_accuracy"] >= 0.85, "model accuracy below deploy threshold"


def test_model_predicts_valid_classes(trained_model_dir):
    model = joblib.load(trained_model_dir / "model.joblib")
    preds = model.predict([[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]])
    assert all(p in (0, 1, 2) for p in preds)
