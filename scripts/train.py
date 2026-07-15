"""Train a simple classifier on the Iris dataset and save it as a versioned artifact.

Usage:
    python scripts/train.py

Output:
    app/model/model.joblib
    app/model/metadata.json
"""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parent.parent / "app" / "model"
MODEL_VERSION = "0.1.0"


def main() -> None:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {accuracy:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "test_accuracy": round(accuracy, 4),
        "feature_names": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
        "target_names": ["setosa", "versicolor", "virginica"],
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
