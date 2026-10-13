
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_FILE = (
    PROJECT_ROOT / "artifacts" / "embeddings" / "crop_embeddings.npz"
)
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"


def train_model():
    if not EMBEDDING_FILE.is_file():
        raise FileNotFoundError(
            "Embeddings not found. Run "
            "'uv run python 1_Training/embedding/extract.py' first."
        )

    with np.load(EMBEDDING_FILE, allow_pickle=False) as data:
        X_train = data["X_train"]
        y_train = data["y_train"].astype(str)

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=1500,
            random_state=42,
        ),
    )

    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "mlp.joblib"
    joblib.dump(model, model_path)

    print(f"Training images: {len(y_train)}")
    print(f"Classes: {sorted(set(y_train))}")
    print(f"Model saved to: {model_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    train_model()
