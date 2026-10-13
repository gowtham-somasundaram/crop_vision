
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "1_Training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

EMBEDDING_FILE = (
    PROJECT_ROOT / "artifacts" / "embeddings" / "crop_embeddings.npz"
)
MODEL_FILE = PROJECT_ROOT / "artifacts" / "models" / "mlp.joblib"
RESULTS_DIR = PROJECT_ROOT / "3_Experiments" / "embedding_audit" / "results"


def main(top_k: int = 5):
    if not EMBEDDING_FILE.is_file():
        raise FileNotFoundError(f"Embeddings not found: {EMBEDDING_FILE}")

    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

    with np.load(EMBEDDING_FILE, allow_pickle=False) as data:
        X_train = data["X_train"].astype(np.float32)
        y_train = data["y_train"].astype(str)
        paths_train = data["paths_train"].astype(str)
        X_test = data["X_test"].astype(np.float32)
        y_test = data["y_test"].astype(str)
        paths_test = data["paths_test"].astype(str)

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Embedding and label counts do not match.")

    if not 1 <= top_k <= len(X_train):
        raise ValueError(f"top_k must be between 1 and {len(X_train)}.")

    # Embeddings are already L2-normalized by the extraction pipeline.
    similarities = cosine_similarity(X_test, X_train)
    model = joblib.load(MODEL_FILE)
    predictions = model.predict(X_test).astype(str)

    classes = sorted(set(y_train))
    rows = []

    for i, test_path in enumerate(paths_test):
        order = np.argsort(similarities[i])[::-1]
        top_indices = order[:top_k]

        record = {
            "test_image": test_path,
            "actual_class": y_test[i],
            "predicted_class": predictions[i],
            "is_correct": bool(y_test[i] == predictions[i]),
        }

        # Closest training example belonging to each class.
        for class_name in classes:
            class_indices = np.flatnonzero(y_train == class_name)
            best_idx = class_indices[
                np.argmax(similarities[i, class_indices])
            ]
            record[f"max_similarity_{class_name}"] = float(
                similarities[i, best_idx]
            )
            record[f"closest_{class_name}_image"] = paths_train[best_idx]

        for rank, train_idx in enumerate(top_indices, start=1):
            record[f"neighbor_{rank}_image"] = paths_train[train_idx]
            record[f"neighbor_{rank}_class"] = y_train[train_idx]
            record[f"neighbor_{rank}_similarity"] = float(
                similarities[i, train_idx]
            )

        rows.append(record)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "embedding_audit.csv"

    results = pd.DataFrame(rows)
    results.to_csv(output_path, index=False)

    errors_path = RESULTS_DIR / "embedding_audit_errors_only.csv"
    results.loc[~results["is_correct"]].to_csv(errors_path, index=False)

    print(f"Test images audited: {len(results)}")
    print(f"Incorrect predictions: {(~results['is_correct']).sum()}")
    print(f"Full audit: {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Errors only: {errors_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
