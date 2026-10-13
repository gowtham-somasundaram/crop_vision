
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "1_Training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from data import load_image_paths
from embedding.dinov3 import DINOv3Embedder


DATASET_DIR = PROJECT_ROOT / "Dataset" / "independent_images"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
RESULTS_DIR = (
    PROJECT_ROOT / "3_Experiments" / "independent_evaluation" / "results"
)

MODEL_FILES = {
    "MLP": "mlp.joblib",
    "RBF_SVM": "rbf_svm.joblib",
    "KNN": "knn.joblib",
    "Random_Forest": "random_forest.joblib",
}


def main():
    
    if not DATASET_DIR.is_dir():
        print(
            "Independent evaluation skipped: dataset not found at "
            f"{DATASET_DIR.relative_to(PROJECT_ROOT)}"
        )
        print("Add a separate labeled dataset when one is available.")
        return


    image_paths, labels = load_image_paths(DATASET_DIR)

    print(f"Independent images: {len(image_paths)}")
    print(f"Classes: {sorted(set(labels))}")

    embedder = DINOv3Embedder()
    X = embedder.embed_images(image_paths).numpy()
    y_true = np.asarray(labels, dtype=str)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    reports = {}
    prediction_columns = {
        "image": [
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in image_paths
        ],
        "actual_class": y_true,
    }

    for model_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename

        if not model_path.is_file():
            print(f"Skipping {model_name}: model not found at {model_path}")
            continue

        model = joblib.load(model_path)
        y_pred = model.predict(X).astype(str)

        metrics_rows.append(
            {
                "classifier": model_name,
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(
                    y_true, y_pred
                ),
                "macro_precision": precision_score(
                    y_true, y_pred, average="macro", zero_division=0
                ),
                "macro_recall": recall_score(
                    y_true, y_pred, average="macro", zero_division=0
                ),
                "macro_f1": f1_score(
                    y_true, y_pred, average="macro", zero_division=0
                ),
                "weighted_f1": f1_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
            }
        )

        reports[model_name] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

        prediction_columns[f"{model_name}_prediction"] = y_pred

        print(
            f"{model_name}: "
            f"accuracy={accuracy_score(y_true, y_pred):.4f}, "
            f"balanced_accuracy={balanced_accuracy_score(y_true, y_pred):.4f}, "
            f"macro_f1={f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
        )

    if not metrics_rows:
        raise FileNotFoundError(
            f"No saved classifier models found in {MODEL_DIR}"
        )

    pd.DataFrame(metrics_rows).to_csv(
        RESULTS_DIR / "independent_metrics.csv", index=False
    )
    pd.DataFrame(prediction_columns).to_csv(
        RESULTS_DIR / "independent_predictions.csv", index=False
    )

    import json

    with (RESULTS_DIR / "classification_reports.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(reports, file, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
