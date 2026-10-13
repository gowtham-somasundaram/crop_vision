
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys

TRAINING_DIR = Path(__file__).resolve().parents[2] / "1_Training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from classifier.models.mlp import build_model as build_mlp
from classifier.models.rbf_svm import build_model as build_rbf_svm
from classifier.models.knn import build_model as build_knn
from classifier.models.random_forest import build_model as build_random_forest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_FILE = (
    PROJECT_ROOT / "artifacts" / "embeddings" / "crop_embeddings.npz"
)
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
RESULTS_DIR = (
    PROJECT_ROOT
    / "3_Experiments"
    / "classifier_comparison"
    / "results"
)


def build_classifiers():
    """Build all candidate classifiers from their own modules."""
    return {
        "MLP": build_mlp(),
        "RBF_SVM": build_rbf_svm(),
        "KNN": build_knn(),
        "Random_Forest": build_random_forest(),
    }


def main():
    if not EMBEDDING_FILE.is_file():
        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDING_FILE}"
        )

    with np.load(EMBEDDING_FILE, allow_pickle=False) as data:
        X_train = data["X_train"]
        y_train = data["y_train"].astype(str)
        X_test = data["X_test"]
        y_test = data["y_test"].astype(str)

    if X_train.shape[0] != len(y_train):
        raise ValueError("Training embeddings and labels do not match.")

    if X_test.shape[0] != len(y_test):
        raise ValueError("Test embeddings and labels do not match.")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train/test embedding dimensions differ.")

    classes = sorted(set(y_train) | set(y_test))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    reports = {}
    predictions = {}

    classifiers = build_classifiers()

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Classes: {classes}")

    for name, model in classifiers.items():
        print(f"\nTraining {name}...")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)

        for warning in caught:
            print(f"Warning for {name}: {warning.message}")

        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        metrics = {
            "classifier": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "macro_precision": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "macro_recall": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "macro_f1": f1_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "weighted_f1": f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
        }
        results.append(metrics)

        reports[name] = classification_report(
            y_test,
            y_pred,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )

        model_path = MODEL_DIR / f"{name.lower()}.joblib"
        joblib.dump(model, model_path)

        cm = confusion_matrix(y_test, y_pred, labels=classes)
        np.savetxt(
            RESULTS_DIR / f"{name.lower()}_confusion_matrix.csv",
            cm,
            delimiter=",",
            fmt="%d",
        )

        print(
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

    # Sort for reporting only; no model is designated in advance.
    results.sort(key=lambda row: row["macro_f1"], reverse=True)

    import csv

    metrics_path = RESULTS_DIR / "classifier_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    report_path = RESULTS_DIR / "classification_reports.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(reports, file, indent=2)

    prediction_path = RESULTS_DIR / "test_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual_class", *predictions.keys()])

        for i, actual in enumerate(y_test):
            writer.writerow(
                [actual, *[predictions[name][i] for name in predictions]]
            )

    print("\n=== Classifier comparison (sorted by macro F1) ===")
    for row in results:
        print(
            f"{row['classifier']:15s} "
            f"Accuracy={row['accuracy']:.4f} "
            f"Balanced Acc={row['balanced_accuracy']:.4f} "
            f"Macro F1={row['macro_f1']:.4f}"
        )

    print(f"\nMetrics saved to: {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"Reports saved to: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Models saved under: {MODEL_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
