from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

# PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_PATH = (
    PROJECT_ROOT / "artifacts" / "embeddings" / "crop_embeddings.npz"
)

TRAINING_DIR = PROJECT_ROOT / "1_Training"
sys.path.insert(0, str(TRAINING_DIR))

from classifier.models.mlp import build_model

RESULTS_DIR = (
    PROJECT_ROOT / "3_Experiments" / "data_efficiency" / "results"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# EXPERIMENT CONFIGURATION
DATA_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
RANDOM_SEEDS = [42, 21, 7, 100, 2026]

# LOAD EXISTING EMBEDDINGS
if not EMBEDDINGS_PATH.exists():
    raise FileNotFoundError(
        f"Embeddings not found: {EMBEDDINGS_PATH}\n"
        "Run 1_Training/embedding/extract.py first."
    )

data = np.load(EMBEDDINGS_PATH, allow_pickle=True)

X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

print("=" * 65)
print("CROPVISION: TRAINING-DATA EFFICIENCY EXPERIMENT")
print("=" * 65)
print(f"Full training set: {X_train.shape}")
print(f"Fixed test set:    {X_test.shape}")
print(f"Classes:           {len(np.unique(y_train))}")

# TRAIN AND EVALUATE
results = []

for fraction in DATA_FRACTIONS:
    subset_size = max(1, int(round(len(X_train) * fraction)))

    for seed in RANDOM_SEEDS:

        # Keep the full training set intact at 100%.
        # Smaller subsets are sampled with stratification.
        if fraction == 1.0:
            X_subset = X_train
            y_subset = y_train
        else:
            X_subset, _, y_subset, _ = train_test_split(
                X_train,
                y_train,
                train_size=fraction,
                random_state=seed,
                stratify=y_train,
            )

        # Build a fresh classifier for each experiment.
        model = build_model()

        # Vary MLP initialization seed as well as subset seed.
        model.set_params(mlpclassifier__random_state=seed)

        model.fit(X_subset, y_subset)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        results.append({
            "training_fraction": int(fraction * 100),
            "training_images": len(X_subset),
            "seed": seed,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "macro_f1": macro_f1,
        })

        print(
            f"{int(fraction * 100):3d}% | "
            f"{len(X_subset):3d} images | "
            f"seed={seed:4d} | "
            f"accuracy={accuracy:.4f} | "
            f"macro_f1={macro_f1:.4f}"
        )

# SAVE INDIVIDUAL RUNS
results_df = pd.DataFrame(results)

raw_path = RESULTS_DIR / "data_efficiency_runs.csv"
results_df.to_csv(raw_path, index=False)

# SUMMARY: MEAN AND STANDARD DEVIATION
summary = (
    results_df.groupby(
        ["training_fraction", "training_images"]
    )
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    )
    .reset_index()
)

summary_path = RESULTS_DIR / "data_efficiency_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n" + "=" * 65)
print("SUMMARY: MEAN TEST PERFORMANCE")
print("=" * 65)

print(
    summary[
        [
            "training_fraction",
            "training_images",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
        ]
    ].round(4).to_string(index=False)
)

# PLOT: DATA SIZE VS ACCURACY
plt.figure(figsize=(9, 5))

plt.errorbar(
    summary["training_fraction"],
    summary["accuracy_mean"],
    yerr=summary["accuracy_std"].fillna(0),
    marker="o",
    capsize=5,
    label="Accuracy",
)

plt.errorbar(
    summary["training_fraction"],
    summary["macro_f1_mean"],
    yerr=summary["macro_f1_std"].fillna(0),
    marker="s",
    capsize=5,
    label="Macro F1",
)

plt.xticks(DATA_FRACTIONS, ["25%", "50%", "75%", "100%"])
plt.xlabel("Percentage of Training Data")
plt.ylabel("Test Score")
plt.title("CropVision: Training Data Efficiency")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = RESULTS_DIR / "data_efficiency_plot.png"
plt.savefig(plot_path, dpi=200)
plt.close()

print("\nExperiment completed.")
print(f"Individual runs: {raw_path}")
print(f"Summary:         {summary_path}")
print(f"Plot:            {plot_path}")