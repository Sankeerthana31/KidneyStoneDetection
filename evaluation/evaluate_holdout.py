
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PREPROCESSING_DIR))

from dataset_loader import DATASET_PATH, get_classification_data
from tf_dataset import create_tf_dataset
from models.clfnet import build_clfnet


# Use the saved Fold 5 checkpoint
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "saved"
    / "clfnet_fold_5_best.keras"
)

print("=" * 45)
print("       HOLDOUT DATASET EVALUATION")
print("=" * 45)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Load dataset metadata
df = get_classification_data(DATASET_PATH)

# Select holdout images only
holdout_df = df[df["split"] == "holdout"].copy()

if holdout_df.empty:
    raise ValueError(
        "No holdout images found. Check dataset_loader.py "
        "and the holdout folder structure."
    )

print("\nModel:", MODEL_PATH.name)
print("Holdout images:", len(holdout_df))
print("\nActual class distribution:")
print(holdout_df["class_name"].value_counts())

# Create dataset in the same way as existing evaluation
holdout_dataset = create_tf_dataset(
    holdout_df,
    shuffle=False
)

# Build architecture and load checkpoint
model = build_clfnet()
model.load_weights(MODEL_PATH)

print("\nMaking predictions...")
predictions = model.predict(holdout_dataset, verbose=1)

predicted_labels = np.argmax(predictions, axis=1)
actual_labels = holdout_df["label"].to_numpy()

if len(actual_labels) != len(predicted_labels):
    raise ValueError(
        "Actual and predicted label counts do not match."
    )

accuracy = accuracy_score(actual_labels, predicted_labels)

print("\n" + "=" * 45)
print("HOLDOUT RESULTS")
print("=" * 45)
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

# Explicit label order: 0 = Negative, 1 = Positive
cm = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=[0, 1]
)

print("\nConfusion matrix (rows=actual, columns=predicted):")
print("                 Pred Negative  Pred Positive")
print(f"Actual Negative      {cm[0, 0]:5d}          {cm[0, 1]:5d}")
print(f"Actual Positive      {cm[1, 0]:5d}          {cm[1, 1]:5d}")

print("\nClassification report:")
print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        digits=4,
        zero_division=0,
    )
)

print("\nHoldout evaluation complete.")
