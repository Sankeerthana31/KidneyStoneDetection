import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PREPROCESSING_DIR))


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)

from dataset_loader import (
    DATASET_PATH,
    get_classification_data
)

from tf_dataset import create_tf_dataset

from models.clfnet import build_clfnet


# ============================================================
# CONFIGURATION
# ============================================================

FOLD = "Fold_1"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "saved"
    / "clfnet_fold1.keras"
)


# ============================================================
# START
# ============================================================

print("========================================")
print("    ClfNet TRAINING PREDICTION CHECK")
print("========================================")

print("\nFold:", FOLD)

print("\nModel:")
print(MODEL_PATH)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    print("\nERROR: Model file not found!")

    print(MODEL_PATH)

    sys.exit()


# ============================================================
# LOAD DATASET
# ============================================================

print("\n========================================")
print("       LOADING DATASET")
print("========================================")

df = get_classification_data(DATASET_PATH)

print("\nTotal images:", len(df))


# ============================================================
# SELECT TRAINING DATA
# ============================================================

train_df = df[
    (df["split"] == "Training") &
    (df["fold"] == FOLD)
].copy()


print("\n========================================")
print("        FOLD 1 TRAINING DATA")
print("========================================")

print("\nTraining images:", len(train_df))

print("\nActual class distribution:")
print(
    train_df["class_name"].value_counts()
)


# ============================================================
# CREATE DATASET
# ============================================================

print("\n========================================")
print("       CREATING TRAINING DATASET")
print("========================================")

train_dataset = create_tf_dataset(
    train_df,
    shuffle=False
)

print("\nTraining dataset created.")


# ============================================================
# BUILD MODEL
# ============================================================

print("\n========================================")
print("          BUILDING ClfNet")
print("========================================")

model = build_clfnet()

print("\nModel created.")

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)

print("\nTotal parameters:")
print(model.count_params())


# ============================================================
# LOAD WEIGHTS
# ============================================================

print("\n========================================")
print("       LOADING TRAINED WEIGHTS")
print("========================================")

model.load_weights(MODEL_PATH)

print("\nWeights loaded successfully.")


# ============================================================
# MAKE TRAINING PREDICTIONS
# ============================================================

print("\n========================================")
print("      MAKING TRAINING PREDICTIONS")
print("========================================")

predictions = model.predict(
    train_dataset,
    verbose=1
)

print("\nPrediction shape:")
print(predictions.shape)


# ============================================================
# CONVERT TO CLASS LABELS
# ============================================================

predicted_labels = np.argmax(
    predictions,
    axis=1
)

actual_labels = train_df["label"].values


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)


print("\n========================================")
print("       TRAINING EVALUATION")
print("========================================")

print(
    f"\nTraining accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    actual_labels,
    predicted_labels
)


print("\n========================================")
print("        CONFUSION MATRIX")
print("========================================")

print("\n                 Predicted")
print("               Negative Positive")

print(
    f"Actual Negative   "
    f"{cm[0][0]:4d}     "
    f"{cm[0][1]:4d}"
)

print(
    f"Actual Positive   "
    f"{cm[1][0]:4d}     "
    f"{cm[1][1]:4d}"
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n========================================")
print("       PREDICTION DISTRIBUTION")
print("========================================")

unique, counts = np.unique(
    predicted_labels,
    return_counts=True
)

for label, count in zip(unique, counts):

    if label == 0:
        class_name = "Negative"
    else:
        class_name = "Positive"

    print(
        f"{class_name}: {count}"
    )


# ============================================================
# PREDICTION CONFIDENCE
# ============================================================

print("\n========================================")
print("       PREDICTION CONFIDENCE")
print("========================================")

negative_confidence = predictions[:, 0]
positive_confidence = predictions[:, 1]

print(
    "\nAverage Negative probability:"
)

print(
    f"{negative_confidence.mean():.4f}"
)

print(
    "\nAverage Positive probability:"
)

print(
    f"{positive_confidence.mean():.4f}"
)


# ============================================================
# SAMPLE POSITIVE TRAINING IMAGES
# ============================================================

print("\n========================================")
print("    SAMPLE POSITIVE TRAINING PREDICTIONS")
print("========================================")

positive_indices = np.where(
    actual_labels == 1
)[0]

for i in positive_indices[:10]:

    predicted_class = (
        "Negative"
        if predicted_labels[i] == 0
        else "Positive"
    )

    print("\nImage:")
    print(
        Path(
            train_df.iloc[i]["image_path"]
        ).name
    )

    print("Actual: Positive")

    print("Predicted:")
    print(predicted_class)

    print(
        "Negative probability:",
        f"{predictions[i][0]:.4f}"
    )

    print(
        "Positive probability:",
        f"{predictions[i][1]:.4f}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("       DIAGNOSTIC COMPLETE")
print("========================================")