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
    confusion_matrix,
    classification_report
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
    / "clfnet_fold1_best.keras"
)


# ============================================================
# START
# ============================================================

print("========================================")
print("       ClfNet MODEL EVALUATION")
print("========================================")

print("\nFold:", FOLD)

print("\nModel path:")
print(MODEL_PATH)


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not MODEL_PATH.exists():

    print("\nERROR: Model file not found!")

    print("\nExpected model:")
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
# SELECT FOLD 1 TEST DATA
# ============================================================

test_df = df[
    (df["split"] == "Test") &
    (df["fold"] == FOLD)
].copy()


print("\n========================================")
print(f"        {FOLD.upper()} TEST DATA")
print("========================================")

print("\nTest images:", len(test_df))

print("\nActual class distribution:")
print(test_df["class_name"].value_counts())


# ============================================================
# CREATE TEST DATASET
# ============================================================

print("\n========================================")
print("       CREATING TEST DATASET")
print("========================================")

test_dataset = create_tf_dataset(
    test_df,
    shuffle=False
)

print("\nTest dataset created successfully.")


# ============================================================
# BUILD MODEL
# ============================================================

print("\n========================================")
print("          BUILDING ClfNet")
print("========================================")

model = build_clfnet()

print("\nClfNet architecture created successfully.")

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)

print("\nTotal parameters:")
print(model.count_params())


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

print("\n========================================")
print("       LOADING TRAINED WEIGHTS")
print("========================================")

try:

    model.load_weights(MODEL_PATH)

    print("\nTrained weights loaded successfully.")

except Exception as e:

    print("\nERROR: Could not load trained weights.")

    print("\nError:")
    print(e)

    sys.exit()


# ============================================================
# MAKE PREDICTIONS
# ============================================================

print("\n========================================")
print("          MAKING PREDICTIONS")
print("========================================")

predictions = model.predict(
    test_dataset,
    verbose=1
)

print("\nPredictions completed.")

print("\nPrediction shape:")
print(predictions.shape)


# ============================================================
# CONVERT PREDICTIONS TO CLASS LABELS
# ============================================================

predicted_labels = np.argmax(
    predictions,
    axis=1
)

actual_labels = test_df["label"].values


# ============================================================
# CHECK LABEL COUNT
# ============================================================

print("\n========================================")
print("          LABEL CHECK")
print("========================================")

print("\nActual labels:", len(actual_labels))

print("Predicted labels:", len(predicted_labels))


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)


print("\n========================================")
print("          EVALUATION RESULT")
print("========================================")

print("\nAccuracy:")
print(f"{accuracy:.4f}")

print(
    f"\nAccuracy percentage: "
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
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("      CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        actual_labels,
        predicted_labels,
        target_names=[
            "Negative",
            "Positive"
        ],
        digits=4
    )
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
# SAMPLE PREDICTIONS
# ============================================================

print("\n========================================")
print("        SAMPLE PREDICTIONS")
print("========================================")

for i in range(min(10, len(test_df))):

    # Actual class
    if actual_labels[i] == 0:
        actual_class = "Negative"
    else:
        actual_class = "Positive"

    # Predicted class
    if predicted_labels[i] == 0:
        predicted_class = "Negative"
    else:
        predicted_class = "Positive"

    # Confidence
    confidence = predictions[i][
        predicted_labels[i]
    ]

    print("\n----------------------------------------")

    print("Image:")
    print(
        Path(
            test_df.iloc[i]["image_path"]
        ).name
    )

    print("Actual:")
    print(actual_class)

    print("Predicted:")
    print(predicted_class)

    print("Confidence:")
    print(f"{confidence:.4f}")


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("       EVALUATION COMPLETE")
print("========================================")