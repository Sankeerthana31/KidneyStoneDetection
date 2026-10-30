import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PREPROCESSING_DIR))

import numpy as np

from dataset_loader import DATASET_PATH, get_classification_data
from tf_dataset import create_tf_dataset
from models.clfnet import build_clfnet


FOLD = "Fold_1"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "saved"
    / "clfnet_fold1.keras"
)


print("========================================")
print("       POSITIVE IMAGE DIAGNOSTIC")
print("========================================")


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = get_classification_data(DATASET_PATH)

train_df = df[
    (df["split"] == "Training") &
    (df["fold"] == FOLD) &
    (df["class_name"] == "Positive")
]

print("\nPositive training images:", len(train_df))

print("\nExpected label:")
print("Positive = 1")


# --------------------------------------------------
# 2. TAKE FIRST 32 POSITIVE IMAGES
# --------------------------------------------------

positive_df = train_df.head(32).copy()

print("\nImages used for testing:", len(positive_df))


# --------------------------------------------------
# 3. CREATE DATASET
# --------------------------------------------------

dataset = create_tf_dataset(
    positive_df,
    shuffle=False
)

images, labels = next(iter(dataset))

print("\nImage batch shape:")
print(images.shape)

print("\nLabels:")
print(labels.numpy())


# --------------------------------------------------
# 4. BUILD MODEL
# --------------------------------------------------

print("\n========================================")
print("          BUILDING MODEL")
print("========================================")

model = build_clfnet()

print("\nModel created.")


# --------------------------------------------------
# 5. LOAD WEIGHTS
# --------------------------------------------------

print("\n========================================")
print("       LOADING SAVED WEIGHTS")
print("========================================")

model.load_weights(MODEL_PATH)

print("\nWeights loaded successfully.")


# --------------------------------------------------
# 6. INFERENCE MODE
# --------------------------------------------------

print("\n========================================")
print("       INFERENCE MODE")
print("       training=False")
print("========================================")

pred_inference = model(
    images,
    training=False
).numpy()

pred_classes = np.argmax(
    pred_inference,
    axis=1
)

print("\nFirst 32 predictions:")

for i in range(len(labels)):

    print(
        f"Image {i+1}: "
        f"Actual=1 | "
        f"Predicted={pred_classes[i]} | "
        f"Negative={pred_inference[i][0]:.4f} | "
        f"Positive={pred_inference[i][1]:.4f}"
    )


positive_predictions = np.sum(
    pred_classes == 1
)

negative_predictions = np.sum(
    pred_classes == 0
)

print("\n========================================")
print("          PREDICTION SUMMARY")
print("========================================")

print("\nPredicted Negative:", negative_predictions)
print("Predicted Positive:", positive_predictions)

print(
    "\nPositive detection rate:",
    f"{positive_predictions / len(labels) * 100:.2f}%"
)

print(
    "\nAverage Positive probability:",
    f"{np.mean(pred_inference[:, 1]):.4f}"
)

print(
    "Average Negative probability:",
    f"{np.mean(pred_inference[:, 0]):.4f}"
)


# --------------------------------------------------
# 7. TRAINING MODE
# --------------------------------------------------

print("\n========================================")
print("          TRAINING MODE")
print("          training=True")
print("========================================")

pred_training = model(
    images,
    training=True
).numpy()

pred_training_classes = np.argmax(
    pred_training,
    axis=1
)

training_positive = np.sum(
    pred_training_classes == 1
)

training_negative = np.sum(
    pred_training_classes == 0
)

print("\nTraining-mode predictions:")

for i in range(len(labels)):

    print(
        f"Image {i+1}: "
        f"Actual=1 | "
        f"Predicted={pred_training_classes[i]} | "
        f"Negative={pred_training[i][0]:.4f} | "
        f"Positive={pred_training[i][1]:.4f}"
    )


print("\n========================================")
print("      TRAINING MODE SUMMARY")
print("========================================")

print("\nPredicted Negative:", training_negative)
print("Predicted Positive:", training_positive)

print(
    "\nPositive detection rate:",
    f"{training_positive / len(labels) * 100:.2f}%"
)

print(
    "\nAverage Positive probability:",
    f"{np.mean(pred_training[:, 1]):.4f}"
)


print("\n========================================")
print("          DIAGNOSTIC COMPLETE")
print("========================================")