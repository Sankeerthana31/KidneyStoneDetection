import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PREPROCESSING_DIR))

import numpy as np
import tensorflow as tf

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
print("     TRAINING vs INFERENCE CHECK")
print("========================================")


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

print("\n========================================")
print("          LOADING DATASET")
print("========================================")

df = get_classification_data(DATASET_PATH)

train_df = df[
    (df["split"] == "Training") &
    (df["fold"] == FOLD)
]

print("\nTraining images:", len(train_df))

print("\nClass distribution:")
print(train_df["class_name"].value_counts())


# --------------------------------------------------
# 2. CREATE DATASET
# --------------------------------------------------

print("\n========================================")
print("       CREATING TF DATASET")
print("========================================")

train_dataset = create_tf_dataset(
    train_df,
    shuffle=False
)

print("\nDataset created successfully.")


# --------------------------------------------------
# 3. BUILD MODEL
# --------------------------------------------------

print("\n========================================")
print("          BUILDING ClfNet")
print("========================================")

model = build_clfnet()

print("\nModel created.")

print("\nTotal parameters:")
print(model.count_params())


# --------------------------------------------------
# 4. LOAD SAVED WEIGHTS
# --------------------------------------------------

print("\n========================================")
print("       LOADING SAVED WEIGHTS")
print("========================================")

model.load_weights(MODEL_PATH)

print("\nSaved weights loaded successfully.")


# --------------------------------------------------
# 5. GET ONE BATCH
# --------------------------------------------------

print("\n========================================")
print("          GETTING BATCH")
print("========================================")

images, labels = next(iter(train_dataset))

print("\nImage batch shape:")
print(images.shape)

print("\nLabel batch shape:")
print(labels.shape)

print("\nActual labels:")
print(labels.numpy())


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
)

pred_inference = pred_inference.numpy()

pred_inference_classes = np.argmax(
    pred_inference,
    axis=1
)

print("\nFirst 20 predictions:")

for i in range(min(20, len(labels))):

    print(
        f"Image {i+1}: "
        f"Actual={labels.numpy()[i]} | "
        f"Predicted={pred_inference_classes[i]} | "
        f"Negative={pred_inference[i][0]:.4f} | "
        f"Positive={pred_inference[i][1]:.4f}"
    )


inference_accuracy = np.mean(
    pred_inference_classes == labels.numpy()
)

print("\nInference accuracy on this batch:")
print(f"{inference_accuracy * 100:.2f}%")


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
)

pred_training = pred_training.numpy()

pred_training_classes = np.argmax(
    pred_training,
    axis=1
)

print("\nFirst 20 predictions:")

for i in range(min(20, len(labels))):

    print(
        f"Image {i+1}: "
        f"Actual={labels.numpy()[i]} | "
        f"Predicted={pred_training_classes[i]} | "
        f"Negative={pred_training[i][0]:.4f} | "
        f"Positive={pred_training[i][1]:.4f}"
    )


training_accuracy = np.mean(
    pred_training_classes == labels.numpy()
)

print("\nTraining-mode accuracy on this batch:")
print(f"{training_accuracy * 100:.2f}%")


# --------------------------------------------------
# 8. COMPARE
# --------------------------------------------------

print("\n========================================")
print("             COMPARISON")
print("========================================")

print(
    "\nInference mode accuracy:",
    f"{inference_accuracy * 100:.2f}%"
)

print(
    "Training mode accuracy:",
    f"{training_accuracy * 100:.2f}%"
)

print("\n========================================")
print("        DIAGNOSTIC RESULT")
print("========================================")

if training_accuracy > inference_accuracy + 0.20:

    print("\nIMPORTANT:")
    print(
        "Training-mode accuracy is much higher "
        "than inference-mode accuracy."
    )

    print(
        "\nThis strongly suggests that "
        "Batch Normalization statistics are "
        "causing the problem."
    )

else:

    print(
        "\nTraining and inference behavior are "
        "relatively similar."
    )

    print(
        "\nThe problem may be related to "
        "another part of the training pipeline."
    )


print("\n========================================")
print("          CHECK COMPLETE")
print("========================================")