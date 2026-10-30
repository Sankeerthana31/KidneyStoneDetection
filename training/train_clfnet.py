import sys
from pathlib import Path

# ========================================
# PROJECT PATHS
# ========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PREPROCESSING_DIR))

# ========================================
# IMPORTS
# ========================================

import tensorflow as tf

from dataset_loader import DATASET_PATH, get_classification_data
from tf_dataset import create_tf_dataset
from models.clfnet import build_clfnet

# ========================================
# SETTINGS
# ========================================

FOLD = "Fold_5"

EPOCHS = 10

LEARNING_RATE = 0.0001

# ========================================
# LOAD DATASET
# ========================================

print("========================================")
print("       ClfNet TRAINING")
print("========================================")

print("\nLoading dataset...")

df = get_classification_data(DATASET_PATH)

# ========================================
# SELECT FOLD 1
# ========================================

train_df = df[
    (df["split"] == "Training") &
    (df["fold"] == FOLD)
]

test_df = df[
    (df["split"] == "Test") &
    (df["fold"] == FOLD)
]

print("\n========================================")
print("          DATASET INFORMATION")
print("========================================")

print("\nFold:", FOLD)

print("Training images:", len(train_df))
print("Test images:", len(test_df))

print("\nTraining class distribution:")
print(train_df["class_name"].value_counts())

print("\nTest class distribution:")
print(test_df["class_name"].value_counts())

# ========================================
# CREATE TF DATASETS
# ========================================

print("\n========================================")
print("       CREATING TF DATASETS")
print("========================================")

train_dataset = create_tf_dataset(
    train_df,
    shuffle=True
)

test_dataset = create_tf_dataset(
    test_df,
    shuffle=False
)

print("\nDatasets created successfully.")

# ========================================
# BUILD MODEL
# ========================================

print("\n========================================")
print("          BUILDING ClfNet")
print("========================================")

model = build_clfnet()

print("\nModel created successfully.")

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)

print("\nTotal parameters:")
print(model.count_params())

# ========================================
# COMPILE MODEL
# ========================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ========================================
# MODEL SAVE PATH
# ========================================

MODEL_DIR = PROJECT_ROOT / "models" / "saved"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR /
    f"clfnet_{FOLD.lower()}_best.keras"
)

# ========================================
# CALLBACKS
# ========================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# ========================================
# TRAIN MODEL
# ========================================

print("\n========================================")
print("          STARTING TRAINING")
print("========================================")

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)

# ========================================
# TRAINING COMPLETE
# ========================================

print("\n========================================")
print("          TRAINING COMPLETE")
print("========================================")

print("\nBest model saved to:")

print(MODEL_PATH)

print("\nTraining accuracy history:")
print(history.history["accuracy"])

print("\nValidation accuracy history:")
print(history.history["val_accuracy"])

print("\nTraining loss history:")
print(history.history["loss"])

print("\nValidation loss history:")
print(history.history["val_loss"])

print("\n========================================")
print("             DONE")
print("========================================")