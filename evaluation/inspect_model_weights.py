import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import tensorflow as tf

from models.clfnet import build_clfnet


# ============================================================
# MODEL PATH
# ============================================================

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
print("       MODEL WEIGHT INSPECTION")
print("========================================")

print("\nModel file:")
print(MODEL_PATH)


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not MODEL_PATH.exists():

    print("\nERROR: Model file does not exist.")

    print(MODEL_PATH)

    sys.exit()


# ============================================================
# BUILD CURRENT ARCHITECTURE
# ============================================================

print("\n========================================")
print("       BUILDING CURRENT ClfNet")
print("========================================")

model = build_clfnet()

print("\nModel created successfully.")

print("\nTotal parameters:")
print(model.count_params())


# ============================================================
# MODEL WEIGHT INFORMATION BEFORE LOADING
# ============================================================

print("\n========================================")
print("     WEIGHTS BEFORE LOADING")
print("========================================")

before_weights = model.get_weights()

print("\nNumber of weight arrays:")
print(len(before_weights))

print("\nFirst 10 weight arrays:")

for i, weight in enumerate(before_weights[:10]):

    print(
        f"{i}: "
        f"shape={weight.shape}, "
        f"mean={np.mean(weight):.6f}, "
        f"std={np.std(weight):.6f}"
    )


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

print("\n========================================")
print("       LOADING SAVED WEIGHTS")
print("========================================")

try:

    model.load_weights(MODEL_PATH)

    print("\nWeights loaded successfully.")

except Exception as e:

    print("\nERROR while loading weights:")

    print(e)

    sys.exit()


# ============================================================
# MODEL WEIGHT INFORMATION AFTER LOADING
# ============================================================

print("\n========================================")
print("      WEIGHTS AFTER LOADING")
print("========================================")

after_weights = model.get_weights()

print("\nNumber of weight arrays:")
print(len(after_weights))

print("\nFirst 10 weight arrays:")

for i, weight in enumerate(after_weights[:10]):

    print(
        f"{i}: "
        f"shape={weight.shape}, "
        f"mean={np.mean(weight):.6f}, "
        f"std={np.std(weight):.6f}"
    )


# ============================================================
# COMPARE WEIGHTS
# ============================================================

print("\n========================================")
print("        WEIGHT COMPARISON")
print("========================================")

different_arrays = 0

total_difference = 0.0

for before, after in zip(
    before_weights,
    after_weights
):

    difference = np.mean(
        np.abs(before - after)
    )

    total_difference += difference

    if difference > 1e-8:
        different_arrays += 1


print("\nWeight arrays changed:")
print(
    f"{different_arrays} / "
    f"{len(before_weights)}"
)

print("\nTotal average weight difference:")
print(
    f"{total_difference:.10f}"
)


# ============================================================
# FINAL CLASSIFIER LAYER
# ============================================================

print("\n========================================")
print("       FINAL CLASSIFIER WEIGHTS")
print("========================================")

final_layer = model.get_layer(
    "stone_prediction"
)

print("\nLayer name:")
print(final_layer.name)

print("\nLayer type:")
print(type(final_layer).__name__)

final_weights = final_layer.get_weights()

for i, weight in enumerate(final_weights):

    print(
        f"\nWeight {i}:"
    )

    print("Shape:")
    print(weight.shape)

    print("Values:")

    print(weight)


# ============================================================
# FINAL LAYER BIAS
# ============================================================

if len(final_weights) > 1:

    bias = final_weights[1]

    print("\n========================================")
    print("          FINAL BIAS")
    print("========================================")

    print("\nBias values:")

    print(bias)

    print("\nNegative class bias:")
    print(f"{bias[0]:.6f}")

    print("\nPositive class bias:")
    print(f"{bias[1]:.6f}")


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("       INSPECTION COMPLETE")
print("========================================")