import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import tensorflow as tf

from models.i2iregnet import build_i2iregnet
from models.losses import half_mse_loss


# ========================================
# TRAINING CONFIGURATION
# ========================================

IMAGE_SIZE = (224, 224, 3)

LEARNING_RATE = 0.0001
EPOCHS = 50


# ========================================
# BUILD MODEL
# ========================================

print("========================================")
print("       I2IRegNet TRAINING")
print("========================================")

print("\nBuilding I2IRegNet...")

model = build_i2iregnet()

print("\nModel created successfully.")

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)


# ========================================
# COMPILE MODEL
# ========================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss=half_mse_loss
)

print("\n========================================")
print("       TRAINING CONFIGURATION")
print("========================================")

print("\nOptimizer: Adam")
print("Learning rate:", LEARNING_RATE)
print("Epochs:", EPOCHS)
print("Loss: Half-MSE")


# ========================================
# DATASET CHECK
# ========================================

print("\n========================================")
print("       DATASET REQUIREMENT")
print("========================================")

print("\nI2IRegNet requires two inputs for training:")
print("1. CT image")
print("2. Ground-truth kidney ROI")

print("\nYour current dataset contains CT images,")
print("but does not contain kidney masks/ROI targets.")

print("\nTherefore, REAL TRAINING IS NOT STARTED.")


# ========================================
# END
# ========================================

print("\n========================================")
print("       TRAINING PIPELINE READY")
print("========================================")