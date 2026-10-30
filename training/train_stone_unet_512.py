
from pathlib import Path
import random

import numpy as np
import tensorflow as tf
import tifffile
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT = Path(r"D:\KidneyStoneDetection")

DATA = PROJECT / "data" / "ct_localization" / "stone_splits"
MODEL_DIR = PROJECT / "models" / "localization"
MODEL_PATH = MODEL_DIR / "stone_unet_512_best.keras"

IMAGE_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 30
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Limit TensorFlow CPU thread usage to help control memory use.
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)


# ============================================================
# 2. LOAD IMAGE-MASK PAIRS
# ============================================================

def load_split(split):
    image_dir = DATA / split / "images"
    mask_dir = DATA / split / "masks"

    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Missing image or mask directory for split: {split}"
        )

    image_paths = sorted(image_dir.glob("*.tif"))
    mask_paths = {p.name: p for p in mask_dir.glob("*.tif")}

    pairs = []

    for image_path in image_paths:
        if image_path.name not in mask_paths:
            raise FileNotFoundError(
                f"Missing mask for {image_path.name}"
            )
        pairs.append((image_path, mask_paths[image_path.name]))

    if not pairs:
        raise RuntimeError(f"No TIFF image-mask pairs found in {split}")

    return pairs


# ============================================================
# 3. PREPROCESS CT IMAGES AND STONE MASKS
# ============================================================

def load_pair(image_path, mask_path):
    image = tifffile.imread(str(image_path)).astype(np.float32)
    mask = tifffile.imread(str(mask_path))

    if image.ndim != 2 or mask.ndim != 2:
        raise ValueError(
            f"Expected grayscale 2D TIFF files: {image_path.name}"
        )

    if image.shape != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch: {image_path.name}"
        )

    # Normalize CT image intensities to [0, 1].
    image = image / 255.0

    # High-valued label pixels identify stone foreground.
    mask = (mask >= 247).astype(np.float32)

    # Add a single channel: (512, 512, 1).
    image = image[..., np.newaxis]
    mask = mask[..., np.newaxis]

    # Bilinear interpolation for images; nearest-neighbor for masks.
    image = tf.image.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        method="bilinear"
    ).numpy()

    mask = tf.image.resize(
        mask,
        (IMAGE_SIZE, IMAGE_SIZE),
        method="nearest"
    ).numpy()

    return image.astype(np.float32), mask.astype(np.float32)


# ============================================================
# 4. BUILD TF.DATA DATASETS
# ============================================================

def make_dataset(split, shuffle=False):
    pairs = load_split(split)

    def generator():
        items = pairs.copy()

        if shuffle:
            random.shuffle(items)

        for image_path, mask_path in items:
            image, mask = load_pair(image_path, mask_path)
            yield image, mask

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                dtype=tf.float32
            )
        )
    )

    if shuffle:
        dataset = dataset.shuffle(
            min(len(pairs), 256),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(BATCH_SIZE)

    # Keep prefetch small to reduce additional memory use.
    dataset = dataset.prefetch(1)

    print(f"{split}: {len(pairs)} image-mask pairs")

    return dataset


# ============================================================
# 5. U-NET ARCHITECTURE
# ============================================================

def conv_block(x, filters):
    x = layers.Conv2D(
        filters, 3, padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters, 3, padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)

    return x


def build_unet():
    inputs = keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
        name="ct_image"
    )

    c1 = conv_block(inputs, 16)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 32)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 64)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 128)
    p4 = layers.MaxPooling2D()(c4)

    bottleneck = conv_block(p4, 256)

    u4 = layers.UpSampling2D()(bottleneck)
    u4 = layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, 128)

    u3 = layers.UpSampling2D()(c5)
    u3 = layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, 64)

    u2 = layers.UpSampling2D()(c6)
    u2 = layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, 32)

    u1 = layers.UpSampling2D()(c7)
    u1 = layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, 16)

    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        name="stone_probability"
    )(c8)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="stone_unet_512"
    )


# ============================================================
# 6. LOSS FUNCTION AND DICE METRIC
# ============================================================

def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)

    return (
        2.0 * intersection + smooth
    ) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def combined_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss = 1.0 - dice_coefficient(y_true, y_pred)

    return tf.reduce_mean(bce) + dice_loss


# ============================================================
# 7. TRAIN MODEL
# ============================================================

if __name__ == "__main__":

    print("\nLoading datasets...")

    train_ds = make_dataset("train", shuffle=True)
    valid_ds = make_dataset("valid")
    test_ds = make_dataset("test")

    print("\nBuilding 512x512 U-Net...")

    model = build_unet()

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-4
        ),
        loss=combined_loss,
        metrics=[dice_coefficient]
    )

    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\nStarting training...")

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("\nLoading the best saved model...")

    best_model = keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={
            "combined_loss": combined_loss,
            "dice_coefficient": dice_coefficient
        }
    )

    print("\nEvaluating on the held-out test set...")

    test_results = best_model.evaluate(
        test_ds,
        return_dict=True,
        verbose=1
    )

    print("\nTest results:")
    for name, value in test_results.items():
        print(f"{name}: {value:.4f}")

    print("\nTraining complete.")
    print("Best model saved to:", MODEL_PATH)
