
from pathlib import Path
import random

import numpy as np
import tensorflow as tf
from PIL import Image

# Dataset paths
BASE_DIR = Path("data/kidney_segmentation")
IMAGE_DIR = BASE_DIR / "images"
MASK_DIR = BASE_DIR / "masks"

# Training configuration
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 10
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Find matching image-mask pairs
image_paths = sorted(IMAGE_DIR.glob("*.png"))
pairs = [
    (p, MASK_DIR / p.name)
    for p in image_paths
    if (MASK_DIR / p.name).exists()
]

if len(pairs) < 2:
    raise RuntimeError("Not enough matching image-mask pairs.")

random.shuffle(pairs)

# Split into training and validation sets
split_index = int(len(pairs) * 0.8)
train_pairs = pairs[:split_index]
val_pairs = pairs[split_index:]

print("Total pairs:", len(pairs))
print("Training pairs:", len(train_pairs))
print("Validation pairs:", len(val_pairs))


def load_pair(image_path, mask_path):
    image = np.array(
        Image.open(image_path).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0

    mask = np.array(
        Image.open(mask_path).convert("L").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    )

    # Convert grayscale mask to binary foreground/background
    mask = (mask > 127).astype(np.float32)[..., np.newaxis]
    return image, mask


def make_dataset(items, training=False):
    def generator():
        for image_path, mask_path in items:
            image, mask = load_pair(image_path, mask_path)
            yield image, mask

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec((IMG_SIZE, IMG_SIZE, 3), tf.float32),
            tf.TensorSpec((IMG_SIZE, IMG_SIZE, 1), tf.float32),
        ),
    )

    if training:
        dataset = dataset.shuffle(min(len(items), 1000), seed=SEED)

    return dataset.batch(BATCH_SIZE).prefetch(1)


def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", activation="relu"
    )(x)
    return x


def build_unet():
    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))

    c1 = conv_block(inputs, 32)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 256)
    p4 = tf.keras.layers.MaxPooling2D()(c4)

    bottleneck = conv_block(p4, 512)

    u4 = tf.keras.layers.UpSampling2D()(bottleneck)
    u4 = tf.keras.layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, 256)

    u3 = tf.keras.layers.UpSampling2D()(c5)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, 128)

    u2 = tf.keras.layers.UpSampling2D()(c6)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, 64)

    u1 = tf.keras.layers.UpSampling2D()(c7)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, 32)

    outputs = tf.keras.layers.Conv2D(
        1, 1, activation="sigmoid"
    )(c8)

    return tf.keras.Model(inputs, outputs, name="kidney_unet")


def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    return (
        2.0 * intersection + smooth
    ) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


model = build_unet()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[dice_coefficient],
)

model.summary()

train_ds = make_dataset(train_pairs, training=True)
val_ds = make_dataset(val_pairs)

output_dir = Path("models/localization")
output_dir.mkdir(parents=True, exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        output_dir / "kidney_unet_best.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    ),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

model.save(output_dir / "kidney_unet_final.keras")
print("Training finished.")
print("Models saved in:", output_dir.resolve())
