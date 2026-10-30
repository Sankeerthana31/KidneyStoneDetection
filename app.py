
from pathlib import Path
from io import BytesIO
import sys

import numpy as np
import streamlit as st
import tensorflow as tf
import tifffile

from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from models.clfnet import build_clfnet

CLASSIFIER_WEIGHTS = PROJECT / "models" / "saved" / "clfnet_fold1_best.keras"
STONE_MODEL_PATH = PROJECT / "models" / "localization" / "stone_unet_512_best.keras"

CLASS_NAMES = ["Negative", "Positive"]
IMAGE_SIZE = 224
STONE_SIZE = 512
SEGMENTATION_THRESHOLD = 0.50


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Kidney Stone Detection",
    page_icon="🔬",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_classifier():
    if not CLASSIFIER_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {CLASSIFIER_WEIGHTS}"
        )

    model = build_clfnet()
    model.load_weights(str(CLASSIFIER_WEIGHTS))
    return model


@st.cache_resource
def load_stone_model():
    if not STONE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Stone U-Net model not found: {STONE_MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        STONE_MODEL_PATH,
        compile=False
    )


# --------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------

def classify_image(uploaded_bytes):

    image = Image.open(BytesIO(uploaded_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    array = np.asarray(image, dtype=np.float32)

    array = tf.keras.applications.mobilenet_v2.preprocess_input(
        array
    )

    model = load_classifier()

    probabilities = model.predict(
        np.expand_dims(array, axis=0),
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))

    return CLASS_NAMES[predicted_index], probabilities


# --------------------------------------------------
# STONE LOCALIZATION
# --------------------------------------------------

def localize_stone(uploaded_bytes):

    original = tifffile.imread(BytesIO(uploaded_bytes))

    if original.ndim != 2:
        raise ValueError(
            "Localization requires a single-channel, 2D grayscale TIFF."
        )

    if original.dtype != np.uint8:
        raise ValueError(
            "Localization currently requires an 8-bit grayscale TIFF."
        )

    model = load_stone_model()

    # Normalize input
    resized_image = tf.image.resize(
        original[..., np.newaxis].astype(np.float32) / 255.0,
        (STONE_SIZE, STONE_SIZE),
        method="bilinear"
    ).numpy()

    # Predict segmentation probability
    probability = model.predict(
        resized_image[np.newaxis, ...],
        verbose=0
    )[0, ..., 0]

    # Generate binary mask
    predicted_small = probability >= SEGMENTATION_THRESHOLD

    predicted_mask = tf.image.resize(
        predicted_small.astype(np.uint8)[..., np.newaxis],
        original.shape,
        method="nearest"
    ).numpy()[..., 0] > 0

    return original, probability, predicted_mask


# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------

def create_visualizations(original, mask):

    original_rgb = np.stack(
        [original] * 3,
        axis=-1
    ).astype(np.uint8)

    # Yellow overlay
    overlay = original_rgb.copy()

    yellow = np.array([255, 255, 0], dtype=np.float32)

    if mask.any():
        overlay[mask] = (
            0.45 * original_rgb[mask].astype(np.float32)
            + 0.55 * yellow
        ).astype(np.uint8)

    # Find bounding box
    coordinates = np.argwhere(mask)

    bbox = None
    crop = None

    if len(coordinates) > 0:

        y_min, x_min = coordinates.min(axis=0)
        y_max, x_max = coordinates.max(axis=0)

        bbox = (
            int(x_min),
            int(y_min),
            int(x_max),
            int(y_max)
        )

        # Draw bounding box
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(overlay, cmap="gray")

        rect = Rectangle(
            (x_min, y_min),
            max(1, x_max - x_min + 1),
            max(1, y_max - y_min + 1),
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rect)
        ax.set_title("Stone Localization with Bounding Box")
        ax.axis("off")

        st.pyplot(fig)
        plt.close(fig)

        # Crop with margin
        margin = max(20, int(max(
            y_max - y_min + 1,
            x_max - x_min + 1
        ) * 3))

        height, width = original.shape

        crop_y1 = max(0, y_min - margin)
        crop_y2 = min(height, y_max + margin + 1)

        crop_x1 = max(0, x_min - margin)
        crop_x2 = min(width, x_max + margin + 1)

        crop = overlay[
            crop_y1:crop_y2,
            crop_x1:crop_x2
        ]

    return original_rgb, mask.astype(np.uint8) * 255, overlay, bbox, crop


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("🔬 Kidney Stone Detection and Localization")

st.write(
    "Deep Learning Framework for Automated Kidney Stone "
    "Localization and Classification in CT Images"
)

st.info(
    "Academic research prototype. This application is not "
    "a clinically validated diagnostic system."
)

st.markdown("---")


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a CT image",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    st.subheader("Uploaded Image")

    try:
        preview = Image.open(BytesIO(file_bytes)).convert("RGB")

        st.image(
            preview,
            caption=uploaded_file.name,
            width="stretch"
        )

    except Exception as error:
        st.warning(
            "The uploaded file could not be displayed as a standard image. "
            "TIFF localization will still be checked separately."
        )

    st.markdown("---")

    if st.button("Analyze CT Image", type="primary"):

        # ==========================================
        # CLASSIFICATION
        # ==========================================

        st.header("1. Stone Classification")

        try:

            with st.spinner("Running classification..."):

                label, probabilities = classify_image(file_bytes)

            if label == "Positive":
                st.warning(f"Classifier result: {label}")
            else:
                st.success(f"Classifier result: {label}")

            col1, col2 = st.columns(2)

            col1.metric(
                "Negative Probability",
                f"{probabilities[0] * 100:.2f}%"
            )

            col2.metric(
                "Positive Probability",
                f"{probabilities[1] * 100:.2f}%"
            )

            st.caption(
                "These are model output probabilities, not clinical certainty."
            )

        except Exception as error:

            st.error(f"Classification failed: {error}")

        # ==========================================
        # LOCALIZATION
        # ==========================================

        st.markdown("---")
        st.header("2. Stone Localization")

        if uploaded_file.name.lower().endswith((".tif", ".tiff")):

            try:

                with st.spinner("Generating stone segmentation..."):

                    original, probability, mask = localize_stone(file_bytes)

                original_rgb, mask_image, overlay, bbox, crop = (
                    create_visualizations(original, mask)
                )

                st.subheader("Stone Segmentation Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.image(
                        original_rgb,
                        caption="Original CT Image",
                        width="stretch"
                    )

                with col2:
                    st.image(
                        mask_image,
                        caption="Predicted Stone Mask",
                        width="stretch"
                    )

                with col3:
                    st.image(
                        overlay,
                        caption="Stone Localization Overlay",
                        width="stretch"
                    )

                st.markdown("---")
                st.subheader("Localization Measurements")

                foreground_pixels = int(mask.sum())

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Predicted Foreground Pixels",
                    foreground_pixels
                )

                col3.metric(
                    "Segmentation Threshold",
                    f"{SEGMENTATION_THRESHOLD:.2f}"
                )

                if bbox is not None:

                    x1, y1, x2, y2 = bbox

                    col2.metric(
                        "Bounding Box Size",
                        f"{x2 - x1 + 1} × {y2 - y1 + 1}"
                    )

                    st.subheader("Enlarged Predicted Region")

                    st.image(
                        crop,
                        caption="Zoomed view of the predicted region",
                        width="stretch"
                    )

                    st.caption(
                        "The crop is enlarged for visualization. "
                        "The original predicted mask has not been expanded."
                    )

                else:

                    st.warning(
                        "The model did not predict any foreground pixels "
                        "at the selected threshold."
                    )

                st.caption(
                    "The highlighted region is a model-generated segmentation "
                    "prediction and is not a confirmed stone diagnosis."
                )

            except Exception as error:

                st.error(f"Localization failed: {error}")

        else:

            st.info(
                "Stone localization currently supports compatible "
                "8-bit grayscale TIFF CT images only. "
                "Classification supports JPG, JPEG and PNG images."
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Models: ClfNet classifier and 512×512 Stone U-Net. "
    "Research prototype only; not for medical diagnosis."
)
