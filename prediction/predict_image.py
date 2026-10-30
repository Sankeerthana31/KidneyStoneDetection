
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.clfnet import build_clfnet
MODEL_PATH = Path("models/saved/clfnet_fold1_best.keras")
OUTPUT_CSV = PROJECT_ROOT / "prediction" / "prediction_results.csv"

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Negative", "Positive"]


def load_model():
    print("Building ClfNet architecture...")
    model = build_clfnet()

    print("Loading saved checkpoint weights...")
    model.load_weights(str(MODEL_PATH))

    print("Model loaded successfully.\n")
    return model


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    return image_array


def predict_path(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    if path.is_file():
        image_paths = [path]
    else:
        image_paths = sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    if not image_paths:
        print("No supported images found.")
        return

    model = load_model()
    results = []

    for index, image_path in enumerate(image_paths, start=1):
        try:
            image_array = preprocess_image(image_path)
            probabilities = model.predict(
                np.expand_dims(image_array, axis=0),
                verbose=0
            )[0]

            predicted_index = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_index]) * 100

            results.append({
                "image": image_path.name,
                "image_path": str(image_path),
                "predicted_class": CLASS_NAMES[predicted_index],
                "negative_probability_percent": float(probabilities[0]) * 100,
                "positive_probability_percent": float(probabilities[1]) * 100,
                "confidence_percent": confidence,
                "status": "Success"
            })

            print(
                f"[{index}/{len(image_paths)}] {image_path.name}: "
                f"{CLASS_NAMES[predicted_index]} ({confidence:.2f}%)"
            )

        except Exception as error:
            results.append({
                "image": image_path.name,
                "image_path": str(image_path),
                "predicted_class": "",
                "negative_probability_percent": "",
                "positive_probability_percent": "",
                "confidence_percent": "",
                "status": f"Error: {error}"
            })

            print(f"[{index}/{len(image_paths)}] {image_path.name}: ERROR - {error}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    successful = sum(row["status"] == "Success" for row in results)

    print("\n========== BATCH SUMMARY ==========")
    print(f"Images found: {len(image_paths)}")
    print(f"Successful predictions: {successful}")
    print(f"Failed predictions: {len(image_paths) - successful}")
    print(f"CSV saved to: {OUTPUT_CSV}")
    print("===================================")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            'Usage: python prediction\\predict_image.py "path\\to\\image_or_folder"'
        )
        sys.exit(1)

    predict_path(sys.argv[1])
