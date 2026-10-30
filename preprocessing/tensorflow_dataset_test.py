import tensorflow as tf

from dataset_loader import (
    DATASET_PATH,
    get_classification_data
)

from image_preprocessing import preprocess_image


# ============================================================
# LOAD DATASET INFORMATION
# ============================================================

print("Loading dataset information...")

df = get_classification_data(DATASET_PATH)

print("Total images:", len(df))


# ============================================================
# TAKE FIRST 10 IMAGES FOR TESTING
# ============================================================

test_df = df.head(10).copy()

print("\nTesting with:", len(test_df), "images")


# ============================================================
# LOAD ONE IMAGE
# ============================================================

image_path = test_df.iloc[0]["image_path"]
label = test_df.iloc[0]["label"]

image = preprocess_image(image_path)

print("\nFirst image:")
print("Path:", image_path)
print("Label:", label)
print("Image shape:", image.shape)
print("Image dtype:", image.dtype)


# ============================================================
# CREATE TENSOR
# ============================================================

image_tensor = tf.convert_to_tensor(image)

print("\nTensor information:")
print("Shape:", image_tensor.shape)
print("Data type:", image_tensor.dtype)


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

image_tensor = tf.expand_dims(image_tensor, axis=0)

label_tensor = tf.convert_to_tensor(
    [label],
    dtype=tf.int32
)

print("\nBatch information:")
print("Image batch shape:", image_tensor.shape)
print("Label batch shape:", label_tensor.shape)


# ============================================================
# CREATE TF.DATA DATASET
# ============================================================

dataset = tf.data.Dataset.from_tensor_slices(
    (image_tensor, label_tensor)
)

dataset = dataset.batch(1)


# ============================================================
# READ FROM DATASET
# ============================================================

for images, labels in dataset:

    print("\nTensorFlow dataset output:")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    break


print("\n========================================")
print("TENSORFLOW DATASET TEST COMPLETE")
print("========================================")