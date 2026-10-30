import tensorflow as tf

from dataset_loader import (
    DATASET_PATH,
    get_classification_data
)

from image_preprocessing import preprocess_image


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def load_image(image_path, label):
    image = preprocess_image(image_path)

    image = tf.convert_to_tensor(image, dtype=tf.float32)

    return image, label


def create_tf_dataset(dataframe, shuffle=False):

    image_paths = dataframe["image_path"].values
    labels = dataframe["label"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    dataset = dataset.map(
        lambda image_path, label:
        tf.numpy_function(
            load_image,
            [image_path, label],
            [tf.float32, tf.int64]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.map(
        lambda image, label: (
            tf.ensure_shape(
                image,
                [IMAGE_SIZE[0], IMAGE_SIZE[1], 3]
            ),
            tf.ensure_shape(label, [])
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


def check_fold(df, fold_name):

    print("\n========================================")
    print(f"        {fold_name}")
    print("========================================")

    train_df = df[
        (df["split"] == "Training") &
        (df["fold"] == fold_name)
    ]

    test_df = df[
        (df["split"] == "Test") &
        (df["fold"] == fold_name)
    ]

    print("\nTraining images:", len(train_df))
    print("Test images:", len(test_df))

    print("\nTraining classes:")
    print(train_df["class_name"].value_counts())

    print("\nTest classes:")
    print(test_df["class_name"].value_counts())

    print("\nCreating TensorFlow datasets...")

    train_dataset = create_tf_dataset(
        train_df,
        shuffle=True
    )

    test_dataset = create_tf_dataset(
        test_df,
        shuffle=False
    )

    print("\nChecking training batch...")

    train_images, train_labels = next(
        iter(train_dataset)
    )

    print(
        "Training image batch shape:",
        train_images.shape
    )

    print(
        "Training label batch shape:",
        train_labels.shape
    )

    print(
        "Training image dtype:",
        train_images.dtype
    )

    print(
        "Training label dtype:",
        train_labels.dtype
    )

    print("\nChecking test batch...")

    test_images, test_labels = next(
        iter(test_dataset)
    )

    print(
        "Test image batch shape:",
        test_images.shape
    )

    print(
        "Test label batch shape:",
        test_labels.shape
    )


if __name__ == "__main__":

    print("========================================")
    print("     FIVE-FOLD DATASET VERIFICATION")
    print("========================================")

    print("\nLoading dataset information...")

    df = get_classification_data(DATASET_PATH)

    print("\nTotal images:", len(df))

    for fold_number in range(1, 6):

        fold_name = f"Fold_{fold_number}"

        check_fold(
            df,
            fold_name
        )

    print("\n========================================")
    print("       FIVE-FOLD TEST COMPLETE")
    print("========================================")