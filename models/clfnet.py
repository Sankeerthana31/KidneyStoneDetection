import tensorflow as tf
from tensorflow.keras import layers, Model


# ============================================================
# SHALLOW ENCODER
# ============================================================

def build_shallow_encoder(inputs):

    x = inputs

    filters = [128, 32, 96, 64, 256, 224]

    for i, f in enumerate(filters):

        x = layers.Conv2D(
            filters=f,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name=f"shallow_conv_{i + 1}"
        )(x)

        # Batch Normalization is kept frozen so that
        # training and inference use the same statistics.
        bn = layers.BatchNormalization(
            name=f"shallow_bn_{i + 1}"
        )

        bn.trainable = False

        x = bn(x)

        x = layers.ReLU(
            name=f"shallow_relu_{i + 1}"
        )(x)

        x = layers.MaxPooling2D(
            pool_size=2,
            name=f"shallow_pool_{i + 1}"
        )(x)

    return x


# ============================================================
# CHANNEL ATTENTION
# ============================================================

def channel_attention(x):

    channels = x.shape[-1]

    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)

    reduced_channels = max(channels // 8, 1)

    shared_dense1 = layers.Dense(
        reduced_channels,
        activation="relu"
    )

    shared_dense2 = layers.Dense(
        channels,
        activation="sigmoid"
    )

    avg_attention = shared_dense2(
        shared_dense1(avg_pool)
    )

    max_attention = shared_dense2(
        shared_dense1(max_pool)
    )

    attention = layers.Add()([
        avg_attention,
        max_attention
    ])

    attention = layers.Activation(
        "sigmoid"
    )(attention)

    attention = layers.Reshape(
        (1, 1, channels)
    )(attention)

    return x * attention


# ============================================================
# SPATIAL ATTENTION
# ============================================================


def spatial_attention(x):

    avg_pool = layers.Lambda(
        lambda t: tf.reduce_mean(
            t,
            axis=-1,
            keepdims=True
        ),
        output_shape=(7, 7, 1)
    )(x)

    max_pool = layers.Lambda(
        lambda t: tf.reduce_max(
            t,
            axis=-1,
            keepdims=True
        ),
        output_shape=(7, 7, 1)
    )(x)

    combined = layers.Concatenate(
        axis=-1
    )([
        avg_pool,
        max_pool
    ])

    attention = layers.Conv2D(
        filters=1,
        kernel_size=7,
        padding="same",
        activation="sigmoid",
        name="spatial_attention_map"
    )(combined)

    return x * attention


# ============================================================
# CBAM
# ============================================================

def cbam(x):

    x = channel_attention(x)

    x = spatial_attention(x)

    return x


# ============================================================
# ClfNet
# ============================================================

def build_clfnet():

    inputs = layers.Input(
        shape=(224, 224, 3),
        name="ct_input"
    )

    # --------------------------------------------------------
    # Shallow encoder
    # --------------------------------------------------------

    shallow_features = build_shallow_encoder(
        inputs
    )

    # --------------------------------------------------------
    # Deep encoder
    # --------------------------------------------------------

    deep_encoder = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    # Freeze ImageNet pretrained encoder
    deep_encoder.trainable = False

    deep_features = deep_encoder(
        inputs,
        training=False
    )

    # --------------------------------------------------------
    # Resize shallow features
    # --------------------------------------------------------

    shallow_features = layers.Resizing(
        7,
        7,
        name="shallow_resize"
    )(shallow_features)

    # --------------------------------------------------------
    # Feature fusion
    # --------------------------------------------------------

    fused_features = layers.Concatenate(
        axis=-1,
        name="feature_fusion"
    )([
        shallow_features,
        deep_features
    ])

    # --------------------------------------------------------
    # Batch normalization after fusion
    # --------------------------------------------------------

    fusion_bn = layers.BatchNormalization(
        name="fusion_batch_normalization"
    )

    fusion_bn.trainable = False

    fused_features = fusion_bn(
        fused_features
    )

    # --------------------------------------------------------
    # 1x1 channel reduction
    # --------------------------------------------------------

    fused_features = layers.Conv2D(
        filters=256,
        kernel_size=1,
        padding="same",
        name="channel_reduction"
    )(fused_features)

    # --------------------------------------------------------
    # CBAM attention
    # --------------------------------------------------------

    fused_features = cbam(
        fused_features
    )

    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    x = layers.ReLU(
        name="classifier_relu"
    )(fused_features)

    x = layers.Dropout(
        0.5,
        name="classifier_dropout"
    )(x)

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    outputs = layers.Dense(
        2,
        activation="softmax",
        name="stone_prediction"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="ClfNet"
    )

    return model


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("========================================")
    print("             ClfNet TEST")
    print("========================================")

    model = build_clfnet()

    print("\nModel created successfully.")

    print("\nInput shape:")
    print(model.input_shape)

    print("\nOutput shape:")
    print(model.output_shape)

    print("\nTotal parameters:")
    print(model.count_params())

    print("\nTrainable parameters:")
    print(
        sum(
            tf.keras.backend.count_params(w)
            for w in model.trainable_weights
        )
    )

    print("\nNon-trainable parameters:")
    print(
        sum(
            tf.keras.backend.count_params(w)
            for w in model.non_trainable_weights
        )
    )

    print("\n========================================")
    print("       BATCH NORMALIZATION STATUS")
    print("========================================")

    for layer in model.layers:

        if isinstance(
            layer,
            layers.BatchNormalization
        ):

            print(
                layer.name,
                "-> trainable:",
                layer.trainable
            )

    print("\n========================================")
    print("           ClfNet TEST COMPLETE")
    print("========================================")