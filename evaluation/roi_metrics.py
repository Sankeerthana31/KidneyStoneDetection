import numpy as np


def calculate_mse(y_true, y_pred):
    """
    Mean Squared Error.
    """

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    return np.mean((y_true - y_pred) ** 2)


def calculate_rmse(y_true, y_pred):
    """
    Root Mean Squared Error.
    """

    mse = calculate_mse(y_true, y_pred)

    return np.sqrt(mse)


def calculate_mae(y_true, y_pred):
    """
    Mean Absolute Error.
    """

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    return np.mean(np.abs(y_true - y_pred))


def calculate_iou(y_true, y_pred, threshold=0.5):
    """
    Intersection over Union.

    The regression output is thresholded to create
    a binary ROI before calculating IoU.
    """

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    true_binary = y_true > threshold
    pred_binary = y_pred > threshold

    intersection = np.logical_and(
        true_binary,
        pred_binary
    ).sum()

    union = np.logical_or(
        true_binary,
        pred_binary
    ).sum()

    if union == 0:
        return 1.0

    return intersection / union


def calculate_dice(y_true, y_pred, threshold=0.5):
    """
    Dice Similarity Coefficient (DSC).
    """

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    true_binary = y_true > threshold
    pred_binary = y_pred > threshold

    intersection = np.logical_and(
        true_binary,
        pred_binary
    ).sum()

    total = true_binary.sum() + pred_binary.sum()

    if total == 0:
        return 1.0

    return (2.0 * intersection) / total