import tensorflow as tf


def half_mse_loss(y_true, y_pred):
    """
    Half Mean Squared Error used for the I2IRegNet regression task.

    Paper:
        Lreg = 1/(2N) * sum((ROI - ROI_hat)^2)
    """

    squared_error = tf.square(y_true - y_pred)

    return 0.5 * tf.reduce_mean(squared_error)