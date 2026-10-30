import numpy as np

from roi_metrics import (
    calculate_mse,
    calculate_rmse,
    calculate_mae,
    calculate_iou,
    calculate_dice
)


print("========================================")
print("        ROI METRICS TEST")
print("========================================")


y_true = np.array([
    [1.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]
])


y_pred = np.array([
    [1.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]
])


print("\nGround truth:")
print(y_true)

print("\nPrediction:")
print(y_pred)


mse = calculate_mse(y_true, y_pred)
rmse = calculate_rmse(y_true, y_pred)
mae = calculate_mae(y_true, y_pred)
iou = calculate_iou(y_true, y_pred)
dice = calculate_dice(y_true, y_pred)


print("\n========================================")
print("          ROI METRICS")
print("========================================")

print("\nMSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("IoU:", iou)
print("Dice:", dice)

print("\n========================================")
print("       METRICS TEST COMPLETE")
print("========================================")