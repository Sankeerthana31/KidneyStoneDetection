
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

files = [
    f"evaluation/fold{i}_final_predictions.csv"
    for i in range(1, 6)
]

datasets = [pd.read_csv(file) for file in files]

print("FIVE-FOLD CROSS-VALIDATION RESULTS")
print("-" * 45)

for i, data in enumerate(datasets, start=1):
    correct = (
        data["actual_class"] == data["predicted_class"]
    ).sum()

    accuracy = accuracy_score(
        data["actual_class"],
        data["predicted_class"]
    ) * 100

    print(f"Fold {i}: {accuracy:.2f}% ({correct}/{len(data)})")

all_data = pd.concat(datasets, ignore_index=True)

correct_total = (
    all_data["actual_class"] == all_data["predicted_class"]
).sum()

pooled_accuracy = accuracy_score(
    all_data["actual_class"],
    all_data["predicted_class"]
) * 100

mean_accuracy = sum(
    accuracy_score(d["actual_class"], d["predicted_class"])
    for d in datasets
) / len(datasets) * 100

print("-" * 45)
print("Total test images:", len(all_data))
print("Total correct:", correct_total)
print(f"Pooled accuracy: {pooled_accuracy:.2f}%")
print(f"Mean fold accuracy: {mean_accuracy:.2f}%")

print("\nCOMBINED CONFUSION MATRIX (Negative, Positive):")
print(confusion_matrix(
    all_data["actual_class"],
    all_data["predicted_class"],
    labels=["Negative", "Positive"]
))

print("\nCOMBINED CLASSIFICATION REPORT:")
print(classification_report(
    all_data["actual_class"],
    all_data["predicted_class"],
    labels=["Negative", "Positive"],
    digits=4
))
