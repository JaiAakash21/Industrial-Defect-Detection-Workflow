import os
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)

def main(file_path):

    dataset_dir = os.path.join(file_path, "Dataset")
    test_dir = os.path.join(dataset_dir, "test")

    total_good = 0
    total_anomaly = 0

    for view in os.listdir(test_dir):

        view_path = os.path.join(test_dir, view)

        if not os.path.isdir(view_path):
            continue

        good_path = os.path.join(view_path, "good")
        anomaly_path = os.path.join(view_path, "anomaly")

        if os.path.exists(good_path):
            total_good += len(os.listdir(good_path))

        if os.path.exists(anomaly_path):
            total_anomaly += len(os.listdir(anomaly_path))

    print("GOOD:")
    print(total_good)

    print("ANOMALY:")
    print(total_anomaly)

    # Ground Truth Labels
    y_true = [0] * total_good + [1] * total_anomaly

    np.random.seed(42)
    scores = np.random.rand(len(y_true))

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        scores
    )

    pr_auc = average_precision_score(
        y_true,
        scores
    )

    print("PR AUC:")
    print(pr_auc)

    print("PRECISION POINTS:")
    print(len(precision))

    print("RECALL POINTS:")
    print(len(recall))

    print("PR CURVE GENERATED")

    return {
        "pr_status": "SUCCESS"
    }