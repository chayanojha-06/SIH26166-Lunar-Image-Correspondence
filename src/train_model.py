import csv
import os
import pickle

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# SIH26166 - TRAIN ACTUAL AI MODEL
# CPU-FIRST RANDOM FOREST
# ============================================================

DATA_FILE = (
    "data/features/"
    "training_features.csv"
)

MODEL_DIR = "models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "lunar_correspondence_rf.pkl"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

print("=" * 70)
print("SIH26166 LUNAR CORRESPONDENCE")
print("CPU AI MODEL TRAINING")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(
    DATA_FILE
):

    raise FileNotFoundError(
        f"Training data not found:\n{DATA_FILE}\n\n"
        "Run generate_training_data.py first."
    )

X = []
y = []

with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(
        file
    )

    for row in reader:

        features = [
            float(
                row["match_count"]
            ),

            float(
                row["mean_distance"]
            ),

            float(
                row["median_distance"]
            ),

            float(
                row["min_distance"]
            ),

            float(
                row["max_distance"]
            ),

            float(
                row["homography_ratio"]
            ),

            float(
                row["fundamental_ratio"]
            ),

            float(
                row["homography_error"]
            ),

            float(
                row["fundamental_error"]
            ),

            float(
                row["spatial_coverage"]
            ),

            float(
                row["homography_inliers"]
            ),

            float(
                row["fundamental_inliers"]
            )
        ]

        X.append(
            features
        )

        y.append(
            int(
                row["label"]
            )
        )

X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int32
)

print(
    "\nSamples:",
    len(X)
)

print(
    "Features:",
    X.shape[1]
)

print(
    "Positive:",
    np.sum(y == 1)
)

print(
    "Negative:",
    np.sum(y == 0)
)

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_test)
)

# ============================================================
# RANDOM FOREST
# ============================================================

print("\nTraining CPU Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

# ============================================================
# VALIDATION
# ============================================================

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1]

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

print("\n")
print("=" * 70)
print("MODEL VALIDATION")
print("=" * 70)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)

print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Different Region",
            "Same Region"
        ],
        zero_division=0
    )
)

print(
    "Confusion matrix:"
)

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_names = [
    "match_count",
    "mean_distance",
    "median_distance",
    "min_distance",
    "max_distance",
    "homography_ratio",
    "fundamental_ratio",
    "homography_error",
    "fundamental_error",
    "spatial_coverage",
    "homography_inliers",
    "fundamental_inliers"
]

importance = model.feature_importances_

print("\n")
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

for name, value in sorted(
    zip(
        feature_names,
        importance
    ),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{name:<25}"
        f"{value * 100:>8.2f}%"
    )

# ============================================================
# SAVE MODEL
# ============================================================

model_package = {
    "model": model,
    "features": feature_names,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
}

with open(
    MODEL_FILE,
    "wb"
) as file:

    pickle.dump(
        model_package,
        file
    )

print("\n")
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    MODEL_FILE
)

print(
    "\nCPU training completed successfully."
)

print("=" * 70)