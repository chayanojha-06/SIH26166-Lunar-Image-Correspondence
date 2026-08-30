import cv2
import numpy as np
import os
import csv
import random

# ============================================================
# SIH26166 - TRAINING DATA GENERATOR
# CPU-FIRST LUNAR CORRESPONDENCE
# ============================================================

DATA_DIR = "data/raw"
FEATURE_DIR = "data/features"
OUTPUT_FILE = os.path.join(
    FEATURE_DIR,
    "training_features.csv"
)

os.makedirs(FEATURE_DIR, exist_ok=True)

SIFT_FEATURES = 8000
LOWE_RATIO = 0.80

POSITIVE_PAIRS = 300
NEGATIVE_PAIRS = 300

random.seed(42)
np.random.seed(42)

# ============================================================
# LOAD IMAGES
# ============================================================

image_files = []

for file in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, file)

    img = cv2.imread(path)

    if img is not None:
        image_files.append(path)

if len(image_files) < 2:
    raise RuntimeError(
        "Need at least 2 valid lunar images in data/raw/"
    )

print("=" * 70)
print("SIH26166 TRAINING DATA GENERATOR")
print("=" * 70)

print("\nImages found:", len(image_files))

# ============================================================
# SIFT
# ============================================================

sift = cv2.SIFT_create(
    nfeatures=SIFT_FEATURES,
    contrastThreshold=0.02,
    edgeThreshold=10
)

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    return keypoints, descriptors


# ============================================================
# IMAGE AUGMENTATION
# ============================================================

def augment_image(image):

    h, w = image.shape[:2]

    scale = random.uniform(
        0.75,
        1.25
    )

    new_w = max(
        32,
        int(w * scale)
    )

    new_h = max(
        32,
        int(h * scale)
    )

    result = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    angle = random.uniform(
        -15,
        15
    )

    center = (
        result.shape[1] // 2,
        result.shape[0] // 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    result = cv2.warpAffine(
        result,
        matrix,
        (
            result.shape[1],
            result.shape[0]
        ),
        borderMode=cv2.BORDER_REFLECT
    )

    # Brightness
    brightness = random.uniform(
        0.75,
        1.25
    )

    result = np.clip(
        result.astype(np.float32) *
        brightness,
        0,
        255
    ).astype(np.uint8)

    # Contrast
    alpha = random.uniform(
        0.75,
        1.25
    )

    result = cv2.convertScaleAbs(
        result,
        alpha=alpha,
        beta=0
    )

    return result


# ============================================================
# MATCHING
# ============================================================

def get_matches(
    des1,
    des2
):

    if des1 is None or des2 is None:
        return []

    if len(des1) < 2 or len(des2) < 2:
        return []

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    raw = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good = []

    for pair in raw:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    return good


# ============================================================
# GEOMETRIC FEATURES
# ============================================================

def calculate_features(
    kp1,
    kp2,
    matches,
    image_shape
):

    h, w = image_shape[:2]

    total_matches = len(matches)

    if total_matches == 0:
        return None

    distances = np.array(
        [
            m.distance
            for m in matches
        ],
        dtype=np.float32
    )

    mean_distance = float(
        np.mean(distances)
    )

    median_distance = float(
        np.median(distances)
    )

    min_distance = float(
        np.min(distances)
    )

    max_distance = float(
        np.max(distances)
    )

    # --------------------------------------------------------
    # Geometric verification
    # --------------------------------------------------------

    pts1 = np.float32(
        [
            kp1[m.queryIdx].pt
            for m in matches
        ]
    )

    pts2 = np.float32(
        [
            kp2[m.trainIdx].pt
            for m in matches
        ]
    )

    homography_inliers = 0
    fundamental_inliers = 0
    homography_error = 999.0
    fundamental_error = 999.0

    if len(matches) >= 4:

        H, hmask = cv2.findHomography(
            pts1,
            pts2,
            cv2.RANSAC,
            4.0,
            maxIters=2000,
            confidence=0.995
        )

        if hmask is not None:

            hmask = hmask.ravel()

            homography_inliers = int(
                np.sum(hmask)
            )

            idx = np.where(
                hmask == 1
            )[0]

            if len(idx) >= 4:

                p1 = pts1[idx].reshape(
                    -1,
                    1,
                    2
                )

                projected = cv2.perspectiveTransform(
                    p1,
                    H
                ).reshape(
                    -1,
                    2
                )

                error = (
                    projected -
                    pts2[idx]
                )

                homography_error = float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                error ** 2,
                                axis=1
                            )
                        )
                    )
                )

    if len(matches) >= 8:

        F, fmask = cv2.findFundamentalMat(
            pts1,
            pts2,
            cv2.FM_RANSAC,
            2.0,
            0.995,
            2000
        )

        if fmask is not None:

            fmask = fmask.ravel()

            fundamental_inliers = int(
                np.sum(fmask)
            )

            idx = np.where(
                fmask == 1
            )[0]

            if F is not None and len(idx) >= 8:

                p1 = np.hstack(
                    [
                        pts1[idx],
                        np.ones(
                            (
                                len(idx),
                                1
                            )
                        )
                    ]
                )

                p2 = np.hstack(
                    [
                        pts2[idx],
                        np.ones(
                            (
                                len(idx),
                                1
                            )
                        )
                    ]
                )

                lines = (
                    F @ p1.T
                ).T

                numerator = np.abs(
                    np.sum(
                        p2 * lines,
                        axis=1
                    )
                )

                denominator = np.sqrt(
                    lines[:, 0] ** 2 +
                    lines[:, 1] ** 2
                )

                denominator[
                    denominator < 1e-8
                ] = 1e-8

                errors = (
                    numerator /
                    denominator
                )

                fundamental_error = float(
                    np.sqrt(
                        np.mean(
                            errors ** 2
                        )
                    )
                )

    # --------------------------------------------------------
    # Spatial distribution
    # --------------------------------------------------------

    grid = np.zeros(
        (4, 4),
        dtype=np.uint8
    )

    for x, y in pts1:

        col = min(
            3,
            int(
                x / max(w, 1) * 4
            )
        )

        row = min(
            3,
            int(
                y / max(h, 1) * 4
            )
        )

        grid[row, col] = 1

    coverage = float(
        np.sum(grid) / 16.0
    )

    # --------------------------------------------------------
    # Normalized features
    # --------------------------------------------------------

    return [
        total_matches / 200.0,
        mean_distance / 500.0,
        median_distance / 500.0,
        min_distance / 500.0,
        max_distance / 500.0,

        homography_inliers / max(
            total_matches,
            1
        ),

        fundamental_inliers / max(
            total_matches,
            1
        ),

        homography_error / 100.0,
        fundamental_error / 10.0,

        coverage,

        homography_inliers / 20.0,
        fundamental_inliers / 20.0
    ]


# ============================================================
# LOAD BASE IMAGES
# ============================================================

images = []

for path in image_files:

    image = cv2.imread(path)

    if image is not None:
        images.append(image)

print(
    "Valid images:",
    len(images)
)

# ============================================================
# CREATE TRAINING SAMPLES
# ============================================================

rows = []

# ============================================================
# POSITIVE PAIRS
# ============================================================

print("\nGenerating positive samples...")

for i in range(
    POSITIVE_PAIRS
):

    image = random.choice(
        images
    )

    reference = image.copy()

    target = augment_image(
        image
    )

    kp1, des1 = extract_features(
        reference
    )

    kp2, des2 = extract_features(
        target
    )

    matches = get_matches(
        des1,
        des2
    )

    features = calculate_features(
        kp1,
        kp2,
        matches,
        reference.shape
    )

    if features is None:
        continue

    rows.append(
        features + [1]
    )

    if (
        i + 1
    ) % 25 == 0:

        print(
            f"Positive: {i + 1}/{POSITIVE_PAIRS}"
        )


# ============================================================
# NEGATIVE PAIRS
# ============================================================

print("\nGenerating negative samples...")

for i in range(
    NEGATIVE_PAIRS
):

    image1, image2 = random.sample(
        images,
        2
    )

    reference = image1.copy()
    target = image2.copy()

    kp1, des1 = extract_features(
        reference
    )

    kp2, des2 = extract_features(
        target
    )

    matches = get_matches(
        des1,
        des2
    )

    features = calculate_features(
        kp1,
        kp2,
        matches,
        reference.shape
    )

    if features is None:

        features = [
            0.0
        ] * 12

    rows.append(
        features + [0]
    )

    if (
        i + 1
    ) % 25 == 0:

        print(
            f"Negative: {i + 1}/{NEGATIVE_PAIRS}"
        )


# ============================================================
# SAVE CSV
# ============================================================

headers = [
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
    "fundamental_inliers",

    "label"
]

random.shuffle(
    rows
)

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow(
        headers
    )

    writer.writerows(
        rows
    )


# ============================================================
# SUMMARY
# ============================================================

positive = sum(
    1
    for row in rows
    if row[-1] == 1
)

negative = sum(
    1
    for row in rows
    if row[-1] == 0
)

print("\n")
print("=" * 70)
print("TRAINING DATA CREATED")
print("=" * 70)

print(
    "Total samples:",
    len(rows)
)

print(
    "Positive samples:",
    positive
)

print(
    "Negative samples:",
    negative
)

print(
    "\nSaved:",
    OUTPUT_FILE
)

print("=" * 70)