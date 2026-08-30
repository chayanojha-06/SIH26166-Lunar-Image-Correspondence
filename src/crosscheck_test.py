import cv2
import numpy as np


ORIGINAL_IMAGE = "data/raw/image1.jpg"
TEST_IMAGE = "data/processed/extreme_combined.jpg"


# ==========================================
# CROSS-CHECK MATCHING
# ==========================================

def crosscheck_matches(des1, des2):

    bf = cv2.BFMatcher(
        cv2.NORM_L2
    )

    # Image 1 -> Image 2
    matches_12 = bf.knnMatch(
        des1,
        des2,
        k=2
    )

    # Image 2 -> Image 1
    matches_21 = bf.knnMatch(
        des2,
        des1,
        k=2
    )

    forward = {}

    for pair in matches_12:

        if len(pair) == 2:

            m, n = pair

            if m.distance < 0.75 * n.distance:

                forward[m.queryIdx] = m.trainIdx

    backward = {}

    for pair in matches_21:

        if len(pair) == 2:

            m, n = pair

            if m.distance < 0.75 * n.distance:

                backward[m.queryIdx] = m.trainIdx

    mutual = []

    for query_idx, train_idx in forward.items():

        if train_idx in backward:

            if backward[train_idx] == query_idx:

                mutual.append(
                    cv2.DMatch(
                        _queryIdx=query_idx,
                        _trainIdx=train_idx,
                        _imgIdx=0,
                        _distance=0
                    )
                )

    return mutual


# ==========================================
# RANSAC + RMSE
# ==========================================

def calculate_geometry(
    kp1,
    kp2,
    matches
):

    if len(matches) < 4:

        return 0, 0, None

    points1 = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ])

    points2 = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ])

    H, mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        5.0
    )

    if mask is None:

        return 0, 0, None

    inliers = int(
        np.sum(mask)
    )

    if H is None or inliers < 4:

        return inliers, 0, None

    p1 = points1[
        mask.ravel() == 1
    ]

    p2 = points2[
        mask.ravel() == 1
    ]

    projected = cv2.perspectiveTransform(
        p1.reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)

    errors = projected - p2

    rmse = np.sqrt(
        np.mean(
            np.sum(
                errors ** 2,
                axis=1
            )
        )
    )

    return inliers, inliers / len(matches) * 100, rmse


# ==========================================
# LOAD IMAGES
# ==========================================

image1 = cv2.imread(
    ORIGINAL_IMAGE
)

image2 = cv2.imread(
    TEST_IMAGE
)

if image1 is None or image2 is None:

    print("ERROR: Could not load images.")
    exit()


print("=" * 85)
print("SIH26166 CROSS-CHECK MATCHING TEST")
print("=" * 85)


# ==========================================
# SIFT
# ==========================================

gray1 = cv2.cvtColor(
    image1,
    cv2.COLOR_BGR2GRAY
)

gray2 = cv2.cvtColor(
    image2,
    cv2.COLOR_BGR2GRAY
)

sift = cv2.SIFT_create(
    nfeatures=10000,
    contrastThreshold=0.03
)

kp1, des1 = sift.detectAndCompute(
    gray1,
    None
)

kp2, des2 = sift.detectAndCompute(
    gray2,
    None
)


# ==========================================
# NORMAL FLANN MATCHING
# ==========================================

flann = cv2.FlannBasedMatcher(
    dict(
        algorithm=1,
        trees=5
    ),
    dict(
        checks=100
    )
)

matches = flann.knnMatch(
    des1,
    des2,
    k=2
)

normal_matches = []

for pair in matches:

    if len(pair) == 2:

        m, n = pair

        if m.distance < 0.75 * n.distance:

            normal_matches.append(m)


normal_inliers, normal_ratio, normal_rmse = calculate_geometry(
    kp1,
    kp2,
    normal_matches
)


# ==========================================
# CROSS-CHECK
# ==========================================

mutual_matches = crosscheck_matches(
    des1,
    des2
)

cross_inliers, cross_ratio, cross_rmse = calculate_geometry(
    kp1,
    kp2,
    mutual_matches
)


# ==========================================
# RESULTS
# ==========================================

print("\n")

print("=" * 85)

print(
    f"{'METHOD':<25}"
    f"{'MATCHES':>12}"
    f"{'INLIERS':>12}"
    f"{'RATIO':>14}"
    f"{'RMSE':>12}"
)

print("-" * 85)


normal_rmse_text = (
    f"{normal_rmse:.4f}"
    if normal_rmse is not None
    else "N/A"
)

cross_rmse_text = (
    f"{cross_rmse:.4f}"
    if cross_rmse is not None
    else "N/A"
)


print(
    f"{'Normal Lowe Ratio':<25}"
    f"{len(normal_matches):>12}"
    f"{normal_inliers:>12}"
    f"{normal_ratio:>13.2f}%"
    f"{normal_rmse_text:>12}"
)

print(
    f"{'Lowe + Cross-Check':<25}"
    f"{len(mutual_matches):>12}"
    f"{cross_inliers:>12}"
    f"{cross_ratio:>13.2f}%"
    f"{cross_rmse_text:>12}"
)


print("=" * 85)

print("\nTest completed successfully!")