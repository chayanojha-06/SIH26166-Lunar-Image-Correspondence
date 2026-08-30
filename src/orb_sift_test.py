import cv2
import numpy as np


IMAGE1 = "data/raw/image1.jpg"
IMAGE2 = "data/raw/image2.jpg"

RANSAC_THRESHOLD = 5.0


def evaluate(points1, points2):

    if len(points1) < 4:
        return 0, 0, None

    H, mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        RANSAC_THRESHOLD
    )

    if H is None or mask is None:
        return 0, 0, None

    mask = mask.ravel()

    inliers1 = points1[mask == 1]
    inliers2 = points2[mask == 1]

    inlier_count = len(inliers1)

    ratio = (
        inlier_count / len(points1)
    ) * 100

    rmse = None

    if inlier_count >= 4:

        projected = cv2.perspectiveTransform(
            inliers1.reshape(-1, 1, 2),
            H
        ).reshape(-1, 2)

        errors = projected - inliers2

        rmse = np.sqrt(
            np.mean(
                np.sum(
                    errors ** 2,
                    axis=1
                )
            )
        )

    return inlier_count, ratio, rmse


print("=" * 85)
print("SIH26166 SIFT + ORB COMPARISON")
print("=" * 85)

image1 = cv2.imread(IMAGE1)
image2 = cv2.imread(IMAGE2)

if image1 is None or image2 is None:
    print("ERROR: Could not load images.")
    exit()

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


# ============================================================
# SIFT
# ============================================================

print("\nRunning SIFT...")

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

good_sift = []

for pair in matches:

    if len(pair) == 2:

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_sift.append(m)

points1 = np.float32([
    kp1[m.queryIdx].pt
    for m in good_sift
])

points2 = np.float32([
    kp2[m.trainIdx].pt
    for m in good_sift
])

sift_inliers, sift_ratio, sift_rmse = evaluate(
    points1,
    points2
)


# ============================================================
# ORB
# ============================================================

print("Running ORB...")

orb = cv2.ORB_create(
    nfeatures=10000,
    scaleFactor=1.2,
    nlevels=8
)

okp1, odes1 = orb.detectAndCompute(
    gray1,
    None
)

okp2, odes2 = orb.detectAndCompute(
    gray2,
    None
)

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)

orb_matches = bf.knnMatch(
    odes1,
    odes2,
    k=2
)

good_orb = []

for pair in orb_matches:

    if len(pair) == 2:

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_orb.append(m)

orb_points1 = np.float32([
    okp1[m.queryIdx].pt
    for m in good_orb
])

orb_points2 = np.float32([
    okp2[m.trainIdx].pt
    for m in good_orb
])

orb_inliers, orb_ratio, orb_rmse = evaluate(
    orb_points1,
    orb_points2
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 85)

print(
    f"{'METHOD':<15}"
    f"{'FEATURES':>12}"
    f"{'GOOD':>12}"
    f"{'INLIERS':>12}"
    f"{'RATIO':>12}"
    f"{'RMSE':>12}"
)

print("-" * 85)

sift_rmse_text = (
    f"{sift_rmse:.4f}"
    if sift_rmse is not None
    else "N/A"
)

orb_rmse_text = (
    f"{orb_rmse:.4f}"
    if orb_rmse is not None
    else "N/A"
)

print(
    f"{'SIFT':<15}"
    f"{len(kp1):>12}"
    f"{len(good_sift):>12}"
    f"{sift_inliers:>12}"
    f"{sift_ratio:>11.2f}%"
    f"{sift_rmse_text:>12}"
)

print(
    f"{'ORB':<15}"
    f"{len(okp1):>12}"
    f"{len(good_orb):>12}"
    f"{orb_inliers:>12}"
    f"{orb_ratio:>11.2f}%"
    f"{orb_rmse_text:>12}"
)

print("=" * 85)

if orb_inliers > sift_inliers:
    print("\nBEST METHOD: ORB")
else:
    print("\nBEST METHOD: SIFT")

print("\nSIFT + ORB COMPARISON COMPLETED")
print("=" * 85)
