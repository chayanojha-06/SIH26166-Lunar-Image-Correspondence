import cv2
import numpy as np


IMAGE1 = "data/raw/image1.jpg"
IMAGE2 = "data/raw/image2.jpg"

LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 5.0


# ============================================================
# LOAD IMAGES
# ============================================================

image1 = cv2.imread(IMAGE1)
image2 = cv2.imread(IMAGE2)

if image1 is None or image2 is None:
    print("ERROR: Could not load images.")
    exit()

print("=" * 90)
print("SIH26166 GEOMETRIC MODEL COMPARISON")
print("=" * 90)

print("\nReference:", image1.shape)
print("Target   :", image2.shape)


# ============================================================
# SIFT FEATURES
# ============================================================

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

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

print("\nFeatures:")
print("Reference:", len(kp1))
print("Target   :", len(kp2))


# ============================================================
# FLANN MATCHING
# ============================================================

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


# ============================================================
# LOWE RATIO TEST
# ============================================================

good_matches = []

for pair in matches:

    if len(pair) != 2:
        continue

    m, n = pair

    if m.distance < LOWE_RATIO * n.distance:
        good_matches.append(m)


print("\nGood matches:", len(good_matches))


if len(good_matches) < 4:
    print("Not enough matches for geometric verification.")
    exit()


points1 = np.float32([
    kp1[m.queryIdx].pt
    for m in good_matches
])

points2 = np.float32([
    kp2[m.trainIdx].pt
    for m in good_matches
])


# ============================================================
# HOMOGRAPHY
# ============================================================

def test_homography():

    H, mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        RANSAC_THRESHOLD
    )

    if H is None or mask is None:
        return 0, 0, None

    mask = mask.ravel()

    p1 = points1[mask == 1]
    p2 = points2[mask == 1]

    inliers = len(p1)

    if inliers < 4:
        return inliers, 0, None

    projected = cv2.perspectiveTransform(
        p1.reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)

    error = projected - p2

    rmse = np.sqrt(
        np.mean(
            np.sum(
                error ** 2,
                axis=1
            )
        )
    )

    ratio = (
        inliers /
        len(good_matches)
    ) * 100

    return inliers, ratio, rmse


# ============================================================
# AFFINE
# ============================================================

def test_affine():

    A, mask = cv2.estimateAffine2D(
        points1,
        points2,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD,
        maxIters=5000,
        confidence=0.99
    )

    if A is None or mask is None:
        return 0, 0, None

    mask = mask.ravel()

    p1 = points1[mask == 1]
    p2 = points2[mask == 1]

    inliers = len(p1)

    if inliers < 3:
        return inliers, 0, None

    projected = cv2.transform(
        p1.reshape(-1, 1, 2),
        A
    ).reshape(-1, 2)

    error = projected - p2

    rmse = np.sqrt(
        np.mean(
            np.sum(
                error ** 2,
                axis=1
            )
        )
    )

    ratio = (
        inliers /
        len(good_matches)
    ) * 100

    return inliers, ratio, rmse


# ============================================================
# FUNDAMENTAL MATRIX
# ============================================================

def test_fundamental():

    F, mask = cv2.findFundamentalMat(
        points1,
        points2,
        cv2.FM_RANSAC,
        RANSAC_THRESHOLD,
        0.99
    )

    if F is None or mask is None:
        return 0, 0, None

    mask = mask.ravel()

    p1 = points1[mask == 1]
    p2 = points2[mask == 1]

    inliers = len(p1)

    if inliers < 8:
        return inliers, 0, None

    # Symmetric epipolar error
    lines1 = cv2.computeCorrespondEpilines(
        p2.reshape(-1, 1, 2),
        2,
        F
    ).reshape(-1, 3)

    lines2 = cv2.computeCorrespondEpilines(
        p1.reshape(-1, 1, 2),
        1,
        F
    ).reshape(-1, 3)

    numerator1 = np.abs(
        np.sum(
            lines1 * np.hstack(
                (p1, np.ones((len(p1), 1)))
            ),
            axis=1
        )
    )

    numerator2 = np.abs(
        np.sum(
            lines2 * np.hstack(
                (p2, np.ones((len(p2), 1)))
            ),
            axis=1
        )
    )

    denominator1 = np.sqrt(
        lines1[:, 0] ** 2 +
        lines1[:, 1] ** 2
    )

    denominator2 = np.sqrt(
        lines2[:, 0] ** 2 +
        lines2[:, 1] ** 2
    )

    errors = (
        (numerator1 / denominator1) ** 2 +
        (numerator2 / denominator2) ** 2
    )

    rmse = np.sqrt(
        np.mean(errors)
    )

    ratio = (
        inliers /
        len(good_matches)
    ) * 100

    return inliers, ratio, rmse


# ============================================================
# RUN MODELS
# ============================================================

print("\n")
print("=" * 90)
print("RUNNING GEOMETRIC MODELS")
print("=" * 90)

print("\nTesting Homography...")
homography_result = test_homography()

print("Testing Affine...")
affine_result = test_affine()

print("Testing Fundamental Matrix...")
fundamental_result = test_fundamental()


# ============================================================
# RESULTS
# ============================================================

results = [
    (
        "Homography",
        homography_result
    ),
    (
        "Affine",
        affine_result
    ),
    (
        "Fundamental",
        fundamental_result
    )
]


print("\n")
print("=" * 90)

print(
    f"{'MODEL':<20}"
    f"{'INLIERS':>12}"
    f"{'RATIO':>14}"
    f"{'RMSE':>14}"
)

print("-" * 90)

for name, result in results:

    inliers, ratio, rmse = result

    rmse_text = (
        f"{rmse:.4f}"
        if rmse is not None
        else "N/A"
    )

    print(
        f"{name:<20}"
        f"{inliers:>12}"
        f"{ratio:>13.2f}%"
        f"{rmse_text:>14}"
    )


# ============================================================
# BEST MODEL
# ============================================================

valid = [
    x for x in results
    if x[1][0] >= 4
]

if valid:

    best = max(
        valid,
        key=lambda x: (
            x[1][0],
            x[1][1],
            -(x[1][2] if x[1][2] is not None else 999)
        )
    )

    name = best[0]
    inliers, ratio, rmse = best[1]

    print("\n")
    print("=" * 90)
    print("BEST GEOMETRIC MODEL")
    print("=" * 90)

    print("\nModel:", name)
    print("Inliers:", inliers)
    print(f"Inlier ratio: {ratio:.2f}%")

    if rmse is not None:
        print(f"RMSE: {rmse:.4f} pixels")

else:

    print("\nNo valid geometric model found.")


print("\n")
print("=" * 90)
print("GEOMETRIC MODEL COMPARISON COMPLETED")
print("=" * 90)