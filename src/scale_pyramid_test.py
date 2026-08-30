import cv2
import numpy as np
import os


# ============================================================
# SIH26166 SCALE PYRAMID MATCHING
# ============================================================

IMAGE1 = "data/raw/image1.jpg"
IMAGE2 = "data/raw/image2.jpg"

SCALES = [1.0, 0.75, 0.50, 0.35]

LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 5.0


# ============================================================
# LOAD
# ============================================================

image1 = cv2.imread(IMAGE1)
image2 = cv2.imread(IMAGE2)

if image1 is None or image2 is None:
    print("ERROR: Could not load images.")
    exit()

print("=" * 90)
print("SIH26166 SCALE PYRAMID MATCHING")
print("=" * 90)

print("\nReference:", image1.shape)
print("Target   :", image2.shape)


# ============================================================
# SIFT
# ============================================================

sift = cv2.SIFT_create(
    nfeatures=10000,
    contrastThreshold=0.03
)


# ============================================================
# MATCH FUNCTION
# ============================================================

def test_scale(scale):

    # Resize reference image
    scaled = cv2.resize(
        image1,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    gray1 = cv2.cvtColor(
        scaled,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        image2,
        cv2.COLOR_BGR2GRAY
    )

    kp1, des1 = sift.detectAndCompute(
        gray1,
        None
    )

    kp2, des2 = sift.detectAndCompute(
        gray2,
        None
    )

    if des1 is None or des2 is None:
        return len(kp1), len(kp2), 0, 0, 0, None

    # --------------------------------------------------------
    # FLANN
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LOWE RATIO
    # --------------------------------------------------------

    good = []

    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    if len(good) < 4:

        return (
            len(kp1),
            len(kp2),
            len(good),
            0,
            0,
            None
        )

    # --------------------------------------------------------
    # POINTS
    # --------------------------------------------------------

    points1 = np.float32([
        kp1[m.queryIdx].pt
        for m in good
    ])

    points2 = np.float32([
        kp2[m.trainIdx].pt
        for m in good
    ])

    # --------------------------------------------------------
    # RANSAC
    # --------------------------------------------------------

    H, mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        RANSAC_THRESHOLD
    )

    if H is None or mask is None:

        return (
            len(kp1),
            len(kp2),
            len(good),
            0,
            0,
            None
        )

    mask = mask.ravel()

    inlier_points1 = points1[
        mask == 1
    ]

    inlier_points2 = points2[
        mask == 1
    ]

    inliers = len(
        inlier_points1
    )

    ratio = (
        inliers /
        len(good)
    ) * 100

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = None

    if inliers >= 4:

        projected = cv2.perspectiveTransform(
            inlier_points1.reshape(-1, 1, 2),
            H
        ).reshape(-1, 2)

        errors = (
            projected -
            inlier_points2
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        errors ** 2,
                        axis=1
                    )
                )
            )
        )

    return (
        len(kp1),
        len(kp2),
        len(good),
        inliers,
        ratio,
        rmse
    )


# ============================================================
# RUN TESTS
# ============================================================

results = []

print()
print("=" * 90)
print("TESTING SCALE PYRAMID")
print("=" * 90)

for scale in SCALES:

    print(
        f"\nTesting reference scale: {scale:.2f}x"
    )

    result = test_scale(scale)

    results.append(
        (scale, result)
    )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 95)

print(
    f"{'SCALE':<12}"
    f"{'REF FEAT':>12}"
    f"{'TARGET FEAT':>14}"
    f"{'GOOD':>10}"
    f"{'INLIERS':>12}"
    f"{'RATIO':>12}"
    f"{'RMSE':>12}"
)

print("-" * 95)

for scale, result in results:

    (
        ref_features,
        target_features,
        good,
        inliers,
        ratio,
        rmse
    ) = result

    rmse_text = (
        f"{rmse:.4f}"
        if rmse is not None
        else "N/A"
    )

    print(
        f"{scale:<12.2f}"
        f"{ref_features:>12}"
        f"{target_features:>14}"
        f"{good:>10}"
        f"{inliers:>12}"
        f"{ratio:>11.2f}%"
        f"{rmse_text:>12}"
    )


# ============================================================
# BEST SCALE
# ============================================================

valid_results = [
    x for x in results
    if x[1][3] >= 4
]

if valid_results:

    best_scale, best_result = max(
        valid_results,
        key=lambda x: (
            x[1][3],
            x[1][4],
            -(
                x[1][5]
                if x[1][5] is not None
                else 999
            )
        )
    )

    print()
    print("=" * 90)
    print("BEST SCALE")
    print("=" * 90)

    print(
        f"\nScale: {best_scale:.2f}x"
    )

    print(
        "Good matches:",
        best_result[2]
    )

    print(
        "RANSAC inliers:",
        best_result[3]
    )

    print(
        f"Inlier ratio: "
        f"{best_result[4]:.2f}%"
    )

    if best_result[5] is not None:

        print(
            f"RMSE: "
            f"{best_result[5]:.4f} pixels"
        )

else:

    print()
    print(
        "No reliable geometric solution found."
    )


print()
print("=" * 90)
print("SCALE PYRAMID TEST COMPLETED")
print("=" * 90)