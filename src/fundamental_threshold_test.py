import cv2
import numpy as np


IMAGE1 = "data/raw/image1.jpg"
IMAGE2 = "data/raw/image2.jpg"

THRESHOLDS = [1.0, 2.0, 3.0, 5.0, 8.0]


print("=" * 90)
print("SIH26166 FUNDAMENTAL MATRIX RANSAC THRESHOLD TEST")
print("=" * 90)


# ============================================================
# LOAD
# ============================================================

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


# ============================================================
# FLANN
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
# LOWE RATIO
# ============================================================

good = []

for pair in matches:

    if len(pair) != 2:
        continue

    m, n = pair

    if m.distance < 0.75 * n.distance:
        good.append(m)


points1 = np.float32([
    kp1[m.queryIdx].pt
    for m in good
])

points2 = np.float32([
    kp2[m.trainIdx].pt
    for m in good
])


print("\nFeatures:")
print("Reference:", len(kp1))
print("Target   :", len(kp2))

print("\nGood matches:", len(good))


# ============================================================
# TEST THRESHOLDS
# ============================================================

results = []


print("\n")
print("=" * 90)
print("RANSAC THRESHOLD RESULTS")
print("=" * 90)

for threshold in THRESHOLDS:

    F, mask = cv2.findFundamentalMat(
        points1,
        points2,
        cv2.FM_RANSAC,
        threshold,
        0.99
    )

    if F is None or mask is None:

        results.append(
            (
                threshold,
                0,
                0,
                None
            )
        )

        continue


    mask = mask.ravel()

    p1 = points1[
        mask == 1
    ]

    p2 = points2[
        mask == 1
    ]

    inliers = len(p1)

    ratio = (
        inliers /
        len(good)
    ) * 100


    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = None

    if inliers >= 8:

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


        pts1_h = np.hstack(
            (
                p1,
                np.ones(
                    (len(p1), 1)
                )
            )
        )

        pts2_h = np.hstack(
            (
                p2,
                np.ones(
                    (len(p2), 1)
                )
            )
        )


        error1 = np.abs(
            np.sum(
                lines1 * pts1_h,
                axis=1
            )
        ) / np.sqrt(
            lines1[:, 0] ** 2 +
            lines1[:, 1] ** 2
        )


        error2 = np.abs(
            np.sum(
                lines2 * pts2_h,
                axis=1
            )
        ) / np.sqrt(
            lines2[:, 0] ** 2 +
            lines2[:, 1] ** 2
        )


        rmse = np.sqrt(
            np.mean(
                error1 ** 2 +
                error2 ** 2
            )
        )


    results.append(
        (
            threshold,
            inliers,
            ratio,
            rmse
        )
    )


# ============================================================
# PRINT
# ============================================================

print()

print(
    f"{'THRESHOLD':<15}"
    f"{'INLIERS':>12}"
    f"{'RATIO':>14}"
    f"{'RMSE':>14}"
)

print("-" * 90)


for threshold, inliers, ratio, rmse in results:

    rmse_text = (
        f"{rmse:.4f}"
        if rmse is not None
        else "N/A"
    )

    print(
        f"{threshold:<15.1f}"
        f"{inliers:>12}"
        f"{ratio:>13.2f}%"
        f"{rmse_text:>14}"
    )


# ============================================================
# BEST
# ============================================================

valid = [
    r for r in results
    if r[1] >= 8
]


if valid:

    best = max(
        valid,
        key=lambda r: (
            r[1],
            r[2],
            -(r[3] if r[3] is not None else 999)
        )
    )

    print("\n")
    print("=" * 90)
    print("BEST FUNDAMENTAL MATRIX CONFIGURATION")
    print("=" * 90)

    print(
        f"\nThreshold: {best[0]:.1f} pixels"
    )

    print(
        f"Inliers: {best[1]}"
    )

    print(
        f"Inlier ratio: {best[2]:.2f}%"
    )

    if best[3] is not None:

        print(
            f"RMSE: {best[3]:.4f} pixels"
        )

else:

    print("\nNo stable fundamental solution found.")


print("\n")
print("=" * 90)
print("FUNDAMENTAL THRESHOLD TEST COMPLETED")
print("=" * 90)