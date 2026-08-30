import cv2
import numpy as np


ORIGINAL_IMAGE = "data/raw/image1.jpg"
TEST_IMAGE = "data/processed/extreme_combined.jpg"


def test_ratio(image1, image2, ratio_threshold):

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(
        nfeatures=10000,
        contrastThreshold=0.03
    )

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return 0, 0, 0, None

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=100)
    )

    matches = flann.knnMatch(
        des1,
        des2,
        k=2
    )

    good = []

    for pair in matches:

        if len(pair) == 2:

            m, n = pair

            if m.distance < ratio_threshold * n.distance:
                good.append(m)

    inliers = 0
    rmse = None

    if len(good) >= 4:

        points1 = np.float32([
            kp1[m.queryIdx].pt
            for m in good
        ])

        points2 = np.float32([
            kp2[m.trainIdx].pt
            for m in good
        ])

        H, mask = cv2.findHomography(
            points1,
            points2,
            cv2.RANSAC,
            5.0
        )

        if mask is not None:

            inliers = int(np.sum(mask))

            if H is not None and inliers >= 4:

                p1 = points1[mask.ravel() == 1]
                p2 = points2[mask.ravel() == 1]

                projected = cv2.perspectiveTransform(
                    p1.reshape(-1, 1, 2),
                    H
                ).reshape(-1, 2)

                errors = projected - p2

                rmse = np.sqrt(
                    np.mean(
                        np.sum(errors ** 2, axis=1)
                    )
                )

    ratio = (
        inliers / len(good) * 100
        if len(good) > 0
        else 0
    )

    return (
        len(good),
        inliers,
        ratio,
        rmse
    )


# ==========================================
# LOAD
# ==========================================

image1 = cv2.imread(ORIGINAL_IMAGE)
image2 = cv2.imread(TEST_IMAGE)

if image1 is None or image2 is None:

    print("ERROR: Could not load images.")
    exit()


print("=" * 80)
print("SIH26166 ADAPTIVE RATIO TEST")
print("=" * 80)


# ==========================================
# TEST DIFFERENT RATIOS
# ==========================================

thresholds = [
    0.65,
    0.70,
    0.75,
    0.80,
    0.85
]


print()
print(
    f"{'RATIO':<12}"
    f"{'GOOD':>12}"
    f"{'INLIERS':>12}"
    f"{'INLIER %':>14}"
    f"{'RMSE':>12}"
)

print("-" * 80)


for threshold in thresholds:

    good, inliers, ratio, rmse = test_ratio(
        image1,
        image2,
        threshold
    )

    rmse_text = (
        f"{rmse:.4f}"
        if rmse is not None
        else "N/A"
    )

    print(
        f"{threshold:<12.2f}"
        f"{good:>12}"
        f"{inliers:>12}"
        f"{ratio:>13.2f}%"
        f"{rmse_text:>12}"
    )


print("=" * 80)
print("TEST COMPLETED")
print("=" * 80)