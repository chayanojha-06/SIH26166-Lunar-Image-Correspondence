import cv2
import numpy as np


ORIGINAL_IMAGE = "data/raw/image1.jpg"
TEST_IMAGE = "data/processed/extreme_combined.jpg"


def match_images(image1, image2, upscale=1):

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    if upscale > 1:
        gray2 = cv2.resize(
            gray2,
            None,
            fx=upscale,
            fy=upscale,
            interpolation=cv2.INTER_CUBIC
        )

    sift = cv2.SIFT_create(
        nfeatures=10000,
        contrastThreshold=0.03
    )

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return len(kp2), 0, 0, 0, None

    index_params = dict(
        algorithm=1,
        trees=5
    )

    search_params = dict(
        checks=100
    )

    flann = cv2.FlannBasedMatcher(
        index_params,
        search_params
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

            if m.distance < 0.75 * n.distance:
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

    ratio = (
        inliers / len(good) * 100
        if len(good) > 0
        else 0
    )

    return (
        len(kp2),
        len(good),
        inliers,
        ratio,
        rmse
    )


# ==========================================
# LOAD IMAGES
# ==========================================

original = cv2.imread(ORIGINAL_IMAGE)
extreme = cv2.imread(TEST_IMAGE)

if original is None or extreme is None:

    print("ERROR: Could not load images.")
    exit()


print("=" * 80)
print("SIH26166 EXTREME CASE UPSCALE TEST")
print("=" * 80)

print("\nOriginal image loaded.")
print("Extreme combined image loaded.")


# ==========================================
# BASELINE
# ==========================================

baseline = match_images(
    original,
    extreme,
    upscale=1
)


# ==========================================
# 2X UPSCALE
# ==========================================

upscaled = match_images(
    original,
    extreme,
    upscale=2
)


# ==========================================
# RESULTS
# ==========================================

print("\n")

print("=" * 80)

print(
    f"{'METHOD':<20}"
    f"{'FEATURES':>12}"
    f"{'GOOD':>10}"
    f"{'INLIERS':>12}"
    f"{'RATIO':>12}"
    f"{'RMSE':>12}"
)

print("-" * 80)


for name, result in [
    ("Baseline", baseline),
    ("2x Upscale", upscaled)
]:

    features, good, inliers, ratio, rmse = result

    rmse_text = (
        f"{rmse:.4f}"
        if rmse is not None
        else "N/A"
    )

    print(
        f"{name:<20}"
        f"{features:>12}"
        f"{good:>10}"
        f"{inliers:>12}"
        f"{ratio:>11.2f}%"
        f"{rmse_text:>12}"
    )


print("=" * 80)

print("\nTest completed successfully!")