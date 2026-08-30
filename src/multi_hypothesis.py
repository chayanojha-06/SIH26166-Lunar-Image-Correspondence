import cv2
import numpy as np


ORIGINAL_IMAGE = "data/raw/image1.jpg"
TEST_IMAGE = "data/processed/extreme_combined.jpg"


# ==========================================
# PREPROCESSING
# ==========================================

def preprocess(image, method):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    if method == "Baseline":
        return gray

    if method == "CLAHE":

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        return clahe.apply(gray)

    if method == "Upscale":

        return cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

    if method == "Upscale + CLAHE":

        upscaled = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        return clahe.apply(upscaled)

    return gray


# ==========================================
# MATCHING
# ==========================================

def run_matching(image1, image2, method):

    img1 = preprocess(image1, "Baseline")
    img2 = preprocess(image2, method)

    sift = cv2.SIFT_create(
        nfeatures=10000,
        contrastThreshold=0.03
    )

    kp1, des1 = sift.detectAndCompute(
        img1,
        None
    )

    kp2, des2 = sift.detectAndCompute(
        img2,
        None
    )

    if des1 is None or des2 is None:

        return (
            len(kp2),
            0,
            0,
            0,
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

            inliers = int(
                np.sum(mask)
            )

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

original = cv2.imread(
    ORIGINAL_IMAGE
)

extreme = cv2.imread(
    TEST_IMAGE
)

if original is None or extreme is None:

    print("ERROR: Could not load images.")
    exit()


print("=" * 95)
print("SIH26166 MULTI-HYPOTHESIS MATCHING")
print("=" * 95)

print("\nOriginal image loaded.")
print("Extreme image loaded.")


# ==========================================
# HYPOTHESES
# ==========================================

methods = [
    "Baseline",
    "CLAHE",
    "Upscale",
    "Upscale + CLAHE"
]


results = []


# ==========================================
# RUN
# ==========================================

for method in methods:

    print("\nRunning:", method)

    result = run_matching(
        original,
        extreme,
        method
    )

    features, good, inliers, ratio, rmse = result

    results.append(
        (
            method,
            features,
            good,
            inliers,
            ratio,
            rmse
        )
    )


# ==========================================
# PRINT RESULTS
# ==========================================

print("\n")

print("=" * 95)

print(
    f"{'METHOD':<25}"
    f"{'FEATURES':>12}"
    f"{'GOOD':>10}"
    f"{'INLIERS':>12}"
    f"{'RATIO':>12}"
    f"{'RMSE':>12}"
)

print("-" * 95)


for result in results:

    method, features, good, inliers, ratio, rmse = result

    rmse_text = (
        f"{rmse:.4f}"
        if rmse is not None
        else "N/A"
    )

    print(
        f"{method:<25}"
        f"{features:>12}"
        f"{good:>10}"
        f"{inliers:>12}"
        f"{ratio:>11.2f}%"
        f"{rmse_text:>12}"
    )


# ==========================================
# SELECT BEST
# ==========================================

valid_results = [
    r for r in results
    if r[3] >= 4
]


if valid_results:

    # Primary criterion:
    # highest number of RANSAC inliers

    best = max(
        valid_results,
        key=lambda r: (
            r[3],
            r[4],
            -r[5] if r[5] is not None else -999
        )
    )

    print("\n")
    print("=" * 95)
    print("BEST HYPOTHESIS")
    print("=" * 95)

    print("Method:", best[0])
    print("Features:", best[1])
    print("Good matches:", best[2])
    print("RANSAC inliers:", best[3])
    print(f"Inlier ratio: {best[4]:.2f}%")

    if best[5] is not None:
        print(f"RMSE: {best[5]:.4f} pixels")

else:

    print("\nNo reliable geometric solution found.")


print("=" * 95)
print("MULTI-HYPOTHESIS TEST COMPLETED")
print("=" * 95)
