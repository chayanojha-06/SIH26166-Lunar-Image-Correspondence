import cv2
import numpy as np
import os
import glob


ORIGINAL_IMAGE = "data/raw/image1.jpg"
DATASET_FOLDER = "data/processed"


# ==========================================
# MATCHING FUNCTION
# ==========================================

def match_images(image1, image2):

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(
        nfeatures=10000,
        contrastThreshold=0.03
    )

    keypoints1, descriptors1 = sift.detectAndCompute(
        gray1, None
    )

    keypoints2, descriptors2 = sift.detectAndCompute(
        gray2, None
    )

    if descriptors1 is None or descriptors2 is None:
        return 0, 0, 0, 0, None

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
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    for pair in matches:

        if len(pair) == 2:

            m, n = pair

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    inliers = 0
    rmse = None

    if len(good_matches) >= 4:

        points1 = np.float32([
            keypoints1[m.queryIdx].pt
            for m in good_matches
        ])

        points2 = np.float32([
            keypoints2[m.trainIdx].pt
            for m in good_matches
        ])

        homography, mask = cv2.findHomography(
            points1,
            points2,
            cv2.RANSAC,
            5.0
        )

        if mask is not None:

            inliers = int(np.sum(mask))

            if homography is not None:

                inlier_points1 = points1[
                    mask.ravel() == 1
                ]

                inlier_points2 = points2[
                    mask.ravel() == 1
                ]

                projected = cv2.perspectiveTransform(
                    inlier_points1.reshape(-1, 1, 2),
                    homography
                ).reshape(-1, 2)

                errors = projected - inlier_points2

                rmse = np.sqrt(
                    np.mean(
                        np.sum(errors ** 2, axis=1)
                    )
                )

    ratio = (
        (inliers / len(good_matches)) * 100
        if len(good_matches) > 0
        else 0
    )

    return (
        len(keypoints2),
        len(good_matches),
        inliers,
        ratio,
        rmse
    )


# ==========================================
# LOAD ORIGINAL
# ==========================================

original = cv2.imread(ORIGINAL_IMAGE)

if original is None:

    print("ERROR: Could not load original image.")
    exit()

print("=" * 95)
print("SIH26166 AUTOMATED DATASET BENCHMARK")
print("=" * 95)

print("\nOriginal image loaded.")


# ==========================================
# FIND ALL TEST IMAGES
# ==========================================

files = glob.glob(
    os.path.join(
        DATASET_FOLDER,
        "*.jpg"
    )
)

files = sorted(files)


# ==========================================
# RUN BENCHMARK
# ==========================================

results = []

print(
    f"\nFound {len(files)} test images."
)

for file in files:

    name = os.path.basename(file)

    print(
        f"Testing: {name}"
    )

    transformed = cv2.imread(file)

    if transformed is None:
        continue

    (
        features,
        good,
        inliers,
        ratio,
        rmse
    ) = match_images(
        original,
        transformed
    )

    results.append(
        (
            name,
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
    f"{'TEST':<25}"
    f"{'FEATURES':>12}"
    f"{'GOOD':>10}"
    f"{'INLIERS':>12}"
    f"{'RATIO':>12}"
    f"{'RMSE':>12}"
)

print("-" * 95)


for result in results:

    (
        name,
        features,
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
        f"{name:<25}"
        f"{features:>12}"
        f"{good:>10}"
        f"{inliers:>12}"
        f"{ratio:>11.2f}%"
        f"{rmse_text:>12}"
    )


print("=" * 95)

print(
    f"\nBenchmark completed: {len(results)} tests."
)
