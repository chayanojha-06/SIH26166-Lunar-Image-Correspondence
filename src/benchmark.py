import cv2
import numpy as np


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_IMAGE = "data/raw/image1.jpg"


# ==========================================
# CREATE TRANSFORMED IMAGE
# ==========================================

def transform_image(image, test_name):

    # ======================================
    # SCALE 30%
    # ======================================

    if test_name == "Scale 30%":

        return cv2.resize(
            image,
            None,
            fx=0.3,
            fy=0.3,
            interpolation=cv2.INTER_AREA
        )

    # ======================================
    # ROTATION 45°
    # ======================================

    elif test_name == "Rotation 45°":

        height, width = image.shape[:2]

        center = (width // 2, height // 2)

        matrix = cv2.getRotationMatrix2D(
            center,
            45,
            1.0
        )

        return cv2.warpAffine(
            image,
            matrix,
            (width, height)
        )

    # ======================================
    # DARKER ILLUMINATION
    # ======================================

    elif test_name == "Brightness -60":

        return cv2.convertScaleAbs(
            image,
            alpha=1.0,
            beta=-60
        )

    # ======================================
    # LOW CONTRAST
    # ======================================

    elif test_name == "Contrast 0.5x":

        return cv2.convertScaleAbs(
            image,
            alpha=0.5,
            beta=0
        )

    # ======================================
    # EXTREME COMBINATION
    # ======================================

    elif test_name == "Extreme Combined":

        # Scale
        transformed = cv2.resize(
            image,
            None,
            fx=0.3,
            fy=0.3,
            interpolation=cv2.INTER_AREA
        )

        # Rotation
        height, width = transformed.shape[:2]

        center = (
            width // 2,
            height // 2
        )

        matrix = cv2.getRotationMatrix2D(
            center,
            45,
            1.0
        )

        transformed = cv2.warpAffine(
            transformed,
            matrix,
            (width, height)
        )

        # Brightness
        transformed = cv2.convertScaleAbs(
            transformed,
            alpha=1.0,
            beta=-60
        )

        # Contrast
        transformed = cv2.convertScaleAbs(
            transformed,
            alpha=0.5,
            beta=0
        )

        return transformed

    return image


# ==========================================
# MATCH TWO IMAGES
# ==========================================

def match_images(image1, image2):

    gray1 = cv2.cvtColor(
        image1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        image2,
        cv2.COLOR_BGR2GRAY
    )

    # SIFT
    sift = cv2.SIFT_create(
        nfeatures=10000,
        contrastThreshold=0.03
    )

    keypoints1, descriptors1 = sift.detectAndCompute(
        gray1,
        None
    )

    keypoints2, descriptors2 = sift.detectAndCompute(
        gray2,
        None
    )

    if descriptors1 is None or descriptors2 is None:
        return (
            len(keypoints1),
            len(keypoints2),
            0,
            0,
            None,
            0
        )

    # FLANN
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

    # Lowe Ratio Test
    good_matches = []

    for pair in matches:

        if len(pair) == 2:

            m, n = pair

            if m.distance < 0.75 * n.distance:

                good_matches.append(m)

    # RANSAC
    inlier_matches = []

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

            for i, match in enumerate(good_matches):

                if mask[i][0]:

                    inlier_matches.append(match)

        # RMSE
        if (
            homography is not None
            and len(inlier_matches) >= 4
        ):

            inlier_points1 = np.float32([
                keypoints1[m.queryIdx].pt
                for m in inlier_matches
            ])

            inlier_points2 = np.float32([
                keypoints2[m.trainIdx].pt
                for m in inlier_matches
            ])

            projected = cv2.perspectiveTransform(
                inlier_points1.reshape(-1, 1, 2),
                homography
            ).reshape(-1, 2)

            errors = projected - inlier_points2

            squared_errors = np.sum(
                errors ** 2,
                axis=1
            )

            rmse = np.sqrt(
                np.mean(squared_errors)
            )

    # Inlier ratio
    if len(good_matches) > 0:

        inlier_ratio = (
            len(inlier_matches)
            / len(good_matches)
        ) * 100

    else:

        inlier_ratio = 0

    return (
        len(keypoints1),
        len(keypoints2),
        len(good_matches),
        len(inlier_matches),
        rmse,
        inlier_ratio
    )


# ==========================================
# MAIN BENCHMARK
# ==========================================

print("=" * 70)

print("SIH26166 LUNAR IMAGE CORRESPONDENCE")
print("AUTOMATED ROBUSTNESS BENCHMARK")

print("=" * 70)


# Load original image

original = cv2.imread(INPUT_IMAGE)

if original is None:

    print("ERROR: Could not load image.")

    exit()


print("\nOriginal image loaded.")
print("Original size:", original.shape)


# Tests

tests = [
    "Scale 30%",
    "Rotation 45°",
    "Brightness -60",
    "Contrast 0.5x",
    "Extreme Combined"
]



results = []


# ==========================================
# RUN ALL TESTS
# ==========================================

for test_name in tests:

    print("\nRunning:", test_name)

    transformed = transform_image(
        original,
        test_name
    )

    (
        features1,
        features2,
        good_matches,
        inliers,
        rmse,
        ratio
    ) = match_images(
        original,
        transformed
    )

    results.append(
        (
            test_name,
            features2,
            good_matches,
            inliers,
            ratio,
            rmse
        )
    )


# ==========================================
# PRINT RESULTS
# ==========================================

print("\n")
print("=" * 85)

print(
    f"{'TEST':<20}"
    f"{'FEATURES':>12}"
    f"{'GOOD':>10}"
    f"{'INLIERS':>12}"
    f"{'RATIO':>12}"
    f"{'RMSE':>12}"
)

print("-" * 85)


for result in results:

    (
        test_name,
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
        f"{test_name:<20}"
        f"{features:>12}"
        f"{good:>10}"
        f"{inliers:>12}"
        f"{ratio:>11.2f}%"
        f"{rmse_text:>12}"
    )


print("=" * 85)

print("\nBenchmark completed successfully!")
