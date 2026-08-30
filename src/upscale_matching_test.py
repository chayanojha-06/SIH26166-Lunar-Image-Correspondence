import cv2
import numpy as np


ORIGINAL_IMAGE = "data/raw/image1.jpg"
TEST_IMAGE = "data/processed/scale_20.jpg"


# ==========================================
# MATCHING FUNCTION
# ==========================================

def match_images(image1, image2, upscale=1):

    gray1 = cv2.cvtColor(
        image1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        image2,
        cv2.COLOR_BGR2GRAY
    )

    # Upscale transformed image
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

    keypoints1, descriptors1 = sift.detectAndCompute(
        gray1,
        None
    )

    keypoints2, descriptors2 = sift.detectAndCompute(
        gray2,
        None
    )

    if descriptors1 is None or descriptors2 is None:

        return 0, 0, 0, 0, None

    # ======================================
    # FLANN
    # ======================================

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

    # ======================================
    # LOWE RATIO TEST
    # ======================================

    good_matches = []

    for pair in matches:

        if len(pair) == 2:

            m, n = pair

            if m.distance < 0.75 * n.distance:

                good_matches.append(m)

    # ======================================
    # RANSAC
    # ======================================

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

            inliers = int(
                np.sum(mask)
            )

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

                errors = (
                    projected -
                    inlier_points2
                )

                rmse = np.sqrt(
                    np.mean(
                        np.sum(
                            errors ** 2,
                            axis=1
                        )
                    )
                )

    ratio = (
        inliers / len(good_matches) * 100
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
# LOAD IMAGES
# ==========================================

original = cv2.imread(
    ORIGINAL_IMAGE
)

test_image = cv2.imread(
    TEST_IMAGE
)

if original is None or test_image is None:

    print("ERROR: Could not load images.")
    exit()


print("=" * 75)
print("SIH26166 UPSCALE MATCHING TEST")
print("=" * 75)

print("\nTest image:", TEST_IMAGE)


# ==========================================
# BASELINE
# ==========================================

baseline = match_images(
    original,
    test_image,
    upscale=1
)


# ==========================================
# 2X UPSCALE
# ==========================================

upscaled = match_images(
    original,
    test_image,
    upscale=2
)


# ==========================================
# RESULTS
# ==========================================

print("\n")
print("=" * 75)

print(
    f"{'METHOD':<20}"
    f"{'FEATURES':>12}"
    f"{'GOOD':>10}"
    f"{'INLIERS':>12}"
    f"{'RATIO':>12}"
    f"{'RMSE':>12}"
)

print("-" * 75)


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


print("=" * 75)

print("\nTest completed successfully!")