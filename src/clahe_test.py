import cv2
import numpy as np


INPUT_IMAGE = "data/raw/image1.jpg"


def apply_clahe(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced


def match_images(gray1, gray2):

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

        return 0, 0, 0, None

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

    return (
        len(keypoints1),
        len(good_matches),
        inliers,
        rmse
    )


# ==========================================
# LOAD ORIGINAL
# ==========================================

original = cv2.imread(INPUT_IMAGE)

if original is None:

    print("ERROR: Image could not be loaded.")

    exit()


# ==========================================
# CREATE EXTREME TEST
# ==========================================

transformed = cv2.resize(
    original,
    None,
    fx=0.3,
    fy=0.3,
    interpolation=cv2.INTER_AREA
)

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

transformed = cv2.convertScaleAbs(
    transformed,
    alpha=0.5,
    beta=-60
)


# ==========================================
# NORMAL PIPELINE
# ==========================================

gray_original = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2GRAY
)

gray_transformed = cv2.cvtColor(
    transformed,
    cv2.COLOR_BGR2GRAY
)

normal_result = match_images(
    gray_original,
    gray_transformed
)


# ==========================================
# CLAHE PIPELINE
# ==========================================

clahe_original = apply_clahe(
    original
)

clahe_transformed = apply_clahe(
    transformed
)

clahe_result = match_images(
    clahe_original,
    clahe_transformed
)


# ==========================================
# RESULTS
# ==========================================

print("\n==========================================")
print("SIH26166 CLAHE IMPROVEMENT TEST")
print("==========================================")

print("\nNORMAL SIFT PIPELINE")

print(
    "Features:",
    normal_result[0]
)

print(
    "Good matches:",
    normal_result[1]
)

print(
    "RANSAC inliers:",
    normal_result[2]
)

if normal_result[1] > 0:

    print(
        "Inlier ratio:",
        f"{normal_result[2] / normal_result[1] * 100:.2f}%"
    )

print(
    "RMSE:",
    normal_result[3]
)


print("\nCLAHE + SIFT PIPELINE")

print(
    "Features:",
    clahe_result[0]
)

print(
    "Good matches:",
    clahe_result[1]
)

print(
    "RANSAC inliers:",
    clahe_result[2]
)

if clahe_result[1] > 0:

    print(
        "Inlier ratio:",
        f"{clahe_result[2] / clahe_result[1] * 100:.2f}%"
    )

print(
    "RMSE:",
    clahe_result[3]
)


print("\n==========================================")
print("TEST COMPLETED")
print("==========================================")