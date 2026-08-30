import cv2
import numpy as np


INPUT_IMAGE = "data/raw/image1.jpg"


# ==========================================
# PERSPECTIVE TRANSFORMATION
# ==========================================

def create_perspective_image(image):

    height, width = image.shape[:2]

    # Original corner points
    source_points = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])

    # Move corners to simulate viewpoint change
    destination_points = np.float32([
        [width * 0.08, height * 0.05],
        [width * 0.92, 0],
        [width * 0.82, height],
        [width * 0.18, height * 0.95]
    ])

    matrix = cv2.getPerspectiveTransform(
        source_points,
        destination_points
    )

    transformed = cv2.warpPerspective(
        image,
        matrix,
        (width, height)
    )

    return transformed


# ==========================================
# MATCHING FUNCTION
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
            None
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

    # Lowe ratio test
    good_matches = []

    for pair in matches:

        if len(pair) == 2:

            m, n = pair

            if m.distance < 0.75 * n.distance:

                good_matches.append(m)

    # RANSAC
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

    return (
        len(keypoints1),
        len(keypoints2),
        len(good_matches),
        inliers,
        rmse
    )


# ==========================================
# LOAD IMAGE
# ==========================================

original = cv2.imread(INPUT_IMAGE)

if original is None:

    print("ERROR: Could not load image.")

    exit()


print("Original image loaded!")
print("Original size:", original.shape)


# ==========================================
# CREATE PERSPECTIVE VERSION
# ==========================================

perspective = create_perspective_image(
    original
)

cv2.imwrite(
    "data/raw/image1_perspective.jpg",
    perspective
)

print(
    "Perspective image created!"
)


# ==========================================
# RUN MATCHING
# ==========================================

(
    features1,
    features2,
    good_matches,
    inliers,
    rmse
) = match_images(
    original,
    perspective
)


# ==========================================
# CALCULATE RATIO
# ==========================================

if good_matches > 0:

    ratio = (
        inliers /
        good_matches
    ) * 100

else:

    ratio = 0


# ==========================================
# RESULTS
# ==========================================

print("\n==========================================")
print("SIH26166 PERSPECTIVE TEST")
print("==========================================")

print(
    "Original features:",
    features1
)

print(
    "Perspective features:",
    features2
)

print(
    "Good matches:",
    good_matches
)

print(
    "RANSAC inliers:",
    inliers
)

print(
    f"Inlier ratio: {ratio:.2f}%"
)

if rmse is not None:

    print(
        f"RMSE: {rmse:.4f} pixels"
    )

else:

    print("RMSE: N/A")

print("==========================================")