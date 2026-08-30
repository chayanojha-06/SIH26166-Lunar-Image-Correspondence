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

    img1 = preprocess(
        image1,
        "Baseline"
    )

    img2 = preprocess(
        image2,
        method
    )

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
# CONFIDENCE SCORE
# ==========================================

def calculate_confidence(
    good_matches,
    inliers,
    inlier_ratio,
    rmse
):

    if good_matches == 0:
        return 0.0

    # Match quantity score
    match_score = min(
        good_matches / 100,
        1.0
    )

    # Inlier count score
    inlier_score = min(
        inliers / 50,
        1.0
    )

    # Geometric consistency
    ratio_score = min(
        inlier_ratio / 100,
        1.0
    )

    # RMSE score
    if rmse is None:

        rmse_score = 0.0

    else:

        rmse_score = max(
            0.0,
            1.0 - (rmse / 5.0)
        )

    # Weighted score
    confidence = (
        0.15 * match_score +
        0.40 * inlier_score +
        0.30 * ratio_score +
        0.15 * rmse_score
    )

    return confidence * 100


# ==========================================
# DECISION
# ==========================================

def get_decision(confidence, inliers):

    if inliers < 4:

        return "INSUFFICIENT GEOMETRIC EVIDENCE"

    if confidence >= 75:

        return "RELIABLE CORRESPONDENCE"

    if confidence >= 50:

        return "MODERATE CONFIDENCE"

    return "LOW CONFIDENCE"


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


print("=" * 65)
print("SIH26166 CORRESPONDENCE CONFIDENCE")
print("=" * 65)


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


for method in methods:

    result = run_matching(
        original,
        test_image,
        method
    )

    features, good, inliers, ratio, rmse = result

    confidence = calculate_confidence(
        good,
        inliers,
        ratio,
        rmse
    )

    results.append(
        (
            method,
            features,
            good,
            inliers,
            ratio,
            rmse,
            confidence
        )
    )


# ==========================================
# SELECT BEST
# ==========================================

best = max(
    results,
    key=lambda x: x[6]
)


(
    method,
    features,
    good,
    inliers,
    ratio,
    rmse,
    confidence
) = best


decision = get_decision(
    confidence,
    inliers
)


# ==========================================
# DISPLAY
# ==========================================

print()

print("Best Method       :", method)
print("Features          :", features)
print("Good Matches      :", good)
print("RANSAC Inliers    :", inliers)
print(f"Inlier Ratio      : {ratio:.2f}%")

if rmse is not None:
    print(f"RMSE              : {rmse:.4f} px")
else:
    print("RMSE              : N/A")

print()
print(f"Confidence Score  : {confidence:.2f}%")
print("Decision           :", decision)

print()
print("=" * 65)
print("CONFIDENCE ANALYSIS COMPLETED")
print("=" * 65)