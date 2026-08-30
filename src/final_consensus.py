import cv2
import numpy as np
import os


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE1 = "data/raw/image1.jpg"
IMAGE2 = "data/raw/image2.jpg"

LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 2.0


# ============================================================
# LOAD IMAGES
# ============================================================

image1 = cv2.imread(IMAGE1)
image2 = cv2.imread(IMAGE2)

if image1 is None or image2 is None:
    print("ERROR: Could not load images.")
    exit()

print("=" * 90)
print("SIH26166 FINAL MULTI-SIGNAL CONSENSUS")
print("=" * 90)

print("\nImages loaded successfully.")
print("Reference:", image1.shape)
print("Target   :", image2.shape)


# ============================================================
# SIFT FEATURE EXTRACTION
# ============================================================

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create(
    nfeatures=10000,
    contrastThreshold=0.03
)

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print("\nFeatures:")
print("Reference:", len(kp1))
print("Target   :", len(kp2))


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

good_matches = []

for pair in matches:

    if len(pair) != 2:
        continue

    m, n = pair

    if m.distance < LOWE_RATIO * n.distance:
        good_matches.append(m)


print("\nGood matches:", len(good_matches))


if len(good_matches) < 8:
    print("Not enough matches for final consensus.")
    exit()


points1 = np.float32([
    kp1[m.queryIdx].pt
    for m in good_matches
])

points2 = np.float32([
    kp2[m.trainIdx].pt
    for m in good_matches
])


# ============================================================
# HOMOGRAPHY
# ============================================================

H, h_mask = cv2.findHomography(
    points1,
    points2,
    cv2.RANSAC,
    5.0
)

if H is not None and h_mask is not None:

    h_mask = h_mask.ravel()

    homography_inliers = int(
        np.sum(h_mask)
    )

else:

    h_mask = np.zeros(
        len(good_matches),
        dtype=np.uint8
    )

    homography_inliers = 0


homography_ratio = (
    homography_inliers /
    len(good_matches)
) * 100


# ============================================================
# FUNDAMENTAL MATRIX
# ============================================================

F, f_mask = cv2.findFundamentalMat(
    points1,
    points2,
    cv2.FM_RANSAC,
    RANSAC_THRESHOLD,
    0.99
)

if F is not None and f_mask is not None:

    f_mask = f_mask.ravel()

    fundamental_inliers = int(
        np.sum(f_mask)
    )

else:

    f_mask = np.zeros(
        len(good_matches),
        dtype=np.uint8
    )

    fundamental_inliers = 0


fundamental_ratio = (
    fundamental_inliers /
    len(good_matches)
) * 100


# ============================================================
# FUNDAMENTAL RMSE
# ============================================================

fundamental_rmse = None

if fundamental_inliers >= 8 and F is not None:

    p1 = points1[f_mask == 1]
    p2 = points2[f_mask == 1]

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
            np.ones((len(p1), 1))
        )
    )

    pts2_h = np.hstack(
        (
            p2,
            np.ones((len(p2), 1))
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

    fundamental_rmse = np.sqrt(
        np.mean(
            error1 ** 2 +
            error2 ** 2
        )
    )


# ============================================================
# HOMOGRAPHY RMSE
# ============================================================

homography_rmse = None

if homography_inliers >= 4 and H is not None:

    p1 = points1[h_mask == 1]
    p2 = points2[h_mask == 1]

    projected = cv2.perspectiveTransform(
        p1.reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)

    error = projected - p2

    homography_rmse = np.sqrt(
        np.mean(
            np.sum(
                error ** 2,
                axis=1
            )
        )
    )


# ============================================================
# SPATIAL COVERAGE
# ============================================================

def calculate_spatial_coverage(points, width, height):

    if len(points) == 0:
        return 0.0

    grid_x = 4
    grid_y = 4

    occupied = set()

    for x, y in points:

        gx = int(
            min(
                grid_x - 1,
                max(
                    0,
                    x / width * grid_x
                )
            )
        )

        gy = int(
            min(
                grid_y - 1,
                max(
                    0,
                    y / height * grid_y
                )
            )
        )

        occupied.add(
            (gx, gy)
        )

    return (
        len(occupied) /
        (grid_x * grid_y)
    ) * 100


target_height, target_width = image2.shape[:2]

fundamental_points = points2[
    f_mask == 1
]

spatial_coverage = calculate_spatial_coverage(
    fundamental_points,
    target_width,
    target_height
)


# ============================================================
# CONSENSUS
# ============================================================

# Fundamental support
fundamental_score = min(
    fundamental_ratio,
    100
)

# Homography support
homography_score = min(
    homography_ratio,
    100
)

# Spatial support
spatial_score = spatial_coverage


# Agreement between the two geometric models
if (
    fundamental_inliers > 0
    and homography_inliers > 0
):

    overlap = np.sum(
        (
            f_mask == 1
        ) &
        (
            h_mask == 1
        )
    )

    union = np.sum(
        (
            f_mask == 1
        ) |
        (
            h_mask == 1
        )
    )

    if union > 0:

        model_agreement = (
            overlap /
            union
        ) * 100

    else:

        model_agreement = 0

else:

    model_agreement = 0


# ============================================================
# FINAL SCORE
# ============================================================

confidence = (
    0.40 * fundamental_score +
    0.25 * homography_score +
    0.20 * spatial_score +
    0.15 * model_agreement
)

confidence = min(
    100,
    max(
        0,
        confidence
    )
)


# ============================================================
# DECISION
# ============================================================

if confidence >= 70:

    decision = "HIGH CONFIDENCE"

elif confidence >= 45:

    decision = "MEDIUM CONFIDENCE"

else:

    decision = "LOW CONFIDENCE"


# ============================================================
# REPORT
# ============================================================

print("\n")
print("=" * 90)
print("GEOMETRIC CONSENSUS")
print("=" * 90)

print(
    f"\nHomography inliers : {homography_inliers}"
)

print(
    f"Homography ratio   : {homography_ratio:.2f}%"
)

if homography_rmse is not None:

    print(
        f"Homography RMSE    : {homography_rmse:.4f} px"
    )

print(
    f"\nFundamental inliers: {fundamental_inliers}"
)

print(
    f"Fundamental ratio  : {fundamental_ratio:.2f}%"
)

if fundamental_rmse is not None:

    print(
        f"Fundamental RMSE   : {fundamental_rmse:.4f} px"
    )

print(
    f"\nSpatial coverage   : {spatial_coverage:.2f}%"
)

print(
    f"Model agreement    : {model_agreement:.2f}%"
)


print("\n")
print("=" * 90)
print("FINAL CONSENSUS SCORE")
print("=" * 90)

print(
    f"\nConfidence: {confidence:.2f}%"
)

print(
    f"Decision  : {decision}"
)


# ============================================================
# SAVE REPORT
# ============================================================

os.makedirs(
    "outputs/reports",
    exist_ok=True
)

report_path = (
    "outputs/reports/"
    "final_consensus.txt"
)

with open(
    report_path,
    "w"
) as f:

    f.write(
        "SIH26166 FINAL CORRESPONDENCE REPORT\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Good matches: {len(good_matches)}\n"
    )

    f.write(
        f"Homography inliers: "
        f"{homography_inliers}\n"
    )

    f.write(
        f"Homography ratio: "
        f"{homography_ratio:.2f}%\n"
    )

    f.write(
        f"Fundamental inliers: "
        f"{fundamental_inliers}\n"
    )

    f.write(
        f"Fundamental ratio: "
        f"{fundamental_ratio:.2f}%\n"
    )

    f.write(
        f"Spatial coverage: "
        f"{spatial_coverage:.2f}%\n"
    )

    f.write(
        f"Model agreement: "
        f"{model_agreement:.2f}%\n"
    )

    f.write(
        f"Confidence: "
        f"{confidence:.2f}%\n"
    )

    f.write(
        f"Decision: "
        f"{decision}\n"
    )


print(
    "\nReport saved to:",
    report_path
)


print("\n")
print("=" * 90)
print("FINAL CONSENSUS COMPLETED")
print("=" * 90)