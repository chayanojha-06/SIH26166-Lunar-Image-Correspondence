import cv2
import numpy as np
import os


# ============================================================
# SIH26166 SPATIAL-AWARE GEOMETRIC VERIFICATION
# ============================================================

ORIGINAL_IMAGE = "data/raw/image1.jpg"
TARGET_IMAGE = "data/raw/image2.jpg"

RANSAC_THRESHOLD = 5.0
LOWE_RATIO = 0.75


# ============================================================
# LOAD
# ============================================================

image1 = cv2.imread(ORIGINAL_IMAGE)
image2 = cv2.imread(TARGET_IMAGE)

if image1 is None or image2 is None:
    print("ERROR: Could not load images.")
    exit()

print("=" * 90)
print("SIH26166 SPATIAL-AWARE GEOMETRIC VERIFICATION")
print("=" * 90)

print("\nImages loaded successfully.")
print("Reference:", image1.shape)
print("Target   :", image2.shape)


# ============================================================
# GRAYSCALE
# ============================================================

gray1 = cv2.cvtColor(
    image1,
    cv2.COLOR_BGR2GRAY
)

gray2 = cv2.cvtColor(
    image2,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# SIFT
# ============================================================

sift = cv2.SIFT_create(
    nfeatures=10000,
    contrastThreshold=0.03
)

kp1, des1 = sift.detectAndCompute(
    gray1,
    None
)

kp2, des2 = sift.detectAndCompute(
    gray2,
    None
)

print("\nFeatures:")
print("Reference:", len(kp1))
print("Target   :", len(kp2))


# ============================================================
# FLANN MATCHING
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
# LOWE RATIO TEST
# ============================================================

good_matches = []

for pair in matches:

    if len(pair) != 2:
        continue

    m, n = pair

    if m.distance < LOWE_RATIO * n.distance:

        good_matches.append(m)


print("\nGood matches:", len(good_matches))


# ============================================================
# RANSAC
# ============================================================

if len(good_matches) < 4:

    print("\nNot enough matches for RANSAC.")
    exit()


points1 = np.float32([
    kp1[m.queryIdx].pt
    for m in good_matches
])

points2 = np.float32([
    kp2[m.trainIdx].pt
    for m in good_matches
])


H, mask = cv2.findHomography(
    points1,
    points2,
    cv2.RANSAC,
    RANSAC_THRESHOLD
)


if H is None or mask is None:

    print("\nHomography estimation failed.")
    exit()


mask = mask.ravel()

inlier_points1 = points1[mask == 1]
inlier_points2 = points2[mask == 1]

inlier_count = len(inlier_points1)

inlier_ratio = (
    inlier_count /
    len(good_matches)
) * 100


# ============================================================
# RMSE
# ============================================================

projected = cv2.perspectiveTransform(
    inlier_points1.reshape(-1, 1, 2),
    H
).reshape(-1, 2)

errors = projected - inlier_points2

rmse = np.sqrt(
    np.mean(
        np.sum(
            errors ** 2,
            axis=1
        )
    )
)


# ============================================================
# SPATIAL COVERAGE
# ============================================================

height, width = gray1.shape

grid_rows = 4
grid_cols = 4

occupied_cells = set()

for point in inlier_points1:

    x, y = point

    col = int(
        x / width * grid_cols
    )

    row = int(
        y / height * grid_rows
    )

    col = min(
        max(col, 0),
        grid_cols - 1
    )

    row = min(
        max(row, 0),
        grid_rows - 1
    )

    occupied_cells.add(
        (row, col)
    )


total_cells = (
    grid_rows *
    grid_cols
)

spatial_coverage = (
    len(occupied_cells) /
    total_cells
) * 100


# ============================================================
# CONVEX HULL COVERAGE
# ============================================================

hull_ratio = 0.0

if len(inlier_points1) >= 3:

    hull = cv2.convexHull(
        inlier_points1
    )

    hull_area = cv2.contourArea(
        hull
    )

    image_area = (
        width *
        height
    )

    hull_ratio = (
        hull_area /
        image_area
    ) * 100


# ============================================================
# QUALITY SCORE
# ============================================================

inlier_score = min(
    inlier_count / 20.0,
    1.0
)

ratio_score = min(
    inlier_ratio / 100.0,
    1.0
)

rmse_score = max(
    0.0,
    1.0 - rmse / 5.0
)

spatial_score = min(
    spatial_coverage / 50.0,
    1.0
)

hull_score = min(
    hull_ratio / 30.0,
    1.0
)


quality_score = (
    0.30 * inlier_score +
    0.25 * ratio_score +
    0.15 * rmse_score +
    0.20 * spatial_score +
    0.10 * hull_score
) * 100


# ============================================================
# DECISION
# ============================================================

if (
    inlier_count >= 15
    and inlier_ratio >= 50
    and rmse < 2.0
    and spatial_coverage >= 25
):

    decision = "HIGH CONFIDENCE"

elif (
    inlier_count >= 8
    and inlier_ratio >= 30
    and rmse < 3.0
):

    decision = "MODERATE CONFIDENCE"

else:

    decision = "LOW CONFIDENCE"


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 90)
print("GEOMETRIC VERIFICATION RESULTS")
print("=" * 90)

print(
    "\nGood matches:",
    len(good_matches)
)

print(
    "RANSAC inliers:",
    inlier_count
)

print(
    f"Inlier ratio: {inlier_ratio:.2f}%"
)

print(
    f"RMSE: {rmse:.4f} pixels"
)

print(
    f"Spatial coverage: {spatial_coverage:.2f}%"
)

print(
    f"Convex hull coverage: {hull_ratio:.2f}%"
)

print(
    f"\nQuality score: {quality_score:.2f}%"
)

print(
    "Decision:",
    decision
)


# ============================================================
# VISUALIZATION
# ============================================================

inlier_matches = [
    good_matches[i]
    for i in range(
        len(good_matches)
    )
    if mask[i]
]

visualization = cv2.drawMatches(
    image1,
    kp1,
    image2,
    kp2,
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


output_dir = os.path.join(
    "outputs",
    "visualizations"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

output_path = os.path.join(
    output_dir,
    "spatial_verification.jpg"
)

cv2.imwrite(
    output_path,
    visualization
)

print(
    "\nVisualization saved to:",
    output_path
)

print("\n")
print("=" * 90)
print("SPATIAL VERIFICATION COMPLETED")
print("=" * 90)