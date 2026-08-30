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
print("SIH26166 ROBUST LOCALIZATION")
print("=" * 90)

print("\nImages loaded successfully.")
print("Reference:", image1.shape)
print("Target   :", image2.shape)


# ============================================================
# SIFT
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

if len(good_matches) < 8:
    print("ERROR: Not enough matches.")
    exit()


# ============================================================
# MATCH POINTS
# ============================================================

points1 = np.float32([
    kp1[m.queryIdx].pt
    for m in good_matches
])

points2 = np.float32([
    kp2[m.trainIdx].pt
    for m in good_matches
])


# ============================================================
# FUNDAMENTAL MATRIX RANSAC
# ============================================================

F, mask = cv2.findFundamentalMat(
    points1,
    points2,
    cv2.FM_RANSAC,
    RANSAC_THRESHOLD,
    0.99
)

if F is None or mask is None:
    print("ERROR: Fundamental matrix estimation failed.")
    exit()

mask = mask.ravel()

inlier_points = points1[mask == 1]
inlier_target = points2[mask == 1]

print("\nRANSAC inliers:", len(inlier_points))


# ============================================================
# ROBUST SPATIAL CLUSTERING
# ============================================================

# We use DBSCAN to identify spatially coherent groups.
# This prevents isolated matches from defining the region.

from sklearn.cluster import DBSCAN


if len(inlier_points) >= 3:

    clustering = DBSCAN(
        eps=300,
        min_samples=2
    ).fit(inlier_points)

    labels = clustering.labels_

    valid_labels = [
        label
        for label in set(labels)
        if label != -1
    ]

else:

    labels = np.full(
        len(inlier_points),
        -1
    )

    valid_labels = []


# ============================================================
# FIND LARGEST CLUSTER
# ============================================================

if len(valid_labels) > 0:

    cluster_sizes = {}

    for label in valid_labels:

        cluster_sizes[label] = np.sum(
            labels == label
        )

    best_cluster = max(
        cluster_sizes,
        key=cluster_sizes.get
    )

    cluster_points = inlier_points[
        labels == best_cluster
    ]

    cluster_target = inlier_target[
        labels == best_cluster
    ]

else:

    best_cluster = None

    cluster_points = inlier_points

    cluster_target = inlier_target


cluster_count = len(cluster_points)


# ============================================================
# CLUSTER STATISTICS
# ============================================================

x = cluster_points[:, 0]
y = cluster_points[:, 1]

min_x = float(np.min(x))
max_x = float(np.max(x))

min_y = float(np.min(y))
max_y = float(np.max(y))

center_x = float(np.mean(x))
center_y = float(np.mean(y))

box_width = max_x - min_x
box_height = max_y - min_y

image_height, image_width = image1.shape[:2]

image_area = image_width * image_height

box_area = box_width * box_height

coverage = (
    box_area /
    image_area
) * 100

normalized_x = (
    center_x /
    image_width
) * 100

normalized_y = (
    center_y /
    image_height
) * 100


# ============================================================
# CLUSTER RATIO
# ============================================================

cluster_ratio = (
    cluster_count /
    len(inlier_points)
) * 100


# ============================================================
# LOCALIZATION CONFIDENCE
# ============================================================

# More points in the same cluster = stronger localization.
#
# This is NOT the overall correspondence confidence.
# It only measures spatial concentration.

if cluster_count >= 10:
    localization_confidence = 90.0

elif cluster_count >= 8:
    localization_confidence = 75.0

elif cluster_count >= 6:
    localization_confidence = 60.0

elif cluster_count >= 4:
    localization_confidence = 40.0

else:
    localization_confidence = 20.0


# ============================================================
# VISUALIZATION
# ============================================================

visual = image1.copy()


# Draw all RANSAC inliers
for px, py in inlier_points:

    cv2.circle(
        visual,
        (int(px), int(py)),
        8,
        (255, 0, 0),
        -1
    )


# Draw selected cluster
for px, py in cluster_points:

    cv2.circle(
        visual,
        (int(px), int(py)),
        14,
        (0, 255, 0),
        -1
    )


# Bounding box
cv2.rectangle(
    visual,
    (int(min_x), int(min_y)),
    (int(max_x), int(max_y)),
    (0, 255, 255),
    5
)


# Centroid
cv2.drawMarker(
    visual,
    (int(center_x), int(center_y)),
    (0, 0, 255),
    cv2.MARKER_CROSS,
    50,
    5
)


# ============================================================
# TEXT
# ============================================================

cv2.putText(
    visual,
    "ROBUST MATCH LOCALIZATION",
    (40, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.3,
    (255, 255, 255),
    3
)

cv2.putText(
    visual,
    f"RANSAC inliers: {len(inlier_points)}",
    (40, 105),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (255, 255, 255),
    2
)

cv2.putText(
    visual,
    f"Main cluster: {cluster_count}",
    (40, 145),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (255, 255, 255),
    2
)

cv2.putText(
    visual,
    f"Localization: {localization_confidence:.1f}%",
    (40, 185),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (255, 255, 255),
    2
)


# ============================================================
# SAVE VISUALIZATION
# ============================================================

os.makedirs(
    "outputs/visualizations",
    exist_ok=True
)

os.makedirs(
    "outputs/reports",
    exist_ok=True
)

OUTPUT = (
    "outputs/visualizations/"
    "robust_localization.jpg"
)

REPORT = (
    "outputs/reports/"
    "robust_localization.txt"
)

cv2.imwrite(
    OUTPUT,
    visual
)


# ============================================================
# SAVE REPORT
# ============================================================

with open(REPORT, "w") as f:

    f.write(
        "SIH26166 ROBUST LOCALIZATION REPORT\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Good matches: {len(good_matches)}\n"
    )

    f.write(
        f"RANSAC inliers: {len(inlier_points)}\n"
    )

    f.write(
        f"Main cluster points: {cluster_count}\n"
    )

    f.write(
        f"Cluster ratio: {cluster_ratio:.2f}%\n\n"
    )

    f.write(
        f"Centroid X: {center_x:.2f} px\n"
    )

    f.write(
        f"Centroid Y: {center_y:.2f} px\n"
    )

    f.write(
        f"Normalized X: {normalized_x:.2f}%\n"
    )

    f.write(
        f"Normalized Y: {normalized_y:.2f}%\n\n"
    )

    f.write(
        f"Bounding box width: {box_width:.2f} px\n"
    )

    f.write(
        f"Bounding box height: {box_height:.2f} px\n"
    )

    f.write(
        f"Region coverage: {coverage:.4f}%\n\n"
    )

    f.write(
        f"Localization confidence: "
        f"{localization_confidence:.2f}%\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 90)
print("ROBUST LOCALIZATION RESULTS")
print("=" * 90)

print(
    f"\nRANSAC inliers       : "
    f"{len(inlier_points)}"
)

print(
    f"Main cluster points  : "
    f"{cluster_count}"
)

print(
    f"Cluster ratio       : "
    f"{cluster_ratio:.2f}%"
)

print(
    f"\nCentroid             : "
    f"({center_x:.2f}, {center_y:.2f}) px"
)

print(
    f"Normalized location : "
    f"({normalized_x:.2f}%, {normalized_y:.2f}%)"
)

print(
    f"\nBounding box         : "
    f"{box_width:.2f} x {box_height:.2f} px"
)

print(
    f"Region coverage      : "
    f"{coverage:.4f}%"
)

print(
    f"\nLocalization confidence: "
    f"{localization_confidence:.2f}%"
)

print(
    "\nVisualization saved to:"
)

print(OUTPUT)

print(
    "\nReport saved to:"
)

print(REPORT)

print("\n")
print("=" * 90)
print("ROBUST LOCALIZATION COMPLETED")
print("=" * 90)