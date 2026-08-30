import cv2
import numpy as np
import os


IMAGE1 = "data/raw/image1.jpg"
IMAGE2 = "data/raw/image2.jpg"

OUTPUT = "outputs/visualizations/localization.jpg"
REPORT = "outputs/reports/localization.txt"


# ============================================================
# LOAD
# ============================================================

image1 = cv2.imread(IMAGE1)
image2 = cv2.imread(IMAGE2)

if image1 is None or image2 is None:
    print("ERROR: Could not load images.")
    exit()

print("=" * 90)
print("SIH26166 MATCHED REGION LOCALIZATION")
print("=" * 90)


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
# FLANN + LOWE
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

good = []

for pair in matches:

    if len(pair) == 2:

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

print("\nGood matches:", len(good))

if len(good) < 8:
    print("ERROR: Not enough matches.")
    exit()


# ============================================================
# POINTS
# ============================================================

points1 = np.float32([
    kp1[m.queryIdx].pt
    for m in good
])

points2 = np.float32([
    kp2[m.trainIdx].pt
    for m in good
])


# ============================================================
# FUNDAMENTAL MATRIX
# ============================================================

F, mask = cv2.findFundamentalMat(
    points1,
    points2,
    cv2.FM_RANSAC,
    2.0,
    0.99
)

if F is None or mask is None:
    print("ERROR: Fundamental matrix failed.")
    exit()

mask = mask.ravel()

inlier_points = points1[mask == 1]

inlier_target = points2[mask == 1]

inliers = len(inlier_points)

ratio = (
    inliers /
    len(good)
) * 100


# ============================================================
# LOCALIZATION
# ============================================================

x = inlier_points[:, 0]
y = inlier_points[:, 1]

min_x = float(np.min(x))
max_x = float(np.max(x))
min_y = float(np.min(y))
max_y = float(np.max(y))

center_x = float(np.mean(x))
center_y = float(np.mean(y))

image_height, image_width = image1.shape[:2]

normalized_x = (
    center_x /
    image_width
) * 100

normalized_y = (
    center_y /
    image_height
) * 100


# ============================================================
# BOUNDING BOX AREA
# ============================================================

box_width = max_x - min_x
box_height = max_y - min_y

box_area = box_width * box_height

image_area = image_width * image_height

coverage = (
    box_area /
    image_area
) * 100


# ============================================================
# TARGET CENTROID
# ============================================================

target_center_x = float(
    np.mean(
        inlier_target[:, 0]
    )
)

target_center_y = float(
    np.mean(
        inlier_target[:, 1]
    )
)


# ============================================================
# VISUALIZATION
# ============================================================

visual = image1.copy()

# Draw every verified point
for px, py in inlier_points:

    cv2.circle(
        visual,
        (int(px), int(py)),
        10,
        (0, 255, 0),
        -1
    )

# Bounding box
cv2.rectangle(
    visual,
    (
        int(min_x),
        int(min_y)
    ),
    (
        int(max_x),
        int(max_y)
    ),
    (0, 255, 255),
    5
)

# Centroid
cv2.drawMarker(
    visual,
    (
        int(center_x),
        int(center_y)
    ),
    (0, 0, 255),
    cv2.MARKER_CROSS,
    40,
    5
)


# ============================================================
# LABELS
# ============================================================

cv2.putText(
    visual,
    "LOCALIZED MATCH REGION",
    (40, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.4,
    (255, 255, 255),
    3
)

text1 = f"Verified inliers: {inliers}"

text2 = (
    f"Location: "
    f"({normalized_x:.1f}%, {normalized_y:.1f}%)"
)

cv2.putText(
    visual,
    text1,
    (40, 110),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (255, 255, 255),
    2
)

cv2.putText(
    visual,
    text2,
    (40, 150),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (255, 255, 255),
    2
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "outputs/visualizations",
    exist_ok=True
)

os.makedirs(
    "outputs/reports",
    exist_ok=True
)

cv2.imwrite(
    OUTPUT,
    visual
)


# ============================================================
# REPORT
# ============================================================

with open(
    REPORT,
    "w"
) as f:

    f.write(
        "SIH26166 MATCHED REGION LOCALIZATION\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Good matches: {len(good)}\n"
    )

    f.write(
        f"RANSAC inliers: {inliers}\n"
    )

    f.write(
        f"Inlier ratio: {ratio:.2f}%\n\n"
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
        f"Reference image coverage: {coverage:.4f}%\n"
    )

    f.write(
        f"\nVisualization: {OUTPUT}\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 90)
print("LOCALIZATION RESULTS")
print("=" * 90)

print(
    f"\nVerified inliers : {inliers}"
)

print(
    f"Inlier ratio     : {ratio:.2f}%"
)

print(
    f"\nCentroid         : "
    f"({center_x:.2f}, {center_y:.2f}) px"
)

print(
    f"Normalized       : "
    f"({normalized_x:.2f}%, {normalized_y:.2f}%)"
)

print(
    f"\nBounding box     : "
    f"{box_width:.2f} x {box_height:.2f} px"
)

print(
    f"Region coverage  : "
    f"{coverage:.4f}%"
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
print("LOCALIZATION COMPLETED")
print("=" * 90)