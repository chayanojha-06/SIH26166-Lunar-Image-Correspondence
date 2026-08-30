import cv2
import numpy as np
import os


IMAGE1 = "data/raw/image1.jpg"
IMAGE2 = "data/raw/image2.jpg"

OUTPUT = "outputs/visualizations/fundamental_verified.jpg"

LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 2.0


# ============================================================
# LOAD
# ============================================================

image1 = cv2.imread(IMAGE1)
image2 = cv2.imread(IMAGE2)

if image1 is None or image2 is None:
    print("ERROR: Could not load images.")
    exit()

print("=" * 90)
print("SIH26166 FUNDAMENTAL MATRIX VISUALIZATION")
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

good = []

for pair in matches:

    if len(pair) != 2:
        continue

    m, n = pair

    if m.distance < LOWE_RATIO * n.distance:
        good.append(m)


print("\nFeatures:")
print("Reference:", len(kp1))
print("Target   :", len(kp2))

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
    RANSAC_THRESHOLD,
    0.99
)

if F is None or mask is None:
    print("ERROR: Fundamental matrix estimation failed.")
    exit()


mask = mask.ravel()

inlier_indices = np.where(mask == 1)[0]

inliers = len(inlier_indices)

ratio = (
    inliers /
    len(good)
) * 100


# ============================================================
# RMSE
# ============================================================

p1 = points1[mask == 1]
p2 = points2[mask == 1]

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


rmse = np.sqrt(
    np.mean(
        error1 ** 2 +
        error2 ** 2
    )
)


# ============================================================
# CREATE MATCH VISUALIZATION
# ============================================================

# Resize target so both images fit reasonably
max_height = 900

scale1 = max_height / image1.shape[0]
scale2 = max_height / image2.shape[0]

scale = min(
    scale1,
    scale2
)

display1 = cv2.resize(
    image1,
    None,
    fx=scale,
    fy=scale
)

display2 = cv2.resize(
    image2,
    None,
    fx=scale,
    fy=scale
)

h1, w1 = display1.shape[:2]
h2, w2 = display2.shape[:2]

canvas_height = max(
    h1,
    h2
)

canvas_width = w1 + w2

canvas = np.zeros(
    (
        canvas_height,
        canvas_width,
        3
    ),
    dtype=np.uint8
)

canvas[:h1, :w1] = display1
canvas[:h2, w1:w1 + w2] = display2


# ============================================================
# DRAW VERIFIED CORRESPONDENCES
# ============================================================

for idx in inlier_indices:

    x1, y1 = points1[idx]
    x2, y2 = points2[idx]

    x1 *= scale
    y1 *= scale

    x2 *= scale
    y2 *= scale

    pt1 = (
        int(x1),
        int(y1)
    )

    pt2 = (
        int(x2 + w1),
        int(y2)
    )

    cv2.circle(
        canvas,
        pt1,
        7,
        (0, 255, 0),
        -1
    )

    cv2.circle(
        canvas,
        pt2,
        7,
        (0, 255, 0),
        -1
    )

    cv2.line(
        canvas,
        pt1,
        pt2,
        (0, 255, 0),
        2
    )


# ============================================================
# LABELS
# ============================================================

cv2.putText(
    canvas,
    "REFERENCE",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (255, 255, 255),
    2
)

cv2.putText(
    canvas,
    "TARGET",
    (w1 + 20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (255, 255, 255),
    2
)

text = (
    f"Fundamental Matrix | "
    f"Inliers: {inliers}/{len(good)} | "
    f"Ratio: {ratio:.2f}%"
)

cv2.putText(
    canvas,
    text,
    (20, canvas_height - 45),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.75,
    (255, 255, 255),
    2
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT),
    exist_ok=True
)

cv2.imwrite(
    OUTPUT,
    canvas
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 90)
print("VERIFIED CORRESPONDENCE RESULTS")
print("=" * 90)

print(
    "\nGood matches:",
    len(good)
)

print(
    "RANSAC inliers:",
    inliers
)

print(
    f"Inlier ratio: {ratio:.2f}%"
)

print(
    f"RMSE: {rmse:.4f} pixels"
)

print(
    "\nVisualization saved to:"
)

print(
    OUTPUT
)

print("\n")
print("=" * 90)
print("FUNDAMENTAL VISUALIZATION COMPLETED")
print("=" * 90)