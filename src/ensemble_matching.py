import cv2
import numpy as np
import os


IMAGE1 = "data/raw/image1.jpg"
IMAGE2 = "data/raw/image2.jpg"

LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 5.0


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(image, method):

    if method == "Baseline":
        return image

    if method == "CLAHE":

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        return cv2.cvtColor(
            enhanced,
            cv2.COLOR_GRAY2BGR
        )

    if method == "Upscale":

        return cv2.resize(
            image,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC
        )

    if method == "Upscale + CLAHE":

        upscaled = cv2.resize(
            image,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(
            upscaled,
            cv2.COLOR_BGR2GRAY
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        return cv2.cvtColor(
            enhanced,
            cv2.COLOR_GRAY2BGR
        )

    return image


# ============================================================
# GET MATCHES
# ============================================================

def get_matches(image1, image2, method):

    processed1 = preprocess(
        image1,
        method
    )

    processed2 = preprocess(
        image2,
        method
    )

    gray1 = cv2.cvtColor(
        processed1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        processed2,
        cv2.COLOR_BGR2GRAY
    )

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

    if des1 is None or des2 is None:
        return [], 0, 0

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

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    # Convert coordinates back to original scale
    scale = 2.0 if "Upscale" in method else 1.0

    correspondences = []

    for m in good:

        p1 = np.array(
            kp1[m.queryIdx].pt,
            dtype=np.float32
        ) / scale

        p2 = np.array(
            kp2[m.trainIdx].pt,
            dtype=np.float32
        ) / scale

        correspondences.append(
            (
                p1,
                p2,
                float(m.distance)
            )
        )

    return (
        correspondences,
        len(kp1),
        len(kp2)
    )


# ============================================================
# LOAD
# ============================================================

image1 = cv2.imread(IMAGE1)
image2 = cv2.imread(IMAGE2)

if image1 is None or image2 is None:

    print("ERROR: Images could not be loaded.")
    exit()


print("=" * 90)
print("SIH26166 ENSEMBLE FEATURE MATCHING")
print("=" * 90)

print("\nReference:", image1.shape)
print("Target   :", image2.shape)


# ============================================================
# RUN HYPOTHESES
# ============================================================

methods = [
    "Baseline",
    "CLAHE",
    "Upscale",
    "Upscale + CLAHE"
]

all_correspondences = []

print("\n")
print("=" * 90)
print("COLLECTING CORRESPONDENCES")
print("=" * 90)

for method in methods:

    print(
        "\nRunning:",
        method
    )

    matches, f1, f2 = get_matches(
        image1,
        image2,
        method
    )

    print(
        "Features:",
        f1,
        "/",
        f2
    )

    print(
        "Good matches:",
        len(matches)
    )

    for p1, p2, distance in matches:

        all_correspondences.append(
            (
                tuple(
                    np.round(p1, 1)
                ),
                tuple(
                    np.round(p2, 1)
                ),
                distance,
                method
            )
        )


# ============================================================
# DEDUPLICATE
# ============================================================

unique = {}

for p1, p2, distance, method in all_correspondences:

    key = (
        p1,
        p2
    )

    if key not in unique:

        unique[key] = (
            p1,
            p2,
            distance,
            method
        )

    elif distance < unique[key][2]:

        unique[key] = (
            p1,
            p2,
            distance,
            method
        )


ensemble = list(
    unique.values()
)


print("\n")
print("=" * 90)

print(
    "Total collected:",
    len(all_correspondences)
)

print(
    "Unique correspondences:",
    len(ensemble)
)


# ============================================================
# RANSAC
# ============================================================

if len(ensemble) < 4:

    print("\nNot enough correspondences.")
    exit()


points1 = np.float32([
    x[0]
    for x in ensemble
])

points2 = np.float32([
    x[1]
    for x in ensemble
])


H, mask = cv2.findHomography(
    points1,
    points2,
    cv2.RANSAC,
    RANSAC_THRESHOLD
)


if H is None or mask is None:

    print("\nRANSAC failed.")
    exit()


mask = mask.ravel()

inlier_points1 = points1[
    mask == 1
]

inlier_points2 = points2[
    mask == 1
]

inliers = len(
    inlier_points1
)

ratio = (
    inliers /
    len(ensemble)
) * 100


# ============================================================
# RMSE
# ============================================================

projected = cv2.perspectiveTransform(
    inlier_points1.reshape(-1, 1, 2),
    H
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


# ============================================================
# RESULT
# ============================================================

print("\n")
print("=" * 90)
print("ENSEMBLE RANSAC RESULTS")
print("=" * 90)

print(
    "\nTotal correspondences:",
    len(ensemble)
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


# ============================================================
# INLIER SOURCES
# ============================================================

source_counts = {}

for i, item in enumerate(ensemble):

    if mask[i]:

        method = item[3]

        source_counts[method] = (
            source_counts.get(
                method,
                0
            ) + 1
        )


print("\nInliers by hypothesis:")

for method, count in source_counts.items():

    print(
        f"{method:<22}: {count}"
    )


# ============================================================
# VISUALIZATION
# ============================================================

inlier_matches = []

# Build synthetic keypoints
for p1, p2, _, _ in ensemble:

    pass


canvas1 = image1.copy()
canvas2 = image2.copy()

h1, w1 = canvas1.shape[:2]
h2, w2 = canvas2.shape[:2]

scale_visual = h1 / h2

canvas2 = cv2.resize(
    canvas2,
    None,
    fx=scale_visual,
    fy=scale_visual
)

canvas = np.hstack(
    (
        canvas1,
        canvas2
    )
)

for i in range(
    len(ensemble)
):

    if mask[i] == 0:
        continue

    x1, y1 = ensemble[i][0]

    x2, y2 = ensemble[i][1]

    x2 *= scale_visual
    y2 *= scale_visual

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
        5,
        (0, 255, 0),
        -1
    )

    cv2.circle(
        canvas,
        pt2,
        5,
        (0, 255, 0),
        -1
    )

    cv2.line(
        canvas,
        pt1,
        pt2,
        (0, 255, 0),
        1
    )


output_dir = "outputs/visualizations"

os.makedirs(
    output_dir,
    exist_ok=True
)

output_path = os.path.join(
    output_dir,
    "ensemble_correspondence.jpg"
)

cv2.imwrite(
    output_path,
    canvas
)

print(
    "\nVisualization saved to:",
    output_path
)

print("\n")
print("=" * 90)
print("ENSEMBLE MATCHING COMPLETED")
print("=" * 90)