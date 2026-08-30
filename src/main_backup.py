import cv2
import numpy as np
import os
from datetime import datetime


# ============================================================
# SIH26166
# FINAL ROBUST MULTI-SIGNAL LOCALIZATION PIPELINE
# ============================================================

REFERENCE_IMAGE = "data/raw/image1.jpg"
TARGET_IMAGE = "data/raw/image2.jpg"

VIS_DIR = "outputs/visualizations"
REPORT_DIR = "outputs/reports"

os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_FEATURES = 10000
LOWE_RATIO = 0.75

F_RANSAC_THRESHOLD = 2.0
H_RANSAC_THRESHOLD = 5.0

RANSAC_CONFIDENCE = 0.99
RANSAC_ITERATIONS = 3000

MIN_GOOD_MATCHES = 12
MIN_INLIERS = 8

GRID_ROWS = 4
GRID_COLS = 4


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("SIH26166 LUNAR IMAGE CORRESPONDENCE SYSTEM")
print("FINAL ROBUST MULTI-SIGNAL LOCALIZATION PIPELINE")
print("=" * 90)


# ============================================================
# LOAD IMAGES
# ============================================================

reference = cv2.imread(REFERENCE_IMAGE)
target = cv2.imread(TARGET_IMAGE)

if reference is None:
    raise FileNotFoundError(
        f"Reference image not found: {REFERENCE_IMAGE}"
    )

if target is None:
    raise FileNotFoundError(
        f"Target image not found: {TARGET_IMAGE}"
    )

print("\nImages loaded successfully.")
print("Reference:", reference.shape)
print("Target   :", target.shape)


# ============================================================
# PREPROCESSING
# ============================================================

def clahe_image(image):

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


def upscale_image(image):

    return cv2.resize(
        image,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )


def build_hypotheses(image):

    return {
        "Baseline": image.copy(),
        "CLAHE": clahe_image(image),
        "Upscale": upscale_image(image),
        "Upscale + CLAHE":
            clahe_image(
                upscale_image(image)
            )
    }


# ============================================================
# SIFT
# ============================================================

def extract_features(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    sift = cv2.SIFT_create(
        nfeatures=MAX_FEATURES,
        contrastThreshold=0.03
    )

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    return keypoints, descriptors


# ============================================================
# MATCHING
# ============================================================

def match_features(des1, des2):

    if des1 is None or des2 is None:
        return []

    matcher = cv2.FlannBasedMatcher(
        dict(
            algorithm=1,
            trees=5
        ),
        dict(
            checks=80
        )
    )

    raw_matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good = []

    for pair in raw_matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    return good


# ============================================================
# POINT EXTRACTION
# ============================================================

def get_points(kp1, kp2, matches):

    points1 = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ])

    points2 = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ])

    return points1, points2


# ============================================================
# FUNDAMENTAL MATRIX
# ============================================================

def fundamental_model(kp1, kp2, matches):

    if len(matches) < 8:

        return None, np.zeros(
            len(matches),
            dtype=np.uint8
        )

    p1, p2 = get_points(
        kp1,
        kp2,
        matches
    )

    F, mask = cv2.findFundamentalMat(
        p1,
        p2,
        cv2.FM_RANSAC,
        F_RANSAC_THRESHOLD,
        RANSAC_CONFIDENCE,
        RANSAC_ITERATIONS
    )

    if mask is None:

        return F, np.zeros(
            len(matches),
            dtype=np.uint8
        )

    mask = mask.ravel().astype(
        np.uint8
    )

    return F, mask


# ============================================================
# HOMOGRAPHY
# ============================================================

def homography_model(kp1, kp2, matches):

    if len(matches) < 4:

        return None, np.zeros(
            len(matches),
            dtype=np.uint8
        )

    p1, p2 = get_points(
        kp1,
        kp2,
        matches
    )

    H, mask = cv2.findHomography(
        p1,
        p2,
        cv2.RANSAC,
        H_RANSAC_THRESHOLD,
        maxIters=RANSAC_ITERATIONS,
        confidence=RANSAC_CONFIDENCE
    )

    if mask is None:

        return H, np.zeros(
            len(matches),
            dtype=np.uint8
        )

    return H, mask.ravel().astype(
        np.uint8
    )


# ============================================================
# REPROJECTION ERROR
# ============================================================

def homography_rmse(
    kp1,
    kp2,
    matches,
    mask,
    H
):

    if H is None:
        return None

    indices = np.where(mask == 1)[0]

    if len(indices) < 4:
        return None

    src = np.float32([
        kp1[matches[i].queryIdx].pt
        for i in indices
    ]).reshape(-1, 1, 2)

    dst = np.float32([
        kp2[matches[i].trainIdx].pt
        for i in indices
    ]).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(
        src,
        H
    )

    errors = (
        projected.reshape(-1, 2)
        - dst.reshape(-1, 2)
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    errors ** 2,
                    axis=1
                )
            )
        )
    )


# ============================================================
# SPATIAL ANALYSIS
# ============================================================

def spatial_analysis(
    kp1,
    matches,
    mask,
    image_shape
):

    indices = np.where(
        mask == 1
    )[0]

    if len(indices) == 0:

        return {
            "coverage": 0.0,
            "support": 0.0,
            "concentration": 0.0,
            "x": 0.0,
            "y": 0.0,
            "bbox": None,
            "grid": np.zeros(
                (GRID_ROWS, GRID_COLS),
                dtype=int
            )
        }

    points = np.float32([
        kp1[matches[i].queryIdx].pt
        for i in indices
    ])

    height, width = image_shape[:2]

    grid = np.zeros(
        (GRID_ROWS, GRID_COLS),
        dtype=int
    )

    for x, y in points:

        col = min(
            GRID_COLS - 1,
            max(
                0,
                int(
                    x / width * GRID_COLS
                )
            )
        )

        row = min(
            GRID_ROWS - 1,
            max(
                0,
                int(
                    y / height * GRID_ROWS
                )
            )
        )

        grid[row, col] += 1

    occupied = np.sum(
        grid > 0
    )

    coverage = (
        occupied /
        (GRID_ROWS * GRID_COLS)
    ) * 100.0

    best_cell = np.unravel_index(
        np.argmax(grid),
        grid.shape
    )

    best_row, best_col = best_cell

    strongest = int(
        grid[
            best_row,
            best_col
        ]
    )

    neighbour_votes = 0

    for r in range(
        max(0, best_row - 1),
        min(GRID_ROWS, best_row + 2)
    ):

        for c in range(
            max(0, best_col - 1),
            min(GRID_COLS, best_col + 2)
        ):

            if (
                r == best_row
                and c == best_col
            ):
                continue

            neighbour_votes += int(
                grid[r, c]
            )

    support = (
        (strongest + neighbour_votes) /
        len(points)
    ) * 100.0

    concentration = (
        strongest /
        len(points)
    ) * 100.0

    x1 = points[:, 0].min()
    x2 = points[:, 0].max()
    y1 = points[:, 1].min()
    y2 = points[:, 1].max()

    centroid_x = float(
        np.mean(points[:, 0])
    )

    centroid_y = float(
        np.mean(points[:, 1])
    )

    return {
        "coverage": float(coverage),
        "support": float(
            min(100.0, support)
        ),
        "concentration": float(
            concentration
        ),
        "x": centroid_x,
        "y": centroid_y,
        "bbox": (
            float(x1),
            float(y1),
            float(x2),
            float(y2)
        ),
        "grid": grid
    }


# ============================================================
# MODEL QUALITY
# ============================================================

def calculate_quality(
    good,
    inliers,
    rmse,
    spatial,
    model_agreement
):

    if good == 0:
        return 0.0

    ratio = (
        inliers /
        good
    )

    # Match count signal
    match_signal = min(
        100.0,
        inliers * 8.0
    )

    # Geometric consistency
    ratio_signal = ratio * 100.0

    # Spatial distribution
    spatial_signal = (
        0.7 * spatial["coverage"] +
        0.3 * spatial["support"]
    )

    # RMSE signal
    if rmse is None:
        rmse_signal = 0.0

    elif rmse <= 1.0:
        rmse_signal = 100.0

    elif rmse <= 3.0:
        rmse_signal = 80.0

    elif rmse <= 5.0:
        rmse_signal = 60.0

    elif rmse <= 10.0:
        rmse_signal = 30.0

    else:
        rmse_signal = 0.0

    agreement_signal = (
        model_agreement * 100.0
    )

    score = (
        0.20 * match_signal +
        0.25 * ratio_signal +
        0.20 * spatial_signal +
        0.20 * rmse_signal +
        0.15 * agreement_signal
    )

    return float(
        min(
            100.0,
            max(0.0, score)
        )
    )


# ============================================================
# HYPOTHESES
# ============================================================

ref_hypotheses = build_hypotheses(
    reference
)

target_hypotheses = build_hypotheses(
    target
)


# ============================================================
# RUN
# ============================================================

print("\n")
print("=" * 90)
print("MULTI-HYPOTHESIS GEOMETRIC ANALYSIS")
print("=" * 90)

results = []


for name in ref_hypotheses:

    print("\nTesting:", name)

    ref_img = ref_hypotheses[name]
    target_img = target_hypotheses[name]

    kp1, des1 = extract_features(
        ref_img
    )

    kp2, des2 = extract_features(
        target_img
    )

    good = match_features(
        des1,
        des2
    )

    print(
        f"Features: {len(kp1)} / {len(kp2)}"
    )

    print(
        f"Good matches: {len(good)}"
    )

    if len(good) < MIN_GOOD_MATCHES:

        print("Not enough matches.")

        continue

    F, fmask = fundamental_model(
        kp1,
        kp2,
        good
    )

    H, hmask = homography_model(
        kp1,
        kp2,
        good
    )

    f_inliers = int(
        np.sum(fmask)
    )

    h_inliers = int(
        np.sum(hmask)
    )

    f_ratio = (
        f_inliers /
        len(good)
    ) * 100.0

    h_ratio = (
        h_inliers /
        len(good)
    ) * 100.0

    f_spatial = spatial_analysis(
        kp1,
        good,
        fmask,
        reference.shape
    )

    h_spatial = spatial_analysis(
        kp1,
        good,
        hmask,
        reference.shape
    )

    f_rmse = None

    # Fundamental matrix is not a direct
    # image-to-image transformation.
    # Therefore its residual is NOT used
    # as a fake pixel reprojection error.

    h_rmse = homography_rmse(
        kp1,
        kp2,
        good,
        hmask,
        H
    )

    # Compare models using normalized
    # agreement rather than raw equality.

    if max(
        f_inliers,
        h_inliers
    ) > 0:

        agreement = (
            min(
                f_inliers,
                h_inliers
            ) /
            max(
                f_inliers,
                h_inliers
            )
        )

    else:

        agreement = 0.0

    # Select the model.
    #
    # Fundamental matrix is preferred for
    # general viewpoint changes.
    #
    # Homography is preferred when it has
    # strong spatially consistent support.

    if (
        h_inliers >= MIN_INLIERS
        and h_rmse is not None
        and h_rmse <= 5.0
        and h_ratio >= 25.0
    ):

        selected_model = "Homography"
        selected_mask = hmask
        selected_inliers = h_inliers
        selected_ratio = h_ratio
        selected_rmse = h_rmse
        selected_spatial = h_spatial

    elif f_inliers >= MIN_INLIERS:

        selected_model = "Fundamental"
        selected_mask = fmask
        selected_inliers = f_inliers
        selected_ratio = f_ratio
        selected_rmse = None
        selected_spatial = f_spatial

    elif h_inliers >= MIN_INLIERS:

        selected_model = "Homography"
        selected_mask = hmask
        selected_inliers = h_inliers
        selected_ratio = h_ratio
        selected_rmse = h_rmse
        selected_spatial = h_spatial

    else:

        print("No reliable geometric model.")

        continue

    confidence = calculate_quality(
        len(good),
        selected_inliers,
        selected_rmse,
        selected_spatial,
        agreement
    )

    print(
        "Geometric model:",
        selected_model
    )

    print(
        "RANSAC inliers:",
        selected_inliers
    )

    print(
        "Inlier ratio:",
        f"{selected_ratio:.2f}%"
    )

    print(
        "RMSE:",
        "N/A"
        if selected_rmse is None
        else f"{selected_rmse:.4f} px"
    )

    print(
        "Spatial support:",
        f"{selected_spatial['support']:.2f}%"
    )

    print(
        "Model agreement:",
        f"{agreement * 100:.2f}%"
    )

    print(
        "Confidence:",
        f"{confidence:.2f}%"
    )

    results.append({
        "name": name,
        "model": selected_model,
        "features_ref": len(kp1),
        "features_target": len(kp2),
        "good": len(good),
        "inliers": selected_inliers,
        "ratio": selected_ratio,
        "rmse": selected_rmse,
        "spatial": selected_spatial,
        "agreement": agreement,
        "confidence": confidence,
        "kp1": kp1,
        "kp2": kp2,
        "matches": good,
        "mask": selected_mask
    })


# ============================================================
# NO RESULT
# ============================================================

if not results:

    print("\nNo reliable correspondence found.")

    report_path = (
        REPORT_DIR +
        "/final_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SIH26166 FINAL PIPELINE\n"
        )

        f.write(
            "RESULT: NO RELIABLE CORRESPONDENCE\n"
        )

    raise SystemExit


# ============================================================
# SELECT BEST
# ============================================================

best = max(
    results,
    key=lambda x: (
        x["confidence"],
        x["inliers"],
        x["ratio"]
    )
)


# ============================================================
# FINAL DECISION
# ============================================================

confidence = best["confidence"]
inliers = best["inliers"]
ratio = best["ratio"]
spatial = best["spatial"]

# Guard against tiny perfect-match sets.
#
# 8/8 matches is NOT allowed to become
# "high confidence" simply because RANSAC
# happened to fit them perfectly.

if inliers < 8:

    decision = "REJECT"

elif (
    inliers < 12
    or ratio < 30.0
):

    if confidence >= 60:
        decision = "MODERATE CONFIDENCE"
    else:
        decision = "LOW CONFIDENCE"

elif confidence >= 75:

    decision = "HIGH CONFIDENCE"

elif confidence >= 50:

    decision = "MODERATE CONFIDENCE"

else:

    decision = "LOW CONFIDENCE"


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 90)
print("FINAL CORRESPONDENCE REPORT")
print("=" * 90)

print(
    f"\n{'METHOD':<24}"
    f"{'MODEL':<14}"
    f"{'GOOD':>8}"
    f"{'INLIERS':>10}"
    f"{'RATIO':>12}"
    f"{'RMSE':>12}"
    f"{'CONF.':>12}"
)

print("-" * 90)

for r in results:

    rmse = (
        f"{r['rmse']:.4f}"
        if r["rmse"] is not None
        else "N/A"
    )

    print(
        f"{r['name']:<24}"
        f"{r['model']:<14}"
        f"{r['good']:>8}"
        f"{r['inliers']:>10}"
        f"{r['ratio']:>11.2f}%"
        f"{rmse:>12}"
        f"{r['confidence']:>11.2f}%"
    )


# ============================================================
# LOCALIZATION
# ============================================================

x = spatial["x"]
y = spatial["y"]

norm_x = (
    x /
    reference.shape[1]
) * 100.0

norm_y = (
    y /
    reference.shape[0]
) * 100.0


print("\n")
print("=" * 90)
print("FINAL LOCALIZATION")
print("=" * 90)

print(
    "\nSelected method:",
    best["name"]
)

print(
    "Geometric model:",
    best["model"]
)

print(
    "Good matches:",
    best["good"]
)

print(
    "Verified inliers:",
    best["inliers"]
)

print(
    "Inlier ratio:",
    f"{ratio:.2f}%"
)

print(
    "Geometric RMSE:",
    "N/A"
    if best["rmse"] is None
    else f"{best['rmse']:.4f} px"
)

print(
    "Spatial coverage:",
    f"{spatial['coverage']:.2f}%"
)

print(
    "Spatial support:",
    f"{spatial['support']:.2f}%"
)

print(
    "Localization:",
    f"({x:.2f}, {y:.2f}) px"
)

print(
    "Normalized:",
    f"({norm_x:.2f}%, {norm_y:.2f}%)"
)

print(
    "\nFINAL CONFIDENCE:",
    f"{confidence:.2f}%"
)

print(
    "DECISION:",
    decision
)


# ============================================================
# VISUALIZATION
# ============================================================

visual = reference.copy()

height, width = reference.shape[:2]

cell_width = width / GRID_COLS
cell_height = height / GRID_ROWS


# Draw grid

for i in range(1, GRID_COLS):

    gx = int(
        i * cell_width
    )

    cv2.line(
        visual,
        (gx, 0),
        (gx, height),
        (255, 255, 255),
        2
    )


for i in range(1, GRID_ROWS):

    gy = int(
        i * cell_height
    )

    cv2.line(
        visual,
        (0, gy),
        (width, gy),
        (255, 255, 255),
        2
    )


# Draw verified inliers

for i, match in enumerate(
    best["matches"]
):

    if best["mask"][i] != 1:
        continue

    px, py = best[
        "kp1"
    ][match.queryIdx].pt

    cv2.circle(
        visual,
        (int(px), int(py)),
        12,
        (0, 255, 0),
        -1
    )


# Bounding box

bbox = spatial["bbox"]

if bbox is not None:

    bx1, by1, bx2, by2 = bbox

    cv2.rectangle(
        visual,
        (int(bx1), int(by1)),
        (int(bx2), int(by2)),
        (255, 0, 0),
        6
    )


# Localization centroid

cv2.drawMarker(
    visual,
    (int(x), int(y)),
    (0, 0, 255),
    cv2.MARKER_CROSS,
    100,
    8
)


# Text panel

cv2.rectangle(
    visual,
    (20, 20),
    (850, 190),
    (0, 0, 0),
    -1
)

cv2.putText(
    visual,
    "SIH26166 FINAL LOCALIZATION",
    (40, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.1,
    (255, 255, 255),
    3
)

cv2.putText(
    visual,
    f"Model: {best['model']}",
    (40, 100),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)

cv2.putText(
    visual,
    f"Inliers: {inliers} | Ratio: {ratio:.1f}%",
    (40, 135),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)

cv2.putText(
    visual,
    f"Confidence: {confidence:.1f}% | {decision}",
    (40, 170),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)


# ============================================================
# SAVE VISUALIZATION
# ============================================================

visual_path = (
    VIS_DIR +
    "/final_result.jpg"
)

cv2.imwrite(
    visual_path,
    visual
)


# ============================================================
# SAVE REPORT
# ============================================================

report_path = (
    REPORT_DIR +
    "/final_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SIH26166 LUNAR IMAGE CORRESPONDENCE SYSTEM\n"
    )

    f.write(
        "FINAL ROBUST MULTI-SIGNAL LOCALIZATION PIPELINE\n"
    )

    f.write("=" * 80 + "\n\n")

    f.write(
        f"Generated: {datetime.now()}\n\n"
    )

    f.write(
        f"Reference: {REFERENCE_IMAGE}\n"
    )

    f.write(
        f"Target: {TARGET_IMAGE}\n\n"
    )

    f.write(
        "HYPOTHESIS RESULTS\n"
    )

    f.write("-" * 80 + "\n")

    for r in results:

        f.write(
            f"\nMethod: {r['name']}\n"
        )

        f.write(
            f"Geometric model: {r['model']}\n"
        )

        f.write(
            f"Reference features: "
            f"{r['features_ref']}\n"
        )

        f.write(
            f"Target features: "
            f"{r['features_target']}\n"
        )

        f.write(
            f"Good matches: "
            f"{r['good']}\n"
        )

        f.write(
            f"RANSAC inliers: "
            f"{r['inliers']}\n"
        )

        f.write(
            f"Inlier ratio: "
            f"{r['ratio']:.2f}%\n"
        )

        f.write(
            f"RMSE: "
            f"{r['rmse']}\n"
        )

        f.write(
            f"Spatial coverage: "
            f"{r['spatial']['coverage']:.2f}%\n"
        )

        f.write(
            f"Spatial support: "
            f"{r['spatial']['support']:.2f}%\n"
        )

        f.write(
            f"Model agreement: "
            f"{r['agreement'] * 100:.2f}%\n"
        )

        f.write(
            f"Confidence: "
            f"{r['confidence']:.2f}%\n"
        )


    f.write(
        "\n\nFINAL RESULT\n"
    )

    f.write("-" * 80 + "\n")

    f.write(
        f"Selected method: "
        f"{best['name']}\n"
    )

    f.write(
        f"Geometric model: "
        f"{best['model']}\n"
    )

    f.write(
        f"Good matches: "
        f"{best['good']}\n"
    )

    f.write(
        f"Verified inliers: "
        f"{inliers}\n"
    )

    f.write(
        f"Inlier ratio: "
        f"{ratio:.2f}%\n"
    )

    f.write(
        f"RMSE: "
        f"{best['rmse']}\n"
    )

    f.write(
        f"Spatial coverage: "
        f"{spatial['coverage']:.2f}%\n"
    )

    f.write(
        f"Spatial support: "
        f"{spatial['support']:.2f}%\n"
    )

    f.write(
        f"Localization X: "
        f"{x:.2f} px\n"
    )

    f.write(
        f"Localization Y: "
        f"{y:.2f} px\n"
    )

    f.write(
        f"Normalized X: "
        f"{norm_x:.2f}%\n"
    )

    f.write(
        f"Normalized Y: "
        f"{norm_y:.2f}%\n"
    )

    f.write(
        f"Confidence: "
        f"{confidence:.2f}%\n"
    )

    f.write(
        f"Decision: "
        f"{decision}\n"
    )


# ============================================================
# COMPLETED
# ============================================================

print("\n")
print("=" * 90)
print("FINAL PIPELINE COMPLETED")
print("=" * 90)

print(
    "\nVisualization:"
)

print(
    visual_path
)

print(
    "\nReport:"
)

print(
    report_path
)

print("=" * 90)