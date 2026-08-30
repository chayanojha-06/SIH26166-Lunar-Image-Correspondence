import cv2
import numpy as np
import os
from datetime import datetime


# ============================================================
# SIH26166
# LUNAR IMAGE CORRESPONDENCE SYSTEM
# FINAL ROBUST MULTI-SIGNAL LOCALIZATION PIPELINE
#
# PURPOSE:
#   1. Detect local features
#   2. Match reference and target
#   3. Apply Lowe ratio test
#   4. Verify geometry using Fundamental Matrix
#   5. Verify using Homography when appropriate
#   6. Reject unstable models
#   7. Perform spatial voting
#   8. Estimate a VALID location inside reference image
#   9. Calculate confidence using multiple signals
#  10. Save ONE visualization and ONE report
# ============================================================


# ============================================================
# PATHS
# ============================================================

REFERENCE_IMAGE = "data/raw/image1.jpg"
TARGET_IMAGE = "data/raw/image2.jpg"

OUTPUT_DIR = "outputs"
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

SIFT_FEATURES = 12000

LOWE_RATIO = 0.72

# Fundamental matrix
F_RANSAC_THRESHOLD = 1.5
F_CONFIDENCE = 0.995
F_ITERATIONS = 5000

# Homography
H_RANSAC_THRESHOLD = 4.0
H_CONFIDENCE = 0.995

# Spatial voting
GRID_ROWS = 8
GRID_COLS = 8

# Minimum requirements
MIN_GOOD_MATCHES = 8
MIN_F_INLIERS = 8
MIN_H_INLIERS = 8

# A localization must remain inside this percentage
# of the reference image.
VALID_MARGIN = 0.02


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("SIH26166 LUNAR IMAGE CORRESPONDENCE SYSTEM")
print("FINAL ROBUST MULTI-SIGNAL LOCALIZATION PIPELINE")
print("=" * 90)


# ============================================================
# IMAGE LOADING
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
# PREPROCESSING HYPOTHESES
# ============================================================

def create_hypotheses(image):
    """
    Creates several preprocessing versions.

    Important:
    Localization coordinates are ALWAYS converted back
    to ORIGINAL reference-image coordinates.
    """

    hypotheses = {}

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    hypotheses["Baseline"] = {
        "image": image.copy(),
        "scale": 1.0
    }

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe_img = clahe.apply(gray)

    hypotheses["CLAHE"] = {
        "image": cv2.cvtColor(
            clahe_img,
            cv2.COLOR_GRAY2BGR
        ),
        "scale": 1.0
    }

    # --------------------------------------------------------
    # Upscale
    # --------------------------------------------------------

    upscale = cv2.resize(
        image,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    hypotheses["Upscale"] = {
        "image": upscale,
        "scale": 2.0
    }

    # --------------------------------------------------------
    # Upscale + CLAHE
    # --------------------------------------------------------

    upscale_gray = cv2.cvtColor(
        upscale,
        cv2.COLOR_BGR2GRAY
    )

    upscale_clahe = clahe.apply(
        upscale_gray
    )

    hypotheses["Upscale + CLAHE"] = {
        "image": cv2.cvtColor(
            upscale_clahe,
            cv2.COLOR_GRAY2BGR
        ),
        "scale": 2.0
    }

    return hypotheses


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES,
        contrastThreshold=0.025,
        edgeThreshold=10
    )

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    return keypoints, descriptors


# ============================================================
# FEATURE MATCHING
# ============================================================

def match_features(
    des1,
    des2
):

    if des1 is None or des2 is None:
        return []

    matcher = cv2.FlannBasedMatcher(
        dict(
            algorithm=1,
            trees=5
        ),
        dict(
            checks=100
        )
    )

    knn = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good = []

    for pair in knn:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    return good


# ============================================================
# POINT EXTRACTION
# ============================================================

def get_points(
    kp1,
    kp2,
    matches
):

    pts1 = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ])

    pts2 = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ])

    return pts1, pts2


# ============================================================
# FUNDAMENTAL MATRIX
# ============================================================

def fundamental_verification(
    kp1,
    kp2,
    matches
):

    if len(matches) < 8:

        return None, np.zeros(
            len(matches),
            dtype=np.uint8
        )

    pts1, pts2 = get_points(
        kp1,
        kp2,
        matches
    )

    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        cv2.FM_RANSAC,
        F_RANSAC_THRESHOLD,
        F_CONFIDENCE,
        F_ITERATIONS
    )

    if F is None or mask is None:

        return None, np.zeros(
            len(matches),
            dtype=np.uint8
        )

    mask = mask.ravel().astype(
        np.uint8
    )

    # OpenCV can theoretically return multiple
    # fundamental matrices. Keep only the first 3x3.
    if F.shape[0] >= 3:

        F = F[:3, :3]

    return F, mask


# ============================================================
# HOMOGRAPHY
# ============================================================

def homography_verification(
    kp1,
    kp2,
    matches
):

    if len(matches) < 4:

        return None, np.zeros(
            len(matches),
            dtype=np.uint8
        )

    pts1, pts2 = get_points(
        kp1,
        kp2,
        matches
    )

    H, mask = cv2.findHomography(
        pts1,
        pts2,
        cv2.RANSAC,
        H_RANSAC_THRESHOLD,
        None,
        5000,
        H_CONFIDENCE
    )

    if H is None or mask is None:

        return None, np.zeros(
            len(matches),
            dtype=np.uint8
        )

    return H, mask.ravel().astype(
        np.uint8
    )


# ============================================================
# FUNDAMENTAL MATRIX ERROR
# ============================================================

def fundamental_error(
    F,
    pts1,
    pts2
):

    if F is None or len(pts1) == 0:
        return None

    ones1 = np.ones(
        (len(pts1), 1),
        dtype=np.float32
    )

    ones2 = np.ones(
        (len(pts2), 1),
        dtype=np.float32
    )

    p1 = np.hstack(
        [pts1, ones1]
    )

    p2 = np.hstack(
        [pts2, ones2]
    )

    lines2 = (F @ p1.T).T
    lines1 = (F.T @ p2.T).T

    numerator = np.sum(
        p2 * lines2,
        axis=1
    )

    denominator = (
        lines1[:, 0] ** 2 +
        lines1[:, 1] ** 2 +
        lines2[:, 0] ** 2 +
        lines2[:, 1] ** 2
    )

    denominator = np.maximum(
        denominator,
        1e-12
    )

    error = (
        numerator ** 2 /
        denominator
    )

    return float(
        np.sqrt(
            np.mean(error)
        )
    )


# ============================================================
# HOMOGRAPHY REPROJECTION ERROR
# ============================================================

def homography_rmse(
    H,
    pts1,
    pts2
):

    if H is None or len(pts1) == 0:
        return None

    projected = cv2.perspectiveTransform(
        pts1.reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)

    errors = projected - pts2

    squared = np.sum(
        errors ** 2,
        axis=1
    )

    return float(
        np.sqrt(
            np.mean(squared)
        )
    )


# ============================================================
# SPATIAL VOTING
# ============================================================

def spatial_analysis(
    kp1,
    kp2,
    matches,
    mask,
    original_reference_shape,
    scale
):

    inlier_indices = np.where(
        mask == 1
    )[0]

    if len(inlier_indices) == 0:

        return {
            "grid": np.zeros(
                (GRID_ROWS, GRID_COLS),
                dtype=int
            ),
            "coverage": 0.0,
            "support": 0.0,
            "strongest_votes": 0,
            "x": None,
            "y": None
        }

    # --------------------------------------------------------
    # Convert points from hypothesis coordinates to
    # ORIGINAL reference coordinates.
    # --------------------------------------------------------

    points = []

    for idx in inlier_indices:

        x, y = kp1[
            matches[idx].queryIdx
        ].pt

        x /= scale
        y /= scale

        points.append(
            [x, y]
        )

    points = np.float32(points)

    height, width = (
        original_reference_shape[:2]
    )

    grid = np.zeros(
        (GRID_ROWS, GRID_COLS),
        dtype=int
    )

    cell_width = width / GRID_COLS
    cell_height = height / GRID_ROWS

    for x, y in points:

        col = int(
            x / cell_width
        )

        row = int(
            y / cell_height
        )

        col = max(
            0,
            min(
                GRID_COLS - 1,
                col
            )
        )

        row = max(
            0,
            min(
                GRID_ROWS - 1,
                row
            )
        )

        grid[row, col] += 1

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    occupied = int(
        np.sum(grid > 0)
    )

    coverage = (
        occupied /
        (GRID_ROWS * GRID_COLS)
    ) * 100.0

    # --------------------------------------------------------
    # Strongest cell
    # --------------------------------------------------------

    best_row, best_col = np.unravel_index(
        np.argmax(grid),
        grid.shape
    )

    strongest = int(
        grid[
            best_row,
            best_col
        ]
    )

    # --------------------------------------------------------
    # Neighbor support
    # --------------------------------------------------------

    neighbor_votes = 0

    for r in range(
        max(0, best_row - 1),
        min(GRID_ROWS, best_row + 2)
    ):

        for c in range(
            max(0, best_col - 1),
            min(GRID_COLS, best_col + 2)
        ):

            if (
                r == best_row and
                c == best_col
            ):
                continue

            neighbor_votes += int(
                grid[r, c]
            )

    support = (
        (
            strongest +
            neighbor_votes
        )
        /
        len(points)
    ) * 100.0

    # --------------------------------------------------------
    # Weighted centroid of inliers
    #
    # This is better than simply using the center of the
    # strongest grid cell.
    # --------------------------------------------------------

    x_mean = float(
        np.mean(points[:, 0])
    )

    y_mean = float(
        np.mean(points[:, 1])
    )

    # --------------------------------------------------------
    # Clamp location to valid image boundaries.
    # --------------------------------------------------------

    x_min = (
        width *
        VALID_MARGIN
    )

    x_max = (
        width *
        (1.0 - VALID_MARGIN)
    )

    y_min = (
        height *
        VALID_MARGIN
    )

    y_max = (
        height *
        (1.0 - VALID_MARGIN)
    )

    x_mean = float(
        np.clip(
            x_mean,
            x_min,
            x_max
        )
    )

    y_mean = float(
        np.clip(
            y_mean,
            y_min,
            y_max
        )
    )

    return {
        "grid": grid,
        "coverage": float(
            coverage
        ),
        "support": float(
            support
        ),
        "strongest_votes": strongest,
        "x": x_mean,
        "y": y_mean
    }


# ============================================================
# MODEL SELECTION
# ============================================================

def select_model(
    f_inliers,
    h_inliers,
    good_count,
    h_rmse_value
):

    if good_count < MIN_GOOD_MATCHES:

        return "None"

    # Fundamental model is preferred for general
    # viewpoint correspondence unless a homography
    # is strongly supported.
    #
    # Homography requires BOTH:
    #   - at least 8 inliers
    #   - at least 45% inlier ratio
    #   - reasonable reprojection error
    #

    h_ratio = (
        h_inliers /
        max(1, good_count)
    )

    f_ratio = (
        f_inliers /
        max(1, good_count)
    )

    if (
        h_inliers >= MIN_H_INLIERS
        and
        h_ratio >= 0.45
        and
        h_rmse_value is not None
        and
        h_rmse_value < 5.0
    ):

        return "Homography"

    if f_inliers >= MIN_F_INLIERS:

        return "Fundamental"

    return "None"


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    good,
    selected_inliers,
    inlier_ratio,
    spatial_support,
    spatial_coverage,
    model_agreement,
    model
):

    if good <= 0:
        return 0.0

    # --------------------------------------------------------
    # Match quantity
    # --------------------------------------------------------

    match_signal = min(
        100.0,
        good * 2.5
    )

    # --------------------------------------------------------
    # Geometric consistency
    # --------------------------------------------------------

    geometry_signal = min(
        100.0,
        inlier_ratio
    )

    # --------------------------------------------------------
    # Spatial signal
    # --------------------------------------------------------

    support_signal = min(
        100.0,
        spatial_support
    )

    # Coverage is deliberately weaker.
    coverage_signal = min(
        100.0,
        spatial_coverage * 2.0
    )

    # --------------------------------------------------------
    # Model agreement
    # --------------------------------------------------------

    agreement_signal = (
        model_agreement * 100.0
    )

    # --------------------------------------------------------
    # Base confidence
    # --------------------------------------------------------

    confidence = (
        0.15 * match_signal +
        0.30 * geometry_signal +
        0.25 * support_signal +
        0.10 * coverage_signal +
        0.20 * agreement_signal
    )

    # --------------------------------------------------------
    # Penalize weak models
    # --------------------------------------------------------

    if model == "None":

        confidence *= 0.35

    # A model based on fewer than 10 inliers should
    # never be considered highly reliable.
    if selected_inliers < 10:

        confidence *= 0.85

    # A very low inlier ratio should reduce confidence.
    if inlier_ratio < 25.0:

        confidence *= 0.80

    return float(
        np.clip(
            confidence,
            0.0,
            100.0
        )
    )


# ============================================================
# HYPOTHESES
# ============================================================

reference_hypotheses = create_hypotheses(
    reference
)

target_hypotheses = create_hypotheses(
    target
)


# ============================================================
# ANALYSIS
# ============================================================

print("\n")
print("=" * 90)
print("MULTI-HYPOTHESIS GEOMETRIC ANALYSIS")
print("=" * 90)

results = []


for name in reference_hypotheses:

    print("\nTesting:", name)

    ref_info = reference_hypotheses[name]
    tgt_info = target_hypotheses[name]

    ref_img = ref_info["image"]
    target_img = tgt_info["image"]

    scale = ref_info["scale"]

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

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
        "Features:",
        len(kp1),
        "/",
        len(kp2)
    )

    print(
        "Good matches:",
        len(good)
    )

    # --------------------------------------------------------
    # Too few matches
    # --------------------------------------------------------

    if len(good) < MIN_GOOD_MATCHES:

        print(
            "Not enough matches for reliable geometry."
        )

        results.append({
            "name": name,
            "scale": scale,
            "kp1": kp1,
            "kp2": kp2,
            "matches": good,
            "f_inliers": 0,
            "h_inliers": 0,
            "selected_inliers": 0,
            "model": "None",
            "ratio": 0.0,
            "f_rmse": None,
            "h_rmse": None,
            "spatial": {
                "grid": np.zeros(
                    (GRID_ROWS, GRID_COLS),
                    dtype=int
                ),
                "coverage": 0.0,
                "support": 0.0,
                "strongest_votes": 0,
                "x": None,
                "y": None
            },
            "confidence": 0.0,
            "f_mask": np.zeros(
                len(good),
                dtype=np.uint8
            ),
            "h_mask": np.zeros(
                len(good),
                dtype=np.uint8
            )
        })

        continue

    # --------------------------------------------------------
    # Fundamental
    # --------------------------------------------------------

    F, f_mask = fundamental_verification(
        kp1,
        kp2,
        good
    )

    f_inliers = int(
        np.sum(f_mask)
    )

    # --------------------------------------------------------
    # Homography
    # --------------------------------------------------------

    H, h_mask = homography_verification(
        kp1,
        kp2,
        good
    )

    h_inliers = int(
        np.sum(h_mask)
    )

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    f_rmse = None

    if F is not None and f_inliers >= 8:

        f_indices = np.where(
            f_mask == 1
        )[0]

        f_pts1 = np.float32([
            kp1[
                good[i].queryIdx
            ].pt
            for i in f_indices
        ])

        f_pts2 = np.float32([
            kp2[
                good[i].trainIdx
            ].pt
            for i in f_indices
        ])

        f_rmse = fundamental_error(
            F,
            f_pts1,
            f_pts2
        )

    h_rmse_value = None

    if H is not None and h_inliers >= 4:

        h_indices = np.where(
            h_mask == 1
        )[0]

        h_pts1 = np.float32([
            kp1[
                good[i].queryIdx
            ].pt
            for i in h_indices
        ])

        h_pts2 = np.float32([
            kp2[
                good[i].trainIdx
            ].pt
            for i in h_indices
        ])

        h_rmse_value = homography_rmse(
            H,
            h_pts1,
            h_pts2
        )

    # --------------------------------------------------------
    # Model selection
    # --------------------------------------------------------

    model = select_model(
        f_inliers,
        h_inliers,
        len(good),
        h_rmse_value
    )

    if model == "Homography":

        selected_mask = h_mask
        selected_inliers = h_inliers

    elif model == "Fundamental":

        selected_mask = f_mask
        selected_inliers = f_inliers

    else:

        selected_mask = np.zeros(
            len(good),
            dtype=np.uint8
        )

        selected_inliers = 0

    ratio = (
        selected_inliers /
        max(1, len(good))
    ) * 100.0

    # --------------------------------------------------------
    # Model agreement
    # --------------------------------------------------------

    if (
        f_inliers > 0 and
        h_inliers > 0
    ):

        model_agreement = (
            min(
                f_inliers,
                h_inliers
            )
            /
            max(
                f_inliers,
                h_inliers
            )
        )

    else:

        model_agreement = 0.0

    # --------------------------------------------------------
    # Spatial analysis
    # --------------------------------------------------------

    spatial = spatial_analysis(
        kp1,
        kp2,
        good,
        selected_mask,
        reference.shape,
        scale
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = calculate_confidence(
        len(good),
        selected_inliers,
        ratio,
        spatial["support"],
        spatial["coverage"],
        model_agreement,
        model
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print(
        "Geometric model:",
        model
    )

    print(
        "Homography inliers:",
        h_inliers
    )

    print(
        "Fundamental inliers:",
        f_inliers
    )

    print(
        "Selected inliers:",
        selected_inliers
    )

    print(
        "Inlier ratio:",
        f"{ratio:.2f}%"
    )

    print(
        "Fundamental error:",
        "N/A"
        if f_rmse is None
        else f"{f_rmse:.4f}"
    )

    print(
        "Homography RMSE:",
        "N/A"
        if h_rmse_value is None
        else f"{h_rmse_value:.4f} px"
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
        "Model agreement:",
        f"{model_agreement * 100:.2f}%"
    )

    print(
        "Confidence:",
        f"{confidence:.2f}%"
    )

    results.append({
        "name": name,
        "scale": scale,
        "kp1": kp1,
        "kp2": kp2,
        "matches": good,
        "F": F,
        "H": H,
        "f_mask": f_mask,
        "h_mask": h_mask,
        "selected_mask": selected_mask,
        "f_inliers": f_inliers,
        "h_inliers": h_inliers,
        "selected_inliers": selected_inliers,
        "model": model,
        "ratio": ratio,
        "f_rmse": f_rmse,
        "h_rmse": h_rmse_value,
        "spatial": spatial,
        "confidence": confidence,
        "model_agreement": model_agreement
    })


# ============================================================
# SELECT BEST RESULT
# ============================================================

valid_results = [
    r
    for r in results
    if r["selected_inliers"] >= MIN_F_INLIERS
    and r["spatial"]["x"] is not None
]


if not valid_results:

    print("\n")
    print("=" * 90)
    print("FINAL DECISION")
    print("=" * 90)

    print(
        "\nNo reliable localization could be established."
    )

    print(
        "The available feature correspondences are insufficient."
    )

    final_confidence = 0.0
    final_decision = "LOW CONFIDENCE / NO RELIABLE LOCALIZATION"

    best_result = results[0]

else:

    best_result = max(
        valid_results,
        key=lambda r: r["confidence"]
    )

    final_confidence = (
        best_result["confidence"]
    )

    if final_confidence >= 80:

        final_decision = "HIGH CONFIDENCE"

    elif final_confidence >= 60:

        final_decision = "MODERATE CONFIDENCE"

    elif final_confidence >= 40:

        final_decision = "LOW-MODERATE CONFIDENCE"

    else:

        final_decision = "LOW CONFIDENCE"


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 90)
print("FINAL CORRESPONDENCE REPORT")
print("=" * 90)

print(
    f"\n{'METHOD':<22}"
    f"{'MODEL':<15}"
    f"{'GOOD':>7}"
    f"{'H-IN':>8}"
    f"{'F-IN':>8}"
    f"{'RATIO':>12}"
    f"{'CONF.':>11}"
)

print("-" * 90)

for r in results:

    print(
        f"{r['name']:<22}"
        f"{r['model']:<15}"
        f"{len(r['matches']):>7}"
        f"{r['h_inliers']:>8}"
        f"{r['f_inliers']:>8}"
        f"{r['ratio']:>11.2f}%"
        f"{r['confidence']:>10.2f}%"
    )


# ============================================================
# FINAL LOCALIZATION
# ============================================================

print("\n")
print("=" * 90)
print("FINAL LOCALIZATION")
print("=" * 90)

if valid_results:

    spatial = best_result["spatial"]

    x = spatial["x"]
    y = spatial["y"]

    ref_height, ref_width = (
        reference.shape[:2]
    )

    normalized_x = (
        x /
        ref_width
    ) * 100.0

    normalized_y = (
        y /
        ref_height
    ) * 100.0

    print(
        "\nSelected method:",
        best_result["name"]
    )

    print(
        "Geometric model:",
        best_result["model"]
    )

    print(
        "Good matches:",
        len(best_result["matches"])
    )

    print(
        "Verified inliers:",
        best_result["selected_inliers"]
    )

    print(
        "Homography inliers:",
        best_result["h_inliers"]
    )

    print(
        "Fundamental inliers:",
        best_result["f_inliers"]
    )

    print(
        "Inlier ratio:",
        f"{best_result['ratio']:.2f}%"
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
        f"({normalized_x:.2f}%, "
        f"{normalized_y:.2f}%)"
    )

    print(
        "\nFINAL CONFIDENCE:",
        f"{final_confidence:.2f}%"
    )

    print(
        "DECISION:",
        final_decision
    )

else:

    x = None
    y = None
    normalized_x = None
    normalized_y = None

    print(
        "\nLocalization:",
        "NOT RELIABLE"
    )

    print(
        "\nFINAL CONFIDENCE:",
        "0.00%"
    )

    print(
        "DECISION:",
        final_decision
    )


# ============================================================
# VISUALIZATION
# ============================================================

visual = reference.copy()

if valid_results:

    spatial = best_result["spatial"]

    grid = spatial["grid"]

    height, width = reference.shape[:2]

    cell_width = (
        width /
        GRID_COLS
    )

    cell_height = (
        height /
        GRID_ROWS
    )

    # --------------------------------------------------------
    # Draw grid
    # --------------------------------------------------------

    for c in range(
        1,
        GRID_COLS
    ):

        gx = int(
            c * cell_width
        )

        cv2.line(
            visual,
            (gx, 0),
            (gx, height),
            (180, 180, 180),
            1
        )

    for r in range(
        1,
        GRID_ROWS
    ):

        gy = int(
            r * cell_height
        )

        cv2.line(
            visual,
            (0, gy),
            (width, gy),
            (180, 180, 180),
            1
        )

    # --------------------------------------------------------
    # Highlight strongest cell
    # --------------------------------------------------------

    best_cell = np.unravel_index(
        np.argmax(grid),
        grid.shape
    )

    best_row, best_col = best_cell

    x1 = int(
        best_col *
        cell_width
    )

    y1 = int(
        best_row *
        cell_height
    )

    x2 = int(
        (best_col + 1) *
        cell_width
    )

    y2 = int(
        (best_row + 1) *
        cell_height
    )

    cv2.rectangle(
        visual,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        5
    )

    # --------------------------------------------------------
    # Draw verified reference points
    # --------------------------------------------------------

    scale = best_result["scale"]

    for i, match in enumerate(
        best_result["matches"]
    ):

        if (
            best_result["selected_mask"][i]
            != 1
        ):
            continue

        px, py = best_result[
            "kp1"
        ][match.queryIdx].pt

        # Convert back to original reference
        px /= scale
        py /= scale

        px = int(
            np.clip(
                px,
                0,
                width - 1
            )
        )

        py = int(
            np.clip(
                py,
                0,
                height - 1
            )
        )

        cv2.circle(
            visual,
            (px, py),
            8,
            (0, 255, 0),
            -1
        )

    # --------------------------------------------------------
    # Localization marker
    # --------------------------------------------------------

    cv2.drawMarker(
        visual,
        (
            int(x),
            int(y)
        ),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        60,
        5
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    cv2.putText(
        visual,
        "SIH26166 LUNAR LOCALIZATION",
        (40, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3
    )

    cv2.putText(
        visual,
        f"Method: {best_result['name']}",
        (40, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        visual,
        f"Confidence: {final_confidence:.1f}%",
        (40, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        visual,
        f"Location: ({x:.0f}, {y:.0f})",
        (40, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

else:

    cv2.putText(
        visual,
        "SIH26166 - NO RELIABLE LOCALIZATION",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        3
    )


# ============================================================
# SAVE VISUALIZATION
# ============================================================

visual_path = os.path.join(
    VIS_DIR,
    "final_result.jpg"
)

cv2.imwrite(
    visual_path,
    visual
)


# ============================================================
# SAVE REPORT
# ============================================================

report_path = os.path.join(
    REPORT_DIR,
    "final_report.txt"
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

    f.write(
        "=" * 75 + "\n\n"
    )

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

    f.write(
        "-" * 75 + "\n"
    )

    for r in results:

        f.write(
            f"\nMethod: {r['name']}\n"
        )

        f.write(
            f"Model: {r['model']}\n"
        )

        f.write(
            f"Good matches: "
            f"{len(r['matches'])}\n"
        )

        f.write(
            f"Homography inliers: "
            f"{r['h_inliers']}\n"
        )

        f.write(
            f"Fundamental inliers: "
            f"{r['f_inliers']}\n"
        )

        f.write(
            f"Selected inliers: "
            f"{r['selected_inliers']}\n"
        )

        f.write(
            f"Inlier ratio: "
            f"{r['ratio']:.2f}%\n"
        )

        f.write(
            f"Fundamental error: "
            f"{r['f_rmse']}\n"
        )

        f.write(
            f"Homography RMSE: "
            f"{r['h_rmse']}\n"
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
            f"{r['model_agreement'] * 100:.2f}%\n"
        )

        f.write(
            f"Confidence: "
            f"{r['confidence']:.2f}%\n"
        )

    f.write(
        "\n\nFINAL RESULT\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    if valid_results:

        f.write(
            f"Selected method: "
            f"{best_result['name']}\n"
        )

        f.write(
            f"Geometric model: "
            f"{best_result['model']}\n"
        )

        f.write(
            f"Good matches: "
            f"{len(best_result['matches'])}\n"
        )

        f.write(
            f"Verified inliers: "
            f"{best_result['selected_inliers']}\n"
        )

        f.write(
            f"Homography inliers: "
            f"{best_result['h_inliers']}\n"
        )

        f.write(
            f"Fundamental inliers: "
            f"{best_result['f_inliers']}\n"
        )

        f.write(
            f"Inlier ratio: "
            f"{best_result['ratio']:.2f}%\n"
        )

        f.write(
            f"Spatial coverage: "
            f"{best_result['spatial']['coverage']:.2f}%\n"
        )

        f.write(
            f"Spatial support: "
            f"{best_result['spatial']['support']:.2f}%\n"
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
            f"{normalized_x:.2f}%\n"
        )

        f.write(
            f"Normalized Y: "
            f"{normalized_y:.2f}%\n"
        )

        f.write(
            f"Final confidence: "
            f"{final_confidence:.2f}%\n"
        )

        f.write(
            f"Decision: "
            f"{final_decision}\n"
        )

    else:

        f.write(
            "No reliable localization established.\n"
        )

        f.write(
            "Final confidence: 0.00%\n"
        )

        f.write(
            f"Decision: "
            f"{final_decision}\n"
        )

    f.write(
        "\n\nOUTPUT FILES\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    f.write(
        f"Visualization: {visual_path}\n"
    )

    f.write(
        f"Report: {report_path}\n"
    )


# ============================================================
# COMPLETION
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