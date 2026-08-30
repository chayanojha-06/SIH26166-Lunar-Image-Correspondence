import cv2
import numpy as np
import os
from datetime import datetime

# ============================================================
# SIH26166 - FINAL LUNAR IMAGE CORRESPONDENCE SYSTEM
# MULTI-HYPOTHESIS + GEOMETRIC VERIFICATION + CONSENSUS
# ============================================================

REFERENCE_IMAGE = "data/raw/image1.jpg"
TARGET_IMAGE = "data/raw/image2.jpg"

OUTPUT_DIR = "outputs"
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------- CONFIG ----------------

SIFT_FEATURES = 12000
LOWE_RATIO = 0.80

FUNDAMENTAL_THRESHOLD = 2.0
HOMOGRAPHY_THRESHOLD = 4.0

RANSAC_CONFIDENCE = 0.999
RANSAC_ITERATIONS = 5000

GRID_ROWS = 4
GRID_COLS = 4

# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("SIH26166 LUNAR IMAGE CORRESPONDENCE SYSTEM")
print("FINAL MULTI-SIGNAL CONSENSUS PIPELINE")
print("=" * 78)

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

def create_hypotheses(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    up = cv2.resize(
        image,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    up_gray = cv2.cvtColor(
        up,
        cv2.COLOR_BGR2GRAY
    )

    return {
        "Baseline": (
            image.copy(),
            1.0
        ),

        "CLAHE": (
            clahe.apply(gray),
            1.0
        ),

        "Upscale": (
            up,
            2.0
        ),

        "Upscale + CLAHE": (
            clahe.apply(up_gray),
            2.0
        )
    }


reference_hypotheses = create_hypotheses(reference)
target_hypotheses = create_hypotheses(target)

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image):

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image

    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES,
        contrastThreshold=0.02,
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

def match_features(des1, des2):

    if des1 is None or des2 is None:
        return []

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    raw = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good = []

    for pair in raw:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    return good


# ============================================================
# FUNDAMENTAL MATRIX
# ============================================================

def calculate_fundamental(
    kp1,
    kp2,
    matches
):

    if len(matches) < 8:

        return (
            None,
            np.zeros(
                len(matches),
                dtype=np.uint8
            )
        )

    pts1 = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ])

    pts2 = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ])

    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        cv2.FM_RANSAC,
        FUNDAMENTAL_THRESHOLD,
        RANSAC_CONFIDENCE,
        RANSAC_ITERATIONS
    )

    if mask is None:

        return (
            F,
            np.zeros(
                len(matches),
                dtype=np.uint8
            )
        )

    mask = mask.ravel().astype(np.uint8)

    if len(mask) != len(matches):

        mask = np.resize(
            mask,
            len(matches)
        )

    return F, mask


# ============================================================
# HOMOGRAPHY
# ============================================================

def calculate_homography(
    kp1,
    kp2,
    matches
):

    if len(matches) < 4:

        return (
            None,
            np.zeros(
                len(matches),
                dtype=np.uint8
            )
        )

    pts1 = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ])

    pts2 = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ])

    H, mask = cv2.findHomography(
        pts1,
        pts2,
        cv2.RANSAC,
        HOMOGRAPHY_THRESHOLD,
        maxIters=RANSAC_ITERATIONS,
        confidence=RANSAC_CONFIDENCE
    )

    if mask is None:

        return (
            H,
            np.zeros(
                len(matches),
                dtype=np.uint8
            )
        )

    return (
        H,
        mask.ravel().astype(np.uint8)
    )


# ============================================================
# HOMOGRAPHY ERROR
# ============================================================

def homography_rmse(
    H,
    kp1,
    kp2,
    matches,
    mask
):

    if H is None:
        return None

    indices = np.where(mask == 1)[0]

    if len(indices) < 4:
        return None

    pts1 = np.float32([
        kp1[matches[i].queryIdx].pt
        for i in indices
    ]).reshape(-1, 1, 2)

    pts2 = np.float32([
        kp2[matches[i].trainIdx].pt
        for i in indices
    ])

    try:

        projected = cv2.perspectiveTransform(
            pts1,
            H
        ).reshape(-1, 2)

        errors = projected - pts2

        distances = np.sqrt(
            np.sum(
                errors ** 2,
                axis=1
            )
        )

        return float(
            np.sqrt(
                np.mean(
                    distances ** 2
                )
            )
        )

    except Exception:

        return None


# ============================================================
# FUNDAMENTAL ERROR
# ============================================================

def fundamental_error(
    F,
    kp1,
    kp2,
    matches,
    mask
):

    if F is None:
        return None

    indices = np.where(mask == 1)[0]

    if len(indices) < 8:
        return None

    p1 = np.float64([
        kp1[matches[i].queryIdx].pt
        for i in indices
    ])

    p2 = np.float64([
        kp2[matches[i].trainIdx].pt
        for i in indices
    ])

    p1 = np.hstack([
        p1,
        np.ones(
            (len(p1), 1)
        )
    ])

    p2 = np.hstack([
        p2,
        np.ones(
            (len(p2), 1)
        )
    ])

    lines = (
        F @ p1.T
    ).T

    numerator = np.abs(
        np.sum(
            p2 * lines,
            axis=1
        )
    )

    denominator = np.sqrt(
        lines[:, 0] ** 2 +
        lines[:, 1] ** 2
    )

    denominator[
        denominator < 1e-12
    ] = 1e-12

    distances = (
        numerator /
        denominator
    )

    return float(
        np.sqrt(
            np.mean(
                distances ** 2
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
    scale,
    reference_shape
):

    indices = np.where(
        mask == 1
    )[0]

    if len(indices) == 0:

        return {
            "coverage": 0.0,
            "dominance": 0.0,
            "centroid": None,
            "points": np.empty(
                (0, 2),
                dtype=np.float32
            ),
            "grid": np.zeros(
                (GRID_ROWS, GRID_COLS),
                dtype=int
            )
        }

    h, w = reference_shape[:2]

    points = []

    for i in indices:

        x, y = kp1[
            matches[i].queryIdx
        ].pt

        x /= scale
        y /= scale

        x = np.clip(
            x,
            0,
            w - 1
        )

        y = np.clip(
            y,
            0,
            h - 1
        )

        points.append(
            [x, y]
        )

    points = np.array(
        points,
        dtype=np.float32
    )

    grid = np.zeros(
        (GRID_ROWS, GRID_COLS),
        dtype=int
    )

    for x, y in points:

        col = min(
            GRID_COLS - 1,
            int(
                x / w *
                GRID_COLS
            )
        )

        row = min(
            GRID_ROWS - 1,
            int(
                y / h *
                GRID_ROWS
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

    dominance = (
        np.max(grid) /
        len(points)
    ) * 100.0

    centroid = np.mean(
        points,
        axis=0
    )

    return {
        "coverage": float(coverage),
        "dominance": float(dominance),
        "centroid": (
            float(centroid[0]),
            float(centroid[1])
        ),
        "points": points,
        "grid": grid
    }


# ============================================================
# MODEL CONSISTENCY
# ============================================================

def consistency(
    h_inliers,
    f_inliers
):

    if h_inliers == 0 or f_inliers == 0:
        return 0.0

    return (
        min(
            h_inliers,
            f_inliers
        ) /
        max(
            h_inliers,
            f_inliers
        )
    ) * 100.0


# ============================================================
# HYPOTHESIS CONFIDENCE
# ============================================================

def calculate_confidence(
    good,
    selected_inliers,
    selected_ratio,
    coverage,
    dominance,
    model_consistency,
    rmse
):

    if good <= 0:
        return 0.0

    # Match quantity
    match_signal = min(
        100.0,
        selected_inliers * 6.0
    )

    # Ratio
    ratio_signal = min(
        100.0,
        selected_ratio
    )

    # Spatial consistency
    spatial_signal = (
        0.55 * coverage +
        0.45 * dominance
    )

    # Model agreement
    agreement_signal = model_consistency

    # Geometric error
    if rmse is None:

        error_signal = 35.0

    else:

        error_signal = (
            100.0 *
            np.exp(
                -rmse / 4.0
            )
        )

        error_signal = np.clip(
            error_signal,
            0,
            100
        )

    score = (
        0.20 * match_signal +
        0.25 * ratio_signal +
        0.20 * spatial_signal +
        0.15 * agreement_signal +
        0.20 * error_signal
    )

    return float(
        np.clip(
            score,
            0,
            100
        )
    )


# ============================================================
# RUN HYPOTHESES
# ============================================================

print("\n")
print("=" * 78)
print("MULTI-HYPOTHESIS ANALYSIS")
print("=" * 78)

results = []

for name in reference_hypotheses:

    print("\nTesting:", name)

    ref_img, ref_scale = (
        reference_hypotheses[name]
    )

    tar_img, _ = (
        target_hypotheses[name]
    )

    kp1, des1 = extract_features(
        ref_img
    )

    kp2, des2 = extract_features(
        tar_img
    )

    matches = match_features(
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
        len(matches)
    )

    if len(matches) < 4:

        print(
            "Too few matches."
        )

        continue

    # --------------------------------------------------------
    # BOTH GEOMETRIC MODELS
    # --------------------------------------------------------

    F, fmask = calculate_fundamental(
        kp1,
        kp2,
        matches
    )

    H, hmask = calculate_homography(
        kp1,
        kp2,
        matches
    )

    f_inliers = int(
        np.sum(fmask)
    )

    h_inliers = int(
        np.sum(hmask)
    )

    f_ratio = (
        f_inliers /
        len(matches)
    ) * 100.0

    h_ratio = (
        h_inliers /
        len(matches)
    ) * 100.0

    f_error = fundamental_error(
        F,
        kp1,
        kp2,
        matches,
        fmask
    )

    h_rmse = homography_rmse(
        H,
        kp1,
        kp2,
        matches,
        hmask
    )

    agreement = consistency(
        h_inliers,
        f_inliers
    )

    # --------------------------------------------------------
    # SELECT MODEL
    # --------------------------------------------------------

    # A homography is preferred when it has reasonable
    # support. Otherwise Fundamental is used.

    if (
        h_inliers >= 6
        and h_ratio >= 12
        and (
            h_inliers >=
            0.70 * f_inliers
        )
    ):

        model = "Homography"
        mask = hmask
        inliers = h_inliers
        ratio = h_ratio
        rmse = h_rmse

    elif f_inliers >= 8:

        model = "Fundamental"
        mask = fmask
        inliers = f_inliers
        ratio = f_ratio
        rmse = f_error

    elif h_inliers >= 4:

        model = "Homography"
        mask = hmask
        inliers = h_inliers
        ratio = h_ratio
        rmse = h_rmse

    else:

        # We still keep the hypothesis so that the consensus
        # stage can inspect it.
        model = "Correspondence"
        mask = np.zeros(
            len(matches),
            dtype=np.uint8
        )

        inliers = 0
        ratio = 0.0
        rmse = None

    spatial = spatial_analysis(
        kp1,
        matches,
        mask,
        ref_scale,
        reference.shape
    )

    if spatial["centroid"] is None:

        location = None

    else:

        location = spatial["centroid"]

    conf = calculate_confidence(
        len(matches),
        inliers,
        ratio,
        spatial["coverage"],
        spatial["dominance"],
        agreement,
        rmse
    )

    result = {
        "name": name,
        "model": model,
        "kp1": kp1,
        "kp2": kp2,
        "matches": matches,
        "mask": mask,
        "scale": ref_scale,
        "good": len(matches),
        "f_inliers": f_inliers,
        "h_inliers": h_inliers,
        "f_ratio": f_ratio,
        "h_ratio": h_ratio,
        "inliers": inliers,
        "ratio": ratio,
        "f_error": f_error,
        "h_rmse": h_rmse,
        "rmse": rmse,
        "agreement": agreement,
        "spatial": spatial,
        "location": location,
        "confidence": conf
    }

    results.append(result)

    print(
        "Homography inliers:",
        h_inliers
    )

    print(
        "Fundamental inliers:",
        f_inliers
    )

    print(
        "Selected model:",
        model
    )

    print(
        "Selected inliers:",
        inliers
    )

    print(
        "Inlier ratio:",
        f"{ratio:.2f}%"
    )

    print(
        "Spatial coverage:",
        f"{spatial['coverage']:.2f}%"
    )

    print(
        "Spatial dominance:",
        f"{spatial['dominance']:.2f}%"
    )

    if location:

        print(
            "Localization:",
            f"({location[0]:.2f}, "
            f"{location[1]:.2f})"
        )

    else:

        print(
            "Localization: INVALID"
        )

    print(
        "Confidence:",
        f"{conf:.2f}%"
    )


# ============================================================
# CHECK
# ============================================================

if not results:

    raise RuntimeError(
        "No usable feature correspondences found."
    )


# ============================================================
# CONSENSUS LOCALIZATION
# ============================================================

print("\n")
print("=" * 78)
print("CROSS-HYPOTHESIS CONSENSUS")
print("=" * 78)

valid = [
    r for r in results
    if r["location"] is not None
    and r["inliers"] >= 4
]

if len(valid) == 0:

    # Last-resort correspondence centroid
    fallback = max(
        results,
        key=lambda r: r["good"]
    )

    print(
        "\nNo geometrically verified location."
    )

    print(
        "Using raw correspondence centroid."
    )

    raw_mask = np.ones(
        len(fallback["matches"]),
        dtype=np.uint8
    )

    spatial = spatial_analysis(
        fallback["kp1"],
        fallback["matches"],
        raw_mask,
        fallback["scale"],
        reference.shape
    )

    final_x, final_y = (
        spatial["centroid"]
    )

    best = fallback

    consensus_score = 20.0

else:

    locations = np.array([
        r["location"]
        for r in valid
    ])

    weights = np.array([
        max(
            1.0,
            r["inliers"]
        ) *
        max(
            0.1,
            r["confidence"] / 100.0
        )
        for r in valid
    ])

    # --------------------------------------------------------
    # Robust center
    # --------------------------------------------------------

    median_x = np.median(
        locations[:, 0]
    )

    median_y = np.median(
        locations[:, 1]
    )

    distances = np.sqrt(
        (
            locations[:, 0] -
            median_x
        ) ** 2
        +
        (
            locations[:, 1] -
            median_y
        ) ** 2
    )

    # Accept hypotheses within a reasonable region of
    # the median estimate.
    adaptive_radius = max(
        180.0,
        min(
            500.0,
            np.median(distances) * 2.5
        )
    )

    consensus_mask = (
        distances <=
        adaptive_radius
    )

    if np.sum(consensus_mask) == 0:

        nearest = np.argmin(
            distances
        )

        consensus_mask[nearest] = True

    selected_locations = (
        locations[consensus_mask]
    )

    selected_weights = (
        weights[consensus_mask]
    )

    final_x = float(
        np.average(
            selected_locations[:, 0],
            weights=selected_weights
        )
    )

    final_y = float(
        np.average(
            selected_locations[:, 1],
            weights=selected_weights
        )
    )

    # Consensus percentage
    consensus_ratio = (
        np.sum(consensus_mask) /
        len(valid)
    )

    # How tightly clustered the accepted locations are
    if len(selected_locations) > 1:

        spread = np.mean(
            np.sqrt(
                (
                    selected_locations[:, 0] -
                    final_x
                ) ** 2
                +
                (
                    selected_locations[:, 1] -
                    final_y
                ) ** 2
            )
        )

        spread_signal = max(
            0.0,
            100.0 -
            min(
                100.0,
                spread / 5.0
            )
        )

    else:

        spread_signal = 30.0

    consensus_score = (
        0.60 *
        consensus_ratio *
        100.0
        +
        0.40 *
        spread_signal
    )

    best = max(
        valid,
        key=lambda r: r["confidence"]
    )

    print(
        "\nValid hypotheses:",
        len(valid)
    )

    print(
        "Consensus hypotheses:",
        int(
            np.sum(consensus_mask)
        )
    )

    print(
        "Consensus radius:",
        f"{adaptive_radius:.1f} px"
    )

    print(
        "Consensus location:",
        f"({final_x:.2f}, "
        f"{final_y:.2f})"
    )

    print(
        "Consensus score:",
        f"{consensus_score:.2f}%"
    )


# ============================================================
# CLAMP FINAL LOCATION
# ============================================================

final_x = float(
    np.clip(
        final_x,
        0,
        reference.shape[1] - 1
    )
)

final_y = float(
    np.clip(
        final_y,
        0,
        reference.shape[0] - 1
    )
)

normalized_x = (
    final_x /
    reference.shape[1]
) * 100.0

normalized_y = (
    final_y /
    reference.shape[0]
) * 100.0


# ============================================================
# FINAL CONFIDENCE
# ============================================================

base_confidence = best["confidence"]

if len(valid) > 0:

    final_confidence = (
        0.65 * base_confidence +
        0.35 * consensus_score
    )

else:

    final_confidence = (
        0.60 * base_confidence +
        0.40 * consensus_score
    )

final_confidence = float(
    np.clip(
        final_confidence,
        0,
        100
    )
)

if (
    final_confidence >= 75
    and best["inliers"] >= 12
):

    decision = "HIGH CONFIDENCE"

elif (
    final_confidence >= 50
    and best["inliers"] >= 8
):

    decision = "MODERATE CONFIDENCE"

else:

    decision = "LOW CONFIDENCE"


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 78)
print("FINAL CORRESPONDENCE REPORT")
print("=" * 78)

print(
    f"\n{'METHOD':<24}"
    f"{'MODEL':<18}"
    f"{'GOOD':>7}"
    f"{'H-IN':>8}"
    f"{'F-IN':>8}"
    f"{'CONF.':>11}"
)

print("-" * 78)

for r in results:

    print(
        f"{r['name']:<24}"
        f"{r['model']:<18}"
        f"{r['good']:>7}"
        f"{r['h_inliers']:>8}"
        f"{r['f_inliers']:>8}"
        f"{r['confidence']:>10.2f}%"
    )


print("\n")
print("=" * 78)
print("FINAL LOCALIZATION")
print("=" * 78)

print(
    "\nSelected evidence:",
    best["name"]
)

print(
    "Model:",
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
    "Best hypothesis confidence:",
    f"{best['confidence']:.2f}%"
)

print(
    "Consensus score:",
    f"{consensus_score:.2f}%"
)

print(
    "FINAL LOCATION:",
    f"({final_x:.2f}, "
    f"{final_y:.2f}) px"
)

print(
    "NORMALIZED:",
    f"({normalized_x:.2f}%, "
    f"{normalized_y:.2f}%)"
)

print(
    "\nFINAL CONFIDENCE:",
    f"{final_confidence:.2f}%"
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

# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

for i in range(
    1,
    GRID_COLS
):

    xg = int(
        i *
        width /
        GRID_COLS
    )

    cv2.line(
        visual,
        (xg, 0),
        (xg, height),
        (255, 255, 255),
        2
    )

for i in range(
    1,
    GRID_ROWS
):

    yg = int(
        i *
        height /
        GRID_ROWS
    )

    cv2.line(
        visual,
        (0, yg),
        (width, yg),
        (255, 255, 255),
        2
    )


# ------------------------------------------------------------
# Draw best hypothesis inliers
# ------------------------------------------------------------

for i, m in enumerate(
    best["matches"]
):

    if best["mask"][i] != 1:
        continue

    px, py = best[
        "kp1"
    ][m.queryIdx].pt

    px /= best["scale"]
    py /= best["scale"]

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
        10,
        (0, 255, 0),
        -1
    )


# ------------------------------------------------------------
# Final localization
# ------------------------------------------------------------

cv2.circle(
    visual,
    (
        int(final_x),
        int(final_y)
    ),
    45,
    (0, 0, 255),
    6
)

cv2.drawMarker(
    visual,
    (
        int(final_x),
        int(final_y)
    ),
    (0, 0, 255),
    cv2.MARKER_CROSS,
    110,
    7
)


# ------------------------------------------------------------
# Text
# ------------------------------------------------------------

cv2.putText(
    visual,
    "SIH26166 FINAL LOCALIZATION",
    (40, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    (255, 255, 255),
    3
)

cv2.putText(
    visual,
    f"Evidence: {best['name']}",
    (40, 105),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.75,
    (255, 255, 255),
    2
)

cv2.putText(
    visual,
    f"Model: {best['model']}",
    (40, 145),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.75,
    (255, 255, 255),
    2
)

cv2.putText(
    visual,
    f"Inliers: {best['inliers']}",
    (40, 185),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.75,
    (255, 255, 255),
    2
)

cv2.putText(
    visual,
    f"Confidence: {final_confidence:.1f}%",
    (40, 225),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.75,
    (255, 255, 255),
    2
)

cv2.putText(
    visual,
    f"Location: ({final_x:.0f}, {final_y:.0f})",
    (40, 265),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.75,
    (255, 255, 255),
    2
)

cv2.putText(
    visual,
    decision,
    (40, 305),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.85,
    (255, 255, 255),
    2
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
        "FINAL MULTI-SIGNAL CONSENSUS PIPELINE\n"
    )

    f.write(
        "=" * 78 +
        "\n\n"
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
        "-" * 78 +
        "\n\n"
    )

    for r in results:

        f.write(
            f"Method: {r['name']}\n"
        )

        f.write(
            f"Model: {r['model']}\n"
        )

        f.write(
            f"Good matches: {r['good']}\n"
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
            f"{r['inliers']}\n"
        )

        f.write(
            f"Inlier ratio: "
            f"{r['ratio']:.2f}%\n"
        )

        f.write(
            f"Spatial coverage: "
            f"{r['spatial']['coverage']:.2f}%\n"
        )

        f.write(
            f"Spatial dominance: "
            f"{r['spatial']['dominance']:.2f}%\n"
        )

        f.write(
            f"Model consistency: "
            f"{r['agreement']:.2f}%\n"
        )

        if r["location"]:

            f.write(
                f"Localization: "
                f"({r['location'][0]:.2f}, "
                f"{r['location'][1]:.2f})\n"
            )

        else:

            f.write(
                "Localization: INVALID\n"
            )

        f.write(
            f"Confidence: "
            f"{r['confidence']:.2f}%\n\n"
        )

    f.write(
        "=" * 78 +
        "\n"
    )

    f.write(
        "FINAL RESULT\n"
    )

    f.write(
        "=" * 78 +
        "\n\n"
    )

    f.write(
        f"Selected evidence: "
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
        f"{best['inliers']}\n"
    )

    f.write(
        f"Best hypothesis confidence: "
        f"{best['confidence']:.2f}%\n"
    )

    f.write(
        f"Consensus score: "
        f"{consensus_score:.2f}%\n"
    )

    f.write(
        f"Final X: "
        f"{final_x:.2f} px\n"
    )

    f.write(
        f"Final Y: "
        f"{final_y:.2f} px\n"
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
        f"{decision}\n"
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n")
print("=" * 78)
print("FINAL PIPELINE COMPLETED")
print("=" * 78)

print("\nVisualization:")
print(visual_path)

print("\nReport:")
print(report_path)

print("=" * 78)