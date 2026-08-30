import cv2
import numpy as np
import os
from datetime import datetime

# ============================================================
# SIH26166
# LUNAR IMAGE CORRESPONDENCE SYSTEM
# FINAL ROBUST MULTI-SIGNAL PIPELINE
#
# Features:
#   - SIFT multi-hypothesis matching
#   - CLAHE preprocessing
#   - Upscaling
#   - Lowe ratio matching
#   - Homography verification
#   - Fundamental matrix verification
#   - Spatial coverage analysis
#   - Cross-hypothesis consensus
#   - Robust localization
#   - Visualization
#   - Detailed report
#
# IMPORTANT:
# Homography is used for localization.
# Fundamental matrix is used as supporting geometric evidence.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

REFERENCE_IMAGE = "data/raw/image1.jpg"
TARGET_IMAGE = "data/raw/image2.jpg"

OUTPUT_DIR = "outputs"
VIS_DIR = os.path.join(
    OUTPUT_DIR,
    "visualizations"
)

REPORT_DIR = os.path.join(
    OUTPUT_DIR,
    "reports"
)

os.makedirs(
    VIS_DIR,
    exist_ok=True
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# ============================================================
# FEATURE CONFIG
# ============================================================

SIFT_FEATURES = 12000

LOWE_RATIO = 0.78

# Homography RANSAC threshold
HOMOGRAPHY_THRESHOLD = 5.0

# Fundamental matrix threshold
FUNDAMENTAL_THRESHOLD = 2.0

RANSAC_CONFIDENCE = 0.999

RANSAC_ITERATIONS = 10000


# ============================================================
# SPATIAL GRID
# ============================================================

GRID_ROWS = 4
GRID_COLS = 4


# ============================================================
# MINIMUM REQUIREMENTS
# ============================================================

MIN_HOMOGRAPHY_INLIERS = 6
MIN_FUNDAMENTAL_INLIERS = 8
MIN_GOOD_MATCHES = 8


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("SIH26166 LUNAR IMAGE CORRESPONDENCE SYSTEM")
print("FINAL ROBUST MULTI-SIGNAL PIPELINE")
print("=" * 78)


# ============================================================
# LOAD IMAGES
# ============================================================

reference = cv2.imread(
    REFERENCE_IMAGE
)

target = cv2.imread(
    TARGET_IMAGE
)


if reference is None:

    raise FileNotFoundError(
        f"Reference image not found:\n"
        f"{REFERENCE_IMAGE}"
    )


if target is None:

    raise FileNotFoundError(
        f"Target image not found:\n"
        f"{TARGET_IMAGE}"
    )


print("\nImages loaded successfully.")

print(
    "Reference:",
    reference.shape
)

print(
    "Target   :",
    target.shape
)


# ============================================================
# PREPROCESSING HYPOTHESES
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


reference_hypotheses = create_hypotheses(
    reference
)

target_hypotheses = create_hypotheses(
    target
)


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
        edgeThreshold=10,
        sigma=1.6
    )


    keypoints, descriptors = (
        sift.detectAndCompute(
            gray,
            None
        )
    )


    if descriptors is None:

        descriptors = None


    return (
        keypoints,
        descriptors
    )


# ============================================================
# FEATURE MATCHING
# ============================================================

def match_features(
    des1,
    des2
):

    if (
        des1 is None
        or des2 is None
    ):

        return []


    matcher = cv2.BFMatcher(
        cv2.NORM_L2
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


        if (
            m.distance
            <
            LOWE_RATIO * n.distance
        ):

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


    try:

        F, mask = cv2.findFundamentalMat(

            pts1,
            pts2,

            cv2.FM_RANSAC,

            FUNDAMENTAL_THRESHOLD,

            RANSAC_CONFIDENCE,

            RANSAC_ITERATIONS

        )

    except Exception:

        return (
            None,
            np.zeros(
                len(matches),
                dtype=np.uint8
            )
        )


    if F is None or mask is None:

        return (
            F,
            np.zeros(
                len(matches),
                dtype=np.uint8
            )
        )


    mask = mask.ravel().astype(
        np.uint8
    )


    if len(mask) != len(matches):

        mask = np.resize(
            mask,
            len(matches)
        )


    return (
        F,
        mask
    )


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


    try:

        H, mask = cv2.findHomography(

            pts1,
            pts2,

            cv2.RANSAC,

            HOMOGRAPHY_THRESHOLD,

            maxIters=RANSAC_ITERATIONS,

            confidence=RANSAC_CONFIDENCE

        )

    except Exception:

        return (
            None,
            np.zeros(
                len(matches),
                dtype=np.uint8
            )
        )


    if H is None or mask is None:

        return (
            H,
            np.zeros(
                len(matches),
                dtype=np.uint8
            )
        )


    return (
        H,
        mask.ravel().astype(
            np.uint8
        )
    )


# ============================================================
# HOMOGRAPHY REPROJECTION ERROR
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


    indices = np.where(
        mask == 1
    )[0]


    if len(indices) < 4:

        return None


    pts1 = np.float32([

        kp1[matches[i].queryIdx].pt

        for i in indices

    ]).reshape(
        -1,
        1,
        2
    )


    pts2 = np.float32([

        kp2[matches[i].trainIdx].pt

        for i in indices

    ])


    try:

        projected = cv2.perspectiveTransform(

            pts1,
            H

        ).reshape(
            -1,
            2
        )


        errors = (
            projected -
            pts2
        )


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
# FUNDAMENTAL MATRIX ERROR
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


    indices = np.where(
        mask == 1
    )[0]


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
            (len(p2), 1
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

        lines[:, 0] ** 2
        +
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

            "dominance": 100.0,

            "centroid": None,

            "points": np.empty(
                (0, 2),
                dtype=np.float32
            ),

            "grid": np.zeros(
                (
                    GRID_ROWS,
                    GRID_COLS
                ),
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

        (
            GRID_ROWS,
            GRID_COLS
        ),

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


        grid[
            row,
            col
        ] += 1


    occupied = np.sum(
        grid > 0
    )


    coverage = (

        occupied /
        (
            GRID_ROWS *
            GRID_COLS
        )

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

        "coverage": float(
            coverage
        ),

        "dominance": float(
            dominance
        ),

        "centroid": (

            float(
                centroid[0]
            ),

            float(
                centroid[1]
            )

        ),

        "points": points,

        "grid": grid
    }


# ============================================================
# HOMOGRAPHY LOCALIZATION
#
# IMPORTANT:
# A Fundamental matrix does NOT directly provide a
# point-to-point mapping.
#
# Therefore localization is obtained from the
# homography inlier centroid.
# ============================================================

def homography_localization(
    H,
    kp1,
    kp2,
    matches,
    hmask,
    scale,
    reference_shape
):

    if H is None:

        return None


    indices = np.where(
        hmask == 1
    )[0]


    if len(indices) < 4:

        return None


    # Target image center
    target_hypothesis_points = np.float32([

        [
            kp2[
                matches[i].trainIdx
            ].pt[0],

            kp2[
                matches[i].trainIdx
            ].pt[1]

        ]

        for i in indices

    ])


    target_center = np.mean(
        target_hypothesis_points,
        axis=0
    ).reshape(
        1,
        1,
        2
    )


    try:

        # H maps reference -> target.
        # Therefore use inverse H to map target -> reference.
        H_inv = np.linalg.inv(H)


        projected = cv2.perspectiveTransform(

            target_center,
            H_inv

        ).reshape(
            2
        )


        x = float(
            projected[0] /
            scale
        )


        y = float(
            projected[1] /
            scale
        )


        h, w = reference_shape[:2]


        if (

            not np.isfinite(x)
            or
            not np.isfinite(y)

        ):

            return None


        # Allow a small tolerance before rejection
        if (

            x < -0.25 * w
            or
            x > 1.25 * w
            or
            y < -0.25 * h
            or
            y > 1.25 * h

        ):

            return None


        x = float(
            np.clip(
                x,
                0,
                w - 1
            )
        )


        y = float(
            np.clip(
                y,
                0,
                h - 1
            )
        )


        return (
            x,
            y
        )


    except Exception:

        return None


# ============================================================
# MODEL CONSISTENCY
# ============================================================

def consistency(
    h_inliers,
    f_inliers
):

    if (
        h_inliers == 0
        or
        f_inliers == 0
    ):

        return 0.0


    return (

        min(
            h_inliers,
            f_inliers
        )
        /
        max(
            h_inliers,
            f_inliers
        )

    ) * 100.0


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    good,
    h_inliers,
    f_inliers,
    h_ratio,
    coverage,
    dominance,
    agreement,
    h_rmse
):

    if good <= 0:

        return 0.0


    # Match quantity
    match_signal = min(
        100.0,
        good / 2.0
    )


    # Homography quality
    inlier_signal = min(
        100.0,
        h_inliers * 8.0
    )


    ratio_signal = min(
        100.0,
        h_ratio * 2.0
    )


    spatial_signal = (

        0.60 * coverage
        +
        0.40 * min(
            dominance,
            60.0
        )

    )


    # Fundamental support
    fundamental_signal = min(
        100.0,
        f_inliers * 6.0
    )


    # Homography reprojection error
    if h_rmse is None:

        error_signal = 20.0

    else:

        error_signal = (

            100.0 *
            np.exp(
                -h_rmse / 5.0
            )

        )


        error_signal = float(
            np.clip(
                error_signal,
                0,
                100
            )
        )


    score = (

        0.10 * match_signal
        +
        0.20 * inlier_signal
        +
        0.20 * ratio_signal
        +
        0.15 * spatial_signal
        +
        0.10 * fundamental_signal
        +
        0.10 * agreement
        +
        0.15 * error_signal

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


    tar_img, tar_scale = (
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


    if len(matches) < MIN_GOOD_MATCHES:

        print(
            "Rejected: insufficient matches"
        )

        continue


    # --------------------------------------------------------
    # FUNDAMENTAL
    # --------------------------------------------------------

    F, fmask = calculate_fundamental(

        kp1,
        kp2,
        matches

    )


    # --------------------------------------------------------
    # HOMOGRAPHY
    # --------------------------------------------------------

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


    h_rmse = homography_rmse(

        H,
        kp1,
        kp2,
        matches,
        hmask

    )


    f_error = fundamental_error(

        F,
        kp1,
        kp2,
        matches,
        fmask

    )


    agreement = consistency(

        h_inliers,
        f_inliers

    )


    # --------------------------------------------------------
    # SPATIAL ANALYSIS
    # --------------------------------------------------------

    spatial = spatial_analysis(

        kp1,
        matches,
        hmask,
        ref_scale,
        reference.shape

    )


    # --------------------------------------------------------
    # LOCALIZATION
    # --------------------------------------------------------

    location = homography_localization(

        H,
        kp1,
        kp2,
        matches,
        hmask,
        ref_scale,
        reference.shape

    )


    # --------------------------------------------------------
    # MODEL SELECTION
    #
    # Fundamental is NOT used for localization.
    # --------------------------------------------------------

    if (

        H is not None
        and
        h_inliers >= MIN_HOMOGRAPHY_INLIERS

        and
        h_ratio >= 8.0

        and
        location is not None

    ):

        model = "Homography"


    elif (

        H is not None
        and
        h_inliers >= 4

        and
        location is not None

    ):

        model = "Homography (weak)"


    else:

        model = "No localization"


        location = None


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = calculate_confidence(

        len(matches),

        h_inliers,

        f_inliers,

        h_ratio,

        spatial["coverage"],

        spatial["dominance"],

        agreement,

        h_rmse

    )


    # Penalize invalid localization
    if location is None:

        confidence *= 0.50


    result = {

        "name": name,

        "model": model,

        "H": H,

        "F": F,

        "kp1": kp1,

        "kp2": kp2,

        "matches": matches,

        "hmask": hmask,

        "fmask": fmask,

        "mask": hmask,

        "scale": ref_scale,

        "good": len(matches),

        "h_inliers": h_inliers,

        "f_inliers": f_inliers,

        "h_ratio": h_ratio,

        "f_ratio": f_ratio,

        "h_rmse": h_rmse,

        "f_error": f_error,

        "agreement": agreement,

        "spatial": spatial,

        "location": location,

        "confidence": confidence

    }


    results.append(
        result
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
        "Selected model:",
        model
    )


    print(
        "Homography ratio:",
        f"{h_ratio:.2f}%"
    )


    print(
        "Fundamental ratio:",
        f"{f_ratio:.2f}%"
    )


    print(
        "Spatial coverage:",
        f"{spatial['coverage']:.2f}%"
    )


    print(
        "Spatial dominance:",
        f"{spatial['dominance']:.2f}%"
    )


    if h_rmse is not None:

        print(
            "Homography RMSE:",
            f"{h_rmse:.2f} px"
        )


    if location is not None:

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
        f"{confidence:.2f}%"
    )


# ============================================================
# CHECK
# ============================================================

if not results:

    raise RuntimeError(

        "No usable feature correspondences found."

    )


# ============================================================
# VALID LOCALIZATION HYPOTHESES
# ============================================================

valid = [

    r

    for r in results

    if (

        r["location"] is not None

        and

        r["h_inliers"] >= 4

    )

]


# ============================================================
# CROSS-HYPOTHESIS CONSENSUS
# ============================================================

print("\n")
print("=" * 78)
print("CROSS-HYPOTHESIS CONSENSUS")
print("=" * 78)


if len(valid) == 0:

    print(
        "\nNo valid homography localization found."
    )


    best = max(

        results,

        key=lambda r:
        r["confidence"]

    )


    final_x = None
    final_y = None

    consensus_score = 0.0

    consensus_mask = []


else:

    locations = np.array([

        r["location"]

        for r in valid

    ])


    # --------------------------------------------------------
    # Robust median
    # --------------------------------------------------------

    median_x = float(
        np.median(
            locations[:, 0]
        )
    )


    median_y = float(
        np.median(
            locations[:, 1]
        )
    )


    distances = np.sqrt(

        (
            locations[:, 0]
            -
            median_x
        ) ** 2

        +

        (
            locations[:, 1]
            -
            median_y
        ) ** 2

    )


    # --------------------------------------------------------
    # Adaptive consensus radius
    # --------------------------------------------------------

    if len(distances) > 1:

        mad = np.median(
            np.abs(
                distances -
                np.median(distances)
            )
        )


        adaptive_radius = max(

            150.0,

            min(

                400.0,

                np.median(distances)
                +
                3.0 * mad

            )

        )

    else:

        adaptive_radius = 150.0


    consensus_mask = (

        distances <=
        adaptive_radius

    )


    # Always keep closest hypothesis
    if not np.any(
        consensus_mask
    ):

        nearest = np.argmin(
            distances
        )

        consensus_mask[
            nearest
        ] = True


    selected = (

        locations[
            consensus_mask
        ]

    )


    # --------------------------------------------------------
    # Confidence weights
    # --------------------------------------------------------

    weights = np.array([

        max(
            1.0,
            r["h_inliers"]
        )
        *
        max(
            0.1,
            r["confidence"] / 100.0
        )

        for r in valid

    ])


    selected_weights = (

        weights[
            consensus_mask
        ]

    )


    final_x = float(

        np.average(

            selected[:, 0],

            weights=selected_weights

        )

    )


    final_y = float(

        np.average(

            selected[:, 1],

            weights=selected_weights

        )

    )


    consensus_ratio = (

        np.sum(consensus_mask)
        /
        len(valid)

    )


    # --------------------------------------------------------
    # Cluster spread
    # --------------------------------------------------------

    if len(selected) > 1:

        spread = float(

            np.mean(

                np.sqrt(

                    (
                        selected[:, 0]
                        -
                        final_x
                    ) ** 2

                    +

                    (
                        selected[:, 1]
                        -
                        final_y
                    ) ** 2

                )

            )

        )

    else:

        spread = 0.0


    spread_signal = max(

        0.0,

        100.0
        -
        min(
            100.0,
            spread / 4.0
        )

    )


    consensus_score = (

        0.65 *
        consensus_ratio *
        100.0

        +

        0.35 *
        spread_signal

    )


    best = max(

        valid,

        key=lambda r:
        r["confidence"]

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
        "Consensus spread:",
        f"{spread:.1f} px"
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
# FINAL LOCATION VALIDATION
# ============================================================

if final_x is not None:

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


else:

    normalized_x = None
    normalized_y = None


# ============================================================
# FINAL CONFIDENCE
# ============================================================

if final_x is None:

    final_confidence = 0.0

    decision = "NO RELIABLE LOCALIZATION"


else:

    base_confidence = (
        best["confidence"]
    )


    final_confidence = (

        0.55 *
        base_confidence

        +

        0.45 *
        consensus_score

    )


    final_confidence = float(

        np.clip(

            final_confidence,

            0,

            100

        )

    )


    # --------------------------------------------------------
    # Confidence decision
    # --------------------------------------------------------

    if (

        final_confidence >= 75

        and

        best["h_inliers"] >= 12

        and

        consensus_score >= 70

    ):

        decision = "HIGH CONFIDENCE"


    elif (

        final_confidence >= 55

        and

        best["h_inliers"] >= 8

        and

        consensus_score >= 60

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
    f"{'MODEL':<22}"
    f"{'GOOD':>7}"
    f"{'H-IN':>8}"
    f"{'F-IN':>8}"
    f"{'CONF.':>11}"

)


print(
    "-" * 78
)


for r in results:

    print(

        f"{r['name']:<24}"
        f"{r['model']:<22}"
        f"{r['good']:>7}"
        f"{r['h_inliers']:>8}"
        f"{r['f_inliers']:>8}"
        f"{r['confidence']:>10.2f}%"

    )


# ============================================================
# FINAL LOCALIZATION
# ============================================================

print("\n")
print("=" * 78)
print("FINAL LOCALIZATION")
print("=" * 78)


if final_x is not None:

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
        "Verified homography inliers:",
        best["h_inliers"]
    )


    print(
        "Supporting fundamental inliers:",
        best["f_inliers"]
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


else:

    print(
        "\nFINAL LOCATION: INVALID"
    )


    print(
        "DECISION:",
        decision
    )


# ============================================================
# VISUALIZATION
# ============================================================

visual = reference.copy()


height, width = (
    reference.shape[:2]
)


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
# Draw BEST HOMOGRAPHY INLIERS
# ------------------------------------------------------------

if final_x is not None:

    for i, m in enumerate(

        best["matches"]

    ):

        if best["hmask"][i] != 1:

            continue


        px, py = best[
            "kp1"
        ][
            m.queryIdx
        ].pt


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

            9,

            (0, 255, 0),

            -1

        )


# ------------------------------------------------------------
# Final localization
# ------------------------------------------------------------

if final_x is not None:

    fx = int(final_x)
    fy = int(final_y)


    # Outer circle
    cv2.circle(

        visual,

        (fx, fy),

        55,

        (0, 0, 255),

        6

    )


    # Crosshair
    cv2.drawMarker(

        visual,

        (fx, fy),

        (0, 0, 255),

        cv2.MARKER_CROSS,

        120,

        7

    )


    # Center dot
    cv2.circle(

        visual,

        (fx, fy),

        10,

        (0, 0, 255),

        -1

    )


# ============================================================
# VISUALIZATION TEXT
# ============================================================

cv2.putText(

    visual,

    "SIH26166 FINAL LOCALIZATION",

    (40, 60),

    cv2.FONT_HERSHEY_SIMPLEX,

    1.2,

    (255, 255, 255),

    3

)


if final_x is not None:

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

        f"H-Inliers: {best['h_inliers']}",

        (40, 185),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (255, 255, 255),

        2

    )


    cv2.putText(

        visual,

        f"F-Inliers: {best['f_inliers']}",

        (40, 225),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (255, 255, 255),

        2

    )


    cv2.putText(

        visual,

        f"Consensus: {consensus_score:.1f}%",

        (40, 265),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (255, 255, 255),

        2

    )


    cv2.putText(

        visual,

        f"Confidence: {final_confidence:.1f}%",

        (40, 305),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (255, 255, 255),

        2

    )


    cv2.putText(

        visual,

        f"Location: ({final_x:.0f}, {final_y:.0f})",

        (40, 345),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (255, 255, 255),

        2

    )


    cv2.putText(

        visual,

        decision,

        (40, 390),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.85,

        (255, 255, 255),

        2

    )


else:

    cv2.putText(

        visual,

        "NO RELIABLE LOCALIZATION",

        (40, 110),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.0,

        (255, 255, 255),

        3

    )


# ============================================================
# SAVE VISUALIZATION
# ============================================================

visual_path = os.path.join(

    VIS_DIR,

    "final_result.jpg"

)


success = cv2.imwrite(

    visual_path,

    visual

)


if not success:

    print(
        "WARNING: visualization could not be saved."
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
        "FINAL ROBUST MULTI-SIGNAL PIPELINE\n"
    )

    f.write(
        "=" * 78 +
        "\n\n"
    )


    f.write(

        f"Generated: "
        f"{datetime.now()}\n\n"

    )


    f.write(

        f"Reference: "
        f"{REFERENCE_IMAGE}\n"

    )


    f.write(

        f"Target: "
        f"{TARGET_IMAGE}\n\n"

    )


    # --------------------------------------------------------
    # Hypothesis results
    # --------------------------------------------------------

    f.write(
        "HYPOTHESIS RESULTS\n"
    )

    f.write(
        "-" * 78 +
        "\n\n"
    )


    for r in results:

        f.write(

            f"Method: "
            f"{r['name']}\n"

        )


        f.write(

            f"Model: "
            f"{r['model']}\n"

        )


        f.write(

            f"Good matches: "
            f"{r['good']}\n"

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

            f"Homography ratio: "
            f"{r['h_ratio']:.2f}%\n"

        )


        f.write(

            f"Fundamental ratio: "
            f"{r['f_ratio']:.2f}%\n"

        )


        f.write(

            f"Homography RMSE: "
            f"{r['h_rmse']}\n"

        )


        f.write(

            f"Fundamental error: "
            f"{r['f_error']}\n"

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


        if r["location"] is not None:

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


    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

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


    if final_x is not None:

        f.write(

            f"Selected evidence: "
            f"{best['name']}\n"

        )


        f.write(

            f"Localization model: "
            f"{best['model']}\n"

        )


        f.write(

            f"Good matches: "
            f"{best['good']}\n"

        )


        f.write(

            f"Verified homography inliers: "
            f"{best['h_inliers']}\n"

        )


        f.write(

            f"Supporting fundamental inliers: "
            f"{best['f_inliers']}\n"

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

    else:

        f.write(
            "Final location: INVALID\n"
        )

        f.write(
            f"Decision: {decision}\n"
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