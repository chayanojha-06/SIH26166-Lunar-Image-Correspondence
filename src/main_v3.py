import cv2
import numpy as np
import os
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ============================================================
# SIH26166 - V3 AI-ASSISTED LUNAR IMAGE CORRESPONDENCE SYSTEM
#
# V3 FEATURES
# ------------------------------------------------------------
# 1. Multi-hypothesis preprocessing
# 2. SIFT feature extraction
# 3. Lowe ratio matching
# 4. Mutual correspondence verification
# 5. Homography verification
# 6. Fundamental matrix verification
# 7. Spatial distribution analysis
# 8. Lightweight CPU ML confidence model
# 9. Multi-signal evidence fusion
# 10. Cross-hypothesis consensus localization
# 11. Visualization
# 12. Automatic text report
#
# IMPORTANT:
# The ML component scores correspondence reliability.
# Geometry remains the primary localization verifier.
#
# CPU-FIRST DESIGN
# No PyTorch
# No TensorFlow
# No CUDA
# No GPU requirement
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

LOWE_RATIO = 0.80

FUNDAMENTAL_THRESHOLD = 2.0
HOMOGRAPHY_THRESHOLD = 4.0

RANSAC_CONFIDENCE = 0.999
RANSAC_ITERATIONS = 5000

GRID_ROWS = 4
GRID_COLS = 4

MIN_MATCHES = 8
MIN_INLIERS = 6

# AI model configuration
AI_ESTIMATORS = 80
AI_RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("SIH26166 LUNAR IMAGE CORRESPONDENCE SYSTEM")
print("V3 AI-ASSISTED CPU-FIRST LOCALIZATION PIPELINE")
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
        edgeThreshold=10
    )

    keypoints, descriptors = (
        sift.detectAndCompute(
            gray,
            None
        )
    )

    return keypoints, descriptors


# ============================================================
# ONE-WAY MATCHING
# ============================================================

def ratio_matches(
    des1,
    des2
):

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
# MUTUAL MATCHING
# ============================================================

def mutual_matching(
    des1,
    des2
):

    forward = ratio_matches(
        des1,
        des2
    )

    backward = ratio_matches(
        des2,
        des1
    )

    if not forward or not backward:
        return []

    backward_pairs = set()

    for m in backward:

        backward_pairs.add(
            (
                m.trainIdx,
                m.queryIdx
            )
        )

    mutual = []

    for m in forward:

        if (
            m.queryIdx,
            m.trainIdx
        ) in backward_pairs:

            mutual.append(m)

    return mutual


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

    if mask is None:

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
        mask.ravel().astype(
            np.uint8
        )
    )


# ============================================================
# HOMOGRAPHY RMSE
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
            (len(p2), 1)
        )
    ])

    try:

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

    except Exception:

        return None


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
        ) /
        max(
            h_inliers,
            f_inliers
        )
    ) * 100.0


# ============================================================
# BASE GEOMETRIC CONFIDENCE
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

    match_signal = min(
        100.0,
        selected_inliers * 6.0
    )

    ratio_signal = min(
        100.0,
        selected_ratio
    )

    spatial_signal = (
        0.55 * coverage +
        0.45 * (
            100.0 -
            dominance
        )
    )

    agreement_signal = (
        model_consistency
    )

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

        0.20 *
        match_signal

        +

        0.25 *
        ratio_signal

        +

        0.20 *
        spatial_signal

        +

        0.15 *
        agreement_signal

        +

        0.20 *
        error_signal
    )

    return float(
        np.clip(
            score,
            0,
            100
        )
    )


# ============================================================
# AI FEATURE GENERATION
# ============================================================

def build_ai_features(
    result
):

    good = float(
        result["good"]
    )

    inliers = float(
        result["inliers"]
    )

    ratio = float(
        result["ratio"]
    )

    h_inliers = float(
        result["h_inliers"]
    )

    f_inliers = float(
        result["f_inliers"]
    )

    coverage = float(
        result["spatial"]["coverage"]
    )

    dominance = float(
        result["spatial"]["dominance"]
    )

    agreement = float(
        result["agreement"]
    )

    if result["f_error"] is None:

        f_error = 20.0

    else:

        f_error = min(
            20.0,
            float(
                result["f_error"]
            )
        )

    if result["h_rmse"] is None:

        h_rmse = 20.0

    else:

        h_rmse = min(
            20.0,
            float(
                result["h_rmse"]
            )
        )

    inlier_ratio = (
        inliers /
        max(
            good,
            1.0
        )
    ) * 100.0

    return np.array([

        np.log1p(good),

        np.log1p(inliers),

        ratio,

        inlier_ratio,

        np.log1p(h_inliers),

        np.log1p(f_inliers),

        coverage,

        dominance,

        agreement,

        f_error,

        h_rmse

    ], dtype=np.float32)


# ============================================================
# LIGHTWEIGHT AI MODEL
# ============================================================

def train_ai_model():

    # --------------------------------------------------------
    # Synthetic evidence training set
    #
    # This is an initialization model.
    # Positive samples represent internally consistent
    # correspondence evidence.
    #
    # Negative samples represent weak/inconsistent evidence.
    # --------------------------------------------------------

    rng = np.random.default_rng(
        AI_RANDOM_STATE
    )

    X = []
    y = []

    # Positive evidence
    for _ in range(400):

        good = rng.uniform(
            40,
            250
        )

        inliers = rng.uniform(
            12,
            min(
                50,
                good
            )
        )

        ratio = rng.uniform(
            8,
            35
        )

        h_in = rng.uniform(
            8,
            25
        )

        f_in = rng.uniform(
            8,
            25
        )

        coverage = rng.uniform(
            30,
            100
        )

        dominance = rng.uniform(
            15,
            45
        )

        agreement = rng.uniform(
            55,
            100
        )

        f_error = rng.uniform(
            0.1,
            3.0
        )

        h_rmse = rng.uniform(
            0.5,
            4.0
        )

        sample = [

            np.log1p(good),

            np.log1p(inliers),

            ratio,

            (
                inliers /
                max(
                    good,
                    1
                )
            ) * 100,

            np.log1p(h_in),

            np.log1p(f_in),

            coverage,

            dominance,

            agreement,

            f_error,

            h_rmse
        ]

        X.append(sample)
        y.append(1)

    # Negative evidence
    for _ in range(400):

        good = rng.uniform(
            10,
            220
        )

        inliers = rng.uniform(
            2,
            12
        )

        ratio = rng.uniform(
            1,
            18
        )

        h_in = rng.uniform(
            2,
            10
        )

        f_in = rng.uniform(
            2,
            13
        )

        coverage = rng.uniform(
            5,
            55
        )

        dominance = rng.uniform(
            35,
            100
        )

        agreement = rng.uniform(
            5,
            65
        )

        f_error = rng.uniform(
            3,
            20
        )

        h_rmse = rng.uniform(
            4,
            20
        )

        sample = [

            np.log1p(good),

            np.log1p(inliers),

            ratio,

            (
                inliers /
                max(
                    good,
                    1
                )
            ) * 100,

            np.log1p(h_in),

            np.log1p(f_in),

            coverage,

            dominance,

            agreement,

            f_error,

            h_rmse
        ]

        X.append(sample)
        y.append(0)

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int32
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    model = RandomForestClassifier(
        n_estimators=AI_ESTIMATORS,
        max_depth=8,
        min_samples_leaf=4,
        random_state=AI_RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_scaled,
        y
    )

    return model, scaler


print("\n")
print("=" * 78)
print("INITIALIZING CPU AI CONFIDENCE MODEL")
print("=" * 78)

ai_model, ai_scaler = (
    train_ai_model()
)

print(
    "AI model: Random Forest"
)

print(
    "Estimators:",
    AI_ESTIMATORS
)

print(
    "Execution: CPU"
)


# ============================================================
# RUN HYPOTHESES
# ============================================================

print("\n")
print("=" * 78)
print("MULTI-HYPOTHESIS ROBUST ANALYSIS")
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

    matches = mutual_matching(
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
        "Mutual good matches:",
        len(matches)
    )

    if len(matches) < MIN_MATCHES:

        print(
            "Rejected: insufficient matches"
        )

        continue

    # --------------------------------------------------------
    # GEOMETRIC MODELS
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
    # MODEL SELECTION
    # --------------------------------------------------------

    if (

        h_inliers >= 8

        and

        h_ratio >= 8

        and

        h_inliers >=
        0.70 * max(
            f_inliers,
            1
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

    elif h_inliers >= 6:

        model = "Homography"

        mask = hmask

        inliers = h_inliers

        ratio = h_ratio

        rmse = h_rmse

    else:

        print(
            "Rejected: weak geometry"
        )

        continue

    # --------------------------------------------------------
    # SPATIAL ANALYSIS
    # --------------------------------------------------------

    spatial = spatial_analysis(
        kp1,
        matches,
        mask,
        ref_scale,
        reference.shape
    )

    location = spatial[
        "centroid"
    ]

    if location is None:

        print(
            "Rejected: no valid localization"
        )

        continue

    # --------------------------------------------------------
    # RESULT OBJECT
    # --------------------------------------------------------

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

        "confidence": 0.0,

        "ai_confidence": 0.0,

        "final_score": 0.0
    }

    # --------------------------------------------------------
    # CLASSICAL CONFIDENCE
    # --------------------------------------------------------

    base_conf = calculate_confidence(

        len(matches),

        inliers,

        ratio,

        spatial[
            "coverage"
        ],

        spatial[
            "dominance"
        ],

        agreement,

        rmse
    )

    result[
        "confidence"
    ] = base_conf

    # --------------------------------------------------------
    # AI FEATURES
    # --------------------------------------------------------

    ai_features = build_ai_features(
        result
    )

    ai_scaled = ai_scaler.transform(
        ai_features.reshape(
            1,
            -1
        )
    )

    ai_probability = ai_model.predict_proba(
        ai_scaled
    )[0, 1] * 100.0

    result[
        "ai_confidence"
    ] = float(
        ai_probability
    )

    # --------------------------------------------------------
    # MULTI-SIGNAL FUSION
    # --------------------------------------------------------

    final_score = (

        0.55 *
        base_conf

        +

        0.45 *
        ai_probability
    )

    result[
        "final_score"
    ] = float(
        np.clip(
            final_score,
            0,
            100
        )
    )

    results.append(
        result
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

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

    print(
        "Classical confidence:",
        f"{base_conf:.2f}%"
    )

    print(
        "AI confidence:",
        f"{ai_probability:.2f}%"
    )

    print(
        "Fused confidence:",
        f"{result['final_score']:.2f}%"
    )

    print(
        "Localization:",
        f"({location[0]:.2f}, "
        f"{location[1]:.2f})"
    )


# ============================================================
# NO RESULTS CHECK
# ============================================================

if not results:

    print("\n")
    print("=" * 78)
    print("NO RELIABLE LOCALIZATION FOUND")
    print("=" * 78)

    print(
        "\nV3 could not obtain sufficient "
        "geometric evidence."
    )

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
            "SIH26166 V3 RESULT\n"
        )

        f.write(
            "NO RELIABLE LOCALIZATION FOUND\n"
        )

        f.write(
            "Insufficient geometric evidence.\n"
        )

    raise RuntimeError(
        "V3 failed: no reliable localization."
    )


# ============================================================
# CROSS-HYPOTHESIS CONSENSUS
# ============================================================

print("\n")
print("=" * 78)
print("AI + GEOMETRIC CROSS-HYPOTHESIS CONSENSUS")
print("=" * 78)

valid = [

    r for r in results

    if r["location"] is not None

    and r["inliers"] >= MIN_INLIERS

    and r["final_score"] >= 40
]


if not valid:

    best = max(
        results,
        key=lambda r:
        r["final_score"]
    )

    final_x, final_y = (
        best["location"]
    )

    consensus_score = (
        best["final_score"] *
        0.50
    )

    print(
        "\nNo strong consensus."
    )

    print(
        "Using best available hypothesis."
    )

else:

    locations = np.array([

        r["location"]

        for r in valid

    ])

    # --------------------------------------------------------
    # Robust median center
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

    adaptive_radius = max(

        180.0,

        min(

            550.0,

            np.median(
                distances
            ) * 2.5
        )
    )

    consensus_mask = (
        distances <=
        adaptive_radius
    )

    if np.sum(
        consensus_mask
    ) == 0:

        nearest = np.argmin(
            distances
        )

        consensus_mask[
            nearest
        ] = True

    selected_locations = (
        locations[
            consensus_mask
        ]
    )

    # --------------------------------------------------------
    # AI-weighted evidence
    # --------------------------------------------------------

    weights = np.array([

        max(
            1.0,
            r["inliers"]
        )

        *

        max(
            0.10,
            r["final_score"] /
            100.0
        )

        *

        max(
            0.10,
            r["ai_confidence"] /
            100.0
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

    consensus_ratio = (

        np.sum(
            consensus_mask
        )

        /

        len(valid)

    )

    if len(
        selected_locations
    ) > 1:

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

        key=lambda r:
        r["final_score"]
    )

    print(
        "\nValid hypotheses:",
        len(valid)
    )

    print(
        "Consensus hypotheses:",
        int(
            np.sum(
                consensus_mask
            )
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
# CLAMP LOCATION
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


# ============================================================
# NORMALIZED LOCATION
# ============================================================

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

base_confidence = (
    best["confidence"]
)

ai_confidence = (
    best["ai_confidence"]
)

final_confidence = (

    0.40 *
    base_confidence

    +

    0.30 *
    ai_confidence

    +

    0.30 *
    consensus_score
)

final_confidence = float(
    np.clip(
        final_confidence,
        0,
        100
    )
)


# ============================================================
# DECISION
# ============================================================

if (

    final_confidence >= 75

    and

    best["inliers"] >= 12

    and

    consensus_score >= 65

):

    decision = "HIGH CONFIDENCE"

elif (

    final_confidence >= 55

    and

    best["inliers"] >= 8

):

    decision = "MODERATE CONFIDENCE"

else:

    decision = "LOW CONFIDENCE"


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 78)
print("FINAL V3 CORRESPONDENCE REPORT")
print("=" * 78)

print(

    f"\n{'METHOD':<24}"
    f"{'MODEL':<18}"
    f"{'GOOD':>7}"
    f"{'H-IN':>8}"
    f"{'F-IN':>8}"
    f"{'AI':>10}"
    f"{'FUSED':>10}"
)

print("-" * 78)

for r in results:

    print(

        f"{r['name']:<24}"

        f"{r['model']:<18}"

        f"{r['good']:>7}"

        f"{r['h_inliers']:>8}"

        f"{r['f_inliers']:>8}"

        f"{r['ai_confidence']:>9.2f}%"

        f"{r['final_score']:>9.2f}%"
    )


print("\n")
print("=" * 78)
print("FINAL V3 LOCALIZATION")
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
    "Classical confidence:",
    f"{best['confidence']:.2f}%"
)

print(
    "AI confidence:",
    f"{best['ai_confidence']:.2f}%"
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
# GRID
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
# DRAW BEST INLIERS
# ------------------------------------------------------------

for i, m in enumerate(
    best["matches"]
):

    if best["mask"][i] != 1:
        continue

    px, py = best[
        "kp1"
    ][m.queryIdx].pt

    px /= best[
        "scale"
    ]

    py /= best[
        "scale"
    ]

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
# FINAL LOCATION
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
# TEXT
# ------------------------------------------------------------

font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(

    visual,

    "SIH26166 V3 AI LOCALIZATION",

    (40, 60),

    font,

    1.2,

    (255, 255, 255),

    3
)

cv2.putText(

    visual,

    f"Evidence: {best['name']}",

    (40, 105),

    font,

    0.75,

    (255, 255, 255),

    2
)

cv2.putText(

    visual,

    f"Model: {best['model']}",

    (40, 145),

    font,

    0.75,

    (255, 255, 255),

    2
)

cv2.putText(

    visual,

    f"Inliers: {best['inliers']}",

    (40, 185),

    font,

    0.75,

    (255, 255, 255),

    2
)

cv2.putText(

    visual,

    f"AI Confidence: {ai_confidence:.1f}%",

    (40, 225),

    font,

    0.75,

    (255, 255, 255),

    2
)

cv2.putText(

    visual,

    f"Final Confidence: {final_confidence:.1f}%",

    (40, 265),

    font,

    0.75,

    (255, 255, 255),

    2
)

cv2.putText(

    visual,

    f"Location: ({final_x:.0f}, {final_y:.0f})",

    (40, 305),

    font,

    0.75,

    (255, 255, 255),

    2
)

cv2.putText(

    visual,

    decision,

    (40, 345),

    font,

    0.85,

    (255, 255, 255),

    2
)


# ============================================================
# SAVE VISUALIZATION
# ============================================================

visual_path = os.path.join(

    VIS_DIR,

    "final_result_v3.jpg"
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

    "final_report_v3.txt"
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
        "V3 AI-ASSISTED CPU-FIRST LOCALIZATION\n"
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
        "AI MODEL\n"
    )

    f.write(
        "Random Forest classifier\n"
    )

    f.write(
        f"Estimators: {AI_ESTIMATORS}\n"
    )

    f.write(
        "Execution: CPU\n\n"
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

        f.write(
            f"Classical confidence: "
            f"{r['confidence']:.2f}%\n"
        )

        f.write(
            f"AI confidence: "
            f"{r['ai_confidence']:.2f}%\n"
        )

        f.write(
            f"Fused confidence: "
            f"{r['final_score']:.2f}%\n"
        )

        if r["location"]:

            f.write(
                f"Localization: "
                f"({r['location'][0]:.2f}, "
                f"{r['location'][1]:.2f})\n"
            )

        f.write("\n")


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
        f"Classical confidence: "
        f"{best['confidence']:.2f}%\n"
    )

    f.write(
        f"AI confidence: "
        f"{best['ai_confidence']:.2f}%\n"
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
print("V3 PIPELINE COMPLETED")
print("=" * 78)

print("\nVisualization:")
print(visual_path)

print("\nReport:")
print(report_path)

print("\nFinal location:")
print(
    f"({final_x:.2f}, {final_y:.2f}) px"
)

print(
    f"Normalized: "
    f"({normalized_x:.2f}%, "
    f"{normalized_y:.2f}%)"
)

print(
    f"AI confidence: "
    f"{ai_confidence:.2f}%"
)

print(
    f"Final confidence: "
    f"{final_confidence:.2f}%"
)

print(
    f"Decision: {decision}"
)

print("=" * 78)