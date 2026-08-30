import os
import pickle
import cv2
import numpy as np
import pandas as pd

# ==============================================================
# SIH26166 FINAL AI-ASSISTED LUNAR LOCALIZATION
# ==============================================================
# Existing model:
# models/lunar_correspondence_rf_v45.pkl
#
# Existing images:
# data/images/
#
# Outputs:
# outputs/visualizations/
# outputs/reports/
# ==============================================================

print("=" * 78)
print("SIH26166 FINAL AI-ASSISTED LUNAR LOCALIZATION")
print("=" * 78)

# ==============================================================
# PATHS
# ==============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "lunar_correspondence_rf_v45.pkl"
)

VIS_DIR = os.path.join(BASE_DIR, "outputs", "visualizations")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ==============================================================
# FEATURE SCHEMA
# MUST MATCH V4.5 MODEL
# ==============================================================

FEATURE_COLUMNS = [
    "max_distance",
    "median_distance",
    "mean_distance",
    "fundamental_error",
    "spatial_coverage",
    "match_count",
    "fundamental_ratio",
    "min_distance",
    "homography_ratio",
    "homography_inliers",
    "homography_error",
    "fundamental_inliers"
]

# ==============================================================
# LOAD AI MODEL
# ==============================================================

print("\n" + "=" * 78)
print("LOADING AI MODEL")
print("=" * 78)

with open(MODEL_PATH, "rb") as f:
    package = pickle.load(f)

if isinstance(package, dict):
    model = package["model"]
    stored_features = package.get("feature_columns", FEATURE_COLUMNS)
else:
    model = package
    stored_features = FEATURE_COLUMNS

if list(stored_features) != FEATURE_COLUMNS:
    raise ValueError("Model feature schema mismatch.")

print("Classifier:", type(model).__name__)
print("Feature schema verified: 12 / 12")
print("Execution: CPU")

# ==============================================================
# LOAD IMAGES
# ==============================================================

print("\n" + "=" * 78)
print("AVAILABLE LUNAR IMAGES")
print("=" * 78)

image_files = sorted(
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

if len(image_files) < 2:
    raise SystemExit("At least two images are required.")

for f in image_files:
    print(" ", f)

# ==============================================================
# SIFT
# ==============================================================

print("\n" + "=" * 78)
print("INITIALIZING SIFT")
print("=" * 78)

MAX_FEATURES = 12000

sift = cv2.SIFT_create(
    nfeatures=MAX_FEATURES,
    contrastThreshold=0.015,
    edgeThreshold=10,
    sigma=1.6
)

print("SIFT initialized.")
print("Maximum features:", MAX_FEATURES)

images = {}
keypoints = {}
descriptors = {}

for filename in image_files:

    path = os.path.join(IMAGE_DIR, filename)

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("WARNING: Cannot read", filename)
        continue

    kp, des = sift.detectAndCompute(image, None)

    images[filename] = image
    keypoints[filename] = kp
    descriptors[filename] = des

    print(
        f"{filename}: {image.shape} | "
        f"{len(kp)} SIFT features"
    )

# ==============================================================
# MATCHER
# ==============================================================

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, float(x)))


# ==============================================================
# PROCESS PAIR
# ==============================================================

def process_pair(target_name, reference_name):

    target = images[target_name]
    reference = images[reference_name]

    kp1 = keypoints[target_name]
    kp2 = keypoints[reference_name]

    des1 = descriptors[target_name]
    des2 = descriptors[reference_name]

    if des1 is None or des2 is None:
        return None

    # ----------------------------------------------------------
    # KNN MATCH
    # ----------------------------------------------------------

    try:
        knn = bf.knnMatch(des1, des2, k=2)
    except Exception:
        return None

    # ----------------------------------------------------------
    # RATIO TEST
    # Slightly relaxed for real lunar imagery
    # ----------------------------------------------------------

    good = []

    for pair in knn:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    good.sort(key=lambda m: m.distance)

    # Limit extreme outliers
    good = good[:300]

    if len(good) < 4:

        return {
            "target": target_name,
            "reference": reference_name,
            "matches": len(good),
            "F_inliers": 0,
            "H_inliers": 0,
            "F_ratio": 0,
            "H_ratio": 0,
            "F_error": 999,
            "H_error": 999,
            "coverage": 0,
            "ai": 0,
            "geometry": 0,
            "localization": 0,
            "score": 0,
            "location": None,
            "decision": "REJECT / INSUFFICIENT EVIDENCE",
            "good": good,
            "H": None,
            "H_mask": None,
            "F_mask": None,
            "kp1": kp1,
            "kp2": kp2
        }

    # ----------------------------------------------------------
    # POINTS
    # ----------------------------------------------------------

    src = np.float32(
        [kp1[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    dst = np.float32(
        [kp2[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    distances = np.array(
        [m.distance for m in good],
        dtype=np.float32
    )

    # ----------------------------------------------------------
    # FUNDAMENTAL MATRIX
    # ----------------------------------------------------------

    F = None
    F_mask = None

    if len(good) >= 8:

        try:

            F, F_mask = cv2.findFundamentalMat(
                src,
                dst,
                cv2.FM_RANSAC,
                2.0,
                0.995
            )

        except Exception:
            F = None
            F_mask = None

    if F_mask is not None and F_mask.size == len(good):
        F_mask = F_mask.ravel().astype(bool)
    else:
        F_mask = np.zeros(len(good), dtype=bool)

    F_inliers = int(F_mask.sum())
    F_ratio = F_inliers / len(good)

    # ----------------------------------------------------------
    # FUNDAMENTAL ERROR
    # ----------------------------------------------------------

    F_error = 999.0

    if F is not None:

        try:

            p1 = src.reshape(-1, 2)
            p2 = dst.reshape(-1, 2)

            lines = cv2.computeCorrespondEpilines(
                p1.reshape(-1, 1, 2),
                1,
                F
            ).reshape(-1, 3)

            numerators = np.abs(
                lines[:, 0] * p2[:, 0]
                + lines[:, 1] * p2[:, 1]
                + lines[:, 2]
            )

            denominators = np.sqrt(
                lines[:, 0] ** 2
                + lines[:, 1] ** 2
            ) + 1e-8

            errors = numerators / denominators

            if F_mask.any():
                F_error = float(
                    np.median(errors[F_mask])
                )
            else:
                F_error = float(np.median(errors))

        except Exception:
            F_error = 999.0

    # ----------------------------------------------------------
    # HOMOGRAPHY
    # ----------------------------------------------------------

    H = None
    H_mask = None

    try:

        H, H_mask = cv2.findHomography(
            src,
            dst,
            cv2.RANSAC,
            5.0,
            maxIters=5000,
            confidence=0.995
        )

    except Exception:
        H = None
        H_mask = None

    if H_mask is not None and H_mask.size == len(good):
        H_mask = H_mask.ravel().astype(bool)
    else:
        H_mask = np.zeros(len(good), dtype=bool)

    H_inliers = int(H_mask.sum())
    H_ratio = H_inliers / len(good)

    # ----------------------------------------------------------
    # HOMOGRAPHY ERROR
    # Calculate symmetric transfer error
    # ----------------------------------------------------------

    H_error = 999.0

    if H is not None and H_inliers > 0:

        try:

            projected = cv2.perspectiveTransform(
                src,
                H
            )

            forward_error = np.linalg.norm(
                projected - dst,
                axis=2
            ).ravel()

            H_inv = np.linalg.inv(H)

            back_projected = cv2.perspectiveTransform(
                dst,
                H_inv
            )

            backward_error = np.linalg.norm(
                back_projected - src,
                axis=2
            ).ravel()

            symmetric_error = (
                forward_error + backward_error
            ) / 2.0

            H_error = float(
                np.median(
                    symmetric_error[H_mask]
                )
            )

        except Exception:
            H_error = 999.0

    # ----------------------------------------------------------
    # SPATIAL COVERAGE
    # ----------------------------------------------------------

    pts = src.reshape(-1, 2)

    x_range = np.ptp(pts[:, 0])
    y_range = np.ptp(pts[:, 1])

    coverage = 100.0 * (
        0.5 * x_range / max(target.shape[1], 1)
        +
        0.5 * y_range / max(target.shape[0], 1)
    )

    coverage = clamp(coverage)

    # ----------------------------------------------------------
    # FEATURES FOR AI
    # ----------------------------------------------------------

    feature_values = {
        "max_distance": float(np.max(distances)),
        "median_distance": float(np.median(distances)),
        "mean_distance": float(np.mean(distances)),
        "fundamental_error": F_error,
        "spatial_coverage": coverage,
        "match_count": len(good),
        "fundamental_ratio": F_ratio,
        "min_distance": float(np.min(distances)),
        "homography_ratio": H_ratio,
        "homography_inliers": H_inliers,
        "homography_error": H_error,
        "fundamental_inliers": F_inliers
    }

    feature_df = pd.DataFrame(
        [[feature_values[c] for c in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    # ----------------------------------------------------------
    # AI
    # ----------------------------------------------------------

    try:

        prediction = int(model.predict(feature_df)[0])

        probabilities = model.predict_proba(feature_df)[0]

        ai_conf = float(np.max(probabilities) * 100)

    except Exception as e:

        print("AI error:", e)

        prediction = 0
        ai_conf = 0.0

    # ----------------------------------------------------------
    # GEOMETRY SCORE
    # ----------------------------------------------------------

    f_count = clamp(F_inliers / 12 * 100)
    f_ratio = clamp(F_ratio * 100)

    if F_error < 999:
        f_error_score = clamp(
            100 * np.exp(-F_error / 4)
        )
    else:
        f_error_score = 0

    h_count = clamp(H_inliers / 8 * 100)
    h_ratio = clamp(H_ratio * 100)

    if H_error < 999:
        h_error_score = clamp(
            100 * np.exp(-H_error / 8)
        )
    else:
        h_error_score = 0

    fundamental_score = (
        0.45 * f_count
        + 0.35 * f_ratio
        + 0.20 * f_error_score
    )

    homography_score = (
        0.45 * h_count
        + 0.35 * h_ratio
        + 0.20 * h_error_score
    )

    geometry = (
        0.50 * fundamental_score
        + 0.35 * homography_score
        + 0.15 * coverage
    )

    # Strong penalty when geometry is genuinely weak
    if F_inliers < 5:
        geometry *= 0.60

    if H_inliers < 4:
        geometry *= 0.65

    if coverage < 30:
        geometry *= 0.70

    geometry = clamp(geometry)

    # ----------------------------------------------------------
    # LOCALIZATION
    # ----------------------------------------------------------

    location = None
    localization = 0.0

    # We need actual geometric agreement.
    if (
        H is not None
        and F_inliers >= 6
        and H_inliers >= 4
        and F_ratio >= 0.25
        and H_ratio >= 0.20
        and H_error <= 15
    ):

        try:

            h, w = target.shape

            center = np.float32(
                [[[w / 2.0, h / 2.0]]]
            )

            mapped = cv2.perspectiveTransform(
                center,
                H
            )

            x = float(mapped[0, 0, 0])
            y = float(mapped[0, 0, 1])

            rh, rw = reference.shape

            if (
                0 <= x < rw
                and 0 <= y < rh
            ):

                location = (x, y)

                ratio_score = (
                    0.5 * clamp(F_ratio * 100)
                    +
                    0.5 * clamp(H_ratio * 100)
                )

                inlier_score = (
                    0.55 * clamp(F_inliers / 12 * 100)
                    +
                    0.45 * clamp(H_inliers / 8 * 100)
                )

                error_score = clamp(
                    100 * np.exp(-H_error / 10)
                )

                localization = clamp(
                    0.40 * ratio_score
                    + 0.35 * inlier_score
                    + 0.15 * error_score
                    + 0.10 * coverage
                )

        except Exception:
            location = None

    # ----------------------------------------------------------
    # FINAL SCORE
    # ----------------------------------------------------------

    # AI = supporting evidence.
    # Geometry + localization = primary evidence.

    score = (
        0.20 * ai_conf
        + 0.45 * geometry
        + 0.35 * localization
    )

    # AI cannot rescue failed geometry.
    if F_inliers < 5 or H_inliers < 4:
        score *= 0.65

    # ----------------------------------------------------------
    # DECISION
    # ----------------------------------------------------------

    if (
        score >= 72
        and ai_conf >= 75
        and geometry >= 55
        and localization >= 55
        and F_inliers >= 7
        and H_inliers >= 4
    ):

        decision = "ACCEPT / STRONG CORRESPONDENCE"

    elif (
        score >= 60
        and geometry >= 45
        and localization >= 40
        and F_inliers >= 6
        and H_inliers >= 4
    ):

        decision = "ACCEPT / MODERATE CORRESPONDENCE"

    elif (
        score >= 45
        and geometry >= 35
        and F_inliers >= 5
        and H_inliers >= 4
    ):

        decision = "REVIEW / POSSIBLE CORRESPONDENCE"

    else:

        decision = "REJECT / INSUFFICIENT EVIDENCE"

    return {
        "target": target_name,
        "reference": reference_name,
        "matches": len(good),
        "F_inliers": F_inliers,
        "H_inliers": H_inliers,
        "F_ratio": F_ratio,
        "H_ratio": H_ratio,
        "F_error": F_error,
        "H_error": H_error,
        "coverage": coverage,
        "ai": ai_conf,
        "prediction": prediction,
        "geometry": geometry,
        "localization": localization,
        "score": clamp(score),
        "location": location,
        "decision": decision,
        "good": good,
        "H": H,
        "H_mask": H_mask,
        "F_mask": F_mask,
        "kp1": kp1,
        "kp2": kp2
    }


# ==============================================================
# RUN ALL PAIRS
# ==============================================================

print("\n" + "=" * 78)
print("FINAL REAL IMAGE-TO-IMAGE CORRESPONDENCE")
print("=" * 78)

results = []

for target in image_files:

    for reference in image_files:

        if target == reference:
            continue

        print(
            f"\nTARGET: {target} -> "
            f"REFERENCE: {reference}"
        )

        result = process_pair(target, reference)

        if result is None:
            continue

        results.append(result)

        print(f"Good matches       : {result['matches']}")
        print(f"Homography inliers : {result['H_inliers']}")
        print(f"Fundamental inliers: {result['F_inliers']}")
        print(f"Spatial coverage   : {result['coverage']:.2f}%")
        print(
            f"AI prediction      : "
            f"{'POSITIVE' if result['prediction'] else 'NEGATIVE'}"
        )
        print(f"AI confidence      : {result['ai']:.2f}%")
        print(f"Geometry score     : {result['geometry']:.2f}%")
        print(
            f"Localization conf. : "
            f"{result['localization']:.2f}%"
        )
        print(f"Final score        : {result['score']:.2f}%")

        if result["location"]:

            x, y = result["location"]

            print(
                f"Localization       : "
                f"({x:.2f}, {y:.2f}) px"
            )

        else:

            print(
                "Localization       : NOT RELIABLE"
            )

        print(
            "Decision           :",
            result["decision"]
        )

# ==============================================================
# RANK
# ==============================================================

results.sort(
    key=lambda r: r["score"],
    reverse=True
)

print("\n" + "=" * 78)
print("FINAL CORRESPONDENCE RANKING")
print("=" * 78)

for i, r in enumerate(results, 1):

    print(
        f"\n{i}. {r['target']} -> {r['reference']}"
    )

    print(f"   AI confidence : {r['ai']:.2f}%")
    print(f"   Geometry      : {r['geometry']:.2f}%")
    print(f"   Localization  : {r['localization']:.2f}%")
    print(f"   Matches       : {r['matches']}")
    print(f"   H-inliers     : {r['H_inliers']}")
    print(f"   F-inliers     : {r['F_inliers']}")
    print(f"   Coverage      : {r['coverage']:.2f}%")
    print(f"   Final score   : {r['score']:.2f}%")
    print(f"   Decision      : {r['decision']}")

# ==============================================================
# BEST RESULT
# ==============================================================

best = results[0]

print("\n" + "=" * 78)
print("FINAL LUNAR LOCALIZATION")
print("=" * 78)

print("\nBEST CORRESPONDENCE:")
print(
    f"{best['target']} -> {best['reference']}"
)

print(f"AI confidence      : {best['ai']:.2f}%")
print(f"Geometry score     : {best['geometry']:.2f}%")
print(
    f"Localization conf. : "
    f"{best['localization']:.2f}%"
)
print(f"Good matches       : {best['matches']}")
print(f"Fundamental inliers: {best['F_inliers']}")
print(f"Homography inliers : {best['H_inliers']}")
print(f"Homography error   : {best['H_error']:.3f} px")
print(f"Spatial coverage   : {best['coverage']:.2f}%")
print(f"FINAL SCORE        : {best['score']:.2f}%")
print("DECISION           :", best["decision"])

# ==============================================================
# LOCALIZATION
# ==============================================================

reference_image = images[best["reference"]]
target_image = images[best["target"]]

if best["location"] is not None:

    x, y = best["location"]

    rh, rw = reference_image.shape

    nx = x / rw * 100
    ny = y / rh * 100

    print("\n" + "=" * 78)
    print("LOCALIZED POSITION")
    print("=" * 78)

    print(f"X : {x:.2f} px")
    print(f"Y : {y:.2f} px")
    print(f"Normalized X : {nx:.2f}%")
    print(f"Normalized Y : {ny:.2f}%")

# ==============================================================
# VISUALIZATION
# ==============================================================

matches = best["good"]
H_mask = best["H_mask"]
F_mask = best["F_mask"]

inlier_matches = []

# Prefer homography inliers
for i, m in enumerate(matches):

    if (
        H_mask is not None
        and i < len(H_mask)
        and H_mask[i]
    ):
        inlier_matches.append(m)

# If no H inliers, use F inliers
if not inlier_matches:

    for i, m in enumerate(matches):

        if (
            F_mask is not None
            and i < len(F_mask)
            and F_mask[i]
        ):
            inlier_matches.append(m)

if inlier_matches:

    vis = cv2.drawMatches(
        target_image,
        best["kp1"],
        reference_image,
        best["kp2"],
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

else:

    vis = cv2.hconcat(
        [target_image, reference_image]
    )

# --------------------------------------------------------------
# DRAW LOCATION
# --------------------------------------------------------------

if best["location"] is not None:

    x, y = best["location"]

    offset = target_image.shape[1]

    px = int(offset + x)
    py = int(y)

    cv2.circle(
        vis,
        (px, py),
        18,
        (0, 0, 255),
        4
    )

    cv2.drawMarker(
        vis,
        (px, py),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        40,
        4
    )

    cv2.putText(
        vis,
        "LOCALIZED",
        (px + 20, py - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )

# ==============================================================
# SAVE VISUALIZATION
# ==============================================================

visualization_path = os.path.join(
    VIS_DIR,
    "FINAL_LUNAR_LOCALIZATION.jpg"
)

cv2.imwrite(
    visualization_path,
    vis
)

print("\nVisualization saved:")
print(visualization_path)

# ==============================================================
# CSV
# ==============================================================

csv_path = os.path.join(
    REPORT_DIR,
    "FINAL_CORRESPONDENCE_RESULTS.csv"
)

csv_rows = []

for r in results:

    row = {
        "target": r["target"],
        "reference": r["reference"],
        "matches": r["matches"],
        "fundamental_inliers": r["F_inliers"],
        "homography_inliers": r["H_inliers"],
        "fundamental_ratio": r["F_ratio"],
        "homography_ratio": r["H_ratio"],
        "fundamental_error": r["F_error"],
        "homography_error": r["H_error"],
        "spatial_coverage": r["coverage"],
        "ai_confidence": r["ai"],
        "geometry_score": r["geometry"],
        "localization_confidence": r["localization"],
        "final_score": r["score"],
        "decision": r["decision"]
    }

    if r["location"]:

        row["x"] = r["location"][0]
        row["y"] = r["location"][1]

    else:

        row["x"] = None
        row["y"] = None

    csv_rows.append(row)

pd.DataFrame(csv_rows).to_csv(
    csv_path,
    index=False
)

# ==============================================================
# REPORT
# ==============================================================

report_path = os.path.join(
    REPORT_DIR,
    "FINAL_LOCALIZATION_REPORT.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SIH26166 FINAL AI-ASSISTED LUNAR LOCALIZATION\n"
    )

    f.write("=" * 78 + "\n\n")

    f.write("BEST CORRESPONDENCE\n")
    f.write(
        f"{best['target']} -> "
        f"{best['reference']}\n\n"
    )

    f.write(
        f"AI confidence      : {best['ai']:.2f}%\n"
    )

    f.write(
        f"Geometry score     : {best['geometry']:.2f}%\n"
    )

    f.write(
        f"Localization conf. : "
        f"{best['localization']:.2f}%\n"
    )

    f.write(
        f"Good matches       : {best['matches']}\n"
    )

    f.write(
        f"Fundamental inliers: {best['F_inliers']}\n"
    )

    f.write(
        f"Homography inliers : {best['H_inliers']}\n"
    )

    f.write(
        f"Homography error   : "
        f"{best['H_error']:.3f} px\n"
    )

    f.write(
        f"Spatial coverage   : "
        f"{best['coverage']:.2f}%\n"
    )

    f.write(
        f"FINAL SCORE        : "
        f"{best['score']:.2f}%\n"
    )

    f.write(
        f"DECISION           : "
        f"{best['decision']}\n\n"
    )

    if best["location"]:

        x, y = best["location"]

        rh, rw = reference_image.shape

        f.write("LOCALIZED POSITION\n")
        f.write(f"X : {x:.2f} px\n")
        f.write(f"Y : {y:.2f} px\n")
        f.write(
            f"Normalized X : "
            f"{x / rw * 100:.2f}%\n"
        )
        f.write(
            f"Normalized Y : "
            f"{y / rh * 100:.2f}%\n"
        )

    f.write("\n\nALL CORRESPONDENCES\n")
    f.write("=" * 78 + "\n")

    for i, r in enumerate(results, 1):

        f.write(
            f"\n{i}. "
            f"{r['target']} -> "
            f"{r['reference']}\n"
        )

        f.write(
            f"Matches: {r['matches']}\n"
        )

        f.write(
            f"F-inliers: {r['F_inliers']}\n"
        )

        f.write(
            f"H-inliers: {r['H_inliers']}\n"
        )

        f.write(
            f"Coverage: {r['coverage']:.2f}%\n"
        )

        f.write(
            f"AI confidence: {r['ai']:.2f}%\n"
        )

        f.write(
            f"Geometry: {r['geometry']:.2f}%\n"
        )

        f.write(
            f"Localization: "
            f"{r['localization']:.2f}%\n"
        )

        f.write(
            f"Final score: "
            f"{r['score']:.2f}%\n"
        )

        f.write(
            f"Decision: "
            f"{r['decision']}\n"
        )

# ==============================================================
# DONE
# ==============================================================

print("\nCSV saved:")
print(csv_path)

print("\nReport saved:")
print(report_path)

print("\n" + "=" * 78)
print("FINAL PIPELINE COMPLETED")
print("=" * 78)