import cv2
import numpy as np
import os
import sys


# ============================================================
# SIH26166 LUNAR IMAGE CORRESPONDENCE SYSTEM
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "visualizations"
)


# ============================================================
# CONFIGURATION
# ============================================================

SIFT_FEATURES = 10000
SIFT_CONTRAST = 0.03
LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 5.0


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):

    image = cv2.imread(path)

    if image is None:
        print("ERROR: Could not load:")
        print(path)
        return None

    return image


# ============================================================
# PREPROCESSING HYPOTHESES
# ============================================================

def preprocess(image, method):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    if method == "Baseline":

        return gray

    elif method == "CLAHE":

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        return clahe.apply(gray)

    elif method == "Upscale":

        return cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

    elif method == "Upscale + CLAHE":

        upscaled = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        return clahe.apply(upscaled)

    return gray


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image):

    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES,
        contrastThreshold=SIFT_CONTRAST
    )

    keypoints, descriptors = sift.detectAndCompute(
        image,
        None
    )

    return keypoints, descriptors


# ============================================================
# FEATURE MATCHING
# ============================================================

def match_features(
    descriptors1,
    descriptors2
):

    if descriptors1 is None or descriptors2 is None:

        return []

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
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < LOWE_RATIO * n.distance:

            good_matches.append(m)

    return good_matches


# ============================================================
# RANSAC GEOMETRIC VERIFICATION
# ============================================================

def verify_geometry(
    keypoints1,
    keypoints2,
    matches
):

    if len(matches) < 4:

        return (
            0,
            0.0,
            None,
            None
        )

    points1 = np.float32([
        keypoints1[m.queryIdx].pt
        for m in matches
    ])

    points2 = np.float32([
        keypoints2[m.trainIdx].pt
        for m in matches
    ])

    H, mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        RANSAC_THRESHOLD
    )

    if H is None or mask is None:

        return (
            0,
            0.0,
            None,
            None
        )

    mask = mask.ravel()

    inlier_count = int(
        np.sum(mask)
    )

    inlier_ratio = (
        inlier_count / len(matches) * 100
    )

    rmse = None

    if inlier_count >= 4:

        inlier_points1 = points1[
            mask == 1
        ]

        inlier_points2 = points2[
            mask == 1
        ]

        projected = cv2.perspectiveTransform(
            inlier_points1.reshape(-1, 1, 2),
            H
        ).reshape(-1, 2)

        errors = (
            projected -
            inlier_points2
        )

        squared_errors = np.sum(
            errors ** 2,
            axis=1
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    squared_errors
                )
            )
        )

    return (
        inlier_count,
        inlier_ratio,
        rmse,
        H
    )


# ============================================================
# CONFIDENCE SCORE
# ============================================================

def calculate_confidence(
    good_matches,
    inliers,
    inlier_ratio,
    rmse
):

    if good_matches == 0:

        return 0.0

    match_score = min(
        good_matches / 100.0,
        1.0
    )

    inlier_score = min(
        inliers / 50.0,
        1.0
    )

    ratio_score = min(
        inlier_ratio / 100.0,
        1.0
    )

    if rmse is None:

        rmse_score = 0.0

    else:

        rmse_score = max(
            0.0,
            1.0 - rmse / 5.0
        )

    confidence = (
        0.15 * match_score +
        0.40 * inlier_score +
        0.30 * ratio_score +
        0.15 * rmse_score
    )

    return confidence * 100


# ============================================================
# DECISION
# ============================================================

def classify_confidence(
    confidence,
    inliers
):

    if inliers < 4:

        return "INSUFFICIENT GEOMETRIC EVIDENCE"

    if confidence >= 75:

        return "HIGH CONFIDENCE"

    if confidence >= 50:

        return "MODERATE CONFIDENCE"

    return "LOW CONFIDENCE"


# ============================================================
# VISUALIZATION
# ============================================================

def save_visualization(
    image1,
    image2,
    kp1,
    kp2,
    matches,
    H,
    output_path
):

    if H is None or len(matches) < 4:

        return

    points1 = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ])

    points2 = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ])

    H2, mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        RANSAC_THRESHOLD
    )

    if mask is None:

        return

    inlier_matches = [
        matches[i]
        for i in range(
            len(matches)
        )
        if mask[i]
    ]

    result = cv2.drawMatches(
        image1,
        kp1,
        image2,
        kp2,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cv2.imwrite(
        output_path,
        result
    )


# ============================================================
# RUN ONE HYPOTHESIS
# ============================================================

def run_hypothesis(
    image1,
    image2,
    method
):

    processed1 = preprocess(
        image1,
        "Baseline"
    )

    processed2 = preprocess(
        image2,
        method
    )

    kp1, des1 = extract_features(
        processed1
    )

    kp2, des2 = extract_features(
        processed2
    )

    good_matches = match_features(
        des1,
        des2
    )

    (
        inliers,
        inlier_ratio,
        rmse,
        H
    ) = verify_geometry(
        kp1,
        kp2,
        good_matches
    )

    confidence = calculate_confidence(
        len(good_matches),
        inliers,
        inlier_ratio,
        rmse
    )

    return {
        "method": method,
        "features": len(kp2),
        "good": len(good_matches),
        "inliers": inliers,
        "ratio": inlier_ratio,
        "rmse": rmse,
        "confidence": confidence,
        "decision": classify_confidence(
            confidence,
            inliers
        ),
        "kp1": kp1,
        "kp2": kp2,
        "matches": good_matches,
        "H": H
    }


# ============================================================
# MAIN SYSTEM
# ============================================================

def main():

    print("=" * 90)
    print("SIH26166 LUNAR IMAGE CORRESPONDENCE SYSTEM")
    print("=" * 90)

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    if len(sys.argv) >= 3:

        image1_path = sys.argv[1]
        image2_path = sys.argv[2]

    else:

        image1_path = os.path.join(
            RAW_DIR,
            "image1.jpg"
        )

        image2_path = os.path.join(
            RAW_DIR,
            "image2.jpg"
        )

    print()
    print("Reference image:")
    print(image1_path)

    print()
    print("Target image:")
    print(image2_path)

    image1 = load_image(
        image1_path
    )

    image2 = load_image(
        image2_path
    )

    if image1 is None or image2 is None:

        sys.exit(1)

    print()
    print("Images loaded successfully.")

    print(
        "Reference size:",
        image1.shape
    )

    print(
        "Target size:",
        image2.shape
    )

    # --------------------------------------------------------
    # HYPOTHESES
    # --------------------------------------------------------

    methods = [
        "Baseline",
        "CLAHE",
        "Upscale",
        "Upscale + CLAHE"
    ]

    results = []

    print()
    print("=" * 90)
    print("RUNNING MULTI-HYPOTHESIS ANALYSIS")
    print("=" * 90)

    for method in methods:

        print()
        print(
            "Testing:",
            method
        )

        result = run_hypothesis(
            image1,
            image2,
            method
        )

        results.append(
            result
        )

        print(
            "Features:",
            result["features"]
        )

        print(
            "Good matches:",
            result["good"]
        )

        print(
            "RANSAC inliers:",
            result["inliers"]
        )

        print(
            f"Inlier ratio: "
            f"{result['ratio']:.2f}%"
        )

        if result["rmse"] is not None:

            print(
                f"RMSE: "
                f"{result['rmse']:.4f} px"
            )

        else:

            print(
                "RMSE: N/A"
            )

        print(
            f"Confidence: "
            f"{result['confidence']:.2f}%"
        )

    # --------------------------------------------------------
    # SELECT BEST
    # --------------------------------------------------------

    best = max(
        results,
        key=lambda r: (
            r["confidence"],
            r["inliers"],
            r["ratio"]
        )
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("FINAL CORRESPONDENCE REPORT")
    print("=" * 90)

    print()
    print(
        f"{'METHOD':<25}"
        f"{'GOOD':>10}"
        f"{'INLIERS':>12}"
        f"{'RATIO':>12}"
        f"{'RMSE':>12}"
        f"{'CONF.':>12}"
    )

    print("-" * 90)

    for result in results:

        if result["rmse"] is None:

            rmse_text = "N/A"

        else:

            rmse_text = (
                f"{result['rmse']:.4f}"
            )

        print(
            f"{result['method']:<25}"
            f"{result['good']:>10}"
            f"{result['inliers']:>12}"
            f"{result['ratio']:>11.2f}%"
            f"{rmse_text:>12}"
            f"{result['confidence']:>11.2f}%"
        )

    print()
    print("=" * 90)
    print("SELECTED HYPOTHESIS")
    print("=" * 90)

    print()
    print(
        "Method:",
        best["method"]
    )

    print(
        "Good matches:",
        best["good"]
    )

    print(
        "RANSAC inliers:",
        best["inliers"]
    )

    print(
        f"Inlier ratio: "
        f"{best['ratio']:.2f}%"
    )

    if best["rmse"] is not None:

        print(
            f"RMSE: "
            f"{best['rmse']:.4f} px"
        )

    print(
        f"Confidence: "
        f"{best['confidence']:.2f}%"
    )

    print(
        "Decision:",
        best["decision"]
    )

    # --------------------------------------------------------
    # SAVE VISUALIZATION
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        "final_correspondence.jpg"
    )

    save_visualization(
        image1,
        image2,
        best["kp1"],
        best["kp2"],
        best["matches"],
        best["H"],
        output_file
    )

    print()
    print(
        "Visualization saved to:"
    )

    print(
        output_file
    )

    print()
    print("=" * 90)
    print("SYSTEM COMPLETED")
    print("=" * 90)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()