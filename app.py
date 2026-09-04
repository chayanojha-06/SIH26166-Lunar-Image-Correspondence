# ============================================================
# AEGIS | SIH26166
# Lunar Image Correspondence System
# Run: streamlit run app.py
# ============================================================

import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AEGIS | Lunar Correspondence",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "lunar_correspondence_rf_v45.pkl"
RAW_DIR = BASE_DIR / "data" / "raw"

FEATURE_COLUMNS = [
    "distance",
    "distance_ratio",
    "query_x",
    "query_y",
    "train_x",
    "train_y",
    "scale",
    "angle",
    "response",
    "size",
    "octave",
    "class_id",
]


# ============================================================
# HTML HELPER
# IMPORTANT: Do not strip/indent the HTML. Streamlit can render
# indented HTML incorrectly as code blocks in some layouts.
# ============================================================

def html(content):
    st.markdown(content, unsafe_allow_html=True)


# ============================================================
# PREMIUM UI
# ============================================================

html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg:#030507;
    --panel:#090c10;
    --panel2:#0d1116;
    --line:rgba(255,255,255,.075);
    --muted:#737d89;
    --text:#f3f4f5;
    --orange:#ff6035;
    --orange2:#ff815c;
    --green:#70e39b;
}

html,body,[class*="css"] {font-family:"DM Sans",sans-serif;}
.stApp {
    background:
        radial-gradient(circle at 15% 8%,rgba(116,24,8,.17),transparent 26%),
        radial-gradient(circle at 88% 18%,rgba(255,77,30,.075),transparent 22%),
        radial-gradient(circle at 50% 100%,rgba(70,25,15,.10),transparent 32%),
        #030507;
    color:var(--text);
}
.stApp::before {
    content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
    opacity:.45;
    background-image:
        linear-gradient(rgba(255,255,255,.045) 1px,transparent 1px),
        linear-gradient(90deg,rgba(255,255,255,.045) 1px,transparent 1px);
    background-size:40px 40px;
}
.block-container {
    max-width:1420px;
    padding:30px 42px 90px;
    position:relative;
    z-index:2;
}
section[data-testid="stSidebar"],header[data-testid="stHeader"] {display:none;}

.nav {
    display:flex; justify-content:space-between; align-items:center;
    height:64px; border-bottom:1px solid var(--line); margin-bottom:65px;
}
.brand {display:flex;align-items:center;gap:13px;}
.brand-mark {
    width:38px;height:38px;display:flex;align-items:center;justify-content:center;
    border-radius:10px;border:1px solid rgba(255,96,53,.55);
    background:radial-gradient(circle,rgba(255,96,53,.18),transparent 65%);
    color:var(--orange);font-size:15px;
}
.brand-name {font-weight:700;font-size:16px;letter-spacing:4px;}
.brand-caption {
    margin-top:3px;font-family:"Space Mono",monospace;
    font-size:7px;letter-spacing:1.5px;color:#69727d;
}
.nav-right {display:flex;align-items:center;gap:18px;}
.nav-code {
    font-family:"Space Mono",monospace;font-size:8px;
    letter-spacing:1.2px;color:#69727d;
}
.engine {
    display:flex;align-items:center;gap:8px;padding:8px 13px;
    border-radius:20px;border:1px solid rgba(112,227,155,.20);
    background:rgba(112,227,155,.035);color:#8ae4ac;
    font-family:"Space Mono",monospace;font-size:8px;letter-spacing:1px;
}
.engine-dot {
    width:6px;height:6px;border-radius:50%;background:var(--green);
    box-shadow:0 0 12px var(--green);
}

.hero {position:relative;text-align:center;padding:20px 0 75px;}
.hero-kicker {
    font-family:"Space Mono",monospace;color:var(--orange);
    font-size:8px;letter-spacing:4px;margin-bottom:25px;
}
.hero h1 {
    margin:0;font-size:clamp(55px,7.5vw,105px);line-height:.91;
    font-weight:700;letter-spacing:-6px;
    background:linear-gradient(120deg,#fff 15%,#f5ddd6 48%,#ff6035 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.hero-sub {
    max-width:690px;margin:30px auto 0;color:#7d8792;
    font-size:13px;line-height:1.9;
}
.orbit {
    position:absolute;width:330px;height:110px;left:50%;top:60%;
    transform:translate(-50%,-50%) rotate(-13deg);
    border:1px solid rgba(255,96,53,.07);border-radius:50%;
    pointer-events:none;
}

.system-strip {
    display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:30px;
}
.system-card {
    position:relative;padding:18px 20px;min-height:76px;
    border:1px solid var(--line);border-radius:14px;
    background:linear-gradient(145deg,rgba(255,255,255,.035),rgba(255,255,255,.012));
}
.system-label {
    color:#626c77;font-family:"Space Mono",monospace;
    font-size:7px;letter-spacing:1.5px;margin-bottom:9px;
}
.system-value {font-size:13px;font-weight:700;}

.section-label {
    color:var(--orange);font-family:"Space Mono",monospace;
    font-size:8px;letter-spacing:2px;margin-bottom:7px;
}
.section-title {
    color:#f1f2f3;font-size:22px;font-weight:700;
    letter-spacing:-.5px;margin-bottom:20px;
}

.input-panel {
    min-height:210px;border-radius:22px;border:1px solid rgba(255,255,255,.08);
    background:linear-gradient(145deg,rgba(17,21,26,.92),rgba(5,7,9,.97));
    padding:24px;margin-bottom:0;
}
.input-head {
    display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;
}
.input-index {
    font-family:"Space Mono",monospace;font-size:8px;letter-spacing:2px;color:var(--orange);
}
.input-type {
    font-family:"Space Mono",monospace;font-size:7px;letter-spacing:1px;color:#58616b;
}
.input-title {font-size:18px;font-weight:700;margin-bottom:4px;}
.input-description {color:#67717c;font-size:10px;margin-bottom:12px;}

div[data-testid="stFileUploader"] section {
    min-height:145px;border:1px dashed rgba(255,96,53,.28)!important;
    border-radius:15px!important;
    background:radial-gradient(circle at center,rgba(255,75,30,.045),transparent 65%)!important;
}

.preview-wrap {
    border-radius:15px;overflow:hidden;border:1px solid rgba(255,255,255,.08);
    background:#020304;margin-top:14px;
}
.preview-meta {
    display:flex;justify-content:space-between;padding:10px 13px;
    border-top:1px solid rgba(255,255,255,.06);color:#66717c;
    font-family:"Space Mono",monospace;font-size:7px;letter-spacing:.7px;
}

.analysis-bar {
    margin-top:14px;padding:18px 20px;border-radius:17px;
    border:1px solid rgba(255,255,255,.07);
    background:linear-gradient(90deg,rgba(255,255,255,.025),rgba(255,80,30,.035));
}
.analysis-info {
    color:#707b86;font-family:"Space Mono",monospace;font-size:7px;
    line-height:1.9;letter-spacing:.8px;
}

.stButton>button {
    width:100%!important;min-height:51px!important;border-radius:12px!important;
    border:1px solid rgba(255,130,95,.5)!important;
    background:linear-gradient(135deg,#ff6740,#d93316)!important;
    color:#fff!important;font-size:11px!important;font-weight:700!important;
    letter-spacing:.8px!important;
}

.result-shell {
    margin-top:60px;padding:30px;border-radius:25px;
    border:1px solid rgba(255,255,255,.075);
    background:linear-gradient(145deg,rgba(15,18,23,.94),rgba(5,7,9,.98));
}
.result-heading {font-size:26px;font-weight:700;}
.result-caption {color:#68727d;font-size:10px;margin-top:5px;}

.score-card {
    position:relative;min-height:270px;display:flex;align-items:center;
    justify-content:center;flex-direction:column;border-radius:21px;
    border:1px solid rgba(255,96,53,.20);
    background:radial-gradient(circle at center,rgba(255,70,25,.12),transparent 62%);
}
.score-kicker {
    font-family:"Space Mono",monospace;color:#6c7680;font-size:8px;letter-spacing:2px;
}
.score-number {
    font-size:72px;line-height:1;font-weight:700;letter-spacing:-5px;margin:8px 0;
    background:linear-gradient(135deg,#fff,#ff7954);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.score-unit {font-family:"Space Mono",monospace;color:#69737d;font-size:8px;}

.badge {
    display:inline-flex;padding:8px 13px;border-radius:20px;
    font-family:"Space Mono",monospace;font-size:7px;letter-spacing:1px;
}
.badge-strong {color:#86e6a8;border:1px solid rgba(90,220,135,.25);background:rgba(90,220,135,.055);}
.badge-moderate {color:#ffd277;border:1px solid rgba(255,190,80,.25);background:rgba(255,190,80,.055);}
.badge-review {color:#ffc16c;border:1px solid rgba(255,180,70,.25);background:rgba(255,180,70,.055);}
.badge-reject {color:#ff8b7b;border:1px solid rgba(255,80,65,.25);background:rgba(255,80,65,.055);}

.metric-grid {display:grid;grid-template-columns:repeat(4,1fr);gap:9px;}
.result-metric {
    min-height:94px;padding:17px;border-radius:14px;
    border:1px solid rgba(255,255,255,.065);background:rgba(255,255,255,.018);
}
.result-metric-label {
    color:#66717c;font-family:"Space Mono",monospace;
    font-size:7px;letter-spacing:1px;margin-bottom:12px;
}
.result-metric-value {color:#f1f3f4;font-size:21px;font-weight:700;}

.visual-panel,.evidence-card,.localization {
    padding:22px;border-radius:20px;border:1px solid rgba(255,255,255,.07);
    background:rgba(255,255,255,.014);
}
.visual-title {font-size:17px;font-weight:700;}
.visual-caption {color:#69737e;font-size:9px;margin-top:4px;margin-bottom:17px;}

.evidence-row {
    display:flex;justify-content:space-between;padding:12px 0;
    border-bottom:1px solid rgba(255,255,255,.05);
}
.evidence-name {color:#818b95;font-size:10px;}
.evidence-value {
    color:#eef1f3;font-family:"Space Mono",monospace;font-size:9px;
}

.coord {font-family:"Space Mono",monospace;font-size:29px;color:#f4f5f6;}
.coord span {color:var(--orange);}
.coord-label {color:#68727d;font-size:7px;letter-spacing:1.2px;margin-bottom:6px;}

.pipeline {display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:15px;}
.pipeline-step {
    padding:17px;min-height:105px;border-radius:14px;
    border:1px solid rgba(255,255,255,.06);background:rgba(255,255,255,.017);
}
.pipeline-num {color:var(--orange);font-family:"Space Mono",monospace;font-size:7px;}
.pipeline-name {margin-top:11px;font-size:10px;font-weight:700;}
.pipeline-desc {margin-top:5px;color:#626d78;font-size:8px;line-height:1.5;}

.scope-card {
    margin-top:55px;padding:28px;border-radius:20px;
    border:1px solid rgba(255,255,255,.065);
    background:linear-gradient(145deg,rgba(255,255,255,.025),rgba(255,255,255,.01));
}
.scope-text {max-width:900px;color:#747e89;font-size:11px;line-height:1.9;}

.footer {
    display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap;
    margin-top:75px;padding-top:23px;border-top:1px solid rgba(255,255,255,.06);
    color:#505963;font-family:"Space Mono",monospace;font-size:7px;letter-spacing:1px;
}

@media(max-width:900px){
    .block-container{padding:20px 16px 60px;}
    .nav-right{display:none;}
    .hero h1{font-size:55px;letter-spacing:-4px;}
    .system-strip,.metric-grid{grid-template-columns:1fr 1fr;}
    .pipeline{grid-template-columns:1fr;}
}
</style>
""")


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


model = load_model()


# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def read_image(uploaded_file):
    if uploaded_file is None:
        return None
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_local_image(path):
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


# ============================================================
# FEATURE EXTRACTION
# IMPORTANT:
# Each row is created from the exact accepted match, including
# its own Lowe ratio. This fixes the misalignment bug in the
# original implementation.
# ============================================================

def feature_rows(query_kp, train_kp, match_records):
    rows = []

    for m, ratio in match_records:
        q = query_kp[m.queryIdx]
        t = train_kp[m.trainIdx]

        octave = int(q.octave)
        octave_value = octave & 255
        layer = (octave >> 8) & 255

        if octave_value >= 128:
            octave_value -= 256

        rows.append({
            "distance": float(m.distance),
            "distance_ratio": float(ratio),
            "query_x": float(q.pt[0]),
            "query_y": float(q.pt[1]),
            "train_x": float(t.pt[0]),
            "train_y": float(t.pt[1]),
            "scale": float(q.size),
            "angle": float(q.angle),
            "response": float(q.response),
            "size": float(q.size),
            "octave": float(octave_value + layer / 255.0),
            "class_id": float(q.class_id),
        })

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


# ============================================================
# MODEL CONFIDENCE
# Returns None when the model cannot be trusted for inference.
# ============================================================

def model_confidence(model, features):
    if model is None or features.empty:
        return None

    try:
        X = features[FEATURE_COLUMNS]

        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            if set(expected) == set(FEATURE_COLUMNS):
                X = X[expected]

        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(X), dtype=float)

            if proba.ndim != 2 or proba.shape[0] != len(X):
                return None

            if proba.shape[1] == 2:
                classes = getattr(model, "classes_", None)
                if classes is not None and len(classes) == 2:
                    positive_index = 1
                    return np.clip(proba[:, positive_index], 0, 1)
                return np.clip(proba[:, 1], 0, 1)

            return np.clip(np.max(proba, axis=1), 0, 1)

        return None

    except Exception:
        return None


# ============================================================
# ROBUST FEATURE QUALITY
# Used when model inference is unavailable.
# This is not labelled as AI confidence.
# ============================================================

def feature_quality(match_records):
    if not match_records:
        return np.array([])

    distances = np.array([m.distance for m, _ in match_records], dtype=float)
    ratios = np.array([r for _, r in match_records], dtype=float)

    d_scale = np.percentile(distances, 90)
    if d_scale <= 0:
        d_scale = max(float(distances.max()), 1.0)

    distance_score = 1.0 - np.clip(distances / d_scale, 0, 1)

    # 0.60 is excellent, 0.75 is the acceptance boundary
    ratio_score = np.clip((0.85 - ratios) / 0.25, 0, 1)

    quality = 0.45 * distance_score + 0.55 * ratio_score
    return np.clip(quality, 0, 1)


# ============================================================
# COVERAGE
# Uses convex hull instead of only bounding-box area.
# ============================================================

def calculate_coverage(points, width, height):
    if points is None or len(points) < 3:
        return 0.0

    image_area = float(width * height)
    if image_area <= 0:
        return 0.0

    pts = np.asarray(points, dtype=np.float32)

    try:
        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
        hull_area = float(cv2.contourArea(hull))
        return float(np.clip(100.0 * hull_area / image_area, 0, 100))
    except Exception:
        return 0.0


# ============================================================
# GEOMETRIC VERIFICATION
# ============================================================

def verify_geometry(query_pts, train_pts):
    n = len(query_pts)

    result = {
        "H": None,
        "hmask": None,
        "H_inliers": 0,
        "F_inliers": 0,
        "H_ratio": 0.0,
        "F_ratio": 0.0,
        "H_error": None,
    }

    if n < 4:
        return result

    H, hmask = cv2.findHomography(
        query_pts,
        train_pts,
        cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=5000,
        confidence=0.995,
    )

    if H is not None and hmask is not None:
        hmask = hmask.ravel().astype(bool)
        H_inliers = int(hmask.sum())

        result["H"] = H
        result["hmask"] = hmask
        result["H_inliers"] = H_inliers
        result["H_ratio"] = H_inliers / max(n, 1)

        if H_inliers > 0:
            projected = cv2.perspectiveTransform(
                query_pts.reshape(-1, 1, 2), H
            ).reshape(-1, 2)

            errors = np.linalg.norm(projected - train_pts, axis=1)
            result["H_error"] = float(np.median(errors[hmask]))

    if n >= 8:
        F, fmask = cv2.findFundamentalMat(
            query_pts,
            train_pts,
            cv2.FM_RANSAC,
            2.0,
            0.995,
            5000,
        )

        if F is not None and fmask is not None:
            fmask = fmask.ravel().astype(bool)
            result["F_inliers"] = int(fmask.sum())
            result["F_ratio"] = result["F_inliers"] / max(n, 1)

    return result


# ============================================================
# SCORE HELPERS
# ============================================================

def saturating_score(value, good, excellent):
    if excellent <= good:
        return 0.0

    if value <= good:
        return 0.0

    return float(np.clip((value - good) / (excellent - good), 0, 1))


def geometry_score(stats, coverage, total_matches):
    H_ratio = stats["H_ratio"]
    F_ratio = stats["F_ratio"]
    H_error = stats["H_error"]

    if H_error is None:
        error_score = 0.0
    else:
        # <=1 px excellent, >=10 px poor
        error_score = np.clip((10.0 - H_error) / 9.0, 0, 1)

    match_score = saturating_score(total_matches, 8, 50)

    # Coverage does not need to span the entire image.
    coverage_score = saturating_score(coverage, 2, 25)

    score = (
        0.35 * H_ratio +
        0.30 * F_ratio +
        0.15 * error_score +
        0.10 * coverage_score +
        0.10 * match_score
    )

    # Hard reliability penalties
    if total_matches < 8:
        score *= 0.35
    elif total_matches < 15:
        score *= 0.65

    if stats["H_inliers"] < 4:
        score *= 0.5

    if stats["F_inliers"] < 8:
        score *= 0.75

    return float(np.clip(score * 100, 0, 100))


def localization_score(stats):
    if stats["H"] is None or stats["H_error"] is None:
        return 0.0

    H_ratio = stats["H_ratio"]
    F_ratio = stats["F_ratio"]
    error_score = np.clip((8.0 - stats["H_error"]) / 7.0, 0, 1)

    score = 0.45 * H_ratio + 0.30 * F_ratio + 0.25 * error_score

    if stats["H_inliers"] < 5:
        score *= 0.5
    if stats["F_inliers"] < 8:
        score *= 0.6

    return float(np.clip(score * 100, 0, 100))


# ============================================================
# VISUALIZATION
# ============================================================

def make_visualization(
    query_img,
    train_img,
    query_kp,
    train_kp,
    matches,
    hmask,
):
    if not matches:
        return None

    if hmask is not None and len(hmask) == len(matches):
        draw_matches = [
            match for match, accepted in zip(matches, hmask) if accepted
        ]
    else:
        draw_matches = matches[:80]

    draw_matches = draw_matches[:100]

    if not draw_matches:
        return None

    vis = cv2.drawMatches(
        query_img,
        query_kp,
        train_img,
        train_kp,
        draw_matches,
        None,
        matchColor=(45, 110, 255),
        singlePointColor=(190, 190, 190),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# ============================================================
# ANALYSIS ENGINE
# ============================================================

def analyze(query_img, train_img, model):
    gray_q = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    gray_t = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(
        nfeatures=12000,
        contrastThreshold=0.015,
        edgeThreshold=10,
        sigma=1.6,
    )

    query_kp, query_des = sift.detectAndCompute(gray_q, None)
    train_kp, train_des = sift.detectAndCompute(gray_t, None)

    if query_des is None or train_des is None:
        raise RuntimeError("Insufficient image features detected.")

    if len(query_kp) < 4 or len(train_kp) < 4:
        raise RuntimeError("Too few detectable image features.")

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = matcher.knnMatch(query_des, train_des, k=2)

    # Store the match and ITS OWN ratio together.
    accepted_records = []

    for pair in raw_matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if n.distance <= 1e-12:
            continue

        ratio = float(m.distance / n.distance)

        if ratio < 0.75:
            accepted_records.append((m, ratio))

    if len(accepted_records) < 4:
        raise RuntimeError(
            "No reliable correspondences found after feature matching."
        )

    # Sort strongest descriptor matches before feature/model analysis.
    accepted_records.sort(
        key=lambda item: (item[1], item[0].distance)
    )

    # Keep enough matches for geometry but avoid pathological thousands.
    accepted_records = accepted_records[:500]

    features = feature_rows(
        query_kp,
        train_kp,
        accepted_records,
    )

    model_probs = model_confidence(model, features)
    heuristic_quality = feature_quality(accepted_records)

    # If the trained model works, use it to rank matches.
    # Otherwise geometry still works using descriptor quality.
    if model_probs is not None and len(model_probs) == len(accepted_records):
        combined_quality = (
            0.70 * model_probs +
            0.30 * heuristic_quality
        )
        model_available = True
    else:
        combined_quality = heuristic_quality
        model_available = False

    order = np.argsort(combined_quality)[::-1]

    max_selected = min(300, len(order))
    selected_indices = order[:max_selected]

    selected_records = [
        accepted_records[i] for i in selected_indices
    ]
    selected_matches = [item[0] for item in selected_records]

    selected_quality = combined_quality[selected_indices]

    query_pts = np.float32([
        query_kp[m.queryIdx].pt
        for m in selected_matches
    ])

    train_pts = np.float32([
        train_kp[m.trainIdx].pt
        for m in selected_matches
    ])

    stats = verify_geometry(query_pts, train_pts)

    coverage = calculate_coverage(
        train_pts,
        train_img.shape[1],
        train_img.shape[0],
    )

    geometry = geometry_score(
        stats,
        coverage,
        len(selected_matches),
    )

    localization = localization_score(stats)

    # Feature confidence is intentionally separated from final confidence.
    feature_confidence = float(
        np.mean(selected_quality) * 100
    )

    localized_x = None
    localized_y = None

    localization_accepted = (
        stats["H"] is not None and
        stats["H_inliers"] >= 6 and
        stats["H_ratio"] >= 0.25 and
        stats["F_inliers"] >= 8 and
        stats["F_ratio"] >= 0.20 and
        stats["H_error"] is not None and
        stats["H_error"] <= 10.0
    )

    if localization_accepted:
        center = np.float32([
            [[
                query_img.shape[1] / 2.0,
                query_img.shape[0] / 2.0
            ]]
        ])

        projected = cv2.perspectiveTransform(
            center,
            stats["H"]
        )[0][0]

        x, y = float(projected[0]), float(projected[1])

        # Only accept coordinates reasonably close to the reference bounds.
        margin = 0.05
        width = train_img.shape[1]
        height = train_img.shape[0]

        if (
            -margin * width <= x <= (1 + margin) * width and
            -margin * height <= y <= (1 + margin) * height
        ):
            localized_x = x
            localized_y = y
        else:
            localization = 0.0

    # Final confidence:
    # Geometry is intentionally dominant because the task is
    # correspondence verification, not generic image similarity.
    if localized_x is not None:
        score = (
            0.25 * feature_confidence +
            0.50 * geometry +
            0.25 * localization
        )
    else:
        score = (
            0.30 * feature_confidence +
            0.70 * geometry
        )

    # Prevent a tiny number of matches from receiving a glamorous score.
    if len(selected_matches) < 8:
        score *= 0.35
    elif len(selected_matches) < 15:
        score *= 0.65

    score = float(np.clip(score, 0, 100))

    # Decision thresholds require supporting evidence, not score alone.
    if (
        score >= 70 and
        geometry >= 60 and
        feature_confidence >= 60 and
        stats["H_inliers"] >= 10 and
        stats["F_inliers"] >= 12 and
        localized_x is not None
    ):
        decision = "STRONG CORRESPONDENCE"

    elif (
        score >= 52 and
        geometry >= 42 and
        stats["H_inliers"] >= 6 and
        stats["F_inliers"] >= 8
    ):
        decision = "MODERATE CORRESPONDENCE"

    elif (
        score >= 30 and
        stats["H_inliers"] >= 4
    ):
        decision = "REVIEW REQUIRED"

    else:
        decision = "REJECT"

    visualization = make_visualization(
        query_img,
        train_img,
        query_kp,
        train_kp,
        selected_matches,
        stats["hmask"],
    )

    return {
        "good_matches": len(selected_matches),
        "raw_good_matches": len(accepted_records),
        "F_inliers": stats["F_inliers"],
        "H_inliers": stats["H_inliers"],
        "F_ratio": stats["F_ratio"] * 100,
        "H_ratio": stats["H_ratio"] * 100,
        "H_error": stats["H_error"],
        "coverage": coverage,
        "feature_confidence": feature_confidence,
        "model_available": model_available,
        "geometry": geometry,
        "localization": localization,
        "score": score,
        "decision": decision,
        "x": localized_x,
        "y": localized_y,
        "visualization": visualization,
        "H": stats["H"],
    }


# ============================================================
# BADGE
# ============================================================

def badge_html(decision):
    if decision == "STRONG CORRESPONDENCE":
        return '<span class="badge badge-strong">● STRONG CORRESPONDENCE</span>'
    if decision == "MODERATE CORRESPONDENCE":
        return '<span class="badge badge-moderate">● MODERATE CORRESPONDENCE</span>'
    if decision == "REVIEW REQUIRED":
        return '<span class="badge badge-review">● REVIEW REQUIRED</span>'
    return '<span class="badge badge-reject">● REJECTED</span>'


# ============================================================
# TOP NAVIGATION
# ============================================================

html("""
<div class="nav">
    <div class="brand">
        <div class="brand-mark">◉</div>
        <div>
            <div class="brand-name">AEGIS</div>
            <div class="brand-caption">LUNAR IMAGE CORRESPONDENCE · SIH26166</div>
        </div>
    </div>
    <div class="nav-right">
        <div class="nav-code">ISRO · SPACE TECHNOLOGY</div>
        <div class="engine">
            <span class="engine-dot"></span>ENGINE ONLINE
        </div>
    </div>
</div>
""")

html("""
<div class="hero">
    <div class="orbit"></div>
    <div class="hero-kicker">MULTI-MODAL · SCALE ROBUST · IMAGE-SPACE ANALYSIS</div>
    <h1>Find the lunar<br>correspondence.</h1>
    <div class="hero-sub">
        Compare two lunar observations using local feature evidence,
        learned match confidence when available, and independent
        geometric verification.
    </div>
</div>
""")

html(f"""
<div class="system-strip">
    <div class="system-card">
        <div class="system-label">LEARNED MODEL</div>
        <div class="system-value">RF-V45</div>
    </div>
    <div class="system-card">
        <div class="system-label">FEATURE ENGINE</div>
        <div class="system-value">SIFT</div>
    </div>
    <div class="system-card">
        <div class="system-label">GEOMETRIC VERIFICATION</div>
        <div class="system-value">RANSAC / H + F</div>
    </div>
    <div class="system-card">
        <div class="system-label">MODEL STATUS</div>
        <div class="system-value" style="color:{'#70e39b' if model else '#ff7046'}">
            {'ONLINE' if model else 'FALLBACK MODE'}
        </div>
    </div>
</div>
""")


# ============================================================
# INPUT
# ============================================================

html("""
<div class="section-label">01 / OBSERVATIONS</div>
<div class="section-title">Compare two lunar observations</div>
""")

left, right = st.columns(2, gap="large")

with left:
    html("""
<div class="input-panel">
<div class="input-head">
<span class="input-index">QUERY / 01</span>
<span class="input-type">TARGET OBSERVATION</span>
</div>
<div class="input-title">Query image</div>
<div class="input-description">
The observation to be tested against the reference image.
</div>
</div>
""")

    query_file = st.file_uploader(
        "Query image",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        key="query",
        label_visibility="collapsed",
    )

    if query_file:
        preview = read_image(query_file)
        if preview is not None:
            st.image(
                cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )
            html(f"""
<div class="preview-meta">
<span>{query_file.name}</span>
<span>{preview.shape[1]} × {preview.shape[0]} PX</span>
</div>
""")

with right:
    html("""
<div class="input-panel">
<div class="input-head">
<span class="input-index">REFERENCE / 02</span>
<span class="input-type">REFERENCE OBSERVATION</span>
</div>
<div class="input-title">Reference image</div>
<div class="input-description">
Candidate lunar scene used for correspondence verification.
</div>
</div>
""")

    reference_file = st.file_uploader(
        "Reference image",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        key="reference",
        label_visibility="collapsed",
    )

    if reference_file:
        preview = read_image(reference_file)
        if preview is not None:
            st.image(
                cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )
            html(f"""
<div class="preview-meta">
<span>{reference_file.name}</span>
<span>{preview.shape[1]} × {preview.shape[0]} PX</span>
</div>
""")


# ============================================================
# ANALYSIS CONTROL
# ============================================================

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

control_left, control_right = st.columns([3, 1], gap="large")

with control_left:
    html("""
<div class="analysis-bar">
<div class="section-label">ANALYSIS CONFIGURATION</div>
<div class="analysis-info">
SIFT FEATURE EXTRACTION · RATIO-TEST MATCHING · LEARNED MATCH CONFIDENCE ·
FUNDAMENTAL MATRIX · HOMOGRAPHY · SPATIAL COVERAGE · IMAGE-SPACE LOCALIZATION
</div>
</div>
""")

with control_right:
    run_analysis = st.button(
        "FIND CORRESPONDENCE",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# LOCAL DEMO DATA
# ============================================================

with st.expander("◉ VERIFIED LOCAL DEMO DATA"):
    demo_files = [
        RAW_DIR / "image1.jpg",
        RAW_DIR / "image1_perspective.jpg",
        RAW_DIR / "image1_transformed.jpg",
        RAW_DIR / "image2.jpg",
    ]

    available = [p.name for p in demo_files if p.exists()]

    if available:
        d1, d2 = st.columns(2)

        with d1:
            default_q = (
                available.index("image1_transformed.jpg")
                if "image1_transformed.jpg" in available
                else 0
            )

            demo_query_name = st.selectbox(
                "Query observation",
                available,
                index=default_q,
            )

        with d2:
            default_r = (
                available.index("image1.jpg")
                if "image1.jpg" in available
                else 0
            )

            demo_reference_name = st.selectbox(
                "Reference observation",
                available,
                index=default_r,
            )

        use_demo = st.checkbox(
            "Use selected repository images",
            value=False,
        )
    else:
        use_demo = False
        st.caption("No local demonstration images found.")


# ============================================================
# RUN ANALYSIS
# ============================================================

if run_analysis:
    query_img = None
    reference_img = None
    query_name = ""
    reference_name = ""

    if use_demo:
        query_path = RAW_DIR / demo_query_name
        reference_path = RAW_DIR / demo_reference_name

        query_img = load_local_image(query_path)
        reference_img = load_local_image(reference_path)

        query_name = demo_query_name
        reference_name = demo_reference_name

    elif query_file is not None and reference_file is not None:
        query_img = read_image(query_file)
        reference_img = read_image(reference_file)

        query_name = query_file.name
        reference_name = reference_file.name

    else:
        st.error(
            "Upload both query and reference images before running analysis."
        )
        st.stop()

    if query_img is None or reference_img is None:
        st.error("Unable to decode one or both images.")
        st.stop()

    with st.spinner("AEGIS / COMPUTING CORRESPONDENCE..."):
        try:
            result = analyze(
                query_img,
                reference_img,
                model,
            )
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    st.session_state["aegis_result"] = result
    st.session_state["aegis_query"] = query_img
    st.session_state["aegis_reference"] = reference_img
    st.session_state["aegis_query_name"] = query_name
    st.session_state["aegis_reference_name"] = reference_name


# ============================================================
# RESULTS
# ============================================================

if "aegis_result" in st.session_state:
    result = st.session_state["aegis_result"]
    query_img = st.session_state["aegis_query"]
    reference_img = st.session_state["aegis_reference"]

    html("""
<div class="result-shell">
<div class="section-label">02 / CORRESPONDENCE ASSESSMENT</div>
<div class="result-heading">Analysis complete</div>
<div class="result-caption">
Correspondence confidence derived from feature evidence and independent geometric verification.
</div>
</div>
""")

    score_col, metrics_col = st.columns([1, 2.15], gap="large")

    with score_col:
        html(f"""
<div class="score-card">
<div class="score-kicker">FINAL CONFIDENCE</div>
<div class="score-number">{result["score"]:.1f}</div>
<div class="score-unit">/ 100</div>
<div style="margin-top:18px;">{badge_html(result["decision"])}</div>
</div>
""")

    with metrics_col:
        html(f"""
<div class="metric-grid">

<div class="result-metric">
<div class="result-metric-label">SELECTED MATCHES</div>
<div class="result-metric-value">{result["good_matches"]}</div>
</div>

<div class="result-metric">
<div class="result-metric-label">FUNDAMENTAL INLIERS</div>
<div class="result-metric-value">{result["F_inliers"]}</div>
</div>

<div class="result-metric">
<div class="result-metric-label">HOMOGRAPHY INLIERS</div>
<div class="result-metric-value">{result["H_inliers"]}</div>
</div>

<div class="result-metric">
<div class="result-metric-label">SPATIAL COVERAGE</div>
<div class="result-metric-value">{result["coverage"]:.1f}%</div>
</div>

<div class="result-metric">
<div class="result-metric-label">FEATURE CONFIDENCE</div>
<div class="result-metric-value">{result["feature_confidence"]:.1f}%</div>
</div>

<div class="result-metric">
<div class="result-metric-label">GEOMETRY</div>
<div class="result-metric-value">{result["geometry"]:.1f}%</div>
</div>

<div class="result-metric">
<div class="result-metric-label">LOCALIZATION</div>
<div class="result-metric-value">{result["localization"]:.1f}%</div>
</div>

<div class="result-metric">
<div class="result-metric-label">TRANSFER ERROR</div>
<div class="result-metric-value">
{result["H_error"] if result["H_error"] is None else f'{result["H_error"]:.3f}'}
</div>
</div>

</div>
""")

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    html("""
<div class="section-label">03 / SOURCE OBSERVATIONS</div>
<div class="section-title">Input imagery</div>
""")

    source1, source2 = st.columns(2, gap="large")

    with source1:
        html("""
<div class="visual-panel">
<div class="visual-title">Query observation</div>
<div class="visual-caption">TARGET IMAGE</div>
</div>
""")

        st.image(
            cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

        html(f"""
<div class="preview-meta">
<span>{st.session_state["aegis_query_name"]}</span>
<span>{query_img.shape[1]} × {query_img.shape[0]} PX</span>
</div>
""")

    with source2:
        html("""
<div class="visual-panel">
<div class="visual-title">Reference observation</div>
<div class="visual-caption">REFERENCE IMAGE</div>
</div>
""")

        st.image(
            cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

        html(f"""
<div class="preview-meta">
<span>{st.session_state["aegis_reference_name"]}</span>
<span>{reference_img.shape[1]} × {reference_img.shape[0]} PX</span>
</div>
""")

    if result["visualization"] is not None:
        st.markdown("<div style='height:35px'></div>", unsafe_allow_html=True)

        html("""
<div class="section-label">04 / CORRESPONDENCE FIELD</div>
<div class="section-title">Geometrically verified feature matches</div>
<div class="visual-caption">
DISPLAYED LINES REPRESENT HOMOGRAPHY-RANSAC INLIERS.
</div>
""")

        st.image(
            result["visualization"],
            use_container_width=True,
        )

    st.markdown("<div style='height:38px'></div>", unsafe_allow_html=True)

    evidence_col, loc_col = st.columns(2, gap="large")

    with evidence_col:
        html("""
<div class="section-label">05 / EVIDENCE</div>
<div class="section-title">Independent measurements</div>
<div class="evidence-card">
""")

        model_status = (
            "Random Forest + descriptor quality"
            if result["model_available"]
            else "Descriptor-quality fallback"
        )

        evidence_items = [
            ("Confidence source", model_status),
            ("Accepted descriptor matches", str(result["raw_good_matches"])),
            ("Selected matches", str(result["good_matches"])),
            ("Fundamental inliers", str(result["F_inliers"])),
            ("Homography inliers", str(result["H_inliers"])),
            ("Fundamental ratio", f'{result["F_ratio"]:.2f}%'),
            ("Homography ratio", f'{result["H_ratio"]:.2f}%'),
            ("Spatial coverage", f'{result["coverage"]:.2f}%'),
            ("Feature confidence", f'{result["feature_confidence"]:.2f}%'),
            ("Geometry score", f'{result["geometry"]:.2f}%'),
            ("Localization score", f'{result["localization"]:.2f}%'),
            (
                "Transfer error",
                "N/A" if result["H_error"] is None
                else f'{result["H_error"]:.3f} px'
            ),
        ]

        for name, value in evidence_items:
            html(
                f'<div class="evidence-row">'
                f'<div class="evidence-name">{name}</div>'
                f'<div class="evidence-value">{value}</div>'
                f'</div>'
            )

        html("</div>")

    with loc_col:
        html("""
<div class="section-label">06 / IMAGE-SPACE LOCALIZATION</div>
<div class="section-title">Estimated correspondence position</div>
""")

        if result["x"] is not None and result["y"] is not None:
            nx = (
                result["x"] /
                reference_img.shape[1] * 100
            )

            ny = (
                result["y"] /
                reference_img.shape[0] * 100
            )

            html(f"""
<div class="localization">
<div style="display:grid;grid-template-columns:1fr 1fr;gap:30px;">
<div>
<div class="coord-label">X POSITION</div>
<div class="coord"><span>{result["x"]:.2f}</span></div>
</div>
<div>
<div class="coord-label">Y POSITION</div>
<div class="coord"><span>{result["y"]:.2f}</span></div>
</div>
</div>

<div style="margin-top:28px;padding-top:20px;border-top:1px solid rgba(255,255,255,.06);display:grid;grid-template-columns:1fr 1fr;gap:30px;">
<div>
<div class="coord-label">NORMALIZED X</div>
<div style="font-family:'Space Mono';font-size:15px;">{nx:.2f}%</div>
</div>
<div>
<div class="coord-label">NORMALIZED Y</div>
<div style="font-family:'Space Mono';font-size:15px;">{ny:.2f}%</div>
</div>
</div>
</div>
""")

            loc_img = reference_img.copy()

            point = (
                int(round(result["x"])),
                int(round(result["y"])),
            )

            for radius, thickness in [(34, 1), (25, 2), (16, 3)]:
                cv2.circle(
                    loc_img,
                    point,
                    radius,
                    (0, 70, 255),
                    thickness,
                )

            cv2.drawMarker(
                loc_img,
                point,
                (0, 70, 255),
                cv2.MARKER_CROSS,
                48,
                2,
            )

            st.image(
                cv2.cvtColor(loc_img, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        else:
            html("""
<div class="localization">
<div style="color:#ff9b87;font-family:'Space Mono';font-size:9px;letter-spacing:1px;">
● LOCALIZATION NOT ACCEPTED
</div>
<div style="margin-top:13px;color:#69737d;font-size:10px;line-height:1.7;">
The image pair produced insufficient geometric evidence for a reliable
reference-image position.
</div>
</div>
""")

    st.markdown("<div style='height:45px'></div>", unsafe_allow_html=True)

    html("""
<div class="section-label">07 / PROCESSING PIPELINE</div>
<div class="section-title">AEGIS correspondence workflow</div>
<div class="pipeline">

<div class="pipeline-step">
<div class="pipeline-num">01</div>
<div class="pipeline-name">IMAGE INPUT</div>
<div class="pipeline-desc">Decode query and reference observations.</div>
</div>

<div class="pipeline-step">
<div class="pipeline-num">02</div>
<div class="pipeline-name">SIFT</div>
<div class="pipeline-desc">Extract scale and rotation-aware local features.</div>
</div>

<div class="pipeline-step">
<div class="pipeline-num">03</div>
<div class="pipeline-name">MATCH QUALITY</div>
<div class="pipeline-desc">Rank correspondences using learned confidence when available.</div>
</div>

<div class="pipeline-step">
<div class="pipeline-num">04</div>
<div class="pipeline-name">RANSAC</div>
<div class="pipeline-desc">Verify fundamental and homography geometry.</div>
</div>

<div class="pipeline-step">
<div class="pipeline-num">05</div>
<div class="pipeline-name">LOCALIZE</div>
<div class="pipeline-desc">Project accepted correspondence into reference image space.</div>
</div>

</div>
""")

    h_error_text = (
        "N/A"
        if result["H_error"] is None
        else f'{result["H_error"]:.4f} px'
    )

    report_text = f"""AEGIS - LUNAR IMAGE CORRESPONDENCE
SIH26166

Query image: {st.session_state["aegis_query_name"]}
Reference image: {st.session_state["aegis_reference_name"]}

Decision: {result["decision"]}
Final correspondence confidence: {result["score"]:.2f}%

Confidence source:
{"Random Forest + descriptor quality" if result["model_available"] else "Descriptor-quality fallback"}

Accepted descriptor matches: {result["raw_good_matches"]}
Selected matches: {result["good_matches"]}
Fundamental inliers: {result["F_inliers"]}
Homography inliers: {result["H_inliers"]}

Fundamental ratio: {result["F_ratio"]:.2f}%
Homography ratio: {result["H_ratio"]:.2f}%
Spatial coverage: {result["coverage"]:.2f}%

Feature confidence: {result["feature_confidence"]:.2f}%
Geometry score: {result["geometry"]:.2f}%
Localization score: {result["localization"]:.2f}%

Homography transfer error: {h_error_text}

Image-space X: {result["x"]}
Image-space Y: {result["y"]}
"""

    csv_df = pd.DataFrame([{
        "query": st.session_state["aegis_query_name"],
        "reference": st.session_state["aegis_reference_name"],
        "decision": result["decision"],
        "final_confidence": result["score"],
        "accepted_descriptor_matches": result["raw_good_matches"],
        "selected_matches": result["good_matches"],
        "fundamental_inliers": result["F_inliers"],
        "homography_inliers": result["H_inliers"],
        "fundamental_ratio": result["F_ratio"],
        "homography_ratio": result["H_ratio"],
        "coverage": result["coverage"],
        "feature_confidence": result["feature_confidence"],
        "geometry": result["geometry"],
        "localization": result["localization"],
        "homography_error_px": result["H_error"],
        "x_px": result["x"],
        "y_px": result["y"],
        "model_available": result["model_available"],
    }])

    html("""
<div class="section-label">EXPORT</div>
<div class="section-title">Analysis artifacts</div>
""")

    r1, r2 = st.columns(2)

    with r1:
        st.download_button(
            "DOWNLOAD ANALYSIS REPORT",
            data=report_text,
            file_name="AEGIS_correspondence_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with r2:
        st.download_button(
            "DOWNLOAD RESULTS CSV",
            data=csv_df.to_csv(index=False),
            file_name="AEGIS_correspondence_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# SCIENTIFIC SCOPE
# ============================================================

html("""
<div class="scope-card">
<div class="section-label">SCIENTIFIC SCOPE</div>
<div class="section-title">What AEGIS measures</div>
<div class="scope-text">

AEGIS is an image-space correspondence prototype for lunar optical imagery.
The final confidence combines local feature quality with independently verified
geometric consistency.

<br><br>

A learned Random Forest model is used only when it can successfully process
the extracted feature representation. When the model is unavailable or
incompatible, the system explicitly falls back to descriptor-quality evidence
rather than silently inventing an AI confidence value.

<br><br>

Homography transfer error describes the consistency of the supplied image pair.
It should not be interpreted as universal lunar localization accuracy.

<br><br>

Image-space coordinates represent pixel positions in the supplied reference
image. They are not lunar latitude and longitude and do not represent
spacecraft telemetry or mission navigation coordinates.

</div>
</div>
""")


# ============================================================
# FOOTER
# ============================================================

html("""
<div class="footer">
<span>AEGIS · SIH26166</span>
<span>LUNAR IMAGE CORRESPONDENCE SYSTEM</span>
<span>IMAGE-SPACE ANALYSIS</span>
<span>SIFT · RANDOM FOREST · RANSAC</span>
</div>
""")
