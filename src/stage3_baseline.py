import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD IMAGES
# ==========================================

image1_path = "data/raw/image1.jpg"
image2_path = "data/raw/image2.jpg"

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

if image1 is None or image2 is None:
    print("ERROR: Could not load images.")
    exit()

print("Both images loaded successfully!")

# ==========================================
# 2. CONVERT TO GRAYSCALE
# ==========================================

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# ==========================================
# 3. RESIZE LARGE IMAGE
# ==========================================

# Image 1 is much larger than Image 2.
# Resize Image 1 to make the scales more comparable.

scale = 0.25

small1 = cv2.resize(
    gray1,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_AREA
)

print("Resized Image 1:", small1.shape)
print("Image 2:", gray2.shape)

# ==========================================
# 4. SIFT FEATURE DETECTION
# ==========================================

sift = cv2.SIFT_create(
    nfeatures=10000,
    contrastThreshold=0.03
)

keypoints1, descriptors1 = sift.detectAndCompute(
    small1,
    None
)

keypoints2, descriptors2 = sift.detectAndCompute(
    gray2,
    None
)

print("Image 1 features:", len(keypoints1))
print("Image 2 features:", len(keypoints2))

# ==========================================
# 5. FLANN MATCHING
# ==========================================

index_params = dict(
    algorithm=1,
    trees=5
)

search_params = dict(
    checks=100
)

flann = cv2.FlannBasedMatcher(
    index_params,
    search_params
)

matches = flann.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)

print("Total comparisons:", len(matches))

# ==========================================
# 6. RATIO TEST
# ==========================================

good_matches = []

for pair in matches:

    if len(pair) == 2:

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

print("Good matches:", len(good_matches))

# ==========================================
# 7. RANSAC
# ==========================================

inlier_matches = []

if len(good_matches) >= 4:

    points1 = np.float32([
        keypoints1[m.queryIdx].pt
        for m in good_matches
    ])

    points2 = np.float32([
        keypoints2[m.trainIdx].pt
        for m in good_matches
    ])

    homography, mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        5.0
    )

    if mask is not None:

        for i, match in enumerate(good_matches):

            if mask[i][0]:
                inlier_matches.append(match)

print("RANSAC verified matches:", len(inlier_matches))

# ==========================================
# 8. DRAW VERIFIED MATCHES
# ==========================================

match_image = cv2.drawMatches(
    small1,
    keypoints1,
    image2,
    keypoints2,
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

match_image = cv2.cvtColor(
    match_image,
    cv2.COLOR_BGR2RGB
)

# ==========================================
# 9. DISPLAY
# ==========================================

plt.figure(figsize=(18, 9))

plt.imshow(match_image)

plt.title(
    f"Verified Lunar Correspondences: "
    f"{len(inlier_matches)}"
)

plt.axis("off")

plt.tight_layout()

plt.show()