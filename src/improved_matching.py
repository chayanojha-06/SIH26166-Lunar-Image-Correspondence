import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD IMAGES
# ==========================================

image1_path = "data/raw/image1.jpg"
image2_path = "data/raw/image1_transformed.jpg"

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

if image1 is None or image2 is None:
    print("ERROR: Could not load images.")
    exit()

print("Both images loaded successfully!")

# ==========================================
# 2. GRAYSCALE
# ==========================================

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# ==========================================
# 3. SIFT
# ==========================================

sift = cv2.SIFT_create(
    nfeatures=10000,
    contrastThreshold=0.03
)

keypoints1, descriptors1 = sift.detectAndCompute(
    gray1, None
)

keypoints2, descriptors2 = sift.detectAndCompute(
    gray2, None
)

print("Original image features:", len(keypoints1))
print("Transformed image features:", len(keypoints2))

# ==========================================
# 4. FLANN
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

# ==========================================
# 5. LOWE RATIO TEST
# ==========================================

good_matches = []

for pair in matches:

    if len(pair) == 2:

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

print("Good matches:", len(good_matches))

# ==========================================
# 6. RANSAC
# ==========================================

inlier_matches = []
homography = None

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

print(
    "RANSAC verified matches:",
    len(inlier_matches)
)

# ==========================================
# 7. INLIER RATIO
# ==========================================

if len(good_matches) > 0:

    inlier_ratio = (
        len(inlier_matches)
        / len(good_matches)
    ) * 100

else:

    inlier_ratio = 0

print(
    f"Inlier ratio: {inlier_ratio:.2f}%"
)

# ==========================================
# 8. RMSE CALCULATION
# ==========================================

rmse = None

if homography is not None and len(inlier_matches) >= 4:

    inlier_points1 = np.float32([
        keypoints1[m.queryIdx].pt
        for m in inlier_matches
    ])

    inlier_points2 = np.float32([
        keypoints2[m.trainIdx].pt
        for m in inlier_matches
    ])

    # Transform Image 1 points into Image 2
    projected_points = cv2.perspectiveTransform(
        inlier_points1.reshape(-1, 1, 2),
        homography
    ).reshape(-1, 2)

    # Calculate errors
    errors = projected_points - inlier_points2

    squared_errors = np.sum(
        errors ** 2,
        axis=1
    )

    rmse = np.sqrt(
        np.mean(squared_errors)
    )

    print(f"RMSE: {rmse:.4f} pixels")

else:

    print("RMSE could not be calculated.")

# ==========================================
# 9. CLEAN VISUALIZATION
# ==========================================

# Sort matches by descriptor distance
sorted_matches = sorted(
    inlier_matches,
    key=lambda x: x.distance
)

# Display only the strongest 100 matches
display_matches = sorted_matches[:100]

result = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    display_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

result = cv2.cvtColor(
    result,
    cv2.COLOR_BGR2RGB
)

# ==========================================
# 10. DISPLAY
# ==========================================

plt.figure(figsize=(18, 9))

plt.imshow(result)

title = (
    f"Lunar Correspondence\n"
    f"Inliers: {len(inlier_matches)} | "
    f"Inlier Ratio: {inlier_ratio:.2f}% | "
    f"RMSE: {rmse:.2f} px"
)

plt.title(title)

plt.axis("off")

plt.tight_layout()

plt.show()

# ==========================================
# 11. SAVE RESULT
# ==========================================

output_path = (
    "outputs/visualizations/"
    "stage5_rmse_result.jpg"
)

cv2.imwrite(
    output_path,
    cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
)

print(
    "Result saved to:",
    output_path
)