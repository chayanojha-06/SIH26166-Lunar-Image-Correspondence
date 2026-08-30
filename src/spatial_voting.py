import cv2
import numpy as np
import os


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE1 = "data/raw/image1.jpg"
IMAGE2 = "data/raw/image2.jpg"

LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 2.0

GRID_ROWS = 4
GRID_COLS = 4


# ============================================================
# LOAD IMAGES
# ============================================================

image1 = cv2.imread(IMAGE1)
image2 = cv2.imread(IMAGE2)

if image1 is None or image2 is None:
    print("ERROR: Could not load images.")
    exit()

print("=" * 90)
print("SIH26166 SPATIAL VOTING LOCALIZATION")
print("=" * 90)

print("\nImages loaded successfully.")
print("Reference:", image1.shape)
print("Target   :", image2.shape)


# ============================================================
# SIFT
# ============================================================

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create(
    nfeatures=10000,
    contrastThreshold=0.03
)

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print("\nFeatures:")
print("Reference:", len(kp1))
print("Target   :", len(kp2))


# ============================================================
# FLANN MATCHING
# ============================================================

flann = cv2.FlannBasedMatcher(
    dict(
        algorithm=1,
        trees=5
    ),
    dict(
        checks=50
    )
)

matches = flann.knnMatch(
    des1,
    des2,
    k=2
)


# ============================================================
# LOWE RATIO
# ============================================================

good = []

for pair in matches:

    if len(pair) == 2:

        m, n = pair

        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)


print("\nGood matches:", len(good))

if len(good) < 8:
    print("ERROR: Not enough matches.")
    exit()


# ============================================================
# MATCH POINTS
# ============================================================

points1 = np.float32([
    kp1[m.queryIdx].pt
    for m in good
])

points2 = np.float32([
    kp2[m.trainIdx].pt
    for m in good
])


# ============================================================
# FUNDAMENTAL MATRIX
# ============================================================

print("\nRunning geometric verification...")

F, mask = cv2.findFundamentalMat(
    points1,
    points2,
    cv2.FM_RANSAC,
    RANSAC_THRESHOLD,
    0.99,
    2000
)

if F is None or mask is None:
    print("ERROR: Fundamental matrix failed.")
    exit()

mask = mask.ravel()

inlier_points = points1[mask == 1]

inlier_target = points2[mask == 1]

inlier_count = len(inlier_points)

inlier_ratio = (
    inlier_count /
    len(good)
) * 100

print("RANSAC inliers:", inlier_count)
print("Inlier ratio:", f"{inlier_ratio:.2f}%")


# ============================================================
# SPATIAL GRID VOTING
# ============================================================

height, width = image1.shape[:2]

cell_width = width / GRID_COLS
cell_height = height / GRID_ROWS

grid_counts = np.zeros(
    (GRID_ROWS, GRID_COLS),
    dtype=int
)

point_cells = []


for point in inlier_points:

    x, y = point

    col = int(
        x / cell_width
    )

    row = int(
        y / cell_height
    )

    col = min(
        GRID_COLS - 1,
        max(0, col)
    )

    row = min(
        GRID_ROWS - 1,
        max(0, row)
    )

    grid_counts[row, col] += 1

    point_cells.append(
        (row, col)
    )


# ============================================================
# FIND STRONGEST CELL
# ============================================================

best_index = np.unravel_index(
    np.argmax(grid_counts),
    grid_counts.shape
)

best_row = best_index[0]
best_col = best_index[1]

best_votes = int(
    grid_counts[
        best_row,
        best_col
    ]
)


# ============================================================
# NEIGHBORING CELL SUPPORT
# ============================================================

neighbor_votes = 0

for row in range(
    max(0, best_row - 1),
    min(GRID_ROWS, best_row + 2)
):

    for col in range(
        max(0, best_col - 1),
        min(GRID_COLS, best_col + 2)
    ):

        if row == best_row and col == best_col:
            continue

        neighbor_votes += int(
            grid_counts[row, col]
        )


local_votes = (
    best_votes +
    neighbor_votes
)

local_support = (
    local_votes /
    max(1, inlier_count)
) * 100


# ============================================================
# CELL CENTER
# ============================================================

cell_center_x = (
    best_col * cell_width
    + cell_width / 2
)

cell_center_y = (
    best_row * cell_height
    + cell_height / 2
)


normalized_x = (
    cell_center_x /
    width
) * 100

normalized_y = (
    cell_center_y /
    height
) * 100


# ============================================================
# SPATIAL CONCENTRATION
# ============================================================

occupied_cells = np.sum(
    grid_counts > 0
)

concentration = (
    best_votes /
    max(1, inlier_count)
) * 100


# ============================================================
# LOCALIZATION SCORE
# ============================================================

# Score is based on:
# 60% strongest-cell support
# 40% neighboring-cell support

localization_score = (
    0.60 * concentration +
    0.40 * local_support
)

localization_score = min(
    100.0,
    localization_score
)


if localization_score >= 70:

    decision = "STRONG LOCALIZATION"

elif localization_score >= 45:

    decision = "MODERATE LOCALIZATION"

else:

    decision = "WEAK LOCALIZATION"


# ============================================================
# PRINT GRID
# ============================================================

print("\n")
print("=" * 90)
print("GRID VOTING RESULTS")
print("=" * 90)

print("\nGrid votes:")

for row in range(GRID_ROWS):

    print(
        " ".join(
            f"{grid_counts[row, col]:3d}"
            for col in range(GRID_COLS)
        )
    )

print(
    "\nStrongest cell:",
    f"Row {best_row + 1}, Column {best_col + 1}"
)

print(
    "Votes in strongest cell:",
    best_votes
)

print(
    "Neighbor support:",
    neighbor_votes
)

print(
    "Local support:",
    f"{local_support:.2f}%"
)

print(
    "Spatial concentration:",
    f"{concentration:.2f}%"
)


# ============================================================
# FINAL LOCALIZATION
# ============================================================

print("\n")
print("=" * 90)
print("LOCALIZATION RESULT")
print("=" * 90)

print(
    f"\nEstimated location:"
)

print(
    f"X = {cell_center_x:.2f} px"
)

print(
    f"Y = {cell_center_y:.2f} px"
)

print(
    f"\nNormalized location:"
)

print(
    f"X = {normalized_x:.2f}%"
)

print(
    f"Y = {normalized_y:.2f}%"
)

print(
    f"\nLocalization score: "
    f"{localization_score:.2f}%"
)

print(
    f"Decision: {decision}"
)


# ============================================================
# VISUALIZATION
# ============================================================

visual = image1.copy()


# Draw grid

for i in range(1, GRID_COLS):

    x = int(
        i * cell_width
    )

    cv2.line(
        visual,
        (x, 0),
        (x, height),
        (255, 255, 255),
        2
    )


for i in range(1, GRID_ROWS):

    y = int(
        i * cell_height
    )

    cv2.line(
        visual,
        (0, y),
        (width, y),
        (255, 255, 255),
        2
    )


# Draw verified points

for px, py in inlier_points:

    cv2.circle(
        visual,
        (int(px), int(py)),
        10,
        (0, 255, 0),
        -1
    )


# Highlight strongest cell

x1 = int(
    best_col * cell_width
)

y1 = int(
    best_row * cell_height
)

x2 = int(
    (best_col + 1) * cell_width
)

y2 = int(
    (best_row + 1) * cell_height
)

cv2.rectangle(
    visual,
    (x1, y1),
    (x2, y2),
    (0, 0, 255),
    8
)


# Mark cell center

cv2.drawMarker(
    visual,
    (
        int(cell_center_x),
        int(cell_center_y)
    ),
    (0, 0, 255),
    cv2.MARKER_CROSS,
    60,
    6
)


# ============================================================
# SAVE VISUALIZATION
# ============================================================

os.makedirs(
    "outputs/visualizations",
    exist_ok=True
)

os.makedirs(
    "outputs/reports",
    exist_ok=True
)

OUTPUT = (
    "outputs/visualizations/"
    "spatial_voting.jpg"
)

REPORT = (
    "outputs/reports/"
    "spatial_voting.txt"
)

cv2.imwrite(
    OUTPUT,
    visual
)


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT,
    "w"
) as f:

    f.write(
        "SIH26166 SPATIAL VOTING LOCALIZATION\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Good matches: {len(good)}\n"
    )

    f.write(
        f"RANSAC inliers: {inlier_count}\n"
    )

    f.write(
        f"Inlier ratio: {inlier_ratio:.2f}%\n\n"
    )

    f.write(
        f"Strongest cell: "
        f"Row {best_row + 1}, "
        f"Column {best_col + 1}\n"
    )

    f.write(
        f"Strongest cell votes: "
        f"{best_votes}\n"
    )

    f.write(
        f"Neighbor votes: "
        f"{neighbor_votes}\n"
    )

    f.write(
        f"Local support: "
        f"{local_support:.2f}%\n"
    )

    f.write(
        f"Spatial concentration: "
        f"{concentration:.2f}%\n\n"
    )

    f.write(
        f"Location X: "
        f"{cell_center_x:.2f} px\n"
    )

    f.write(
        f"Location Y: "
        f"{cell_center_y:.2f} px\n"
    )

    f.write(
        f"Normalized X: "
        f"{normalized_x:.2f}%\n"
    )

    f.write(
        f"Normalized Y: "
        f"{normalized_y:.2f}%\n\n"
    )

    f.write(
        f"Localization score: "
        f"{localization_score:.2f}%\n"
    )

    f.write(
        f"Decision: "
        f"{decision}\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 90)

print(
    "Visualization saved to:"
)

print(OUTPUT)

print(
    "\nReport saved to:"
)

print(REPORT)

print("=" * 90)
print("SPATIAL VOTING COMPLETED")
print("=" * 90)
