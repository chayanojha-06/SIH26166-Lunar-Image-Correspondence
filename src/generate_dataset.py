import cv2
import numpy as np
import os


INPUT_IMAGE = "data/raw/image1.jpg"
OUTPUT_FOLDER = "data/processed"


# ==========================================
# LOAD IMAGE
# ==========================================

image = cv2.imread(INPUT_IMAGE)

if image is None:
    print("ERROR: Could not load image.")
    exit()

print("Original image loaded!")
print("Size:", image.shape)


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# SAVE FUNCTION
# ==========================================

def save_image(name, img):

    path = os.path.join(
        OUTPUT_FOLDER,
        name
    )

    cv2.imwrite(path, img)

    print("Created:", path)


# ==========================================
# 1. SCALE VARIATIONS
# ==========================================

for scale in [0.2, 0.4, 0.6, 0.8]:

    transformed = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    save_image(
        f"scale_{int(scale * 100)}.jpg",
        transformed
    )


# ==========================================
# 2. ROTATION VARIATIONS
# ==========================================

height, width = image.shape[:2]

center = (
    width // 2,
    height // 2
)

for angle in [15, 30, 45, 60]:

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    transformed = cv2.warpAffine(
        image,
        matrix,
        (width, height)
    )

    save_image(
        f"rotation_{angle}.jpg",
        transformed
    )


# ==========================================
# 3. BRIGHTNESS VARIATIONS
# ==========================================

for brightness in [-30, -60, 30, 60]:

    transformed = cv2.convertScaleAbs(
        image,
        alpha=1.0,
        beta=brightness
    )

    save_image(
        f"brightness_{brightness}.jpg",
        transformed
    )


# ==========================================
# 4. CONTRAST VARIATIONS
# ==========================================

for contrast in [0.5, 0.75, 1.5, 2.0]:

    transformed = cv2.convertScaleAbs(
        image,
        alpha=contrast,
        beta=0
    )

    save_image(
        f"contrast_{str(contrast).replace('.', '_')}.jpg",
        transformed
    )


# ==========================================
# 5. BLUR VARIATIONS
# ==========================================

for kernel in [3, 5, 9]:

    transformed = cv2.GaussianBlur(
        image,
        (kernel, kernel),
        0
    )

    save_image(
        f"blur_{kernel}.jpg",
        transformed
    )


# ==========================================
# 6. NOISE
# ==========================================

noise = np.random.normal(
    0,
    15,
    image.shape
).astype(np.float32)

noisy = image.astype(
    np.float32
) + noise

noisy = np.clip(
    noisy,
    0,
    255
).astype(np.uint8)

save_image(
    "noise.jpg",
    noisy
)


# ==========================================
# 7. PERSPECTIVE
# ==========================================

source = np.float32([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
])

destination = np.float32([
    [width * 0.08, height * 0.05],
    [width * 0.92, 0],
    [width * 0.82, height],
    [width * 0.18, height * 0.95]
])

matrix = cv2.getPerspectiveTransform(
    source,
    destination
)

perspective = cv2.warpPerspective(
    image,
    matrix,
    (width, height)
)

save_image(
    "perspective.jpg",
    perspective
)


# ==========================================
# 8. EXTREME COMBINATIONS
# ==========================================

transformed = cv2.resize(
    image,
    None,
    fx=0.3,
    fy=0.3,
    interpolation=cv2.INTER_AREA
)

h, w = transformed.shape[:2]

center = (
    w // 2,
    h // 2
)

matrix = cv2.getRotationMatrix2D(
    center,
    45,
    1.0
)

transformed = cv2.warpAffine(
    transformed,
    matrix,
    (w, h)
)

transformed = cv2.convertScaleAbs(
    transformed,
    alpha=0.5,
    beta=-60
)

save_image(
    "extreme_combined.jpg",
    transformed
)


print("\n==========================================")
print("DATASET GENERATION COMPLETE")
print("==========================================")

print(
    "All generated images are in:",
    OUTPUT_FOLDER
)