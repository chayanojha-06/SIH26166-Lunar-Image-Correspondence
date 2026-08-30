import cv2


IMAGE = "data/processed/scale_20.jpg"


print("=" * 60)
print("SIH26166 MULTI-SCALE FEATURE TEST")
print("=" * 60)


image = cv2.imread(IMAGE)

if image is None:
    print("ERROR: Could not load image.")
    exit()


gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# ==========================================
# TEST DIFFERENT SIFT SETTINGS
# ==========================================

settings = [
    ("Baseline", 10000, 0.03),
    ("More Features", 20000, 0.03),
    ("Low Threshold", 20000, 0.01),
    ("Very Low Threshold", 30000, 0.005)
]


for name, features, threshold in settings:

    sift = cv2.SIFT_create(
        nfeatures=features,
        contrastThreshold=threshold
    )

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    count = len(keypoints)

    print(
        f"{name:<22} "
        f"Features: {count}"
    )


print("=" * 60)
print("TEST COMPLETED")
print("=" * 60)