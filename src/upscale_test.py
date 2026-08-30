import cv2


IMAGE = "data/processed/scale_20.jpg"


print("=" * 65)
print("SIH26166 UPSCALE + SIFT TEST")
print("=" * 65)


image = cv2.imread(IMAGE)

if image is None:
    print("ERROR: Could not load image.")
    exit()


gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


scales = [1, 2, 3, 4]


for scale in scales:

    if scale == 1:

        test_image = gray

    else:

        test_image = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    sift = cv2.SIFT_create(
        nfeatures=10000,
        contrastThreshold=0.03
    )

    keypoints, descriptors = sift.detectAndCompute(
        test_image,
        None
    )

    print(
        f"Upscale {scale}x"
        f"{' ' * 10}"
        f"Features: {len(keypoints)}"
    )


print("=" * 65)
print("TEST COMPLETED")
print("=" * 65)