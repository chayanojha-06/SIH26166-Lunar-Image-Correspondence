import cv2


def match_images(image1_path, image2_path):
    # Load images
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load one or both images.")

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and calculate descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    print(f"Image 1 keypoints: {len(keypoints1)}")
    print(f"Image 2 keypoints: {len(keypoints2)}")

    # Match descriptors
    matcher = cv2.BFMatcher()

    matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    # Lowe's ratio test
    good_matches = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"Good matches: {len(good_matches)}")

    return img1, img2, keypoints1, keypoints2, good_matches