"""SIFT keypoint detection and correspondence matching."""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_correspondences(img1, img2, ratio=0.75):
    """Return SIFT correspondences accepted by Lowe's ratio test."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    if descriptors1 is None or descriptors2 is None:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            [],
            keypoints1,
            keypoints2,
        )

    knn_matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        descriptors1, descriptors2, k=2
    )
    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            good_matches.append(best)

    good_matches.sort(key=lambda match: match.distance)
    points1 = np.float32(
        [keypoints1[match.queryIdx].pt for match in good_matches]
    )
    points2 = np.float32(
        [keypoints2[match.trainIdx].pt for match in good_matches]
    )
    return points1, points2, good_matches, keypoints1, keypoints2


def plot_matches(
    img1, keypoints1, img2, keypoints2, matches, output_path, title
):
    """Draw and save matched keypoints."""
    visualization = cv2.drawMatches(
        img1,
        keypoints1,
        img2,
        keypoints2,
        matches,
        None,
        matchColor=(0, 0, 255),
        singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.figure(figsize=(14, 6))
    plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()
