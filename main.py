"""Run the planar-homography assignment pipeline."""

import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from correspondences import get_correspondences, plot_matches
from homography import apply_homography, ransac_homography


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "materials")
OUT_DIR = os.path.join(PROJECT_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    cover = cv2.imread(os.path.join(DATA_DIR, "cv_cover.jpg"))
    capture = cv2.VideoCapture(os.path.join(DATA_DIR, "book.mov"))
    has_frame, first_frame = capture.read()
    capture.release()
    if cover is None or not has_frame:
        raise FileNotFoundError(
            "could not load cv_cover.jpg or the first frame of book.mov"
        )

    pts_cover, pts_frame, good, kp_cover, kp_frame = get_correspondences(
        cover, first_frame
    )
    print(f"[1.1] good matches after ratio test: {len(good)}")


    # -------------- 1.1 --------------------
    
    match_count = min(50, len(good))
    plot_matches(
        cover,
        kp_cover,
        first_frame,
        kp_frame,
        good[:match_count],
        os.path.join(OUT_DIR, "1_1_matches.png"),
        f"{match_count} SIFT correspondences (KNN k=2 + ratio test)",
    )
    if len(good) < 4:
        raise RuntimeError("not enough correspondences to estimate a homography")


    # --------------- 1.2 ---------------------
    
    homography, inliers = ransac_homography(pts_cover, pts_frame)
    print(f"[1.2] RANSAC inliers: {inliers.sum()} / {len(inliers)}")
    print("H (cover -> frame0) =\n", np.round(homography, 5))

    cv_homography, _ = cv2.findHomography(
        pts_cover, pts_frame, cv2.RANSAC, 3.0
    )
    if cv_homography is None:
        raise RuntimeError("OpenCV could not estimate a reference homography")
    transformed = apply_homography(homography, pts_cover[inliers])
    transformed_cv = apply_homography(
        cv_homography, pts_cover[inliers]
    )
    print(
        "      mean reprojection error (ours)   : %.3f px"
        % np.linalg.norm(
            transformed - pts_frame[inliers], axis=1
        ).mean()
    )
    print(
        "      mean reprojection error (OpenCV) : %.3f px"
        % np.linalg.norm(
            transformed_cv - pts_frame[inliers], axis=1
        ).mean()
    )

    inlier_indexes = np.where(inliers)[0][:10]
    source = pts_cover[inlier_indexes]
    true_destination = pts_frame[inlier_indexes]
    predicted_destination = apply_homography(homography, source)
    predicted_source = apply_homography(
        np.linalg.inv(homography), true_destination
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(cv2.cvtColor(cover, cv2.COLOR_BGR2RGB))
    axes[0].scatter(*source.T, c="lime", s=60, label="original points")
    axes[0].scatter(
        *predicted_source.T,
        c="red",
        marker="x",
        s=60,
        label="frame points mapped back with H$^{-1}$",
    )
    axes[0].set_title("Book cover")
    axes[0].legend(fontsize=7)

    axes[1].imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    axes[1].scatter(
        *true_destination.T, c="lime", s=60, label="original points"
    )
    axes[1].scatter(
        *predicted_destination.T,
        c="red",
        marker="x",
        s=60,
        label="cover points mapped with H",
    )
    axes[1].set_title("First video frame")
    axes[1].legend(fontsize=7)
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "1_2_homography_check.png"), dpi=130
    )
    plt.close()
    
    # ---------------- 1.3 ---------------------

    height, width = cover.shape[:2]
    cover_corners = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float64,
    )
    book_corners = apply_homography(homography, cover_corners)
    print("[1.3] book corners in first frame (TL, TR, BR, BL):")
    print(np.round(book_corners, 1))
    np.save(
        os.path.join(OUT_DIR, "book_corners_frame0.npy"), book_corners
    )
    np.save(
        os.path.join(OUT_DIR, "H_cover_to_frame0.npy"), homography
    )

    visualization = first_frame.copy()
    cv2.polylines(
        visualization,
        [book_corners.astype(np.int32).reshape(-1, 1, 2)],
        True,
        (0, 255, 0),
        3,
    )
    for (x, y), name in zip(
        book_corners, ["TL", "TR", "BR", "BL"]
    ):
        cv2.circle(
            visualization, (int(x), int(y)), 6, (0, 0, 255), -1
        )
        cv2.putText(
            visualization,
            name,
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    cv2.imwrite(
        os.path.join(OUT_DIR, "1_3_book_corners.png"), visualization
    )


if __name__ == "__main__":
    main()