"""Part 2 & Bonus: Image Mosaics (panoramic image stitching)."""

import os

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


def save_homography_check(
    img1,
    img2,
    pts1,
    pts2,
    homography,
    inliers,
    output_path,
    title1="Source view",
    title2="Destination view",
):
    """Verify H by mapping inlier points forward and backward across the two views."""
    inlier_indexes = np.where(inliers)[0][:10]
    source = pts1[inlier_indexes]
    true_destination = pts2[inlier_indexes]
    predicted_destination = apply_homography(homography, source)
    predicted_source = apply_homography(
        np.linalg.inv(homography), true_destination
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].scatter(*source.T, c="lime", s=60, label="original points")
    axes[0].scatter(
        *predicted_source.T,
        c="red",
        marker="x",
        s=60,
        label="dest points mapped back with H$^{-1}$",
    )
    axes[0].set_title(title1)
    axes[0].legend(fontsize=7)

    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].scatter(
        *true_destination.T, c="lime", s=60, label="original points"
    )
    axes[1].scatter(
        *predicted_destination.T,
        c="red",
        marker="x",
        s=60,
        label="source points mapped with H",
    )
    axes[1].set_title(title2)
    axes[1].legend(fontsize=7)

    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()


def warp_channel(channel, valid, y0, x0, y1, x1, w00, w01, w10, w11):
    """Inverse-warp a single 2D channel using bilinear interpolation."""
    out_h, out_w = valid.shape
    out_ch = np.zeros((out_h, out_w), dtype=channel.dtype)
    ch = channel.astype(np.float64)
    interpolated = (
        w00 * ch[y0, x0]
        + w01 * ch[y0, x1]
        + w10 * ch[y1, x0]
        + w11 * ch[y1, x1]
    )
    if np.issubdtype(channel.dtype, np.integer):
        out_ch[valid] = np.clip(np.round(interpolated), 0, 255).astype(
            channel.dtype
        )
    else:
        out_ch[valid] = interpolated.astype(channel.dtype)
    return out_ch


def warp_image(image, homography, return_offset=False, source_mask=None):
    """Warp `image` into the destination plane using inverse mapping and bilinear interpolation.

    For color images, each channel is warped separately and stacked together.
    If `return_offset` is True, returns `(warped, (x_min, y_min), valid_mask)`.
    Otherwise, returns `warped`.
    """
    h_s, w_s = image.shape[:2]
    corners = np.array(
        [[0, 0], [w_s - 1, 0], [w_s - 1, h_s - 1], [0, h_s - 1]],
        dtype=np.float64,
    )
    warped_corners = apply_homography(homography, corners)

    x_min = int(np.floor(warped_corners[:, 0].min()))
    x_max = int(np.ceil(warped_corners[:, 0].max()))
    y_min = int(np.floor(warped_corners[:, 1].min()))
    y_max = int(np.ceil(warped_corners[:, 1].max()))

    out_w = x_max - x_min + 1
    out_h = y_max - y_min + 1

    xs = np.arange(x_min, x_max + 1, dtype=np.float64)
    ys = np.arange(y_min, y_max + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    dst_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    inv_homography = np.linalg.inv(homography)
    src_points = apply_homography(inv_homography, dst_points)
    src_x = src_points[:, 0].reshape(out_h, out_w)
    src_y = src_points[:, 1].reshape(out_h, out_w)

    valid = (
        (src_x >= 0)
        & (src_x <= w_s - 1)
        & (src_y >= 0)
        & (src_y <= h_s - 1)
    )

    vx = src_x[valid]
    vy = src_y[valid]

    x0 = np.floor(vx).astype(np.int32)
    y0 = np.floor(vy).astype(np.int32)
    x1 = np.minimum(x0 + 1, w_s - 1)
    y1 = np.minimum(y0 + 1, h_s - 1)

    dx = vx - x0
    dy = vy - y0

    w00 = (1.0 - dx) * (1.0 - dy)
    w01 = dx * (1.0 - dy)
    w10 = (1.0 - dx) * dy
    w11 = dx * dy

    if image.ndim == 3:
        channels = [
            warp_channel(
                image[:, :, c], valid, y0, x0, y1, x1, w00, w01, w10, w11
            )
            for c in range(image.shape[2])
        ]
        warped = np.dstack(channels)
    else:
        warped = warp_channel(
            image, valid, y0, x0, y1, x1, w00, w01, w10, w11
        )

    if source_mask is not None:
        warped_mask = (
            warp_channel(
                source_mask.astype(np.float64),
                valid,
                y0,
                x0,
                y1,
                x1,
                w00,
                w01,
                w10,
                w11,
            )
            > 0.5
        )
        valid = valid & warped_mask
        warped[~valid] = 0

    if return_offset:
        return warped, (x_min, y_min), valid
    return warped


def create_mosaic(
    warped_source,
    offset,
    warped_mask,
    destination,
    destination_mask=None,
    return_mask=False,
):
    """Merge the warped source image and the destination image into a common canvas."""
    x_min, y_min = offset
    h_w, w_w = warped_source.shape[:2]
    h_d, w_d = destination.shape[:2]

    mosaic_x_min = min(0, x_min)
    mosaic_y_min = min(0, y_min)
    mosaic_x_max = max(w_d - 1, x_min + w_w - 1)
    mosaic_y_max = max(h_d - 1, y_min + h_w - 1)

    mosaic_w = mosaic_x_max - mosaic_x_min + 1
    mosaic_h = mosaic_y_max - mosaic_y_min + 1

    if destination.ndim == 3:
        mosaic = np.zeros(
            (mosaic_h, mosaic_w, destination.shape[2]),
            dtype=destination.dtype,
        )
    else:
        mosaic = np.zeros((mosaic_h, mosaic_w), dtype=destination.dtype)
    combined_mask = np.zeros((mosaic_h, mosaic_w), dtype=bool)

    wx0 = x_min - mosaic_x_min
    wy0 = y_min - mosaic_y_min
    wx1, wy1 = wx0 + w_w, wy0 + h_w

    roi_warped = mosaic[wy0:wy1, wx0:wx1]
    roi_warped[warped_mask] = warped_source[warped_mask]
    combined_mask[wy0:wy1, wx0:wx1] |= warped_mask

    dx0 = -mosaic_x_min
    dy0 = -mosaic_y_min
    dx1, dy1 = dx0 + w_d, dy0 + h_d

    if destination_mask is None:
        mosaic[dy0:dy1, dx0:dx1] = destination
        combined_mask[dy0:dy1, dx0:dx1] = True
    else:
        roi_dest = mosaic[dy0:dy1, dx0:dx1]
        roi_dest[destination_mask] = destination[destination_mask]
        combined_mask[dy0:dy1, dx0:dx1] |= destination_mask

    if return_mask:
        return mosaic, combined_mask
    return mosaic


def stitch_pair(source_img, destination_img, matches_out=None, check_out=None, warped_out=None, mosaic_out=None,
                source_mask=None, destination_mask=None, source_title="Source view", dest_title="Destination view"):
    """Estimate H (source -> destination), warp source, and create the output mosaic."""
    pts_src, pts_dst, good, kp_src, kp_dst = get_correspondences(
        source_img, destination_img
    )

    if matches_out is not None:
        match_count = min(50, len(good))
        plot_matches(
            source_img,
            kp_src,
            destination_img,
            kp_dst,
            good[:match_count],
            matches_out,
            f"{match_count} SIFT correspondences (KNN k=2 + ratio test)",
        )

    homography, inliers = ransac_homography(pts_src, pts_dst)
    reproj_error = np.linalg.norm(
        apply_homography(homography, pts_src[inliers]) - pts_dst[inliers],
        axis=1,
    ).mean()
    print(
        f"RANSAC inliers: {inliers.sum()} / {len(inliers)} "
        f"(mean reprojection error: {reproj_error:.3f} px)"
    )

    if check_out is not None:
        save_homography_check(
            source_img,
            destination_img,
            pts_src,
            pts_dst,
            homography,
            inliers,
            check_out,
            title1=source_title,
            title2=dest_title,
        )

    warped, offset, warped_mask = warp_image(
        source_img, homography, return_offset=True, source_mask=source_mask
    )
    print(
        f"warped source {source_img.shape[1]}x{source_img.shape[0]} "
        f"-> {warped.shape[1]}x{warped.shape[0]} at offset {offset}"
    )
    if warped_out is not None:
        cv2.imwrite(warped_out, warped)

    mosaic, combined_mask = create_mosaic(
        warped,
        offset,
        warped_mask,
        destination_img,
        destination_mask=destination_mask,
        return_mask=True,
    )
    print(
        f"created mosaic of size {mosaic.shape[1]}x{mosaic.shape[0]}"
    )
    if mosaic_out is not None:
        cv2.imwrite(mosaic_out, mosaic)

    return mosaic, combined_mask, homography


def main():
    pano1 = cv2.imread(os.path.join(DATA_DIR, "pano_image1.jpg"))
    pano2 = cv2.imread(os.path.join(DATA_DIR, "pano_image2.jpg"))

    print("Part 2: Panorama of Provided 2 Images (pano_image1 -> pano_image2)")
    _, _, h_pano = stitch_pair(pano1, pano2, matches_out=os.path.join(OUT_DIR, "2_1_matches.png"),
                               check_out=os.path.join(OUT_DIR, "2_1_homography_check.png"),
                               warped_out=os.path.join(OUT_DIR, "2_2_warped.png"),
                               mosaic_out=os.path.join(OUT_DIR, "2_3_mosaic.png"),
                               source_title="Second View (pano_image1)", dest_title="First View (pano_image2)")
    np.save(os.path.join(OUT_DIR, "H_pano1_to_pano2.npy"), h_pano)

    boat1 = cv2.imread(os.path.join(DATA_DIR, "boat1.jpg"))
    boat2 = cv2.imread(os.path.join(DATA_DIR, "boat2.jpg"))

    print("\nPart 2 (Custom Example): 2-Image St. Petersburg Panorama (boat2 -> boat1)")
    stitch_pair(boat2, boat1, matches_out=os.path.join(OUT_DIR, "2_3_custom_boat_matches.png"),
                check_out=os.path.join(OUT_DIR, "2_3_custom_boat_homography_check.png"),
                warped_out=os.path.join(OUT_DIR, "2_3_custom_boat_warped.png"),
                mosaic_out=os.path.join(OUT_DIR, "2_3_custom_boat_mosaic.png"),
                source_title="Right View (boat2)", dest_title="Left View (boat1)")

    sh21 = cv2.imread(os.path.join(DATA_DIR, "shanghai-21.png"))
    sh22 = cv2.imread(os.path.join(DATA_DIR, "shanghai-22.png"))
    sh23 = cv2.imread(os.path.join(DATA_DIR, "shanghai-23.png"))

    print("\nPart 2 (Second Example): 2-Image Shanghai Panorama (shanghai-23 -> shanghai-21)")
    stitch_pair(sh23, sh21, matches_out=os.path.join(OUT_DIR, "2_3_shanghai_2img_matches.png"),
                check_out=os.path.join(OUT_DIR, "2_3_shanghai_2img_homography_check.png"),
                warped_out=os.path.join(OUT_DIR, "2_3_shanghai_2img_warped.png"),
                mosaic_out=os.path.join(OUT_DIR, "2_3_shanghai_2img_mosaic.png"), source_title="Shanghai View 23",
                dest_title="Shanghai View 21")

    print("\nPart 3 (Bonus): 3-Image Shanghai Panorama (Order A: (21 + 22) + 23)")
    mosaic_12, mask_12, _ = stitch_pair(sh21, sh22, matches_out=os.path.join(OUT_DIR, "3_bonus_step1_matches.png"),
                                        mosaic_out=os.path.join(OUT_DIR, "3_bonus_step1_mosaic.png"),
                                        source_title="Shanghai View 21", dest_title="Shanghai View 22")
    stitch_pair(sh23, mosaic_12, matches_out=os.path.join(OUT_DIR, "3_bonus_step2_matches.png"),
                mosaic_out=os.path.join(OUT_DIR, "3_bonus_mosaic_order_12_3.png"), destination_mask=mask_12,
                source_title="Shanghai View 23", dest_title="Stitched (21 + 22)")

    print("\nPart 3 (Bonus): 3-Image Shanghai Panorama (Order B: (22 + 21) -> + 23 in 21 plane)")
    mosaic_21, mask_21, _ = stitch_pair(sh22, sh21, source_title="Shanghai View 22", dest_title="Shanghai View 21")
    stitch_pair(sh23, mosaic_21, mosaic_out=os.path.join(OUT_DIR, "3_bonus_mosaic_order_21_3.png"),
                destination_mask=mask_21, source_title="Shanghai View 23", dest_title="Stitched (22 -> 21)")


if __name__ == "__main__":
    main()
