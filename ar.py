import os

import cv2
import numpy as np

from correspondences import get_correspondences
from homography import apply_homography, ransac_homography

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "materials")
OUT_DIR = os.path.join(PROJECT_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)

MIN_INLIERS = 10


def find_content_box(video, n_samples=10, thresh=10):
    """Find the non-black picture area of a letterboxed video.

    Samples several frames and merges them, so one dark frame can't fool it.
    Returns (y0, y1, x0, x1) so that frame[y0:y1, x0:x1] has no black bars.
    """
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows = cols = None
    for i in np.linspace(0, max(total - 1, 0), n_samples, dtype=int):
        video.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = video.read()
        if not ok:
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        r = gray.mean(axis=1) > thresh  # rows that are not black
        c = gray.mean(axis=0) > thresh  # columns that are not black
        rows = r if rows is None else (rows | r)
        cols = c if cols is None else (cols | c)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind
    if rows is None or not rows.any() or not cols.any():
        return 0, height, 0, width  # nothing detected -> keep the whole frame
    ys, xs = np.where(rows)[0], np.where(cols)[0]
    return ys[0], ys[-1] + 1, xs[0], xs[-1] + 1


def crop_to_aspect(frame, aspect):
    """Center-crop `frame` so that width / height == aspect."""
    h, w = frame.shape[:2]
    if w / h > aspect:  # too wide -> trim left/right
        new_w = int(round(h * aspect))
        x0 = (w - new_w) // 2
        return frame[:, x0:x0 + new_w]
    new_h = int(round(w / aspect))  # too tall -> trim top/bottom
    y0 = (h - new_h) // 2
    return frame[y0:y0 + new_h, :]


def prepare_overlay(frame, cover_shape):
    """Crop a movie frame to the cover's aspect ratio, resize to cover size."""
    cover_h, cover_w = cover_shape[:2]
    cropped = crop_to_aspect(frame, cover_w / cover_h)
    return cv2.resize(cropped, (cover_w, cover_h), interpolation=cv2.INTER_AREA)


def book_corners(cover_shape, homography):
    """Map the 4 cover corners (TL, TR, BR, BL) into the frame (as in 1.3)."""
    h, w = cover_shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                       dtype=np.float64)
    return apply_homography(homography, corners)


def overlay(frame, source, homography):
    """Paste `source` (cover-sized) over the book in `frame` using H."""
    h, w = frame.shape[:2]
    warped = cv2.warpPerspective(source, homography, (w, h))
    corners = book_corners(source.shape, homography)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 255)
    out = frame.copy()
    out[mask > 0] = warped[mask > 0]
    return out


def estimate_homography(cover, frame):
    """H (cover -> frame) from our SIFT matching + our RANSAC, or None."""
    points_cover, points_frame, good, _, _ = get_correspondences(cover, frame)
    if len(good) < MIN_INLIERS:
        return None
    try:
        homography, inliers = ransac_homography(points_cover, points_frame)
    except RuntimeError:
        return None
    if inliers.sum() < MIN_INLIERS:
        return None
    return homography


def main():
    cover = cv2.imread(os.path.join(DATA_DIR, "cv_cover.jpg"))
    book = cv2.VideoCapture(os.path.join(DATA_DIR, "book.mov"))
    ar = cv2.VideoCapture(os.path.join(DATA_DIR, "ar_source.mov"))
    if cover is None or not book.isOpened() or not ar.isOpened():
        raise FileNotFoundError("check cv_cover.jpg / book.mov / ar_source.mov")

    y0, y1, x0, x1 = find_content_box(ar)
    trim = lambda f: f[y0:y1, x0:x1]
    print(f"[bars] picture area: rows {y0}:{y1}, cols {x0}:{x1}")

    ok, ar_raw = ar.read()
    ar.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ar_first = trim(ar_raw)
    cropped = prepare_overlay(ar_first, cover.shape)
    scale = cover.shape[0] / ar_raw.shape[0]
    new_size = (int(round(ar_raw.shape[1] * scale)), cover.shape[0])
    original_small = cv2.resize(ar_raw, new_size)
    cv2.imwrite(os.path.join(OUT_DIR, "1_4_crop.png"),
                np.hstack([original_small, cropped]))
    print(f"[1.4] ar_source frame {ar_raw.shape[1]}x{ar_raw.shape[0]} "
          f"-> trimmed {ar_first.shape[1]}x{ar_first.shape[0]} "
          f"-> cropped {cropped.shape[1]}x{cropped.shape[0]} "
          f"(aspect {cover.shape[1] / cover.shape[0]:.3f})")

    ok, first = book.read()
    h0_path = os.path.join(OUT_DIR, "H_cover_to_frame0.npy")
    h0 = np.load(h0_path) if os.path.exists(h0_path) else estimate_homography(cover, first)
    if h0 is None:
        raise RuntimeError("could not estimate H for the first frame")
    cv2.imwrite(os.path.join(OUT_DIR, "1_5_overlay_first_frame.png"),
                overlay(first, cropped, h0))
    print("[1.5] saved first-frame overlay")

    book.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = book.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(book.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        os.path.join(OUT_DIR, "1_6_ar_output.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    samples = set(np.linspace(0, max(total - 1, 0), 4, dtype=int).tolist())
    last_h, failures, index = h0, 0, 0
    while True:
        ok, frame = book.read()
        if not ok:
            break
        ok_ar, ar_frame = ar.read()
        if not ok_ar:  # ar_source shorter than book.mov -> loop it
            ar.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, ar_frame = ar.read()
        ar_frame = trim(ar_frame)  # strip the black bars
        homography = estimate_homography(cover, frame)
        if homography is None:  # tracking lost: reuse the previous pose
            homography, failures = last_h, failures + 1
        last_h = homography
        result = overlay(frame, prepare_overlay(ar_frame, cover.shape), homography)
        writer.write(result)
        if index in samples:
            cv2.imwrite(os.path.join(OUT_DIR, f"1_6_frame_{index:04d}.png"), result)
        index += 1
        if index % 50 == 0:
            print(f"      processed {index}/{total} frames")
    writer.release()
    book.release()
    ar.release()
    print(f"[1.6] wrote {index} frames, {failures} used the previous H "
          f"-> out/1_6_ar_output.mp4")


if __name__ == "__main__":
    main()