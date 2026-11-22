"""Homography estimation and point transformation."""

import numpy as np


def compute_homography(source_points, destination_points):
    """Estimate H from source to destination with H[2, 2] fixed to 1."""
    source = np.asarray(source_points, dtype=np.float64)
    destination = np.asarray(destination_points, dtype=np.float64)
    point_count = source.shape[0]
    if point_count < 4 or destination.shape[0] != point_count:
        raise ValueError("need at least 4 corresponding point pairs")

    matrix = np.zeros((2 * point_count, 8))
    target = np.zeros(2 * point_count)
    for index, (source_point, destination_point) in enumerate(
        zip(source, destination)
    ):
        x, y = source_point
        destination_x, destination_y = destination_point
        matrix[2 * index] = [x, y, 1, 0, 0, 0, -x * destination_x, -y * destination_x]
        matrix[2 * index + 1] = [0, 0, 0, x, y, 1, -x * destination_y, -y * destination_y]
        target[2 * index] = destination_x
        target[2 * index + 1] = destination_y

    parameters, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    return np.append(parameters, 1.0).reshape(3, 3)


def apply_homography(homography, points):
    """Transform an (N, 2) point array with a 3 x 3 homography."""
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.hstack(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)]
    )
    transformed = homogeneous @ homography.T
    return transformed[:, :2] / transformed[:, 2:3]


def ransac_homography(
    source_points, destination_points, iterations=2000, threshold=3.0, seed=0
):
    """Estimate a robust homography using RANSAC and least squares."""
    source = np.asarray(source_points, dtype=np.float64)
    destination = np.asarray(destination_points, dtype=np.float64)
    point_count = len(source)
    if point_count < 4:
        raise ValueError("need at least 4 corresponding point pairs")

    random = np.random.default_rng(seed)
    best_inliers = np.zeros(point_count, dtype=bool)
    for _ in range(iterations):
        sample = random.choice(point_count, 4, replace=False)
        try:
            candidate = compute_homography(source[sample], destination[sample])
            errors = np.linalg.norm(apply_homography(candidate, source) - destination, axis=1)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not np.all(np.isfinite(errors)):
            continue
        inliers = errors < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers.sum() < 4:
        raise RuntimeError("RANSAC did not find 4 homography inliers")
    
    homography = compute_homography(source[best_inliers], destination[best_inliers])
    
    return homography, best_inliers
