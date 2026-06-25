import numpy as np
import cv2

def generate_silhouette_mask(image_shape, points):
    """
    Generate a binary mask from a silhouette defined by edge segments.

    The points represent edge segments where points[i] is the ith edge with two endpoints.
    points[:, 0, :] forms one side of the silhouette and points[:, 1, :] forms the other side.
    These are combined to form a closed contour which is then filled.

    Args:
        image_shape (tuple): Shape of the image (height, width).
        points (np.ndarray): Array of edge segments with shape (N, 2, 2).
                            points[i, 0, :] and points[i, 1, :] are the two endpoints of edge i.

    Returns:
        tuple: (mask, bbox_xyxy) where:
            - mask (np.ndarray): Binary mask with the silhouette filled.
            - bbox_xyxy (list): Bounding box [x1, y1, x2, y2] of the silhouette.
    """
    # Filter out invalid edges (those with NaN values)
    valid_mask = ~np.isnan(points).any(axis=(1, 2))  # (N,) boolean mask
    valid_points = points[valid_mask]  # (M, 2, 2) where M <= N

    if len(valid_points) == 0:
        # No valid points
        valid_mask = ~np.isnan(points.reshape(-1,2)).any(axis=1)  # (N*2,) boolean mask for all endpoints
        valid_points = points.reshape(-1, 2)[valid_mask]  # (K, 2) where K <= N*2
        
        bbox_xywh = cv2.boundingRect(valid_points.astype(np.int32)) if len(valid_points) > 0 else (0, 0, 0, 0)
        bbox_xyxy = [bbox_xywh[0], bbox_xywh[1], bbox_xywh[0]+bbox_xywh[2], bbox_xywh[1]+bbox_xywh[3]]
        return np.zeros(image_shape, dtype=np.uint8), bbox_xyxy

    # Extract the two sides of the silhouette
    side1 = valid_points[:, 0, :]  # (M, 2) - first endpoints
    side2 = valid_points[:, 1, :]  # (M, 2) - second endpoints

    # Create closed contour by concatenating side1 forward with side2 reversed
    # This creates a closed polygon: side1[0] -> side1[1] -> ... -> side1[-1] -> side2[-1] -> ... -> side2[0] -> side1[0]
    contour_points = np.concatenate([side1, side2[::-1]], axis=0)  # (2*M, 2)

    if len(contour_points) < 3:
        # Not enough points to form a polygon
        return np.zeros(image_shape, dtype=np.uint8), [0, 0, 0, 0]

    # Convert to int32 for OpenCV
    contour_points = contour_points.astype(np.int32)

    # Create mask and fill the polygon
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [contour_points], 255)

    # Compute bounding box
    bbox_xywh = cv2.boundingRect(contour_points)
    bbox_xyxy = [bbox_xywh[0], bbox_xywh[1], bbox_xywh[0]+bbox_xywh[2], bbox_xywh[1]+bbox_xywh[3]]

    return mask, bbox_xyxy


def compute_convex_hull_2d(points_2d, image_shape=None):
    """
    Compute convex hull from unordered 2D points and optionally generate a mask.

    Args:
        points_2d (np.ndarray): Unordered 2D points with shape (N, 2).
        image_shape (tuple, optional): Image shape (height, width) for mask generation.
                                      If None, only returns hull points.

    Returns:
        tuple: (hull_points, mask, bbox_xyxy)
            - hull_points (np.ndarray): Ordered convex hull points (K, 2)
            - mask (np.ndarray or None): Binary mask if image_shape provided
            - bbox_xyxy (list): Bounding box [x1, y1, x2, y2]
    """
    # Filter out invalid points (NaN or inf)
    valid_mask = np.isfinite(points_2d).all(axis=1)
    valid_points = points_2d[valid_mask]

    if len(valid_points) < 3:
        # Need at least 3 points for convex hull
        empty_mask = np.zeros(image_shape, dtype=np.uint8) if image_shape else None
        return np.zeros((0, 2)), empty_mask, [0, 0, 0, 0]

    # Convert to int32 for OpenCV
    valid_points_int = valid_points.astype(np.int32)

    # Compute convex hull
    hull = cv2.convexHull(valid_points_int)
    hull_points = hull.squeeze()  # (K, 2)

    if len(hull_points.shape) == 1:
        # Degenerate case (single point)
        empty_mask = np.zeros(image_shape, dtype=np.uint8) if image_shape else None
        return np.zeros((0, 2)), empty_mask, [0, 0, 0, 0]

    # Compute bounding box
    bbox_xywh = cv2.boundingRect(hull_points)
    bbox_xyxy = [bbox_xywh[0], bbox_xywh[1],
                 bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]]

    # Generate mask if image_shape is provided
    if image_shape is not None:
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [hull_points], 255)
    else:
        mask = None

    return hull_points.astype(np.float32), mask, bbox_xyxy



def upsample_ruled_surface(blade_3d, blade_valid=None, num_samples_along=2, num_samples_across=5):
    """
    Upsample a ruled surface blade geometry by interpolating along and across rulings.

    A ruled surface is defined by a set of line segments (rulings) where:
    - blade_3d[i, 0, :] and blade_3d[i, 1, :] are the two endpoints of the i-th ruling
    - Consecutive rulings blade_3d[i-1] and blade_3d[i] form a surface patch

    Args:
        blade_3d (np.ndarray): Blade 3D points with shape (N, 2, 3) where:
            - N is the number of rulings along the blade length
            - 2 represents the two edges of the blade
            - 3 is the XYZ coordinates
        blade_valid (np.ndarray, optional): Validity mask with shape (N, 2). If None, all points are considered valid.
        num_samples_along (int): Number of interpolation samples between consecutive rulings (default: 2).
            Total rulings after interpolation = N + (N-1) * (num_samples_along - 1)
        num_samples_across (int): Number of interpolation samples across each ruling (default: 5).
            Total points per ruling = num_samples_across

    Returns:
        tuple: (blade_3d_upsampled, blade_valid_upsampled)
            - blade_3d_upsampled (np.ndarray): Upsampled blade points with shape (N_new * num_samples_across, 3)
                where N_new = N + (N-1) * (num_samples_along - 1)
            - blade_valid_upsampled (np.ndarray): Upsampled validity mask with shape (N_new * num_samples_across,)
    """
    N = blade_3d.shape[0]

    if blade_valid is None:
        blade_valid = np.ones((N, 2), dtype=bool)

    # Step 1: Interpolate between consecutive rulings (along the blade length)
    rulings_upsampled = []
    valid_upsampled = []

    for i in range(N):
        # Add the current ruling
        rulings_upsampled.append(blade_3d[i])  # (2, 3)
        valid_upsampled.append(blade_valid[i])  # (2,)

        # Interpolate between current and next ruling
        if i < N - 1:
            for j in range(1, num_samples_along):
                alpha = j / num_samples_along  # Interpolation weight
                interpolated_ruling = (1 - alpha) * blade_3d[i] + alpha * blade_3d[i + 1]  # (2, 3)
                interpolated_valid = np.logical_and(blade_valid[i], blade_valid[i + 1])  # (2,)

                rulings_upsampled.append(interpolated_ruling)
                valid_upsampled.append(interpolated_valid)

    rulings_upsampled = np.array(rulings_upsampled)  # (N_new, 2, 3)
    valid_upsampled = np.array(valid_upsampled)  # (N_new, 2)
    N_new = rulings_upsampled.shape[0]

    # Step 2: Interpolate across each ruling (between the two edges)
    blade_3d_upsampled = []
    blade_valid_upsampled = []

    for i in range(N_new):
        edge0 = rulings_upsampled[i, 0, :]  # (3,) - first edge point
        edge1 = rulings_upsampled[i, 1, :]  # (3,) - second edge point
        valid0 = valid_upsampled[i, 0]
        valid1 = valid_upsampled[i, 1]

        # Interpolate across the ruling
        for k in range(num_samples_across):
            beta = k / (num_samples_across - 1) if num_samples_across > 1 else 0.5  # Interpolation weight
            interpolated_point = (1 - beta) * edge0 + beta * edge1  # (3,)
            interpolated_valid = valid0 and valid1  # Both edges must be valid

            blade_3d_upsampled.append(interpolated_point)
            blade_valid_upsampled.append(interpolated_valid)

    blade_3d_upsampled = np.array(blade_3d_upsampled)  # (N_new * num_samples_across, 3)
    blade_valid_upsampled = np.array(blade_valid_upsampled, dtype=bool)  # (N_new * num_samples_across,)

    return blade_3d_upsampled, blade_valid_upsampled
