import cv2
import numpy as np

import torch
import sys
sys.path.append('.')
sys.path.append('..')


from utils.camera import project_3d_with_intrinsics, project_model_base_to_4k_pinhole, apply_intrinsics_transform_to_2d


# get_points_on_a_grid from spatial_tracker_v2
def meshgrid2d(B, Y, X, stack=False, norm=False, device="cuda"):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y - 1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x
    
def get_points_on_a_grid(grid_size, interp_shape,
                          grid_center=(0, 0), device="cuda"):
    if grid_size == 1:
        return torch.tensor([interp_shape[1] / 2, 
                             interp_shape[0] / 2], device=device)[
            None, None
        ]

    grid_y, grid_x = meshgrid2d(
        1, grid_size, grid_size, stack=False, norm=False, device=device
    )
    step = interp_shape[1] // 64
    if grid_center[0] != 0 or grid_center[1] != 0:
        grid_y = grid_y - grid_size / 2.0
        grid_x = grid_x - grid_size / 2.0
    grid_y = step + grid_y.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[0] - step * 2
    )
    grid_x = step + grid_x.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[1] - step * 2
    )

    grid_y = grid_y + grid_center[0]
    grid_x = grid_x + grid_center[1]
    xy = torch.stack([grid_x, grid_y], dim=-1).to(device)
    return xy





def load_model_base_and_preprocess(frame_est_model_base, valid_joints, img_h, img_w, intr_original, intr_transform, root_idx):
    intr_resized = intr_transform @ intr_original
    
    frame_body_3d = frame_est_model_base['joints3d_h36m']  # (N_body, 3)
    if 'focal_length' in frame_est_model_base:
        focal_length = frame_est_model_base['focal_length'][0]
    elif 'scaled_focal_length' in frame_est_model_base:
        focal_length = frame_est_model_base['scaled_focal_length']
    else:
        focal_length = 75000.0
        
    frame_body_2d = project_model_base_to_4k_pinhole(frame_body_3d, focal_length, img_h, img_w)

    frame_body_2d_on_crop = apply_intrinsics_transform_to_2d(frame_body_2d.copy(), intr_transform)

    skeleton_3d = frame_body_3d[valid_joints>0.5].copy()  # Only valid joints for alignment
    skeleton_2d = frame_body_2d[valid_joints>0.5].copy()
    # Solve PnP using intr_new
    
    success, rvec, tvec = cv2.solvePnP(skeleton_3d, skeleton_2d, intr_original, None,  # No distortion coefficients
                                            flags=cv2.SOLVEPNP_ITERATIVE)

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Transform all skeleton points (including invalid ones)
    # Correct formula: transformed = (R @ (points - mean).T + t).T
    skeleton_3d = (rmat @ skeleton_3d.T + tvec).T  # (17, 3)

    cam_t = skeleton_3d[root_idx]-frame_body_3d[root_idx]  # Get the camera-space coordinates of the root joint after transformation
    skeleton_3d = frame_body_3d + cam_t  # Apply the same translation to all joints to maintain relative structure

    # if is not motionbert, apply to make it project to the same 2D points as original intrinsics; if is motionbert, orthogonal projection and we cannot do anything about it
    return skeleton_3d, frame_body_2d, frame_body_2d_on_crop, intr_resized




def load_point_map(frame_data):
    """
    Reconstruct (3, H, W) point_map from saved data.
    - New format: 'depth' (float16) + 'intrinsics' -> reconstruct X, Y from depth + K
    - Old format: 'point_map' (3, H, W) -> use directly (backward compatibility)
    Also returns depth_conf as float32 in [0, 1].
    """
    #if 'point_map' in frame_data:
    #    point_map = frame_data['point_map'].astype(np.float32)
    #    depth_conf = frame_data['depth_conf'].astype(np.float32)
    #else:
    depth = frame_data['depth'].astype(np.float32)
    intr = frame_data['intrinsics']
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - intr[0, 2]) * depth / intr[0, 0]
    Y = (v - intr[1, 2]) * depth / intr[1, 1]
    point_map = np.stack([X, Y, depth], axis=0)  # (3, H, W)
    try:
        depth_conf = frame_data['depth_conf'].astype(np.float32) / 255.0
    except KeyError:
        depth_conf = np.ones((H, W), dtype=np.float32)
    if 'point_map' in frame_data:
        # For backward compatibility, also check if point_map is available and compare
        point_map_old = frame_data['point_map'].astype(np.float32)
        if not np.allclose(point_map, point_map_old, atol=5e-3):
            print("Warning: Reconstructed point_map differs from saved point_map. Using reconstructed version.")
    return point_map, depth_conf


def extract_3d_from_point_map(frame_data, ref_skeleton_2d, skeleton_valid,
                              mask_blade, blade_grid_size=8,
                              blade_depth_stability_patch=7,
                              blade_depth_stability_thresh=0.005,
                              mask_blade2=None):
    """
    Extract 3D points from dense point_map using GT 2D joints and GT blade mask.
    Uses grid sampling for blade region (similar to run_spatial_tracker_v2.py).

    Args:
        frame_data (dict): Frame data containing 'point_map' (3, H, W).
        ref_skeleton_2d (np.ndarray): Reference 2D joint locations (17, 2).
        skeleton_valid (np.ndarray): Skeleton validity mask (17,).
        mask_blade (np.ndarray): GT blade mask (H, W).
        blade_depth_stability_patch (int): Half-size of local patch for depth stability check.
        blade_depth_stability_thresh (float): Max allowed depth std dev (meters) in the
            local patch; points with higher local variance are discarded as unstable.
        mask_blade2 (np.ndarray, optional): GT blade2 mask (H, W). If provided, blade2_3d_est
            is extracted independently using the same grid sampling logic.

    Returns:
        tuple: (body_3d_est, blade_3d_est, blade2_3d_est)
            - body_3d_est (np.ndarray): Extracted body 3D points (N_valid_joints, 3).
            - blade_3d_est (np.ndarray): Extracted blade 3D points (N_blade_grid_points, 3).
            - blade2_3d_est (np.ndarray or None): Extracted blade2 3D points, or None if mask_blade2 not provided.
    """
    point_map, _ = load_point_map(frame_data)  # (3, H, W)
    H, W = point_map.shape[1], point_map.shape[2]
    depth = point_map[2]  # (H, W)

    # Extract body 3D points using GT 2D joint locations
    valid_body_3d_est = []
    for joint_idx in np.where(skeleton_valid > 0.5)[0]:
        x, y = ref_skeleton_2d[joint_idx]
        x_int, y_int = int(np.clip(x, 0, W-1)), int(np.clip(y, 0, H-1))
        # Sample 3D point from point_map at this 2D location
        point_3d = point_map[:, y_int, x_int]  # (3,)
        valid_body_3d_est.append(point_3d)

    body_3d_est = np.zeros((len(skeleton_valid), 3), dtype=np.float32)  # (17, 3)
    if len(valid_body_3d_est) > 0:
        body_3d_est[skeleton_valid > 0.5] = valid_body_3d_est

    # Extract blade 3D points using grid sampling (similar to run_spatial_tracker_v2.py)
    grid_size_blade = max(H, W) // blade_grid_size  # e.g., 20 pixels
    grid_pts_region_blade = get_points_on_a_grid(grid_size_blade, (H, W), device="cpu")
    grid_pts_region_blade_int = grid_pts_region_blade[0].long().numpy()

    # Filter grid points that fall within the blade mask
    mask_values_region_blade = mask_blade[grid_pts_region_blade_int[..., 1], grid_pts_region_blade_int[..., 0]]
    grid_pts_region_blade_filtered = grid_pts_region_blade[0].numpy()[mask_values_region_blade > 0]

    # Sample 3D points from point_map, discarding points with unstable local depth
    half_p = blade_depth_stability_patch // 2
    blade_3d_est = []
    for pt in grid_pts_region_blade_filtered:
        x, y = int(np.clip(pt[0], 0, W-1)), int(np.clip(pt[1], 0, H-1))

        # Check depth stability in a local patch around (x, y)
        y0, y1 = max(0, y - half_p), min(H, y + half_p + 1)
        x0, x1 = max(0, x - half_p), min(W, x + half_p + 1)
        if depth[y0:y1, x0:x1].std() > blade_depth_stability_thresh:
            continue  # depth is unstable compared to neighbours — skip

        point_3d = point_map[:, y, x]  # (3,)
        blade_3d_est.append(point_3d)

    blade_3d_est = np.array(blade_3d_est) if blade_3d_est else np.zeros((0, 3), dtype=np.float32)

    # Extract blade2 3D points if mask provided
    blade2_3d_est = None
    if mask_blade2 is not None:
        grid_pts_region_blade2 = get_points_on_a_grid(grid_size_blade, (H, W), device="cpu")
        grid_pts_region_blade2_int = grid_pts_region_blade2[0].long().numpy()

        mask_values_region_blade2 = mask_blade2[grid_pts_region_blade2_int[..., 1], grid_pts_region_blade2_int[..., 0]]
        grid_pts_region_blade2_filtered = grid_pts_region_blade2[0].numpy()[mask_values_region_blade2 > 0]

        blade2_3d_est = []
        for pt in grid_pts_region_blade2_filtered:
            x, y = int(np.clip(pt[0], 0, W-1)), int(np.clip(pt[1], 0, H-1))
            y0, y1 = max(0, y - half_p), min(H, y + half_p + 1)
            x0, x1 = max(0, x - half_p), min(W, x + half_p + 1)
            if depth[y0:y1, x0:x1].std() > blade_depth_stability_thresh:
                continue
            blade2_3d_est.append(point_map[:, y, x])
        blade2_3d_est = np.array(blade2_3d_est) if blade2_3d_est else np.zeros((0, 3), dtype=np.float32)

    return body_3d_est, blade_3d_est, blade2_3d_est








def compute_root_aligned_scale(blade_3d_est, body_3d_est, body_3d_ref, skeleton_valid, root_idx):
    """
    Compute 3-DoF translation t and 1-DoF scale s to align body_3d_est (and optionally
    blade_3d_est) while minimising 2D reprojection error.

    Step 1 — translation t: move the root of body_3d_est to the root of body_3d_ref.
        t = body_3d_ref[root_idx] - body_3d_est[root_idx]
        M_i = p_i + t  =>  M[root_idx] == C

    Step 2 — scale s: closed-form least squares such that
        K(s*(M_i - C) + C) ~ K(r_i)
    where r_i is the reference ray for each point.
    Ray-coincidence constraint (K-independent), with d = M_i - C:
        s*(d_x*r_z - r_x*d_z) = r_x*C_z - C_x*r_z   (x)
        s*(d_y*r_z - r_y*d_z) = r_y*C_z - C_y*r_z   (y)
    Solved as: s = dot(a, b) / dot(a, a)

    Args:
        body_3d_est (np.ndarray): (17, 3) estimated body joints in camera space.
        body_3d_ref (np.ndarray): (17, 3) reference body joints (root used for t and C).
        skeleton_valid (np.ndarray): (17,) validity mask.
        root_idx (int): Index of the root joint.
        blade_3d_est (np.ndarray, optional): (N, 3) estimated blade points.
            Blade always uses r = blade_3d_est (preserve original 2D projection).
        body_ref_3d (np.ndarray, optional): (17, 3) reference rays for body joints.
            If None (default), uses body_3d_ref (align body to model_base in 2D).
            Pass body_3d_est to instead preserve the body's original 2D projections.

    Returns:
        t (np.ndarray): translation vector (3,)
        s (float): scale factor
        C (np.ndarray): pivot point = body_3d_ref[root_idx], shape (3,)
    """
    C = body_3d_ref[root_idx]          # (3,) pivot = reference root
    t = C - body_3d_est[root_idx]             # (3,) translation


    def _constraints(P, R, t, C):
        """Linear constraints for s: K(s*(P+t-C)+C) ~ K(R).
        P (N,3): points to transform. R (N,3): reference rays."""
        D = P + t - C                         # (N, 3)
        a_x = D[:, 0] * R[:, 2] - R[:, 0] * D[:, 2]
        b_x = R[:, 0] * C[2] - C[0] * R[:, 2]
        a_y = D[:, 1] * R[:, 2] - R[:, 1] * D[:, 2]
        b_y = R[:, 1] * C[2] - C[1] * R[:, 2]
        return np.concatenate([a_x, a_y]), np.concatenate([b_x, b_y])

    # Body constraints (valid joints only)
    valid = skeleton_valid > 0.5
    a, b = _constraints(body_3d_est[valid], body_3d_est[valid], t, C)

    # Blade constraints: preserve original 2D projection (r = blade_3d_est)
    if blade_3d_est is not None and len(blade_3d_est) > 0:
        a_bl, b_bl = _constraints(blade_3d_est, blade_3d_est, t, C)
        a = np.concatenate([a, a_bl])
        b = np.concatenate([b, b_bl])

    a = a.astype(np.float64)
    b = b.astype(np.float64)

    denom = np.dot(a, a)
    s = float(np.dot(a, b) / denom) if denom > 1e-10 else 1.0

    return t, s, C


def apply_root_align_and_scale(body_3d_est, blade_3d_est, t, s, C):
    """
    Apply translation t then uniform scale s around pivot C to body and blade.

        body_aligned  = s * (body_3d_est  + t - C) + C
        blade_aligned = s * (blade_3d_est + t - C) + C

    Args:
        body_3d_est (np.ndarray): (17, 3) body joints.
        blade_3d_est (np.ndarray): (N, 3) blade points.
        t (np.ndarray): translation (3,).
        s (float): scale factor.
        C (np.ndarray): pivot point (3,).

    Returns:
        body_aligned (np.ndarray): (17, 3)
        blade_aligned (np.ndarray): (N, 3)
    """
    body_aligned  = s * (body_3d_est  + t - C) + C
    blade_aligned = s * (blade_3d_est + t - C) + C
    return body_aligned, blade_aligned



def align_with_root_aligned_and_scale(blade_3d_est, body_3d_est, body_3d_ref, skeleton_valid, root_idx):
    # Stage 2 — root-align (translation) + fine scale to match 2D projection
    t_align, s_align, C_align = compute_root_aligned_scale(blade_3d_est=blade_3d_est,
                                                        body_3d_est=body_3d_est,
                                                        body_3d_ref=body_3d_ref,
                                                        skeleton_valid=skeleton_valid,
                                                        root_idx=root_idx)
    

    body_3d_est, blade_3d_est = apply_root_align_and_scale(body_3d_est=body_3d_est,
                                                           blade_3d_est=blade_3d_est,
                                                            t=t_align, s=s_align, C=C_align
                                                            )
    
    return blade_3d_est, body_3d_est




def align_with_bbox_scale_and_pnp(blade_3d_est, body_3d_est, intr_est, intr_resized, skeleton_3d_ref, skeleton_valid,
                            scale_ratio, verbose=False):
    """
    Process estimated tracks: compute/apply scale ratio, and align with PnP.

    Args:
        blade_3d_est (np.ndarray): Estimated blade 3D tracks (N_blade, 3).
        body_3d_est (np.ndarray): Estimated body 3D tracks, all 17 H36M joints (17, 3).
        intr_est (np.ndarray): Estimated intrinsics (3, 3).
        intr_resized (np.ndarray): Resized/target intrinsics (3, 3).
        skeleton_3d_ref (np.ndarray): Ground truth skeleton 3D points (17, 3).
        skeleton_valid (np.ndarray): Skeleton validity mask (17,).
        scale_ratio (float or None): Pre-computed scale ratio; if None, it is computed from scratch.
        verbose (bool): Enable verbose output.

    Returns:
        tuple: (blade_3d_est, body_3d_est, scale_ratio)
            - blade_3d_est (np.ndarray): Scaled and PnP-aligned blade 3D points (N_blade, 3).
            - body_3d_est (np.ndarray): Scaled and PnP-aligned body 3D points, invalid joints zeroed (17, 3).
            - scale_ratio (float): The scale ratio used (propagated across frames).
    """


    # Step 1: Compute scale ratio (only for first frame)
    if scale_ratio is None:
        scale_ratio, _ = compute_scale_ratio_bbox_comparison(
            skeleton_3d_ref=skeleton_3d_ref,
            skeleton_valid=skeleton_valid,
            body_3d_est=body_3d_est,
            verbose=verbose
        )
        
    # Step 2: Apply scale ratio to estimated 3D points
    body_3d_est = body_3d_est * scale_ratio
    blade_3d_est = blade_3d_est * scale_ratio

    body_3d_est_valid = body_3d_est[skeleton_valid > 0.5]  # (N_valid_body, 3)

    # Step 3: Align with PnP
    align_results = align_3d_with_pnp(
        skeleton_3d=body_3d_est_valid,
        skeleton_valid=np.ones(len(body_3d_est_valid)),
        blade_3d=blade_3d_est,
        blade_valid=np.ones(len(blade_3d_est)),
        intr_old=intr_est,
        intr_new=intr_resized
    )

    body_3d_est_valid, blade_3d_est = align_results[0:2]

    body_3d_est = np.zeros_like(skeleton_3d_ref)
    body_3d_est[skeleton_valid > 0.5] = body_3d_est_valid

    return blade_3d_est, body_3d_est, scale_ratio

def compute_scale_ratio_bbox_comparison(skeleton_3d_ref, skeleton_valid,  body_3d_est, verbose=False):
    """
    Compute scale ratio between ground truth and estimated 3D points.

    Args:
        skeleton_3d_ref (np.ndarray): Ground truth skeleton 3D points (17, 3).
        skeleton_valid (np.ndarray): Skeleton validity mask (17,).
        body_3d_est (np.ndarray): Estimated body 3D tracks, all 17 H36M joints (17, 3).

    Returns:
        tuple: (scale_ratio_avg, bbox_info)
            - scale_ratio_avg (float): Average scale ratio (mean of x and y components)
            - bbox_info (dict): Dictionary with bounding box information
    """
    # For estimated: bounding box of valid body joints only
    all_3d_est = body_3d_est[skeleton_valid > 0.5]  # (N_valid_body, 3)
    bbox_min_est = all_3d_est.min(axis=0)  # (3,) - min x, y, z"
    bbox_max_est = all_3d_est.max(axis=0)  # (3,) - max x, y, z
    scale_est = bbox_max_est - bbox_min_est  # (3,) - scale in x, y, z

    # For ground truth: union skeleton_3d_ref and blade_3d_gt (with validity check)
    # Filter skeleton_3d_ref by skeleton_valid
    skeleton_3d_ref_valid = skeleton_3d_ref[skeleton_valid > 0]  # (N_skel_valid, 3)
    all_3d_ref = skeleton_3d_ref_valid#np.vstack([skeleton_3d_ref_valid, blade_3d_gt_valid])  # (N_skel_valid + N_blade_valid, 3)
    bbox_min_ref = all_3d_ref.min(axis=0)  # (3,) - min x, y, z
    bbox_max_ref = all_3d_ref.max(axis=0)  # (3,) - max x, y, z
    scale_ref = bbox_max_ref - bbox_min_ref  # (3,) - scale in x, y, z

    # Compute scale ratio (gt / est)
    scale_ratio_xyz = scale_ref / scale_est

    # Average scale ratio (mean of x and y components)
    scale_ratio_avg = (scale_ratio_xyz[0] + scale_ratio_xyz[1]) / 2.0

    # Store bounding box information
    bbox_info = {
        'bbox_min_est': bbox_min_est,
        'bbox_max_est': bbox_max_est,
        'scale_est': scale_est,
        'bbox_min_ref': bbox_min_ref,
        'bbox_max_ref': bbox_max_ref,
        'scale_ref': scale_ref,
        'scale_ratio_xyz': scale_ratio_xyz,
        'scale_ratio_avg': scale_ratio_avg
    }

    if verbose:
        print(f"  Estimated bbox: min={bbox_info['bbox_min_est']}, max={bbox_info['bbox_max_est']}")
        print(f"  Estimated scale (x, y, z): {bbox_info['scale_est']}")
        print(f"  Ground truth bbox: min={bbox_info['bbox_min_ref']}, max={bbox_info['bbox_max_ref']}")
        print(f"  Ground truth scale (x, y, z): {bbox_info['scale_ref']}")
        print(f"  Scale ratio (gt/est) - X: {bbox_info['scale_ratio_xyz'][0]:.3f}, Y: {bbox_info['scale_ratio_xyz'][1]:.3f}, Z: {bbox_info['scale_ratio_xyz'][2]:.3f}")

    return scale_ratio_avg, bbox_info





def align_3d_with_pnp(skeleton_3d, skeleton_valid, blade_3d, blade_valid,
                      intr_old, intr_new):
    """
    Use cv2.solvePnP to align 3D skeleton and blade points using two different intrinsics.

    This function:
    1. Projects 3D points with intr_old to get 2D correspondences
    2. Filters valid skeleton and blade points
    3. Concatenates them and solves PnP with intr_new to find rotation R and translation t
    4. Transforms all 3D points (including invalid ones)
    5. Returns transformed 3D with invalid points filled with np.nan

    The purpose is to compensate for the difference between intr_old (computed from dataloader)
    and intr_new (estimated by the tracking algorithm).

    Args:
        skeleton_3d (np.ndarray): (17, 3) array of 3D skeleton joints
        skeleton_valid (np.ndarray): (17,) boolean/float mask for valid skeleton joints
        blade_3d (np.ndarray): (M, 2, 3) array of 3D blade edge points
        blade_valid (np.ndarray): (M, 2) boolean/float mask for valid blade points
        intr_old (np.ndarray): (3, 3) intrinsics matrix from dataloader (crop+scale applied)
        intr_new (np.ndarray): (3, 3) intrinsics matrix estimated by tracking algorithm

    Returns:
        tuple: (skeleton_3d_transformed, blade_3d_transformed, success) where:
            - skeleton_3d_transformed (np.ndarray): (17, 3) transformed skeleton,
                                                     invalid points filled with np.nan
            - blade_3d_transformed (np.ndarray): (M, 2, 3) transformed blade,
                                                  invalid points filled with np.nan
            - success (bool): True if PnP succeeded, False otherwise

    """
    # Import here to avoid circular dependency
    # Project 3D to 2D using intr_old to get correspondences
    skeleton_2d = project_3d_with_intrinsics(skeleton_3d, intr_old)  # (17, 2)
    blade_2d = project_3d_with_intrinsics(blade_3d.reshape(-1, 3), intr_old) # (M*2, 2)
    #.reshape(blade_3d.shape[0], blade_3d.shape[1], 2)  # (M, 2, 2)

    # Filter valid skeleton points
    skeleton_3d_valid = skeleton_3d[skeleton_valid > 0]  # (N_skel, 3)
    skeleton_2d_valid = skeleton_2d[skeleton_valid > 0]  # (N_skel, 2)

    # Filter valid blade points
    blade_3d_flat = blade_3d.reshape(-1, 3)  # (M*2, 3)
    blade_2d_flat = blade_2d.reshape(-1, 2)  # (M*2, 2)
    blade_valid_flat = blade_valid.reshape(-1)  # (M*2,)
    blade_3d_valid = blade_3d_flat[blade_valid_flat > 0]  # (N_blade, 3)
    blade_2d_valid = blade_2d_flat[blade_valid_flat > 0]  # (N_blade, 2)

    # Concatenate all valid points
    points_3d_pnp = np.vstack([skeleton_3d_valid, blade_3d_valid])  # (N_total, 3)
    points_2d_pnp = np.vstack([skeleton_2d_valid, blade_2d_valid])  # (N_total, 2)

    # Need at least 4 points for PnP
    if len(points_3d_pnp) < 4:
        # Return original points with invalid filled with nan
        skeleton_3d_out = skeleton_3d.copy()
        skeleton_3d_out[skeleton_valid <= 0] = np.nan
        blade_3d_out = blade_3d.copy()
        blade_3d_out[blade_valid <= 0] = np.nan
        return skeleton_3d_out, blade_3d_out, None, None, False

    # center the 3D points to improve numerical stability
    #points_3d_mean = points_3d_pnp.mean(axis=0)
    #points_3d_pnp = points_3d_pnp - points_3d_mean

    # Solve PnP using intr_new
    success, rvec, tvec = cv2.solvePnP(points_3d_pnp, points_2d_pnp, intr_new, None,  # No distortion coefficients
                                            flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        # Return original points with invalid filled with nan
        skeleton_3d_out = skeleton_3d.copy()
        skeleton_3d_out[skeleton_valid <= 0] = np.nan
        blade_3d_out = blade_3d.copy()
        blade_3d_out[blade_valid <= 0] = np.nan
        return skeleton_3d_out, blade_3d_out, None, None, False

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Transform all skeleton points (including invalid ones)
    # Correct formula: transformed = (R @ (points - mean).T + t).T

    skeleton_3d_transformed = (rmat @ (skeleton_3d).T + tvec).T  # (17, 3)
    skeleton_3d_transformed[skeleton_valid <= 0] = np.nan

    # Transform all blade points (including invalid ones)
    blade_3d_flat_transformed = (rmat @ (blade_3d_flat).T + tvec).T  # (M*2, 3)
    blade_3d_transformed = blade_3d_flat_transformed.reshape(blade_3d.shape)  # (M, 2, 3)
    blade_3d_transformed[blade_valid <= 0] = np.nan

    return skeleton_3d_transformed, blade_3d_transformed, rvec, tvec, success






