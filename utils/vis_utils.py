import cv2
import numpy as np
from utils.constants import H36M_JOINTS, H36M_JOINT_PAIRS

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def visualize_skeleton_2d(image, skeleton_2d, skeleton_valid, joint_color=(255, 0, 0), bone_color=(255, 0, 0), radius=4, show_joint_names=False):
    """
    Visualize 2D skeleton on the image.

    Args:
        image (np.ndarray): The input image.
        skeleton_2d (np.ndarray): The 2D skeleton of shape (N, 2).
        skeleton_valid (np.ndarray): Validity of each joint.
        joint_color (tuple): Color for joints.
        bone_color (tuple): Color for bones.

    Returns:
        np.ndarray: Image with skeleton overlay.
    """
    vis_image = image.copy()

    # Draw bones
    for joint_start, joint_end in H36M_JOINT_PAIRS:
        if skeleton_valid[joint_start] < 0.5 or skeleton_valid[joint_end] < 0.5:
            continue

        pt_start = tuple(skeleton_2d[joint_start].astype(int))
        pt_end = tuple(skeleton_2d[joint_end].astype(int))
        cv2.line(vis_image, pt_start, pt_end, bone_color, thickness=10)


    # Draw joints
    for joint in range(skeleton_2d.shape[0]):
        if skeleton_valid[joint] < 0.5:
            continue
        pt = tuple(skeleton_2d[joint].astype(int))
        cv2.circle(vis_image, pt, radius=radius, color=joint_color, thickness=-1)
        if show_joint_names and joint in H36M_JOINTS:
            cv2.putText(vis_image, H36M_JOINTS[joint], (pt[0] + radius + 2, pt[1] - radius - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, joint_color, 1, cv2.LINE_AA)

    return vis_image



def visualize_blade_2d(image, blade_edge_2d, blade_valid, point_color=(0, 255, 0), edge_color=(0, 255, 0)):
        vis_image = image.copy()        
        # Draw blade edges
        for iid, ((x1, y1), (x2, y2)) in enumerate(blade_edge_2d.astype(np.int32)):
            if blade_valid[iid, 0] < 0.5 or blade_valid[iid, 1] < 0.5:
                continue
            cv2.line(vis_image, (x1, y1), (x2, y2), edge_color, 2)
        
        for line_id in range(2):
            for i in range(0, blade_edge_2d.shape[0]-1, 1):
                if blade_valid[i, line_id] < 0.5 or blade_valid[i+1, line_id] < 0.5:
                    continue
                pt1 = tuple(blade_edge_2d[i, line_id].astype(int))
                pt2 = tuple(blade_edge_2d[i+1, line_id].astype(int))
                cv2.line(vis_image, pt1, pt2, edge_color, thickness=2)


        return vis_image


def visualize_tracks_2d(image, body_2d=None, blade_2d=None, blade2_2d=None, joint_2d=None, joint_valid=None,
                        body_color=(255, 0, 0), blade_color=(0, 255, 0), blade2_color=(0, 165, 255), joint_color=(0, 0, 255),
                        body_radius=2, blade_radius=3, blade2_radius=3, joint_radius=5):
    """
    Visualize 2D tracked points (body grid, blade grid, and joints) on the image.

    Args:
        image (np.ndarray): The input image (H, W, 3).
        body_2d (np.ndarray): Body grid track points (N_body, 2).
        blade_2d (np.ndarray): Blade grid track points (N_blade, 2).
        joint_2d (np.ndarray): Joint track points (N_joints, 2), can contain NaN for invalid.
        joint_valid (np.ndarray): Joint validity mask (N_joints,).
        body_color (tuple): Color for body grid points (BGR).
        blade_color (tuple): Color for blade grid points (BGR).
        joint_color (tuple): Color for joint points (BGR).
        body_radius (int): Radius for body grid points.
        blade_radius (int): Radius for blade grid points.
        joint_radius (int): Radius for joint points.

    Returns:
        np.ndarray: Image with tracks overlay.
    """
    vis_image = image.copy()

    # Draw body grid tracks
    if body_2d is not None:
        for pt in body_2d:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                cv2.circle(vis_image, (x, y), body_radius, body_color, -1)

    # Draw blade grid tracks
    if blade_2d is not None:
        for pt in blade_2d:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                cv2.circle(vis_image, (x, y), blade_radius, blade_color, -1)

    # Draw blade2 grid tracks
    if blade2_2d is not None:
        for pt in blade2_2d:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                cv2.circle(vis_image, (x, y), blade2_radius, blade2_color, -1)

    # Draw joint tracks
    if joint_2d is not None:
        for idx, pt in enumerate(joint_2d):
            # Skip invalid joints (NaN or marked invalid)
            if np.any(np.isnan(pt)):
                continue
            if joint_valid is not None and joint_valid[idx] < 0.5:
                continue

            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                cv2.circle(vis_image, (x, y), joint_radius, joint_color, -1)
                joint_name = H36M_JOINTS[idx]
                cv2.putText(vis_image, joint_name, (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, joint_color, 1)

                # Draw skeleton connections if using H36M format
                if joint_valid is not None and len(joint_2d) == len(H36M_JOINTS):
                    for joint_start, joint_end in H36M_JOINT_PAIRS:
                        if (joint_valid[joint_start] > 0.5 and joint_valid[joint_end] > 0.5 and
                            not np.any(np.isnan(joint_2d[joint_start])) and
                            not np.any(np.isnan(joint_2d[joint_end]))):
                            pt_start = tuple(joint_2d[joint_start].astype(int))
                            pt_end = tuple(joint_2d[joint_end].astype(int))
                            cv2.line(vis_image, pt_start, pt_end, joint_color, thickness=2)

    return vis_image
