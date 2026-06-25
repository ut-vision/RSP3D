"""
Camera parameters and coordinate transformations.
"""

from typing import Optional

import numpy as np


class CameraParams:
    """
    Camera intrinsic and extrinsic parameters.
    """

    def __init__(
        self,
        image_width: int = 3840,
        image_height: int = 2160,
        fov_deg: Optional[float] = 60,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
        rvec: Optional[np.ndarray] = None,
        tvec: Optional[np.ndarray] = None,
    ):
        self.width = image_width
        self.height = image_height

        if fx is not None and fy is not None:
            self.fx = fx
            self.fy = fy
            self.fov = None
        elif fov_deg is not None:
            self.fov = fov_deg
            self.fx = (image_width / 2) / np.tan(np.deg2rad(fov_deg / 2))
            self.fy = self.fx
        else:
            raise ValueError("Must provide either fov_deg or (fx, fy)")

        self.cx = cx if cx is not None else image_width / 2
        self.cy = cy if cy is not None else image_height / 2

        self.dist_coeffs = dist_coeffs

        self._R = None
        self._t = None
        self._rvec = None
        self._tvec = None

        if R is not None:
            self._R = np.array(R).reshape(3, 3)
        if t is not None:
            self._t = np.array(t).reshape(3, 1)
        if rvec is not None:
            self._rvec = np.array(rvec).reshape(3, 1)
        if tvec is not None:
            self._tvec = np.array(tvec).reshape(3, 1)

        if self._R is None and self._rvec is not None:
            import cv2
            self._R, _ = cv2.Rodrigues(self._rvec)

        if self._t is None and self._tvec is not None:
            self._t = self._tvec

    @property
    def R(self) -> Optional[np.ndarray]:
        return self._R

    @property
    def t(self) -> Optional[np.ndarray]:
        return self._t

    @property
    def has_extrinsics(self) -> bool:
        return self._R is not None and self._t is not None

    def __repr__(self):
        ext_str = ", has_extrinsics" if self.has_extrinsics else ""
        dist_str = ", has_dist" if self.dist_coeffs is not None else ""
        return (f"CameraParams(W={self.width}, H={self.height}, "
                f"fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}{dist_str}{ext_str})")

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        x = points_3d[..., 0]
        y = points_3d[..., 1]
        z = points_3d[..., 2]

        z_safe = np.where(z == 0, 1e-6, z)

        u = self.fx * x / z_safe + self.cx
        v = self.fy * y / z_safe + self.cy

        return np.stack([u, v], axis=-1)

    def world_to_camera(self, points_world: np.ndarray) -> np.ndarray:
        if not self.has_extrinsics:
            raise ValueError("Extrinsic parameters (R, t) not available")

        points_cam = np.einsum('ij,...j->...i', self._R, points_world) + self._t.flatten()
        return points_cam


def project_3d_with_intrinsics(points_3d, intrinsics):
    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2]

    z_safe = np.where(z == 0, 1e-6, z)

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    u = fx * x / z_safe + cx
    v = fy * y / z_safe + cy

    return np.stack([u, v], axis=-1)


def apply_intrinsics_transform_to_2d(points_2d, intrinsics_transform):
    original_shape = points_2d.shape

    if points_2d.ndim == 2:
        points_2d_flat = points_2d
    elif points_2d.ndim == 3:
        points_2d_flat = points_2d.reshape(-1, 2)
    else:
        raise ValueError(f"points_2d must be 2D or 3D array, got shape {points_2d.shape}")

    num_points_flat = points_2d_flat.shape[0]
    points_2d_homo = np.concatenate([
        points_2d_flat,
        np.ones((num_points_flat, 1))
    ], axis=1)

    points_2d_transformed_homo = (intrinsics_transform @ points_2d_homo.T).T
    points_2d_transformed = points_2d_transformed_homo[:, :2]
    return points_2d_transformed.reshape(original_shape)


def project_model_base_to_4k_pinhole(joints3d, focal_length, img_h, img_w):
    cx, cy = img_w / 2.0, img_h / 2.0
    z   = joints3d[:, 2]
    x_2d = focal_length * joints3d[:, 0] / z + cx
    y_2d = focal_length * joints3d[:, 1] / z + cy
    return np.stack([x_2d, y_2d], axis=1)
