"""单像素深度反投影到世界坐标（与 AirSim DepthPlanar + 相机位姿一致）。"""
from __future__ import annotations

import math

import numpy as np


def _quaternion_to_rotation_matrix(q) -> np.ndarray:
    """Quaternion (w,x,y,z) → 3×3 旋转矩阵（世界系）。"""
    w, x, y, z = float(q.w_val), float(q.x_val), float(q.y_val), float(q.z_val)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def to_map_xyz(v, u, depth, image_shape, camera_fov, camera_position, camera_orientation):
    """
    将深度图上一像素 (v,u) 与深度值投影到世界坐标（NED）。

    相机模型：OpenCV 风格，光轴为 +Z，X 右、Y 下；与 AirSim DepthPlanar 常用约定一致。
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    fov_rad = math.radians(float(camera_fov))
    fy = (height / 2.0) / math.tan(fov_rad / 2.0)
    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    x_cam = (float(u) - cx) / fx * float(depth)
    y_cam = (float(v) - cy) / fy * float(depth)
    z_cam = float(depth)

    p_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64)
    R = _quaternion_to_rotation_matrix(camera_orientation)
    cam = np.array(
        [camera_position.x_val, camera_position.y_val, camera_position.z_val],
        dtype=np.float64,
    )
    return R @ p_cam + cam
