"""深度图批量反投影（供 ``map_updating_train.trace_rays_vectorized`` 使用）。"""
from __future__ import annotations

import numpy as np

from uav_search.to_map_test import to_map_xyz


def depth_image_to_world_points(depth_image, camera_fov, camera_position, camera_orientation):
    """
    将整张深度图逐像素投影为世界坐标点，返回形状 ``(h * w, 3)`` 的数组，
    与 ``depth_image.flatten()`` 顺序一致。
    """
    h, w = depth_image.shape
    shape = (h, w)
    out = np.zeros((h * w, 3), dtype=np.float64)
    for v in range(h):
        for u in range(w):
            d = float(depth_image[v, u])
            idx = v * w + u
            if d >= 250:
                out[idx] = (0.0, 0.0, 0.0)
            else:
                out[idx] = to_map_xyz(v, u, d, shape, camera_fov, camera_position, camera_orientation)
    return out
