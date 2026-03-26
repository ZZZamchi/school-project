"""从全网格 attraction / exploration / obstacle 地图抽取与 PPO 一致的 ego 朝向系摘要，供拓扑/VLM 策略使用。"""
from __future__ import annotations

from typing import Any

import numpy as np

from uav_search.action_model_inputs_test import _crop_rotate_and_pad, map_input_preparation

ORIENTATION_IDX_TO_LABEL = {
    0: "北 (机头约朝世界 +X 网格正向，与 env uav_pose 约定一致)",
    1: "西",
    2: "南",
    3: "东",
}


def _sector_means_8_valid_mask(arr2d: np.ndarray, valid2d: np.ndarray) -> np.ndarray:
    """仅在 valid2d 为真的格上按 8 扇区求平均。"""
    h, w = arr2d.shape
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    yy, xx = np.indices((h, w))
    dx = xx.astype(np.float64) - cx
    dy = yy.astype(np.float64) - cy
    ang = (np.degrees(np.arctan2(dx, -dy)) + 360.0) % 360.0
    bins = (np.floor(ang / 45.0).astype(np.int32)) % 8
    vals = arr2d.astype(np.float64)
    out = np.zeros(8, dtype=np.float64)
    for b in range(8):
        m = valid2d & (bins == b)
        if not np.any(m):
            out[b] = 0.0
        else:
            out[b] = float(np.mean(vals[m]))
    return out


def _sector_means_8_masked(arr2d: np.ndarray, invalid_value: float | None) -> np.ndarray:
    """与 vision_topo 相同的 8 扇区角分桶，对无效格（如 attraction 未观测 -1）掩掉。"""
    h, w = arr2d.shape
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    yy, xx = np.indices((h, w))
    dx = xx.astype(np.float64) - cx
    dy = yy.astype(np.float64) - cy
    ang = (np.degrees(np.arctan2(dx, -dy)) + 360.0) % 360.0
    bins = (np.floor(ang / 45.0).astype(np.int32)) % 8
    vals = arr2d.astype(np.float64)
    if invalid_value is None:
        mask = np.ones_like(vals, dtype=bool)
    else:
        mask = vals != float(invalid_value)
    out = np.zeros(8, dtype=np.float64)
    for b in range(8):
        m = mask & (bins == b)
        if not np.any(m):
            out[b] = 0.0
        else:
            out[b] = float(np.mean(vals[m]))
    return out


def _attraction_peak_bearing_masked(a_xy: np.ndarray, observed: np.ndarray) -> tuple[float | None, float | None]:
    valid = np.asarray(observed, dtype=bool)
    if not np.any(valid):
        return None, None
    aa = np.where(valid, a_xy, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(aa)), aa.shape)
    h, w = a_xy.shape
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    peak_dx = float(ix) - cx
    peak_dy = float(iy) - cy
    bearing = (np.degrees(np.arctan2(peak_dx, -peak_dy)) + 360.0) % 360.0
    return bearing, float(a_xy[iy, ix])


def ego_map_context_for_policy(
    attraction_map: np.ndarray,
    exploration_map: np.ndarray,
    obstacle_map: np.ndarray,
    uav_pose: dict[str, Any],
    *,
    z_slice_mode: str = "center",
) -> dict[str, Any]:
    """
    使用与 ``map_input_preparation`` 相同的裁剪与旋转，得到机体系局部体；再压缩为 JSON 友好统计量。
    attraction: (40,40,10,2)；obstacle: (40,40,10)；exploration: (40,40,10)。
    """
    mi = map_input_preparation(attraction_map, exploration_map, obstacle_map, uav_pose)
    a = mi["attraction_map_input"].astype(np.float64, copy=False)
    e = mi["exploration_map_input"].astype(np.float64, copy=False)
    o = mi["obstacle_map_input"].astype(np.float64, copy=False)

    if z_slice_mode == "center":
        zi_a = a.shape[0] // 2
        zi_o = o.shape[0] // 2
    else:
        zi_a = zi_o = 0

    a_xy = np.asarray(a[zi_a])
    e_xy = np.asarray(e[zi_a])
    o_xy = np.asarray(o[zi_o])

    position = np.asarray(uav_pose["position"], dtype=np.float64)
    orientation = int(uav_pose["orientation"])
    a_obs_local = _crop_rotate_and_pad(
        full_map=attraction_map[:, :, :, 1],
        center_coords=position,
        crop_size=(20, 20, 10),
        padding_value=-1.0,
        orientation=orientation,
    )
    # map_input 中 attraction 为 transpose(Z,Y,X)，未转置的裁剪体最后一维为 Z
    attr_observed_xy = np.asarray(a_obs_local[:, :, zi_a]) != -1.0

    attraction_sector_mean_ego8 = [
        round(float(x), 4) for x in _sector_means_8_valid_mask(a_xy, attr_observed_xy)
    ]
    exploration_sector_mean_ego8 = [round(float(x), 4) for x in _sector_means_8_masked(e_xy, -1.0)]
    obstacle_sector_mean_ego8 = [round(float(x), 4) for x in _sector_means_8_masked(o_xy, None)]

    peak_bearing, peak_val = _attraction_peak_bearing_masked(a_xy, attr_observed_xy)
    pos = position.reshape(-1)

    return {
        "frame": "ego_cropped_like_ppo_map_input",
        "grid_position_xyz": [round(float(pos[i]), 3) for i in range(min(3, len(pos)))],
        "orientation_idx": orientation,
        "orientation_label_cn": ORIENTATION_IDX_TO_LABEL.get(orientation, str(orientation)),
        "attraction_sector_mean_ego8": attraction_sector_mean_ego8,
        "exploration_sector_mean_ego8": exploration_sector_mean_ego8,
        "obstacle_sector_mean_ego8": obstacle_sector_mean_ego8,
        "attraction_peak": (
            None
            if peak_bearing is None
            else {
                "bearing_deg_ego": round(peak_bearing, 2),
                "value": round(peak_val, 4) if peak_val is not None else None,
            }
        ),
        "note": "ego8 与 RGB wedge 一致。attraction 扇区仅在 attraction_map[...,1]!=-1（已观测）格上平均；无显著吸引信号时策略应更多依赖 obstacle / exploration / 视觉。",
    }


def fuse_wedge_scores_with_map_context(
    wedge_scores: np.ndarray,
    map_context: dict[str, Any] | None,
    *,
    w_attraction: float = 0.75,
    w_obstacle: float = 0.6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """用 attraction 引导、obstacle 抑制，融合 Sobel wedge 分数。"""
    ws = np.asarray(wedge_scores, dtype=np.float64).reshape(8).copy()
    meta: dict[str, Any] = {"fused": False}
    if not map_context:
        return ws, meta
    a = np.asarray(map_context.get("attraction_sector_mean_ego8", []), dtype=np.float64)
    if a.size != 8:
        return ws, meta
    a_pos = np.maximum(a, 0.0)
    if float(a_pos.max()) > 1e-9:
        a_n = a_pos / float(a_pos.max())
    else:
        a_n = np.ones(8, dtype=np.float64)

    o = np.asarray(map_context.get("obstacle_sector_mean_ego8", []), dtype=np.float64)
    if o.size != 8:
        o = np.zeros(8, dtype=np.float64)
    o = np.clip(o, 0.0, 1.0)

    fused = ws * (1.0 - w_attraction + w_attraction * a_n) * (1.0 - w_obstacle * o)
    if float(np.max(fused)) < 1e-9:
        return ws, {**meta, "fused": False, "reason": "fused_near_zero"}
    meta["fused"] = True
    meta["weights"] = {"w_attraction": w_attraction, "w_obstacle": w_obstacle}
    return fused, meta
