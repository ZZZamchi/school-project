"""
离散航向导航（不调用 PPO）。动作与 ``AirSimDroneEnv`` 一致：0 前飞、1 左转、2 右转、3 掉头、4 下降、5 上升。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from uav_search.topo_text_map.builder import DIR_8

# 与 builder / env 对齐（4=降、5=升，与 moveToZAsync 一致）
ACTION_NAMES_CN = {
    0: "向前飞行 (Forward)",
    1: "左转航向 (Yaw Left)",
    2: "右转航向 (Yaw Right)",
    3: "掉头 (Yaw Back)",
    4: "下降 (Descend)",
    5: "上升 (Ascend)",
}


def _yaw_deg_to_orientation_idx(yaw_deg: float) -> int:
    if -45 <= yaw_deg <= 45:
        return 0
    if -135 < yaw_deg <= -45:
        return 1
    if yaw_deg > 135 or yaw_deg < -135:
        return 2
    return 3


ORIENTATION_LABEL = {0: "航向≈+X", 1: "航向≈-Y", 2: "航向≈-X", 3: "航向≈+Y"}


def _best_heading_idx_for_target(dx: float, dy: float) -> int:
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return 0
    vx, vy = dx / n, dy / n
    dirs = [(1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0)]
    dots = [vx * a + vy * b for a, b in dirs]
    return int(np.argmax(dots))


def _bearing_deg(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx))


def _bearing_to_8label(deg: float) -> str:
    d = deg % 360.0
    labels = ["东", "东北", "北", "西北", "西", "西南", "南", "东南"]
    idx = int((d + 22.5) // 45) % 8
    return labels[idx]


def _angular_diff_deg(a: float, b: float) -> float:
    d = (a - b) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d


# 与 builder.DIR_8、`_bearing_deg`(atan2(dy,dx)) 一致：东=0°，北=90°，逆时针递增
_DIR_LABEL_PREFIX_TO_DEG: Tuple[Tuple[str, float], ...] = (
    ("东北", 45.0),
    ("东南", 315.0),
    ("西北", 135.0),
    ("西南", 225.0),
    ("东", 0.0),
    ("北", 90.0),
    ("西", 180.0),
    ("南", 270.0),
)


def dir_label_to_bearing_deg(label: str) -> float:
    """解析 ``facts_for_llm`` 中 ``北 (North)`` 等形式，返回方位角（度）。"""
    s = label.strip()
    for pref, deg in _DIR_LABEL_PREFIX_TO_DEG:
        if s.startswith(pref):
            return deg
    return 0.0


def choose_frontier_nearest_bearing(frontier_labels: List[str], reference_bearing_deg: float) -> str:
    """在候选前沿方向中选与 ``reference_bearing_deg`` 夹角最小者。"""
    return min(
        frontier_labels,
        key=lambda lab: _angular_diff_deg(dir_label_to_bearing_deg(lab), reference_bearing_deg),
    )


def flight_telemetry_pose_only(client: Any) -> Dict[str, Any]:
    """当前位姿与航向（不含外部参考点）。"""
    import airsim

    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    q = state.kinematics_estimated.orientation
    _, _, yaw_rad = airsim.to_eularian_angles(q)
    yaw_deg = math.degrees(yaw_rad)
    cur_idx = _yaw_deg_to_orientation_idx(yaw_deg)

    return {
        "yaw_deg": round(yaw_deg, 2),
        "orientation_idx": cur_idx,
        "orientation_label": ORIENTATION_LABEL.get(cur_idx, ""),
    }


def topo_nav_decide(
    client: Any,
    target_xyz: np.ndarray,
    grid_size: float = 5.0,
    horiz_near_m: float = 10.0,
    z_deadband_m: float = 4.0,
    grid_position: np.ndarray | None = None,
    grid_margin: float = 3.0,
    horiz_far_m: float = 35.0,
    z_urgent_m: float = 18.0,
    topo_builder: Any | None = None,
) -> Tuple[int, Dict[str, Any]]:
    """返回 (action_id, decision_dict)。可选 ``topo_builder`` 时用拓扑前沿与几何参考对齐。"""
    import airsim

    decision: Dict[str, Any] = {
        "vertical": {"intent": "none", "dz_m": 0.0, "action_if_applied": None},
        "horizontal": {"phase": "unknown", "orientation_idx": None, "desired_heading_idx": None},
        "flight": {},
    }

    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    q = state.kinematics_estimated.orientation
    _, _, yaw_rad = airsim.to_eularian_angles(q)
    yaw_deg = math.degrees(yaw_rad)

    x, y, z = float(pos.x_val), float(pos.y_val), float(pos.z_val)
    tx, ty, tz = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])
    dx, dy, dz = tx - x, ty - y, tz - z
    horiz = math.hypot(dx, dy)
    bearing = _bearing_deg(dx, dy)
    tgt_hidx_geo = _best_heading_idx_for_target(dx, dy)
    cur_idx = _yaw_deg_to_orientation_idx(yaw_deg)

    # 拓扑前沿：在未尝试过的离开方向中选与几何参考方位最接近的 → 四向航向 idx
    tgt_hidx = tgt_hidx_geo
    topo_map_meta: Dict[str, Any] = {"mode": "geometric_only"}
    if topo_builder is not None:
        try:
            facts = topo_builder.facts_for_llm()
            unexplored = list(facts.get("unexplored_directions_from_current_node") or [])
            if not unexplored:
                unexplored = list(DIR_8)
            chosen = choose_frontier_nearest_bearing(unexplored, bearing)
            fb_deg = dir_label_to_bearing_deg(chosen)
            rad = math.radians(fb_deg)
            tgt_hidx = _best_heading_idx_for_target(math.cos(rad), math.sin(rad))
            topo_map_meta = {
                "mode": "frontier_aligned",
                "current_node_id": facts.get("current_node_id"),
                "unexplored_directions": unexplored,
                "chosen_frontier": chosen,
                "chosen_frontier_bearing_deg": round(fb_deg, 2),
                "geometric_heading_idx": int(tgt_hidx_geo),
                "aligned_heading_idx": int(tgt_hidx),
            }
        except Exception as ex:  # noqa: BLE001 — 拓扑决策失败时回退几何
            tgt_hidx = tgt_hidx_geo
            topo_map_meta = {"mode": "geometric_fallback", "error": str(ex)}

    decision["topo_map"] = topo_map_meta
    decision["flight"] = {
        "yaw_deg": round(yaw_deg, 2),
        "orientation_idx": cur_idx,
        "orientation_label": ORIENTATION_LABEL.get(cur_idx, ""),
        "reference_bearing_deg": round(bearing, 2),
        "direction_8": _bearing_to_8label(bearing),
        "horiz_m": round(horiz, 2),
        "dz_m": round(dz, 2),
    }
    decision["horizontal"]["orientation_idx"] = cur_idx
    decision["horizontal"]["desired_heading_idx"] = tgt_hidx
    decision["vertical"]["dz_m"] = round(dz, 3)

    def set_vert(intent: str) -> None:
        decision["vertical"]["intent"] = intent
        if intent == "ascend":
            decision["vertical"]["action_if_applied"] = 5
        elif intent == "descend":
            decision["vertical"]["action_if_applied"] = 4
        else:
            decision["vertical"]["action_if_applied"] = None

    # 栅格边界：优先转向 / 高度壳
    if grid_position is not None:
        gx = float(grid_position[0])
        gy = float(grid_position[1])
        gz = float(grid_position[2])
        if gx < grid_margin or gx > 40.0 - grid_margin - 1 or gy < grid_margin or gy > 40.0 - grid_margin - 1:
            decision["horizontal"]["phase"] = "avoid_oob_turn"
            decision["reason"] = "栅格 XY 近边界，左转避免前飞越界"
            set_vert("none")
            return 1, decision
        if gz <= 0.5 or gz >= 8.5:
            decision["horizontal"]["phase"] = "hold"
            if gz <= 0.5:
                set_vert("ascend")
                decision["reason"] = "栅格 z 近下界，上升"
                return 5, decision
            set_vert("descend")
            decision["reason"] = "栅格 z 近上界，下降"
            return 4, decision

    need_z = abs(dz) > z_deadband_m
    set_vert("ascend" if dz > 0 else "descend") if need_z else set_vert("none")

    # 远距离：先水平；若高度差极大则穿插升降
    if horiz >= horiz_near_m:
        urgent_z = abs(dz) > z_urgent_m
        if urgent_z and horiz < horiz_far_m and need_z:
            decision["horizontal"]["phase"] = "vertical_priority_mid_range"
            decision["reason"] = f"中距({horiz:.0f}m)且|Δz|>{z_urgent_m}m，优先修正高度"
            return (5 if dz > 0 else 4), decision

        if cur_idx != tgt_hidx:
            cw = (tgt_hidx - cur_idx) % 4
            ccw = (cur_idx - tgt_hidx) % 4
            decision["horizontal"]["phase"] = "align_heading"
            decision["reason"] = f"远距离水平对准（当前idx={cur_idx}→期望idx={tgt_hidx}）"
            set_vert("none")
            return (1 if cw <= ccw else 2), decision

        if cur_idx == tgt_hidx:
            if grid_position is not None:
                gx = float(grid_position[0])
                gy = float(grid_position[1])
                if gx <= 6 or gx >= 33 or gy <= 6 or gy >= 33:
                    decision["horizontal"]["phase"] = "avoid_oob_forward"
                    decision["reason"] = "已对准航向但栅格近边界，左转代替前飞"
                    set_vert("none")
                    return 1, decision
            # 已对准：远距离仍可穿插升降（每步决策显式包含 vertical 意图）
            if need_z and abs(dz) > z_deadband_m * 1.25:
                decision["horizontal"]["phase"] = "forward_or_vertical"
                decision["reason"] = "航向已对准；优先微调高度再接近"
                return (5 if dz > 0 else 4), decision
            decision["horizontal"]["phase"] = "forward"
            decision["reason"] = "航向已对准，前飞"
            set_vert("none")
            return 0, decision

    # 近距离：高度优先于最后平飞
    if need_z:
        decision["horizontal"]["phase"] = "near_altitude_trim"
        decision["reason"] = "近距离，按 Δz 升降"
        return (5 if dz > 0 else 4), decision

    decision["horizontal"]["phase"] = "near_forward"
    decision["reason"] = "近距离且高度差在死区内，前飞"
    set_vert("none")
    return 0, decision


def topo_nav_policy_action(
    client: Any,
    target_xyz: np.ndarray,
    grid_size: float = 5.0,
    horiz_near_m: float = 10.0,
    z_deadband_m: float = 4.0,
    grid_position: np.ndarray | None = None,
    grid_margin: float = 3.0,
) -> int:
    """仅返回动作 id（兼容旧接口）。"""
    a, _ = topo_nav_decide(
        client,
        target_xyz,
        grid_size=grid_size,
        horiz_near_m=horiz_near_m,
        z_deadband_m=z_deadband_m,
        grid_position=grid_position,
        grid_margin=grid_margin,
        topo_builder=None,
    )
    return a
