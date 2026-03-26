"""与 ``AirSimDroneEnv`` 对齐的导航**安全壳**（OOB、姿态、防原地打转）及 Sobel 楔分工具。

主决策须使用 ``apex_vl_nav.apex_vl_topo_nav_decide``（Qwen3-VL）；本文件不再提供无 LLM 的完整替代策略。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from uav_search.topo_text_map.topo_nav_policy import (
    ORIENTATION_LABEL,
    _best_heading_idx_for_target,
    _yaw_deg_to_orientation_idx,
)


def score_wedges_from_rgb(rgb: np.ndarray, task_text: str = "") -> np.ndarray:
    del task_text
    try:
        import cv2
    except ImportError:
        return np.ones(8, dtype=np.float64)

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return np.ones(8, dtype=np.float64)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    h, w = mag.shape
    yy, xx = np.indices((h, w))
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    dx = xx.astype(np.float64) - cx
    dy = yy.astype(np.float64) - cy
    ang = (np.degrees(np.arctan2(dx, -dy)) + 360.0) % 360.0
    bins = (np.floor(ang / 45.0).astype(np.int32)) % 8
    scores = np.zeros(8, dtype=np.float64)
    for b in range(8):
        mask = bins == b
        if not np.any(mask):
            scores[b] = 0.0
        else:
            scores[b] = float(np.mean(mag[mask]))
    m = float(np.max(scores)) if scores.size else 1.0
    if m < 1e-6:
        scores[:] = 1.0
    return scores


def _body_wedge_idx_for_world_bearing(frontier_bearing_deg: float, bearing_forward_deg: float) -> int:
    rb = (frontier_bearing_deg - bearing_forward_deg) % 360.0
    if rb > 180.0:
        rb -= 360.0
    return int((int(math.floor((rb + 180.0) / 45.0)) + 4) % 8)


def _empty_decision() -> Dict[str, Any]:
    return {
        "vertical": {"intent": "none", "dz_m": 0.0, "action_if_applied": None},
        "horizontal": {"phase": "unknown", "orientation_idx": None, "desired_heading_idx": None},
        "flight": {},
        "topo_map": {},
        "vision": {},
    }


def set_nav_vert_intent(decision: Dict[str, Any], intent: str) -> None:
    decision["vertical"]["intent"] = intent
    if intent == "ascend":
        decision["vertical"]["action_if_applied"] = 5
    elif intent == "descend":
        decision["vertical"]["action_if_applied"] = 4
    else:
        decision["vertical"]["action_if_applied"] = None


def fill_flight_kinematics(decision: Dict[str, Any], client: Any) -> Tuple[float, float, int]:
    import airsim

    state = client.getMultirotorState()
    q = state.kinematics_estimated.orientation
    _, _, yaw_rad = airsim.to_eularian_angles(q)
    yaw_deg = math.degrees(yaw_rad)
    bearing_forward = yaw_deg % 360.0
    if bearing_forward < 0:
        bearing_forward += 360.0
    cur_idx = _yaw_deg_to_orientation_idx(yaw_deg)
    decision["flight"] = {
        "yaw_deg": round(yaw_deg, 2),
        "orientation_idx": cur_idx,
        "orientation_label": ORIENTATION_LABEL.get(cur_idx, ""),
    }
    return yaw_deg, bearing_forward, cur_idx


def nav_decision_bootstrap(client: Any) -> Tuple[Dict[str, Any], float, float, int]:
    decision = _empty_decision()
    yaw_deg, bearing_forward, cur_idx = fill_flight_kinematics(decision, client)
    return decision, yaw_deg, bearing_forward, cur_idx


def maybe_oob_mutate_and_return_action(
    decision: Dict[str, Any],
    grid_position: np.ndarray | None,
    grid_margin: float = 3.0,
) -> int | None:
    if grid_position is None:
        return None
    gx = float(grid_position[0])
    gy = float(grid_position[1])
    gz = float(grid_position[2])
    if gx < grid_margin or gx > 40.0 - grid_margin - 1 or gy < grid_margin or gy > 40.0 - grid_margin - 1:
        decision["horizontal"]["phase"] = "avoid_oob_turn"
        set_nav_vert_intent(decision, "none")
        # 朝栅格中心 (20,20) 转向，减少「固定左转」在边界附近来回摆动，便于沿边界探索
        dx_c = 20.0 - gx
        dy_c = 20.0 - gy
        tgt_hidx = _best_heading_idx_for_target(dx_c, dy_c)
        fl = decision.get("flight") or {}
        cur_idx = fl.get("orientation_idx")
        if cur_idx is None:
            action_oob = 1
        else:
            cw = (tgt_hidx - int(cur_idx)) % 4
            ccw = (int(cur_idx) - tgt_hidx) % 4
            action_oob = 1 if cw <= ccw else 2
        decision["reason"] = (
            f"栅格 XY 近边界，转向朝向中心(20,20)（目标航向 idx {tgt_hidx}，当前 {cur_idx}）"
        )
        decision["topo_map"] = {
            "mode": "oob_turn_center_seek",
            "grid_center_dx_dy": [round(dx_c, 2), round(dy_c, 2)],
            "target_orientation_idx": int(tgt_hidx),
        }
        return action_oob
    if gz <= 0.5 or gz >= 8.5:
        decision["horizontal"]["phase"] = "hold"
        if gz <= 0.5:
            set_nav_vert_intent(decision, "ascend")
            decision["reason"] = "栅格 z 近下界，上升"
            decision["topo_map"] = {"mode": "z_shell"}
            return 5
        set_nav_vert_intent(decision, "descend")
        decision["reason"] = "栅格 z 近上界，下降"
        decision["topo_map"] = {"mode": "z_shell"}
        return 4
    return None


def apply_yaw_spin_guard(
    topo_builder: Any,
    action_int: int,
    decision: Dict[str, Any],
) -> Tuple[int, Dict[str, Any]]:
    """连续只转弯（1/2/3）时易在原地摆动；强制间歇前飞以打破死锁。"""
    sg = getattr(topo_builder, "_nav_spin_guard", None)
    if sg is None:
        return action_int, decision
    tm = decision.get("topo_map") or {}
    mode = tm.get("mode")
    if mode in ("oob_turn", "oob_turn_center_seek", "z_shell"):
        sg["yaw_only_streak"] = 0
        sg["same_frontier_align_streak"] = 0
        sg["last_chosen_frontier"] = None
        return action_int, decision
    ph = (decision.get("horizontal") or {}).get("phase")
    ch = tm.get("chosen_frontier")
    if action_int in (1, 2, 3):
        sg["yaw_only_streak"] = int(sg.get("yaw_only_streak", 0)) + 1
        if ph == "align_heading_vision_frontier" and ch is not None and ch == sg.get("last_chosen_frontier"):
            sg["same_frontier_align_streak"] = int(sg.get("same_frontier_align_streak", 0)) + 1
        else:
            sg["same_frontier_align_streak"] = 1 if ph == "align_heading_vision_frontier" else 0
        sg["last_chosen_frontier"] = ch
        if sg["yaw_only_streak"] >= 4 or (
            ph == "align_heading_vision_frontier" and int(sg.get("same_frontier_align_streak", 0)) >= 3
        ):
            decision["horizontal"]["phase"] = "anti_spin_forward"
            decision["reason"] = "防原地打转：连续转向过多，强制前飞一步"
            tm2 = dict(tm)
            tm2["spin_guard"] = True
            decision["topo_map"] = tm2
            sg["yaw_only_streak"] = 0
            sg["same_frontier_align_streak"] = 0
            return 0, decision
    elif action_int == 0:
        sg["yaw_only_streak"] = 0
        sg["same_frontier_align_streak"] = 0
    return action_int, decision
