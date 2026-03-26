"""
从离散轨迹构建**结构化拓扑事实**（供 LLM / VLM 填写自然语言）。

- 将 (x,y) 按 ``grid_size`` 栅格化，每个首次进入的栅格成为一个 ``Node_*``。
- 栅格变化时记录边（动作名；**不在此生成**语义描述）。
- ``unexplored_directions_from_current_node``：从当前节点按 8 方位生成尚未沿该方向离开过的方向标签。

**不**在本地生成节点/前沿的自然语言占位符；完整文本拓扑应由 LLM 根据 ``llm_prompts`` + ``facts_for_llm()`` 生成。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from uav_search.topo_text_map.schema import Edge

# 与 uav_env_multi 中 Discrete(6) 对齐：4=z 减小（下降）、5=z 增大（上升）
ACTION_NAMES_CN = {
    0: "向前飞行 (Forward)",
    1: "左转航向 (Yaw Left)",
    2: "右转航向 (Yaw Right)",
    3: "掉头 (Yaw Back)",
    4: "下降 (Descend)",
    5: "上升 (Ascend)",
}

DIR_8 = [
    "北 (North)",
    "东北 (NorthEast)",
    "东 (East)",
    "东南 (SouthEast)",
    "南 (South)",
    "西南 (SouthWest)",
    "西 (West)",
    "西北 (NorthWest)",
]


def _cell(ix: int, iy: int) -> Tuple[int, int]:
    return (ix, iy)


class TopoTextMapBuilder:
    def __init__(
        self,
        task: str,
        grid_size: float = 12.0,
        seed: int = 42,
    ) -> None:
        del seed  # 保留参数以兼容旧调用；不再用于占位随机文案
        self.task = task
        self.grid_size = max(grid_size, 1.0)

        self._cell_to_nid: Dict[Tuple[int, int], str] = {}
        self._nid_order: List[str] = []
        self._visit_count: Dict[str, int] = {}
        self._edges: List[Edge] = []
        self._exited: Dict[str, Set[str]] = {}  # node_id -> set of direction labels used when leaving

        self._last_nid: Optional[str] = None
        self._last_cell: Optional[Tuple[int, int]] = None
        self._last_xyz: Optional[Tuple[float, float, float]] = None
        self._cell_last_z: Dict[Tuple[int, int], float] = {}
        self._step_log: List[Dict[str, Any]] = []
        # apex_vl 防原地打转：连续只转弯时强制前飞一步
        self._nav_spin_guard: Dict[str, Any] = {
            "yaw_only_streak": 0,
            "same_frontier_align_streak": 0,
            "last_chosen_frontier": None,
        }

    def _alloc_node_id(self) -> str:
        k = len(self._nid_order)
        nid = f"Node_{k}"
        self._nid_order.append(nid)
        return nid

    def _world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        ix = int(math.floor(x / self.grid_size))
        iy = int(math.floor(y / self.grid_size))
        return _cell(ix, iy)

    def _bearing_to_dir8(self, dx: float, dy: float) -> str:
        ang = math.degrees(math.atan2(dy, dx))
        if ang < 0:
            ang += 360.0
        idx = int(round(ang / 45.0)) % 8
        return DIR_8[idx]

    def step(self, x: float, y: float, z: float, action: int) -> None:
        """每环境步调用：更新节点、边、前沿记录。"""
        prev_cell = self._last_cell
        cell = self._world_to_cell(x, y)
        self._cell_last_z[cell] = z

        if cell not in self._cell_to_nid:
            nid = self._alloc_node_id()
            self._cell_to_nid[cell] = nid
            self._visit_count[nid] = 1
            self._exited.setdefault(nid, set())
        else:
            nid = self._cell_to_nid[cell]
            if prev_cell is not None and prev_cell != cell:
                self._visit_count[nid] = self._visit_count.get(nid, 0) + 1

        if self._last_cell is not None and self._last_nid is not None and cell != self._last_cell:
            self._edges.append(
                Edge(
                    from_node=self._last_nid,
                    to_node=nid,
                    action_taken=ACTION_NAMES_CN.get(action, str(action)),
                    semantic_reason="",
                )
            )
            if self._last_xyz is not None:
                dx = x - self._last_xyz[0]
                dy = y - self._last_xyz[1]
                dlabel = self._bearing_to_dir8(dx, dy)
                self._exited.setdefault(self._last_nid, set()).add(dlabel)

        self._last_cell = cell
        self._last_nid = nid
        self._last_xyz = (x, y, z)
        self._step_log.append(
            {
                "xyz": [round(x, 3), round(y, 3), round(z, 3)],
                "action_id": int(action),
                "action_name_cn": ACTION_NAMES_CN.get(action, str(action)),
                "grid_cell": [int(cell[0]), int(cell[1])],
                "node_id": nid,
            }
        )

    def facts_for_llm(self) -> Dict[str, Any]:
        """
        供 ``llm_prompts.build_user_prompt_from_facts`` 使用的结构化事实（无最终自然语言描述）。
        LLM 应据此填写完整 ``TopoTextMap`` 的自然语言字段。
        """
        nodes_out: List[Dict[str, Any]] = []
        for nid in self._nid_order:
            cell: Optional[List[int]] = None
            for c, n in self._cell_to_nid.items():
                if n == nid:
                    cell = [int(c[0]), int(c[1])]
                    break
            if cell is None:
                cell = [0, 0]
            ck = (cell[0], cell[1])
            z = float(self._cell_last_z.get(ck, -5.0))
            nodes_out.append(
                {
                    "node_id": nid,
                    "grid_cell": cell or [0, 0],
                    "grid_size_m": self.grid_size,
                    "visit_count": int(self._visit_count.get(nid, 1)),
                    "representative_z_m": round(z, 2),
                }
            )
        edges_out: List[Dict[str, Any]] = []
        for e in self._edges:
            edges_out.append(
                {
                    "from_node": e.from_node,
                    "to_node": e.to_node,
                    "action_taken": e.action_taken,
                }
            )
        cur = self._last_nid or (self._nid_order[0] if self._nid_order else "")
        frontier_dirs: List[str] = []
        if cur:
            used = self._exited.get(cur, set())
            frontier_dirs = [d for d in DIR_8 if d not in used]

        return {
            "task": self.task,
            "discretization": {"grid_size_m": self.grid_size, "mode": "floor(x/grid_size), floor(y/grid_size)"},
            "current_node_id": cur,
            "nodes": nodes_out,
            "transitions_observed": edges_out,
            "unexplored_directions_from_current_node": frontier_dirs,
            "step_sequence": list(self._step_log),
        }

    def to_json_dict(self) -> dict[str, Any]:
        """持久化用：仅含结构化 ``facts_for_llm``，不含本地生成的语义占位。"""
        facts = self.facts_for_llm()
        return {
            "task": self.task,
            "current_location_id": facts.get("current_node_id", ""),
            "facts_for_llm": facts,
        }
