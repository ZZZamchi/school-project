"""
从离散轨迹构建文本拓扑图。

- 将 (x,y) 按 ``grid_size`` 栅格化，每个首次进入的栅格成为一个 ``Node_*``。
- 栅格变化时记录 ``Edge``（动作名 + 简短理由模板）。
- ``unexplored_frontiers``：从当前节点按 8 方位生成尚未沿该方向离开过的前沿（模板化观察文本）。

自然语言字段应由 **LLM** 根据 ``llm_prompts`` 与 ``facts_for_llm()`` 生成；本地模板仅作无 API 时的占位。
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from uav_search.topo_text_map.schema import Edge, Frontier, MapGraph, Node, TopoTextMap

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
        self.task = task
        self.grid_size = max(grid_size, 1.0)
        self._rng = random.Random(seed)

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
        # vision_topo / apex_vl 防原地打转：连续只转弯时强制前飞一步
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

    def _cell_semantic(self, ix: int, iy: int, z: float) -> str:
        """占位：按栅格与高度生成简短中文描述，可换为 VLM。"""
        # CabinLake 类场景：偏水域/岸/林地的粗模板
        h = "低空" if z > -4 else "近水面/低高度"
        biome = self._rng.choice(
            [
                "开阔水域或湖岸附近，远处可见岸线。",
                "树林边缘，树冠遮挡部分视野。",
                "沙滩或浅滩区域，地表纹理较均匀。",
                "湖湾转折处，地形略有起伏。",
            ]
        )
        return f"栅格 ({ix},{iy})，{h}；{biome}"

    def _reason_template(self, action: int, from_nid: str, to_nid: str) -> str:
        return (
            f"从 {from_nid} 执行「{ACTION_NAMES_CN.get(action, str(action))}」后进入新区域 {to_nid}；继续探索。"
        )

    def _bearing_to_dir8(self, dx: float, dy: float) -> str:
        ang = math.degrees(math.atan2(dy, dx))
        if ang < 0:
            ang += 360.0
        idx = int(round(ang / 45.0)) % 8
        return DIR_8[idx]

    def _frontier_observation(self, source: str, direction: str) -> str:
        """占位前沿观察；可接检测/分割结果。"""
        opts = [
            "远景对比度较低，需靠近或换视角确认。",
            "水面反光较强，纹理变化明显。",
            "岸线几何变化，适合作为导航地标。",
            "植被遮挡严重，远景不清晰。",
        ]
        return f"[{source} → {direction}] {self._rng.choice(opts)}"

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
            reason = self._reason_template(action, self._last_nid, nid)
            self._edges.append(
                Edge(
                    from_node=self._last_nid,
                    to_node=nid,
                    action_taken=ACTION_NAMES_CN.get(action, str(action)),
                    semantic_reason=reason,
                )
            )
            # 记录从上一节点沿某方位“已探索”的离开方向
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
        供 ``llm_prompts.build_user_prompt_from_facts`` 使用的**结构化事实**（无最终自然语言描述）。
        LLM 应据此填写 ``semantic_description`` / ``semantic_reason`` / ``visual_observation``。
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

    def build_snapshot(self) -> TopoTextMap:
        if not self._nid_order:
            return TopoTextMap(
                task=self.task,
                current_location_id="",
                map_graph=MapGraph(nodes=[], edges=[], unexplored_frontiers=[]),
            )
        nodes = []
        for nid in self._nid_order:
            nodes.append(
                Node(
                    node_id=nid,
                    semantic_description=self._semantic_for_node(nid),
                    visit_count=int(self._visit_count.get(nid, 1)),
                )
            )
        cur = self._last_nid or self._nid_order[0]
        frontiers = self._compute_frontiers(cur)
        mg = MapGraph(nodes=nodes, edges=list(self._edges), unexplored_frontiers=frontiers)
        return TopoTextMap(task=self.task, current_location_id=cur, map_graph=mg)

    def _semantic_for_node(self, nid: str) -> str:
        for c, n in self._cell_to_nid.items():
            if n == nid:
                ix, iy = c
                z = float(self._cell_last_z.get(c, -5.0))
                return self._cell_semantic(ix, iy, z)
        return "未知区域。"

    def _compute_frontiers(self, current_nid: str) -> List[Frontier]:
        """从当前节点列出尚未作为「沿该方位离开」记录的前沿（简化版）。"""
        out: List[Frontier] = []
        used = self._exited.get(current_nid, set())
        for d in DIR_8:
            if d in used:
                continue
            out.append(
                Frontier(
                    source_node=current_nid,
                    direction=d,
                    visual_observation=self._frontier_observation(current_nid, d),
                )
            )
        return out[:8]

    def to_json_dict(self) -> dict:
        return self.build_snapshot().to_dict()


def build_example_user_format() -> TopoTextMap:
    """与用户示例等价的静态图（用于单测 / 演示）。"""
    from uav_search.topo_text_map.schema import MapGraph

    nodes = [
        Node(
            node_id="Node_0",
            semantic_description="茂密的针叶林上方，视野受限。",
            visit_count=1,
        ),
        Node(
            node_id="Node_1",
            semantic_description="开阔的沙滩，连接着森林和海洋。",
            visit_count=1,
        ),
    ]
    edges = [
        Edge(
            from_node="Node_0",
            to_node="Node_1",
            action_taken="向南飞行 (South)",
            semantic_reason="南方视野开阔，且检测到类似水体的反光。",
        )
    ]
    ufs = [
        Frontier(
            source_node="Node_1",
            direction="东 (East)",
            visual_observation="一望无际的深水区。",
        ),
        Frontier(
            source_node="Node_1",
            direction="东南 (SouthEast)",
            visual_observation="海面上有一个模糊的黑色凸起物。",
        ),
        Frontier(
            source_node="Node_1",
            direction="西 (West)",
            visual_observation="沿着海岸线的礁石区，海浪很大。",
        ),
        Frontier(
            source_node="Node_0",
            direction="北 (North)",
            visual_observation="更深的森林内部。",
        ),
    ]
    mg = MapGraph(nodes=nodes, edges=edges, unexplored_frontiers=ufs)
    return TopoTextMap(task="demo nav task", current_location_id="Node_1", map_graph=mg)
