"""拓扑图与地图摘要单测（unittest）。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import tempfile
import unittest

# 保证从任意 cwd 运行 `python tests/test_topo_text_map.py` 时可导入 uav_search
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from uav_search.topo_text_map.builder import TopoTextMapBuilder
from uav_search.topo_text_map.llm_prompts import build_prompt_pair, export_prompt_bundle, parse_llm_topo_json
from uav_search.topo_text_map.topo_nav_policy import choose_frontier_nearest_bearing, dir_label_to_bearing_deg
from uav_search.topo_text_map.ego_map_context import ego_map_context_for_policy, fuse_wedge_scores_with_map_context
from uav_search.topo_text_map.vision_topo_nav import _body_wedge_idx_for_world_bearing, score_wedges_from_rgb
from uav_search.topo_text_map.schema import Edge, Frontier, MapGraph, Node, TopoTextMap


def _demo_topo_text_map() -> TopoTextMap:
    """单测用：固定 JSON 拓扑样例（与生产/真实场景无关；仅校验 schema 往返）。"""
    nodes = [
        Node(
            node_id="Node_0",
            semantic_description="高架匝道旁的灰色隔音屏，金属反光条呈斜向条纹；下方车流呈断续光带。",
            visit_count=1,
        ),
        Node(
            node_id="Node_1",
            semantic_description="夜间物流园装卸区，集装箱堆垛形成规则矩形阴影，地面标线褪色发白。",
            visit_count=1,
        ),
    ]
    edges = [
        Edge(
            from_node="Node_0",
            to_node="Node_1",
            action_taken="向西南偏航并下降 (SouthWest)",
            semantic_reason="侧风较强，选择贴近建筑物背风侧以减小横漂，同时保持对装卸区入口的视线。",
        )
    ]
    ufs = [
        Frontier(
            source_node="Node_1",
            direction="东 (East)",
            visual_observation="远处航站楼玻璃幕墙呈条状高光，疑似有进近灯光但未确认。",
        ),
        Frontier(
            source_node="Node_1",
            direction="东南 (SouthEast)",
            visual_observation="成片光伏板阵列，强镜面反射导致曝光过饱和，细节不可靠。",
        ),
        Frontier(
            source_node="Node_1",
            direction="西 (West)",
            visual_observation="铁路编组场平行股道密集，枕木纹理重复度高，易产生视错觉。",
        ),
        Frontier(
            source_node="Node_0",
            direction="北 (North)",
            visual_observation="城市天际线轮廓被雾霾柔化，最高楼顶部红色防撞灯周期性闪烁。",
        ),
    ]
    mg = MapGraph(nodes=nodes, edges=edges, unexplored_frontiers=ufs)
    return TopoTextMap(
        task="unittest synthetic topo lattice-v2",
        current_location_id="Node_1",
        map_graph=mg,
    )


class TestTopoTextMap(unittest.TestCase):
    def test_example_roundtrip(self) -> None:
        ex = _demo_topo_text_map()
        d = ex.to_dict()
        s = json.dumps(d, ensure_ascii=False)
        d2 = json.loads(s)
        back = TopoTextMap.from_dict(d2)
        self.assertEqual(back.task, "unittest synthetic topo lattice-v2")
        self.assertEqual(back.current_location_id, "Node_1")
        self.assertEqual(len(back.map_graph.nodes), 2)
        self.assertEqual(len(back.map_graph.edges), 1)
        self.assertEqual(len(back.map_graph.unexplored_frontiers), 4)

    def test_builder_creates_nodes_on_grid_cross(self) -> None:
        b = TopoTextMapBuilder(task="demo nav task", grid_size=20.0, seed=0)
        b.step(0.0, 0.0, -5.0, 0)
        b.step(25.0, 0.0, -5.0, 0)
        b.step(45.0, 2.0, -5.0, 0)
        d = b.to_json_dict()
        facts = d["facts_for_llm"]
        self.assertGreaterEqual(len(facts["nodes"]), 2)
        self.assertGreaterEqual(len(facts["transitions_observed"]), 1)
        self.assertEqual(d["task"], "demo nav task")
        self.assertTrue(d["current_location_id"].startswith("Node_"))

    def test_llm_prompt_contains_facts_and_parse_roundtrip(self) -> None:
        b = TopoTextMapBuilder(task="demo nav task", grid_size=10.0, seed=1)
        b.step(0.0, 0.0, -5.0, 0)
        b.step(12.0, 0.0, -5.0, 0)
        facts = b.facts_for_llm()
        self.assertIn("step_sequence", facts)
        self.assertIn("transitions_observed", facts)
        sys_p, usr_p = build_prompt_pair("demo nav task", facts)
        self.assertIn("JSON", sys_p)
        self.assertIn("step_sequence", usr_p)
        ex = _demo_topo_text_map()
        raw = json.dumps(ex.to_dict(), ensure_ascii=False)
        back = parse_llm_topo_json("```json\n" + raw + "\n```")
        self.assertEqual(back.task, ex.task)

    def test_choose_frontier_nearest_reference_bearing(self) -> None:
        """在给定参考方位下选取最近前沿（DIR_8）。"""
        opts = ["北 (North)", "东 (East)", "西 (West)"]
        self.assertEqual(
            choose_frontier_nearest_bearing(opts, 85.0),
            "北 (North)",
        )
        self.assertEqual(
            choose_frontier_nearest_bearing(opts, 350.0),
            "东 (East)",
        )
        self.assertAlmostEqual(dir_label_to_bearing_deg("东南 (SouthEast)"), 315.0)

    def test_body_wedge_forward_aligns_with_rgb_bins(self) -> None:
        self.assertEqual(_body_wedge_idx_for_world_bearing(30.0, 30.0), 0)
        self.assertEqual(_body_wedge_idx_for_world_bearing(75.0, 30.0), 1)

    def test_ego_map_context_shapes_and_fusion(self) -> None:
        am = np.zeros((40, 40, 10, 2), dtype=np.float32)
        am[..., 1] = -1.0
        am[20, 22, 5, 0] = 0.9
        am[20, 22, 5, 1] = 10.0
        em = np.zeros((40, 40, 10), dtype=np.float32)
        om = np.zeros((40, 40, 10), dtype=np.float32)
        pose = {"position": np.array([20, 21, 5], dtype=np.float64), "orientation": 0}
        ctx = ego_map_context_for_policy(am, em, om, pose)
        self.assertEqual(len(ctx["attraction_sector_mean_ego8"]), 8)
        self.assertEqual(len(ctx["obstacle_sector_mean_ego8"]), 8)
        ws = np.ones(8, dtype=np.float64)
        fused, meta = fuse_wedge_scores_with_map_context(ws, ctx)
        self.assertEqual(fused.shape, (8,))
        self.assertTrue(meta.get("fused", False))

    def test_score_wedges_from_rgb_shape(self) -> None:
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        s = score_wedges_from_rgb(rgb, "locate depot alpha")
        self.assertEqual(s.shape, (8,))

    def test_builder_facts_unexplored_feasible_with_topo_nav(self) -> None:
        b = TopoTextMapBuilder(task="t", grid_size=15.0, seed=0)
        b.step(0.0, 0.0, -5.0, 0)
        facts = b.facts_for_llm()
        self.assertIn("unexplored_directions_from_current_node", facts)
        u = facts["unexplored_directions_from_current_node"]
        self.assertGreater(len(u), 0)

    def test_export_prompt_bundle_writes_files(self) -> None:
        b = TopoTextMapBuilder(task="测试", grid_size=10.0, seed=0)
        b.step(0.0, 0.0, -5.0, 0)
        facts = b.facts_for_llm()
        with tempfile.TemporaryDirectory() as td:
            paths = export_prompt_bundle(td, "测试", facts)
            self.assertTrue(os.path.isfile(paths["system"]))
            self.assertTrue(os.path.isfile(paths["user"]))
            self.assertTrue(os.path.isfile(paths["few_shot_example"]))


if __name__ == "__main__":
    unittest.main()
