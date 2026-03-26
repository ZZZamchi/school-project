# 文本拓扑地图（TopoTextMap）

面向「节点—边—未探索前沿」的结构化拓扑表示；`TopoTextMapBuilder.facts_for_llm()` 提供**仅结构化事实**（无本地语义占位），自然语言拓扑应由 **LLM** 根据 `llm_prompts.py` 生成。

## 包内文件

| 文件 | 职责 |
|------|------|
| `schema.py` | `TopoTextMap` / `Node` / `Edge` / `Frontier` 等数据模型 |
| `builder.py` | 轨迹栅格化、`TopoTextMapBuilder.step`、`facts_for_llm()`、`to_json_dict()` |
| `llm_prompts.py` | `build_prompt_pair` / `export_prompt_bundle` / `parse_llm_topo_json` |
| `vision_topo_nav.py` | 与 `AirSimDroneEnv` 对齐的**安全壳**（OOB、机位、防原地打转）；**不**作为无 LLM 的替代策略用于正式评测 |
| `ego_map_context.py` | 与 PPO 同源机体系地图摘要（attraction/obstacle/exploration） |
| `apex_vl_nav.py` | Qwen3-VL + `facts_for_llm` + 地图摘要（需 `scripts/requirements-apex-vl.txt`）；**无 VLM 回退** |

## 测试与脚本（在仓库 **apex 根目录** 执行）

- 单测：`python3 -m pytest tests -q`（需完整 `uav_search` 依赖）
- AirSim 连通性：`python3 scripts/check_airsim.py`
- 评测：`python3 run_test_official_ppo.py --policy apex_vl --help`

## 推荐数据流

环境轨迹 → `TopoTextMapBuilder.step` → `facts_for_llm()` → **Qwen3-VL**（`apex_vl_topo_nav_decide`）→ 离散动作；评测脚本仅支持 `ppo` 与 `apex_vl`。
