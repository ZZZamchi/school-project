# 文本拓扑地图（TopoTextMap）

面向「节点—边—未探索前沿」的结构化拓扑表示；`facts_for_llm()` 可配合外部 VLM/LLM 使用。本地 `TopoTextMapBuilder` 负责栅格离散化与结构化事实。

## 包内文件

| 文件 | 职责 |
|------|------|
| `schema.py` | `TopoTextMap` / `Node` / `Edge` / `Frontier` 等数据模型 |
| `builder.py` | 轨迹栅格化、`TopoTextMapBuilder.step`、`facts_for_llm()` |
| `llm_prompts.py` | `build_prompt_pair` / `export_prompt_bundle` / `parse_llm_topo_json`（解析器与提示词组装） |
| `vision_topo_nav.py` | RGB + 拓扑前沿启发式决策 |
| `ego_map_context.py` | 与 PPO 同源机体系地图摘要（attraction/obstacle/exploration） |
| `apex_vl_nav.py` | Qwen3-VL + 拓扑 + 地图摘要（需安装 `scripts/requirements-apex-vl.txt`） |

## 测试与脚本（在仓库 **apex 根目录** 执行）

- 单测：`python3 -m unittest discover -v tests`
- AirSim 连通性：`python3 scripts/check_airsim.py`
- 评测：`python3 run_test_official_ppo.py --help`

## 推荐数据流

环境轨迹 → `TopoTextMapBuilder.step` → `facts_for_llm()` →（可选）VLM 推理 → 离散动作；或用 `vision_topo_nav_decide` / `apex_vl_topo_nav_decide` 与 `AirSimDroneEnv` 对齐。
