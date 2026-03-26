"""
由 **LLM 根据 prompt 生成** 文本拓扑图（节点语义、边理由、前沿观察），本模块只负责 **System / User 提示词** 与 **解析模型输出**。

使用方式：
1. 调用 ``build_system_prompt()`` + ``build_user_prompt_from_facts()`` 得到字符串，送入任意聊天 API；
2. 将模型返回的 **纯 JSON**（可含 ```json 围栏）交给 ``parse_llm_topo_json`` → ``TopoTextMap``。

结构化事实由 ``TopoTextMapBuilder.facts_for_llm()`` 提供；**自然语言拓扑字段须由 LLM 根据本模块 prompt 生成**。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from uav_search.topo_text_map.schema import TopoTextMap

# ---------------------------------------------------------------------------
# System Prompt：约束输出为单一 JSON，且字段与用户约定一致
# ---------------------------------------------------------------------------

FEW_SHOT_JSON = r"""
{
  "task": "demo nav task",
  "current_location_id": "Node_1",
  "map_graph": {
    "nodes": [
      {
        "node_id": "Node_0",
        "semantic_description": "茂密的针叶林上方，视野受限。",
        "visit_count": 1
      },
      {
        "node_id": "Node_1",
        "semantic_description": "开阔的沙滩，连接着森林和海洋。",
        "visit_count": 1
      }
    ],
    "edges": [
      {
        "from_node": "Node_0",
        "to_node": "Node_1",
        "action_taken": "向南飞行 (South)",
        "semantic_reason": "南方视野开阔，且检测到类似水体的反光。"
      }
    ],
    "unexplored_frontiers": [
      {
        "source_node": "Node_1",
        "direction": "东 (East)",
        "visual_observation": "一望无际的深水区。"
      }
    ]
  }
}
""".strip()


def build_system_prompt() -> str:
    return f"""你是一个无人机环境导航助手，需要根据下方给出的**结构化轨迹事实**，生成**文本拓扑地图**。

## 输出要求（必须严格遵守）
1. 只输出 **一个合法的 JSON 对象**，不要 Markdown 标题、不要解释性正文。
2. 可省略 JSON 外的内容；若用 markdown 代码围栏，围栏内必须是完整 JSON。
3. JSON 顶层字段必须为：
   - "task": string，自然语言任务描述（与输入一致或略润色）。
   - "current_location_id": string，当前所在节点 id（如 Node_1）。
   - "map_graph": object，包含：
     - "nodes": 数组，每项含 "node_id", "semantic_description"（**由你根据地理/场景常识与输入事实撰写**）, "visit_count"（整数）。
     - "edges": 数组，每项含 "from_node", "to_node", "action_taken", "semantic_reason"（**解释为何该动作导致进入下一节点**）。
     - "unexplored_frontiers": 数组，每项含 "source_node", "direction"（建议格式如「东 (East)」）, "visual_observation"（**对该方向未探索区域的视觉/语义推测**）。

## 语义撰写原则
- **semantic_description**：概括该节点处地表/水体/植被/人造物等，简洁、可导航。
- **semantic_reason**：结合动作与空间关系，说明转移动机（探索、避障、方向选择等）。
- **visual_observation**：对前沿方向的合理推测，可含不确定性（「疑似」「可能」）。

## 输出结构示例（字段必须齐全，内容可替换）
{FEW_SHOT_JSON}
"""


def build_user_prompt_from_facts(
    task: str,
    facts: Dict[str, Any],
    extra_instruction: str = "",
) -> str:
    """
    :param task: 任务语句（自然语言）
    :param facts: 由 ``TopoTextMapBuilder.facts_for_llm()`` 等生成的结构化事实（JSON 可序列化）
    :param extra_instruction: 追加约束，如「前沿至少 4 条」
    """
    body = json.dumps(facts, ensure_ascii=False, indent=2)
    tail = f"\n\n## 附加说明\n{extra_instruction.strip()}\n" if extra_instruction.strip() else ""
    return f"""## 任务
{task}

## 轨迹与离散化事实（请据此生成完整拓扑 JSON 中的自然语言字段）
{body}
{tail}
请根据上述事实，直接输出符合 System 要求的 **JSON 对象**。"""


def build_prompt_pair(
    task: str,
    facts: Dict[str, Any],
    extra_instruction: str = "",
) -> Tuple[str, str]:
    """返回 (system_prompt, user_prompt)，便于 OpenAI / 其它 API 的 messages 格式。"""
    return build_system_prompt(), build_user_prompt_from_facts(task, facts, extra_instruction)


def parse_llm_topo_json(raw: str) -> TopoTextMap:
    """
    从模型输出中解析 ``TopoTextMap``。支持外围 ```json ... ``` 围栏。
    """
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    # 若模型多打了废话，尝试截取第一个 {{ ... }}
    if not text.startswith("{"):
        m = re.search(r"\{[\s\S]*\}\s*$", text)
        if m:
            text = m.group(0).strip()
    data = json.loads(text)
    return TopoTextMap.from_dict(data)


def format_messages_openai(
    task: str,
    facts: Dict[str, Any],
    extra_instruction: str = "",
) -> List[Dict[str, str]]:
    """OpenAI Chat Completions 风格 messages 列表。"""
    sys_p, usr_p = build_prompt_pair(task, facts, extra_instruction)
    return [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": usr_p},
    ]


def export_prompt_bundle(
    out_dir: str,
    task: str,
    facts: Dict[str, Any],
    extra_instruction: str = "",
) -> Dict[str, str]:
    """
    将 System / User / few-shot 参考写入目录，便于审阅或与 Qwen 等本地模型对齐。
    返回写入路径的字典键为 ``system``, ``user``, ``few_shot_example``。
    """
    os.makedirs(out_dir, exist_ok=True)
    sys_p, usr_p = build_prompt_pair(task, facts, extra_instruction)
    paths: Dict[str, str] = {}
    p_sys = os.path.join(out_dir, "system_prompt.txt")
    p_usr = os.path.join(out_dir, "user_prompt.txt")
    p_fs = os.path.join(out_dir, "few_shot_reference.json")
    with open(p_sys, "w", encoding="utf-8") as f:
        f.write(sys_p)
    with open(p_usr, "w", encoding="utf-8") as f:
        f.write(usr_p)
    with open(p_fs, "w", encoding="utf-8") as f:
        f.write(FEW_SHOT_JSON)
    paths["system"] = p_sys
    paths["user"] = p_usr
    paths["few_shot_example"] = p_fs
    readme = os.path.join(out_dir, "README.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "system_prompt.txt — 发给模型的系统提示\n"
            "user_prompt.txt — 含 facts JSON 的用户提示\n"
            "few_shot_reference.json — 内嵌于 system 的示例结构（单独副本便于对照）\n"
        )
    paths["readme"] = readme
    return paths
