"""
APEX 框架下用 **Qwen3-VL** 做动作决策（同步链路）：拓扑 ``facts_for_llm`` 作为时空语义记忆的文本摘要，
RGB 作为视觉观测；输出与 ``AirSimDroneEnv`` 一致的离散动作（论文中 RL/PPO 策略由 VLM+拓扑 替代）。

参见: https://arxiv.org/abs/2602.00551 ；模型: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Tuple

import numpy as np

from uav_search.topo_text_map.vision_topo_nav import (
    apply_yaw_spin_guard,
    maybe_oob_mutate_and_return_action,
    nav_decision_bootstrap,
    set_nav_vert_intent,
    vision_topo_nav_decide,
)

_vl_model: Any = None
_vl_processor: Any = None
_vl_model_path: str | None = None

APEX_VL_DEFAULT_HF_ID = "Qwen/Qwen3-VL-8B-Instruct"

SYSTEM_PROMPT_ZH = """你是无人机导航智能体（APEX 风格管线）。
你还会收到两类记忆：（1）文本拓扑 JSON；（2）与 PPO 训练时 ``map_input_preparation`` 同源的机体系局部地图摘要（attraction / obstacle / exploration 的 8 扇区统计与吸引峰值方位），对应论文中的体素语义-几何地图融合结果在策略输入侧的压缩视图。
请结合第一视角 RGB：attraction 高的方向更可能靠近任务相关区域；obstacle 高的方向应规避。
除非必须对齐航向，否则优先选择前飞（action_id=0）；不要连续多步只输出转弯（1/2/3）而不前飞。
必须只输出一个 JSON 对象，不要 Markdown 代码围栏，不要其它解释。
Schema: {"action_id": <0-5 的整数>, "reason": "<一句中文理由>"}
动作: 0=低空前进, 1=原地左转, 2=原地右转, 3=原地后转, 4=下降, 5=上升。"""


def _default_model_path() -> str:
    return (
        os.environ.get("OFFICIAL_APEX_QWEN3_VL_DIR")
        or os.environ.get("APEX_VL_MODEL_DIR")
        or APEX_VL_DEFAULT_HF_ID
    )


def _compact_facts_for_prompt(facts: dict, max_json_chars: int = 6000) -> dict:
    out: dict = dict(facts)
    seq = out.get("step_sequence")
    if isinstance(seq, list) and len(seq) > 10:
        out["step_sequence_tail"] = seq[-10:]
        del out["step_sequence"]
    s = json.dumps(out, ensure_ascii=False)
    while len(s) > max_json_chars and "step_sequence_tail" in out:
        tail = out["step_sequence_tail"]
        if isinstance(tail, list) and len(tail) > 3:
            out["step_sequence_tail"] = tail[-3:]
            s = json.dumps(out, ensure_ascii=False)
        else:
            break
    if len(s) > max_json_chars:
        out = {"note": "facts 过长已截断", "current_node_id": out.get("current_node_id"), "summary": s[: max_json_chars - 80]}
    return out


def _parse_action_id_from_text(text: str) -> int | None:
    text = text.strip()
    m = re.search(r"\{[^{}]*\"action_id\"[^{}]*\}", text, re.DOTALL)
    chunk = m.group(0) if m else text
    try:
        obj = json.loads(chunk)
        v = int(obj.get("action_id", -1))
        if 0 <= v <= 5:
            return v
    except Exception:
        pass
    m2 = re.search(r'"action_id"\s*:\s*(\d+)', text)
    if m2:
        v = int(m2.group(1))
        if 0 <= v <= 5:
            return v
    m3 = re.search(r"action_id\s*[=:]\s*(\d+)", text, re.I)
    if m3:
        v = int(m3.group(1))
        if 0 <= v <= 5:
            return v
    return None


def _get_vl_model_and_processor(model_id_or_path: str | None = None):
    global _vl_model, _vl_processor, _vl_model_path
    path = model_id_or_path or _default_model_path()
    if _vl_model is not None and _vl_model_path == path:
        return _vl_model, _vl_processor
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as e:
        raise RuntimeError(
            "加载 Qwen3-VL 需要新版 transformers（含 Qwen3VLForConditionalGeneration）。\n"
            "可执行: pip install -r scripts/requirements-apex-vl.txt（自本仓库 apex 根目录）\n"
            "或按模型卡: pip install git+https://github.com/huggingface/transformers\n"
            f"原始错误: {e}"
        ) from e

    dtype = "auto"
    kwargs: dict = {"dtype": dtype, "device_map": "auto"}
    if os.environ.get("APEX_VL_ATTN_IMPL"):
        kwargs["attn_implementation"] = os.environ["APEX_VL_ATTN_IMPL"]

    _vl_processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    _vl_model = Qwen3VLForConditionalGeneration.from_pretrained(path, trust_remote_code=True, **kwargs)
    _vl_model.eval()
    _vl_model_path = path
    return _vl_model, _vl_processor


def _move_model_inputs_to_device(batch: Any, device: Any) -> Any:
    """BatchFeature / dict 兼容：避免 ``.to(device)`` 在部分版本上抛 ``TypeError``。"""
    import torch

    if hasattr(batch, "to"):
        try:
            return batch.to(device)
        except Exception:
            pass
    if isinstance(batch, dict):
        out: dict = {}
        for k, v in batch.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out
    raise TypeError(f"无法将模型输入移到设备：{type(batch)}")


def _vl_generate(
    rgb: np.ndarray,
    user_text: str,
    model_id_or_path: str | None,
    max_new_tokens: int,
    temperature: float,
) -> str:
    from PIL import Image

    import torch

    model, processor = _get_vl_model_and_processor(model_id_or_path)
    image = Image.fromarray(rgb.astype(np.uint8, copy=False))
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT_ZH,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    dev = next(model.parameters()).device
    inputs_on_dev = _move_model_inputs_to_device(inputs, dev)
    gen_kw: dict = {"max_new_tokens": max_new_tokens}
    if temperature and temperature > 0:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = float(temperature)
        gen_kw["top_p"] = 0.9
    else:
        gen_kw["do_sample"] = False

    with torch.no_grad():
        out_ids = model.generate(**inputs_on_dev, **gen_kw)
    in_len = inputs_on_dev["input_ids"].shape[1]
    gen = out_ids[0, in_len:]
    text = processor.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text


def apex_vl_topo_nav_decide(
    client: Any,
    topo_builder: Any,
    rgb: np.ndarray,
    grid_position: np.ndarray | None,
    task_text: str = "",
    grid_margin: float = 3.0,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    model_id_or_path: str | None = None,
    map_context: Dict[str, Any] | None = None,
) -> Tuple[int, Dict[str, Any]]:
    decision, _yaw_deg, _bearing_forward, _cur_idx = nav_decision_bootstrap(client)

    oob_action = maybe_oob_mutate_and_return_action(decision, grid_position, grid_margin)
    if oob_action is not None:
        decision["apex_vl"] = {"skipped": "oob_preempt", "module": "topo_safety_shell"}
        return oob_action, decision

    set_nav_vert_intent(decision, "none")

    facts = topo_builder.facts_for_llm()
    compact = _compact_facts_for_prompt(facts)
    map_block = ""
    if map_context:
        map_block = (
            "地图记忆摘要（机体系，与训练地图张量同源）JSON:\n"
            f"{json.dumps(map_context, ensure_ascii=False, indent=2)}\n\n"
        )
    user_text = (
        f"任务描述（自然语言，无显式目标坐标）:\n{task_text}\n\n"
        f"{map_block}"
        f"拓扑记忆 JSON:\n{json.dumps(compact, ensure_ascii=False, indent=2)}\n\n"
        f"请只输出 JSON: {{\"action_id\": 0-5, \"reason\": \"...\"}}"
    )

    model_path = model_id_or_path or _default_model_path()
    raw = ""
    action_id: int | None = None
    vl_error: str | None = None
    try:
        raw = _vl_generate(
            rgb,
            user_text,
            model_id_or_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        action_id = _parse_action_id_from_text(raw)
    except Exception as e:
        vl_error = f"{type(e).__name__}: {e}"

    if action_id is None:
        fb_action, fb_decision = vision_topo_nav_decide(
            client,
            topo_builder,
            rgb,
            grid_position,
            task_text=task_text,
            grid_margin=grid_margin,
            map_context=map_context,
        )
        fb_decision["apex_vl"] = {
            "fallback": "vision_topo_nav",
            "raw_model_output": raw[:2000] if raw else "",
            "parse_error": vl_error or "unparseable_or_empty",
            "model_path": model_path,
        }
        return fb_action, fb_decision

    decision["horizontal"]["phase"] = "apex_vl_json"
    decision["reason"] = f"Qwen3-VL 决策 action_id={action_id}"
    decision["topo_map"] = {"mode": "apex_vl", "current_node_id": facts.get("current_node_id")}
    if map_context is not None:
        decision["grid_maps"] = {"ego_summary_for_policy": map_context}
    decision["apex_vl"] = {
        "module": "Qwen3-VL-8B-Instruct",
        "model_path": model_path,
        "raw_output": raw[:4000],
        "parsed_action_id": action_id,
        "paper_frame": "VLM(spatio-semantic memory as text topo) replaces RL policy; sync path",
    }
    return apply_yaw_spin_guard(topo_builder, action_id, decision)
