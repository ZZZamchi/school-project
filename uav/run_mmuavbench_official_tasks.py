# -*- coding: utf-8 -*-
"""
MM-UAVBench 官方图像任务评测（多模型、多任务，零样本 MCQ）
支持多种 VLM，从 HuggingFace 拉取题目与图片，生成 txt 报告。
根据硬件自动做线程/显存相关优化。
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime

# 抑制 AMD GPU 上 SDPA 的实验性告警（不影响结果）
warnings.filterwarnings("ignore", message=".*AMD GPU is still experimental.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*TORCH_ROCM_AOTRITON.*", category=UserWarning)

# 减少 HuggingFace 重复告警（在 import 前设置）
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Windows 控制台 UTF-8，减少乱码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from PIL import Image

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("请安装: pip install huggingface_hub")
    sys.exit(1)


def get_hardware_info() -> dict:
    """检测 CPU/GPU/内存，返回用于优化的信息。"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": None,
        "vram_gb": None,
        "cpu_count": os.cpu_count() or 4,
        "ram_gb": None,
    }
    if info["cuda_available"]:
        try:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        except Exception:
            pass
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except Exception:
        info["ram_gb"] = None
    return info


def get_gpu_resource_string() -> str:
    """Return GPU VRAM (and util% if nvidia-smi)."""
    if not torch.cuda.is_available():
        return "GPU: N/A"
    parts = []
    try:
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        parts.append(f"VRAM {alloc:.2f}/{total:.2f} GB")
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=2, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode == 0 and out.stdout.strip():
            parts.append(f"util {out.stdout.strip().split()[0]}%")
    except Exception:
        pass
    return "GPU " + (", ".join(parts) if parts else "N/A")


def get_system_resource_string() -> str:
    """Return system RAM usage if psutil."""
    try:
        import psutil
        r = psutil.virtual_memory()
        return f"RAM {r.used / 1e9:.1f}/{r.total / 1e9:.1f} GB"
    except Exception:
        return ""


def setup_hardware(cpu_threads: int | None = None) -> str:
    """
    根据硬件设置 PyTorch 线程数并返回 device。
    若未传 cpu_threads，则 CPU 下用 min(16, cpu_count)，GPU 下不限制线程数。
    """
    info = get_hardware_info()
    device = "cuda" if info["cuda_available"] else "cpu"

    if device == "cpu":
        n = cpu_threads if cpu_threads is not None else min(8, info["cpu_count"])
        torch.set_num_threads(n)
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("medium")
            except Exception:
                pass
    else:
        n = min(4, info["cpu_count"])
        torch.set_num_threads(n)
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    return device

# 配置
REPO_ID = "daisq/MM-UAVBench"
REPO_TYPE = "dataset"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "data", "mm_uavbench_cache")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 官方全部图像任务
IMAGE_TASKS = [
    "Scene_Classification",
    "Orientation_Classification",
    "Environment_State_Classification",
    "Urban_OCR",
    "Class_Agnostic_Counting",
    "Referring_Expression_Counting",
    "Target_Backtracking",
    "Cross_Object_Reasoning",
    "Intent_Analysis_and_Prediction",
    "Scene_Attribute_Understanding",
    "Scene_Damage_Assessment",
    "Scene_Analysis_and_Prediction",
    "Temporal_Ordering",
    "Ground_Target_Planning",
    "Air_Ground_Collaborative_Planning",
    "Swarm_Collaborative_Planning",
]

# 可用模型：id -> display_name
AVAILABLE_MODELS = {
    "clip_vitb32": "CLIP ViT-B/32 (openai)",
    "siglip_base": "SigLIP Base 224 (google)",
    "clip_vitl14": "CLIP ViT-L/14 (openai)",
    "qwen2vl_2b": "Qwen2-VL-2B-Instruct (Qwen)",
    "qwen2vl_7b": "Qwen2-VL-7B-Instruct (Qwen)",
}


def load_task_qa(task_name: str, max_samples: int | None = None) -> list[dict]:
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"tasks/{task_name}.json",
        repo_type=REPO_TYPE,
        cache_dir=CACHE_DIR,
    )
    with open(path, "r", encoding="utf-8") as f:
        qa_list = json.load(f)
    if max_samples is not None:
        qa_list = qa_list[: max_samples]
    return qa_list


def get_image_path(qa: dict) -> str:
    resources = qa.get("metadata", {}).get("data_resources", [])
    if not resources:
        raise ValueError("no data_resources")
    return resources[0]["path"]


def options_to_ordered_list(options: dict) -> tuple[list[str], list[str]]:
    letters, texts = [], []
    for k in sorted(options.keys()):
        v = options.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        letters.append(k)
        texts.append(v.strip())
    return letters, texts


# 批量推理时每题最多选项数（不足则用首选项填充，保证矩阵形状一致）
BATCH_OPTIONS_PAD = 4


def normalize_answer(s: str) -> str:
    s = (s or "").strip().upper()
    m = re.match(r"^([A-Z])", s)
    return m.group(1) if m else ""


# --------------- 多模型：统一接口 predict(image_path, options) -> str ---------------


def _make_clip_runner(device: str, model_id: str):
    from transformers import CLIPModel, CLIPProcessor
    if model_id == "clip_vitb32":
        name = "openai/clip-vit-base-patch32"
    elif model_id == "clip_vitl14":
        name = "openai/clip-vit-large-patch14"
    else:
        raise ValueError(model_id)
    model = CLIPModel.from_pretrained(name).to(device)
    processor = CLIPProcessor.from_pretrained(name)
    K = BATCH_OPTIONS_PAD

    def _pad_options(letters: list[str], texts: list[str]) -> tuple[list[str], list[str]]:
        if not letters or not texts:
            return [], []
        while len(letters) < K:
            letters.append(letters[0])
            texts.append(texts[0])
        return letters[:K], texts[:K]

    def predict(image_path: str, options: dict) -> str:
        letters, texts = options_to_ordered_list(options)
        if not letters or not texts:
            return ""
        pil = Image.open(image_path).convert("RGB")
        inputs = processor(text=texts, images=pil, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.inference_mode():
            out = model(**inputs)
        logits = out.logits_per_image.squeeze(0).cpu()
        idx = int(logits.argmax().item())
        return letters[idx] if idx < len(letters) else ""

    def predict_batch(batch: list[tuple[str, dict]], preloaded_images: list | None = None) -> list[str]:
        """batch: list of (image_path, options). preloaded_images: optional list of PIL.Image (or None) to skip disk load."""
        n = len(batch)
        out: list[str] = [""] * n
        if n == 0:
            return out
        plis: list = []
        letters_list: list[list[str]] = []
        texts_flat: list[str] = []
        indices: list[int] = []
        for i, (path, options) in enumerate(batch):
            try:
                letters, texts = options_to_ordered_list(options)
                letters, texts = _pad_options(letters, texts)
                if not letters or not texts:
                    continue
                if preloaded_images and i < len(preloaded_images) and preloaded_images[i] is not None:
                    pil = preloaded_images[i]
                else:
                    pil = Image.open(path).convert("RGB")
            except (PermissionError, OSError, IOError, Exception):
                continue
            plis.append(pil)
            letters_list.append(letters)
            texts_flat.extend(texts)
            indices.append(i)
        if not plis:
            return out
        B = len(plis)
        inputs = processor(text=texts_flat, images=plis, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.inference_mode():
            logits_img = model(**inputs).logits_per_image
        logits_img = logits_img.cpu()
        for b, orig_i in enumerate(indices):
            logits_b = logits_img[b, b * K : (b + 1) * K]
            idx = int(logits_b.argmax().item())
            out[orig_i] = letters_list[b][idx] if idx < len(letters_list[b]) else letters_list[b][0]
        return out

    return predict, predict_batch


def _make_siglip_runner(device: str):
    from transformers import SiglipModel, SiglipProcessor
    name = "google/siglip-base-patch16-224"
    model = SiglipModel.from_pretrained(name).to(device)
    processor = SiglipProcessor.from_pretrained(name)
    K = BATCH_OPTIONS_PAD

    def _pad_options(letters: list[str], texts: list[str]) -> tuple[list[str], list[str]]:
        if not letters or not texts:
            return [], []
        while len(letters) < K:
            letters.append(letters[0])
            texts.append(texts[0])
        return letters[:K], texts[:K]

    def predict(image_path: str, options: dict) -> str:
        letters, texts = options_to_ordered_list(options)
        if not letters or not texts:
            return ""
        pil = Image.open(image_path).convert("RGB")
        inputs = processor(text=texts, images=pil, return_tensors="pt", padding="max_length", truncation=True).to(device)
        with torch.inference_mode():
            out = model(**inputs)
        logits = out.logits_per_image.squeeze(0).cpu()
        idx = int(logits.argmax().item())
        return letters[idx] if idx < len(letters) else ""

    def predict_batch(batch: list[tuple[str, dict]], preloaded_images: list | None = None) -> list[str]:
        n = len(batch)
        out: list[str] = [""] * n
        if n == 0:
            return out
        plis, letters_list, texts_flat, indices = [], [], [], []
        for i, (path, options) in enumerate(batch):
            try:
                letters, texts = options_to_ordered_list(options)
                letters, texts = _pad_options(letters, texts)
                if not letters or not texts:
                    continue
                if preloaded_images and i < len(preloaded_images) and preloaded_images[i] is not None:
                    pil = preloaded_images[i]
                else:
                    pil = Image.open(path).convert("RGB")
            except (PermissionError, OSError, IOError, Exception):
                continue
            plis.append(pil)
            letters_list.append(letters)
            texts_flat.extend(texts)
            indices.append(i)
        if not plis:
            return out
        B = len(plis)
        inputs = processor(text=texts_flat, images=plis, return_tensors="pt", padding="max_length", truncation=True).to(device)
        with torch.inference_mode():
            logits_img = model(**inputs).logits_per_image
        logits_img = logits_img.cpu()
        for b, orig_i in enumerate(indices):
            logits_b = logits_img[b, b * K : (b + 1) * K]
            idx = int(logits_b.argmax().item())
            out[orig_i] = letters_list[b][idx] if idx < len(letters_list[b]) else letters_list[b][0]
        return out

    return predict, predict_batch


def _parse_mcq_letter(text: str) -> str:
    """从生成文本中解析出选项字母 A/B/C/D。"""
    text = (text or "").strip().upper()
    for c in text:
        if c in "ABCD":
            return c
    return ""


def _make_qwen2vl_runner(device: str, size: str):
    """Qwen2-VL 生成式 MCQ：构造 prompt，生成回复后解析字母。CPU 下降低分辨率以加速。"""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model_name = "Qwen/Qwen2-VL-2B-Instruct" if size == "2b" else "Qwen/Qwen2-VL-7B-Instruct"
    is_cpu = device == "cpu"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, "bfloat16", torch.float16) if not is_cpu else torch.float32,
        device_map="auto" if not is_cpu else None,
        low_cpu_mem_usage=is_cpu,
    )
    if is_cpu:
        model = model.to("cpu")
    processor = AutoProcessor.from_pretrained(model_name)
    # 限制分辨率：CPU 用更小以加速并省内存，GPU 适中
    try:
        processor.image_processor.min_pixels = 56 * 56
        processor.image_processor.max_pixels = (336 * 336) if is_cpu else (512 * 512)
    except Exception:
        pass

    def predict(image_path: str, options: dict) -> str:
        letters, texts = options_to_ordered_list(options)
        if not letters or not texts:
            return ""
        opt_lines = "\n".join(f"{letters[i]}. {texts[i]}" for i in range(len(letters)))
        prompt = (
            "You are an expert in drone and aerial image analysis. "
            "Answer the following multiple-choice question based on the image. "
            "Reply with ONLY one letter: A, B, C, or D.\n\n"
            f"Question: Based on the image, choose the correct option.\n\n"
            f"Options:\n{opt_lines}\n\nAnswer:"
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            if device == "cuda":
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            else:
                inputs = {k: v.to("cpu") if hasattr(v, "to") else v for k, v in inputs.items()}
            max_tokens = 16 if device == "cpu" else 32
            with torch.inference_mode():
                out_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            input_len = inputs["input_ids"].shape[1]
            gen_ids = out_ids[:, input_len:]
            output_text = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            pred = _parse_mcq_letter(output_text[0] if output_text else "")
            return pred if pred in letters else (letters[0] if letters else "")
        except Exception:
            return letters[0] if letters else ""

    return predict


def get_runner(model_id: str, device: str) -> tuple:
    """返回 (predict_fn, predict_batch_fn)。predict_batch_fn 为 None 表示不支持批量推理。"""
    if model_id in ("clip_vitb32", "clip_vitl14"):
        return _make_clip_runner(device, model_id)
    if model_id == "siglip_base":
        return _make_siglip_runner(device)
    if model_id == "qwen2vl_2b":
        return _make_qwen2vl_runner(device, "2b"), None
    if model_id == "qwen2vl_7b":
        return _make_qwen2vl_runner(device, "7b"), None
    raise ValueError(f"未知模型: {model_id}，可选: {list(AVAILABLE_MODELS.keys())}")


def _load_chunk_images(chunk: list[tuple[str, dict, str, dict]]) -> list:
    """Load images for a chunk in current thread. Returns list of PIL.Image or None (same length as chunk)."""
    out: list = []
    for path, opts, _, _ in chunk:
        try:
            out.append(Image.open(path).convert("RGB"))
        except (PermissionError, OSError, IOError, Exception):
            out.append(None)
    return out


def run_one_task(
    task_name: str,
    max_samples: int | None,
    predict_fn,
    batch_fn=None,
    batch_size: int = 16,
) -> dict:
    try:
        qa_list = load_task_qa(task_name, max_samples=max_samples)
    except Exception as e:
        return {"task": task_name, "correct": 0, "total": 0, "accuracy": 0.0, "results": [], "error": str(e)}
    # 先收集所有能成功下载的 (path, options, gt, qa)
    items: list[tuple[str, dict, str, dict]] = []
    for qa in qa_list:
        try:
            rel_path = get_image_path(qa)
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=rel_path,
                repo_type=REPO_TYPE,
                cache_dir=CACHE_DIR,
            )
        except Exception:
            continue
        options = qa.get("options") or {}
        gt = normalize_answer(qa.get("answer") or "")
        items.append((local_path, options, gt, qa))
    correct, total, results = 0, 0, []
    if batch_fn is not None and batch_size > 1:
        num_chunks = (len(items) + batch_size - 1) // batch_size
        prefetch_queue: queue.Queue = queue.Queue(maxsize=num_chunks)

        def _prefetch_worker():
            for idx in range(1, num_chunks):
                start = idx * batch_size
                chunk = items[start : start + batch_size]
                imgs = _load_chunk_images(chunk)
                prefetch_queue.put((start, imgs))
            prefetch_queue.put((None, None))

        prefetch_thread = threading.Thread(target=_prefetch_worker, daemon=True)
        prefetch_thread.start()
        for idx in range(num_chunks):
            start = idx * batch_size
            chunk = items[start : start + batch_size]
            batch_input = [(path, opts) for path, opts, _, _ in chunk]
            preloaded: list | None = None
            if idx > 0:
                _, preloaded = prefetch_queue.get()
            try:
                preds = batch_fn(batch_input, preloaded_images=preloaded) if preloaded is not None else batch_fn(batch_input)
            except Exception:
                preds = [""] * len(chunk)
            for (path, opts, gt, qa), pred in zip(chunk, preds):
                if not pred:
                    continue
                total += 1
                if pred == gt:
                    correct += 1
                results.append({"question_id": qa.get("question_id"), "gt": gt, "pred": pred})
        try:
            prefetch_queue.get(timeout=2.0)
        except queue.Empty:
            pass
        prefetch_thread.join(timeout=1.0)
    else:
        for local_path, options, gt, qa in items:
            try:
                pred = predict_fn(local_path, options)
            except (PermissionError, OSError, IOError):
                continue
            if not pred:
                continue
            total += 1
            if pred == gt:
                correct += 1
            results.append({"question_id": qa.get("question_id"), "gt": gt, "pred": pred})
    acc = (correct / total * 100) if total else 0.0
    return {"task": task_name, "correct": correct, "total": total, "accuracy": acc, "results": results}


def write_report(
    results_per_model: dict[str, list[dict]],
    report_path: str,
    device: str,
    max_samples: int | None,
    task_list: list[str],
) -> None:
    """results_per_model: model_id -> list of task stats."""
    lines = [
        "=" * 70,
        "MM-UAVBench 图像任务评测报告（多模型 × 多任务，零样本 MCQ）",
        "=" * 70,
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"设备: {device}",
        f"每任务样本数: {'全部' if max_samples is None else max_samples}",
        f"任务数: {len(task_list)}",
        f"模型数: {len(results_per_model)}",
        "",
    ]
    for model_id, all_stats in results_per_model.items():
        name = AVAILABLE_MODELS.get(model_id, model_id)
        lines.append("-" * 70)
        lines.append(f"模型: {model_id}  ({name})")
        lines.append("-" * 70)
        total_correct, total_count = 0, 0
        for s in all_stats:
            if s.get("error"):
                lines.append(f"  [{s['task']}] 加载失败: {s['error']}")
                continue
            lines.append(f"  [{s['task']}] 正确: {s['correct']}/{s['total']}  准确率: {s['accuracy']:.1f}%")
            total_correct += s["correct"]
            total_count += s["total"]
        if total_count:
            lines.append(f"  >> 总体: {total_correct}/{total_count}  平均准确率: {total_correct / total_count * 100:.1f}%")
        lines.append("")
    # 汇总表：每任务各模型准确率
    lines.append("=" * 70)
    lines.append("汇总表（各任务 × 各模型 准确率%）")
    lines.append("=" * 70)
    header = "Task".ljust(36) + "  " + "  ".join(f"{m:14s}" for m in results_per_model.keys())
    lines.append(header)
    for task_name in task_list:
        row = task_name.ljust(36)
        for model_id, all_stats in results_per_model.items():
            s = next((x for x in all_stats if x["task"] == task_name), None)
            if s and not s.get("error"):
                row += f"  {s['accuracy']:13.1f}%"
            else:
                row += "  ---"
        lines.append(row)
    row_total = "总体".ljust(36)
    for model_id, all_stats in results_per_model.items():
        tc = sum(s["correct"] for s in all_stats if not s.get("error"))
        tn = sum(s["total"] for s in all_stats if not s.get("error"))
        acc = (tc / tn * 100) if tn else 0
        row_total += f"  {acc:13.1f}%"
    lines.append(row_total)
    lines.append("")
    lines.append("=" * 70)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已保存: {report_path}")


def cmd_check_hardware() -> int:
    """打印硬件信息与建议（原 check_hardware.py 功能）。"""
    print("=" * 50)
    print("MM-UAVBench 硬件检测")
    print("=" * 50)
    print("Python:", sys.version.split()[0])
    hw = get_hardware_info()
    print("PyTorch:", torch.__version__)
    print("GPU 可用 (CUDA/ROCm):", hw["cuda_available"])
    if hw["cuda_available"]:
        print("  - 设备:", hw.get("device_name") or "N/A")
        print("  - VRAM (GB):", hw.get("vram_gb"))
    print("CPU 核心数:", hw["cpu_count"])
    print("PyTorch 当前线程数:", torch.get_num_threads())
    if hw.get("ram_gb"):
        try:
            import psutil
            r = psutil.virtual_memory()
            print("物理内存 (GB):", hw["ram_gb"], "  已用:", round(r.used / 1e9, 2), "  可用:", round(r.available / 1e9, 2))
        except Exception:
            print("物理内存 (GB):", hw["ram_gb"])
    else:
        print("安装 psutil 可查看内存: pip install psutil")
    print("=" * 50)
    print("建议:")
    if not hw["cuda_available"]:
        print("  - 当前无可用 GPU，将使用 CPU。建议仅跑轻量模型: --models clip_vitb32 siglip_base")
        print("  - 若跑 Qwen2-VL，可用 --cpu-threads 8 或减少 --max-samples")
        print("  - AMD 显卡需安装 ROCm 版 PyTorch（需 Python 3.12），运行 setup_rocm.ps1")
    else:
        vram = hw.get("vram_gb") or 0
        if vram < 8:
            print("  - VRAM < 8GB，建议 --models clip_vitb32 siglip_base，或 qwen2vl_2b 配合较小 --max-samples")
        elif vram < 16:
            print("  - VRAM 8–16GB，可跑 qwen2vl_2b；qwen2vl_7b 建议先小样本试跑")
        else:
            print("  - VRAM >= 16GB，可跑全部模型（含 qwen2vl_7b）")
    print("=" * 50)
    return 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MM-UAVBench 多模型 × 多任务评测")
    parser.add_argument("--check-hardware", action="store_true", help="仅检测硬件并退出")
    parser.add_argument("--tasks", nargs="+", default=None, help="任务名列表，默认全部")
    parser.add_argument("--models", nargs="+", default=["clip_vitb32", "siglip_base"], help="模型 id 列表，默认 clip_vitb32 siglip_base")
    parser.add_argument("--max-samples", type=int, default=20, help="每任务最多样本数，0=全部")
    parser.add_argument("--fast", action="store_true", help="加速预设：每任务 10 题、仅 CLIP+SigLIP、优化线程与后端")
    parser.add_argument("--cpu-threads", type=int, default=None, help="CPU 下 PyTorch 线程数，默认 min(16, cpu_count)")
    parser.add_argument("--report", action="store_true", default=True)
    parser.add_argument("--no-report", action="store_true", help="不生成报告")
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument("--no-monitor", action="store_true", help="不显示 GPU/内存占用与预计剩余时间")
    parser.add_argument("--batch-size", type=int, default=32, help="CLIP/SigLIP batch size (larger => higher GPU use). 0=no batching")
    args = parser.parse_args()

    if args.check_hardware:
        return cmd_check_hardware()

    # --fast 预设：少样本 + 轻量模型（不覆盖用户显式传入的 --max-samples / --models）
    argv_str = " ".join(sys.argv)
    if args.fast:
        if "--max-samples" not in argv_str:
            args.max_samples = 10
        if "--models" not in argv_str:
            args.models = ["clip_vitb32", "siglip_base"]
        if args.cpu_threads is None and "--cpu-threads" not in argv_str:
            args.cpu_threads = min(16, os.cpu_count() or 8)

    task_list = args.tasks or IMAGE_TASKS
    model_ids = args.models
    max_samples = None if args.max_samples == 0 else args.max_samples
    do_report = args.report and not args.no_report
    report_path = args.report_path or os.path.join(RESULTS_DIR, "MM-UAVBench_report.txt")

    for m in model_ids:
        if m not in AVAILABLE_MODELS:
            print(f"未知模型: {m}，可选: {list(AVAILABLE_MODELS.keys())}")
            return 1

    device = setup_hardware(args.cpu_threads)
    hw = get_hardware_info()
    show_monitor = not args.no_monitor
    print("Hardware: ", end="")
    if hw["cuda_available"]:
        print(f"GPU {hw['device_name'] or 'N/A'} (VRAM {hw['vram_gb']}GB)")
        print("  >> Device: cuda (models run on GPU)")
        print("  >> CLIP/SigLIP use little VRAM (~0.5-1GB); high RAM/CPU is from data loading")
    else:
        print(f"CPU ({hw['cpu_count']} cores, threads {torch.get_num_threads()})")
        print("  >> Device: cpu")
    if hw.get("ram_gb"):
        print(f"RAM: {hw['ram_gb']} GB")
    print("Models:", model_ids)
    print("Tasks:", len(task_list))

    total_steps = len(model_ids) * len(task_list)
    start_time = time.time()
    step_done = 0
    total_samples_done = 0

    results_per_model = {}
    for model_id in model_ids:
        print(f"\n>>> Load model: {model_id} ({AVAILABLE_MODELS[model_id]})")
        if show_monitor and torch.cuda.is_available():
            print(f"  [Res] step 0/{total_steps}  |  {get_gpu_resource_string()}", end="")
            ram_str = get_system_resource_string()
            if ram_str:
                print(f"  |  {ram_str}", end="")
            print()
        try:
            predict_fn, batch_fn = get_runner(model_id, device)
        except Exception as e:
            print(f"  Load failed: {e}")
            continue
        batch_size = args.batch_size if (batch_fn is not None and args.batch_size > 0) else 0
        if batch_size:
            print(f"  Batch inference: size={batch_size}")
        all_stats = []
        for task_name in task_list:
            print(f"  --- {task_name} ---")
            stat = run_one_task(task_name, max_samples, predict_fn, batch_fn=batch_fn, batch_size=batch_size or 1)
            all_stats.append(stat)
            if stat.get("error"):
                print(f"    Error: {stat['error']}")
            else:
                print(f"    correct: {stat['correct']}/{stat['total']}, acc: {stat['accuracy']:.1f}%")
            step_done += 1
            total_samples_done += stat.get("total") or 0
            if show_monitor:
                elapsed = time.time() - start_time
                eta_sec = (elapsed / step_done) * (total_steps - step_done) if step_done else 0
                line = f"  [Res] step {step_done}/{total_steps}  |  {get_gpu_resource_string()}"
                sys_str = get_system_resource_string()
                if sys_str:
                    line += f"  |  {sys_str}"
                line += f"  |  done {total_samples_done} q, elapsed {elapsed/60:.1f} min"
                if step_done < total_steps and eta_sec > 0:
                    line += f", ETA {eta_sec/60:.1f} min"
                print(line)
                sys.stdout.flush()
        results_per_model[model_id] = all_stats

    total_elapsed = time.time() - start_time
    print("\n=== Summary ===")
    for model_id, all_stats in results_per_model.items():
        total_ok = sum(s["correct"] for s in all_stats if not s.get("error"))
        total_n = sum(s["total"] for s in all_stats if not s.get("error"))
        acc = (total_ok / total_n * 100) if total_n else 0
        print(f"  {model_id}: {total_ok}/{total_n} = {acc:.1f}%")
    print(f"Total time: {total_elapsed/60:.1f} min ({total_samples_done} questions)")
    if show_monitor and torch.cuda.is_available():
        print(f"Final: {get_gpu_resource_string()}")

    if do_report and results_per_model:
        write_report(results_per_model, report_path, device, max_samples, task_list)
    return 0


if __name__ == "__main__":
    sys.exit(main())
