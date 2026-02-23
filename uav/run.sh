#!/bin/bash
# MM-UAVBench 完整实验：6 模型 × 16 任务 × 3 轮取平均
# 使用 GPU 4-7（2×A100 80GB + 2×L40）

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONUNBUFFERED=1

python3 run_mmuavbench_official_tasks.py \
  --models random_baseline clip_vitb32 clip_vitl14 qwen2vl_2b qwen2vl_7b qwen3vl_8b \
  --max-samples 0 \
  --batch-size 16 \
  --rounds 3
