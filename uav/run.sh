#!/bin/bash
# MM-UAVBench 全任务：多模型 × 19 任务（16 图像 + 3 视频）× 3 轮取平均

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONUNBUFFERED=1

python3 run_mmuavbench_official_tasks.py \
  --models random_baseline clip_vitb32 clip_vitl14 qwen2vl_2b qwen2vl_7b qwen3vl_8b \
  --max-samples 0 \
  --all-tasks \
  --batch-size 16 \
  --rounds 3
