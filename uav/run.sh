#!/bin/bash
# MM-UAVBench 评测，每模型 3 轮取平均
# 使用 4 个空闲 L40: 0,1,2,6

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0,1,2,6
export PYTHONUNBUFFERED=1

python3 run_mmuavbench_official_tasks.py \
  --models random_baseline clip_vitb32 clip_vitl14 qwen2vl_2b qwen2vl_7b qwen3vl_8b \
  --max-samples 0 \
  --batch-size 16 \
  --rounds 3
