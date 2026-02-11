@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
REM 4 models, full run (random_baseline + CLIP-B + CLIP-L + Qwen2-VL-2B)
py -3.12 run_mmuavbench_official_tasks.py --max-samples 0 --models random_baseline clip_vitb32 clip_vitl14 qwen2vl_2b
pause
