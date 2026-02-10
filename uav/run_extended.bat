@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
REM Extended: 3 models (CLIP-B + SigLIP + CLIP-L), full questions. ~12GB VRAM.
REM Qwen2-VL-2B: run separately with --models qwen2vl_2b --max-samples 0
py -3.12 run_mmuavbench_official_tasks.py --max-samples 0 --models clip_vitb32 siglip_base clip_vitl14
pause
