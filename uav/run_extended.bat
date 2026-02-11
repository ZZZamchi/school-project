@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
REM Quick run: 10 samples per task, clip_vitb32 + siglip_base
py -3.12 run_mmuavbench_official_tasks.py --fast
pause
