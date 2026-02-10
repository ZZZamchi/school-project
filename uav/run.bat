@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
REM MM-UAVBench full run (Python 3.12 + ROCm GPU). Quick: py -3.12 run_mmuavbench_official_tasks.py --fast
py -3.12 run_mmuavbench_official_tasks.py --max-samples 0 --models clip_vitb32 siglip_base
pause
