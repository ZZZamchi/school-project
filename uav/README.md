# uav · MM-UAVBench 图像任务

基于 [daisq/MM-UAVBench](https://huggingface.co/datasets/daisq/MM-UAVBench) 数据，本地跑 16 个图像任务的零样本 MCQ，多模型输出 txt 报告。支持随机基线、CLIP、Qwen2-VL 等，AMD 显卡用 ROCm。

## Results (4 models)

- **设备**：cuda（AMD RX 7700 XT）
- **题目**：16 任务、全量 4911 题
- **模型与总体准确率**：

| Model | Overall accuracy |
|-------|------------------|
| random_baseline | 24.8% |
| clip_vitb32 | 27.6% |
| clip_vitl14 | 27.8% |
| qwen2vl_2b | **35.4%** |

完整每任务数字见 `results/MM-UAVBench_report.txt`。任务说明见 `docs/任务与方法说明.md`。

## How to run

```bash
pip install -r requirements.txt
# AMD GPU 需先运行 setup_rocm.ps1（Python 3.12）
```

| 操作 | 命令 |
|------|------|
| **Full run (4 models)** | 双击 `run.bat` 或：`py -3.12 run_mmuavbench_official_tasks.py --max-samples 0 --models random_baseline clip_vitb32 clip_vitl14 qwen2vl_2b` |
| **Quick run** | `py -3.12 run_mmuavbench_official_tasks.py --fast` |

数据会自动下载到 `data/mm_uavbench_cache/`。

## Upload to GitHub

仓库：[ZZZamchi/school-project](https://github.com/ZZZamchi/school-project)（uav 放在仓库内的 `uav` 文件夹）。

**方式一**：在 **uav 目录内**双击 `upload.bat`，会自动在上一级克隆（若无）并同步到 `school-project/uav` 后执行 git add / commit / push。

**方式二**：在**项目上一级目录**手动执行：

```bash
git clone https://github.com/ZZZamchi/school-project.git
cd school-project
mkdir uav 2>nul
xcopy /E /I /Y ..\uav\* uav\
git add uav
git commit -m "uav: MM-UAVBench 4-model results"
git push -u origin main
```

`data/` 已在 .gitignore 中，不会提交；`results/MM-UAVBench_report.txt` 会一并推送。
