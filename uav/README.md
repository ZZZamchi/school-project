# uav · MM-UAVBench 图像任务

用 HuggingFace 上的 [daisq/MM-UAVBench](https://huggingface.co/datasets/daisq/MM-UAVBench) 数据，本地跑 16 个图像任务的零样本 MCQ，多模型出 txt 报告。支持 CLIP、SigLIP、Qwen2-VL，AMD 显卡用 ROCm。

## 已有实验结果

- **设备**：cuda（AMD RX 7700 XT）
- **题目**：16 任务、每任务全量，共 4911 题
- **已跑模型**：
  - clip_vitb32：总体 1357/4911 ≈ **27.6%**
  - clip_vitl14：总体 1364/4911 ≈ **27.8%**
- 环境/场景类（Environment_State_Classification）两个模型都在 67.8%；场景分类、场景属性等略好，朝向/指代计数/回溯等偏低。完整数字见 `results/MM-UAVBench_report.txt`。

## 剩余工作

- 补跑 siglip_base，和 CLIP 一起写进同一份报告。
- 可选：接视频任务、bbox 裁剪、或加 Qwen2-VL / 其他 VLM 做对比。

## 怎么跑

```bash
pip install -r requirements.txt
# AMD GPU 需先跑 setup_rocm.ps1（Python 3.12）

# 两模型全量
run.bat
# 或：py -3.12 run_mmuavbench_official_tasks.py --max-samples 0 --models clip_vitb32 siglip_base

# 三模型（含 clip_vitl14）
run_extended.bat
```

数据会自动下到 `data/mm_uavbench_cache/`。任务说明见 `docs/任务与方法说明.md`。

## 推到 GitHub（uav 文件夹）

仓库：<https://github.com/ZZZamchi/school-project>

在**项目上一级目录**执行（把 `PythonProject` 换成你本机路径）：

```bash
git clone https://github.com/ZZZamchi/school-project.git
cd school-project
mkdir uav
xcopy /E /I /Y ..\uav\* uav\
# 上面会拷到 uav 下，data 已在 .gitignore 里不会进仓库
git add uav
git commit -m "uav: MM-UAVBench 图像任务复现"
git push -u origin main
```

若分支是 `master` 就把 `main` 改成 `master`。
