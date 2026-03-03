# MM-UAVBench 图像+视频任务评测

论文 [arXiv:2512.23219](https://arxiv.org/abs/2512.23219)，数据 [daisq/MM-UAVBench](https://huggingface.co/datasets/daisq/MM-UAVBench)。

16 图像 + 3 视频任务（Event_Prediction / Event_Tracing / Event_Understanding），零样本 MCQ。

## 运行

```bash
pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=0,1,2,6   # 可选
./run.sh
```

仅图像（默认）：
```bash
python3 run_mmuavbench_official_tasks.py --models qwen2vl_7b qwen3vl_8b --max-samples 0
```

**含视频任务**（需支持多帧的模型，如 qwen2vl_7b / qwen3vl_8b）：
```bash
python3 run_mmuavbench_official_tasks.py --models qwen2vl_7b qwen3vl_8b --max-samples 0 --all-tasks
```

`--fast` 快速验证；`--check-hardware` 检测硬件；`--rounds N` 每模型 N 轮取平均。

## 模型

| id | 显存 |
|----|------|
| random_baseline, clip_vitb32, clip_vitl14, siglip_base | 小 |
| qwen2vl_2b | ~5GB |
| qwen2vl_7b, qwen3vl_8b | ~14–18GB |

论文中其他模型（如 InternVL、LLaVA、MiniCPM）可参考 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 或官方 [MM-UAVBench](https://github.com/MM-UAVBench/MM-UAVBench) 接入。

## 输出

`results/MM-UAVBench_report.txt`，汇总表最佳用 * 标记。新实验会合并进现有报告。

## 任务

16 图像 + 3 视频（Event_*）。视频题会从 mp4 均匀采样 8 帧，多帧输入仅 Qwen2-VL/Qwen3-VL 支持。详见 `docs/任务与方法说明.md`。
