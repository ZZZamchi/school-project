# MM-UAVBench 图像任务评测

论文 [arXiv:2512.23219](https://arxiv.org/abs/2512.23219)，数据 [daisq/MM-UAVBench](https://huggingface.co/datasets/daisq/MM-UAVBench)。

16 个图像任务，零样本 MCQ，支持多模型。

## 运行

```bash
pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=6,7   # 可选
./run.sh
```

或直接：

```bash
python3 run_mmuavbench_official_tasks.py --models qwen2vl_7b qwen3vl_8b --max-samples 0
```

`--fast` 快速验证；`--check-hardware` 检测硬件。

## 模型

| id | 显存 |
|----|------|
| random_baseline, clip_vitb32, clip_vitl14, siglip_base | 小 |
| qwen2vl_2b | ~5GB |
| qwen2vl_7b, qwen3vl_8b | ~14–18GB |

## 输出

`results/MM-UAVBench_report.txt`，汇总表最佳用 * 标记。新实验会合并进现有报告。

## 任务

16 图像任务：场景分类、朝向、环境、OCR、计数、回溯、跨物体推理、意图、属性、损毁、分析预测、时序、地面规划、空地协同、集群协同。详见 `docs/任务与方法说明.md`。
