# APEX 课题仓库（论文 [arXiv:2602.00551](https://arxiv.org/pdf/2602.00551)）

## 目录约定

| 路径 | 用途 |
|------|------|
| **本仓库根目录（`apex/`）** | 训练、评测、拓扑与 VLM 扩展的**唯一工作区**（`uav_search/`、`run_*.py`、`scripts/`、`tests/`） |
| **`reference/apex_official/`** | 上游 [github.com/4amGodvzx/apex](https://github.com/4amGodvzx/apex) 的**只读快照**，仅供对照；**不在此目录修改代码**（更新方式见 `reference/README.md`） |

## 训练

```bash
cd /path/to/本仓库/apex
python3 -u run_train_ppo_10k.py
```

完整说明与端口、GPU：**[`docs/TRAIN_OFFICIAL_GPU4.md`](docs/TRAIN_OFFICIAL_GPU4.md)** · 文档索引 **[`docs/README.md`](docs/README.md)**

- 权重：`outputs/models_official/`

## 单测

```bash
cd /path/to/本仓库/apex
python3 -m unittest discover -v tests
# 可选：pip install -r requirements-dev.txt 后使用 python3 -m pytest tests -q
# （系统自带 pytest 3.x 在 Python 3.10 上会失败，请先升级。）
```

## 其它

- `data/uavon/`：UAV-ON 数据与场景包  
- `main.py`：入口提示（训练请用 `run_train_ppo_10k.py`）
