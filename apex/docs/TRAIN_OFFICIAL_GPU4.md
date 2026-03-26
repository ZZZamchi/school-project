# 官方 APEX 环境 + PPO 训练（物理 GPU 4 / 双卡 4+5）

> 文档索引：**[README.md](README.md)**

## 硬件与环境（本机）

### GPU 编号：nvidia-smi 与 UE `GraphicsAdapter`（易混点）

| 变量 | 含义 |
|------|------|
| **`CUDA_VISIBLE_DEVICES`** | **物理 GPU 编号**，与 **`nvidia-smi` 左侧 GPU 序号**一致（如 `4` 表示第 4 号卡）。训练/推理前用此选对卡；建议同时设 **`CUDA_DEVICE_ORDER=PCI_BUS_ID`**（脚本已 `setdefault`）。 |
| **`OFFICIAL_APEX_GRAPHICS_ADAPTER`** | **仅当 Python 自拉起 AirSim/UE** 时传给 `-GraphicsAdapter=`；是 **UE 进程内「可见 GPU」的序号**，**不是** nvidia-smi 物理编号。与 `scripts/start_airsim_single_scene.sh` 一致：先 `CUDA_VISIBLE_DEVICES=<物理号>`，再 **`-GraphicsAdapter=0`**。默认 **`0`**；**不要**把它设成与 `CUDA_VISIBLE_DEVICES` 相同的数字（如 4），否则与官方 shell 启动逻辑不一致。 |
| **`SKIP_LAUNCH=1`（默认）** | 已手动开 AirSim 时 **不读 GraphicsAdapter 起进程**，只需端口/关卡一致；此时以 **`CUDA_VISIBLE_DEVICES`** 让 PyTorch 与 AirSim **同一张物理卡** 即可。 |

- **GPU（短训 / 测试）**：`CUDA_VISIBLE_DEVICES=4`（`run_train_ppo_10k.py`、`run_test_official_ppo.py` 默认）。
- **GPU（完整复现）**：`CUDA_VISIBLE_DEVICES=4,5`（`run_train_official_full.py` 默认）。SB3 PPO 仍只用 **第一张可见卡**（逻辑 `cuda:0` = 物理 **GPU4**）；**GPU5** 可用于同机其它进程或后续扩展，**不是** DataParallel 自动双卡训练。
- **AirSim**：已启动且 **Api 端口与 `OFFICIAL_APEX_BASE_PORT` 一致**（默认 **41460**）。
- **关卡**：`OFFICIAL_APEX_SKIP_LAUNCH=1` 时**不**再拉起 UE，假定当前关卡与 `rl_tasks.json` 中任务一致（默认 `OFFICIAL_APEX_TASK_ID=4`，CabinLake Boat）。

## 训练 10k 并保存权重

```bash
cd /path/to/apex
python3 -u run_train_ppo_10k.py
```

日志：自行重定向（如 `outputs/train_official_20k.log`）  
权重目录：`outputs/models_official/`

- checkpoint / 最终权重前缀：`official_ppo_{总步数}_`（例如 10000 → `official_ppo_10000_final.zip`，20k → `official_ppo_20000_final.zip`）
- 旧文件 `official_ppo_10k_final.zip` 仍为兼容名；测试脚本会按存在性优先选择权重。

### 从已有 10k 权重续训（再追加 20k 步，10k 处 checkpoint）

前提：``outputs/models_official/official_ppo_10k_final.zip`` 存在，且 **task / AirSim** 与当初训练一致。

```bash
cd /path/to/apex
CUDA_VISIBLE_DEVICES=4 bash scripts/run_official_continue_from_10k_nohup.sh
```

- 默认：本段 **追加 20000** 步；**每 10000 步** 存 checkpoint（约在 **+10k** 与 **+20k** 各一盘）。
- 产出：``official_ppo_resume_20000_final.zip``（及 ``official_ppo_resume_20000_*_steps.zip``）。
- 环境变量：``OFFICIAL_APEX_RESUME``、``OFFICIAL_APEX_TOTAL_STEPS``、``OFFICIAL_APEX_CHECKPOINT_FREQ`` 见 ``run_train_ppo_10k.py``。

### 20k 训练 + 退出 SSH（nohup，从零开始）

```bash
cd /path/to/apex
bash scripts/run_official_train_20k_nohup.sh
# tail -f outputs/train_official_20k.log
```

结束后评测（每步 RGB + `traj_xy`/`traj_xz`/`traj_yz`/`traj_3d`）：

```bash
bash scripts/run_official_test_20k.sh
```

可调环境变量：

| 变量 | 默认 | 含义 |
|------|------|------|
| `OFFICIAL_APEX_TOTAL_STEPS` | 10000 | 总环境步数（`20000` 即 20k） |
| `OFFICIAL_APEX_CHECKPOINT_FREQ` | 与总步数相同 | 存盘间隔；20k 训可设 `10000` 以在 10k/20k 各存一次 |
| `OFFICIAL_APEX_N_STEPS` | 1024 | PPO 每轮 rollout 长度 |
| `OFFICIAL_APEX_BATCH_SIZE` | 64 | mini-batch（会自动与 n_steps 对齐因子） |
| `OFFICIAL_APEX_BASE_PORT` | 41460 | AirSim 端口 |
| `OFFICIAL_APEX_TASK_ID` | 4 | `uav_search/task_map/rl_tasks.json` 中的 task |
| `OFFICIAL_APEX_SKIP_LAUNCH` | 1 | 1=连接已有仿真；0=按官方脚本启动 UE |
| `OFFICIAL_APEX_ENV_ROOT` | 自动指向 `data/uavon/envs/TRAIN_ENVS` | 仅 `SKIP_LAUNCH=0` 时用 |

TensorBoard：`outputs/tb_official/`

**墙钟粗估（10k）**：每步约 2～5 s，**10 000 步** ≈ **5.5～14 小时**；PPO 更新开销相对仿真可忽略。**400k** 量级见下文「完整复现」。

---

## 完整复现（论文 Supplement 两阶段课程 + 约 400k 步，GPU 4+5）

与官方快照 `uav_search/models/f_ppo_num_4_final_400000.zip` 命名对齐：**阶段1（默认 200k）** 关闭吸引力权重，**阶段2（默认 200k）** 恢复全奖励；合计 **400k**。

### 依赖与配置清单

| 项 | 说明 |
|----|------|
| Python | 3.10+（与项目一致） |
| PyTorch + CUDA | 与驱动匹配；训练/测试脚本在可见 GPU 上用 `cuda` |
| `stable-baselines3`、`gymnasium`、`airsim`、`opencv-python`、`matplotlib` | 评测保存轨迹图需要 matplotlib |
| **AirSim** | 已启动，**Api 端口** = `OFFICIAL_APEX_BASE_PORT`（默认 **41460**） |
| **关卡** | `OFFICIAL_APEX_SKIP_LAUNCH=1` 时 UE 已打开且关卡与 `rl_tasks.json` 中 `OFFICIAL_APEX_TASK_ID` 任务一致 |
| **数据** | `data/uavon/envs/TRAIN_ENVS` 与 `uav_search/task_map/rl_tasks.json`（环境内已用绝对路径） |

### 训练（完整）

```bash
cd /path/to/apex
# 可选：export CUDA_VISIBLE_DEVICES=4,5   # 脚本默认已是 4,5
python3 -u run_train_official_full.py
```

- **权重输出**：`outputs/models_official/official_ppo_full_stage1_final.zip`（阶段末）、`official_ppo_full_400k_final.zip`（最终）
- **中间 checkpoint**：`official_ppo_full_stage1_*_steps.zip`、`official_ppo_full_stage2_*_steps.zip`
- **TensorBoard**：`outputs/tb_official_full/`

**环境变量（常用）**：

| 变量 | 默认 | 含义 |
|------|------|------|
| `OFFICIAL_APEX_TOTAL_STEPS_STAGE1` | 200000 | 阶段1 步数（W_ATTRACTION=0） |
| `OFFICIAL_APEX_TOTAL_STEPS_STAGE2` | 200000 | 阶段2 步数（W_ATTRACTION=1） |
| `OFFICIAL_APEX_SKIP_CURRICULUM` | 0 | 设为 `1` 则**单阶段**全奖励，总步数见 `OFFICIAL_APEX_TOTAL_STEPS`（默认 STAGE1+STAGE2） |
| `OFFICIAL_APEX_N_STEPS` | 2048 | PPO rollout 长度 |
| `OFFICIAL_APEX_BATCH_SIZE` | 64 | mini-batch |
| `OFFICIAL_APEX_CHECKPOINT_FREQ` | 50000 | checkpoint 间隔 |
| `OFFICIAL_APEX_W_*` | 见 `uav_env_multi._compute_reward` | 可调稀疏/探索等权重（阶段1 脚本会设 `W_ATTRACTION=0`） |

**一键脚本（仓库根目录）**：

```bash
bash scripts/run_official_full_repro_gpu45.sh
# 只训练：TRAIN_ONLY=1 bash scripts/run_official_full_repro_gpu45.sh
# 只测试：TEST_ONLY=1 bash scripts/run_official_full_repro_gpu45.sh
```

### 时间粗估（400k 步 × 真实 AirSim）

在 **2～5 s/步** 量级下，**400k 步** 约 **9～23 天** 墙钟（量级估算，以本机为准）。可用 `OFFICIAL_APEX_TOTAL_STEPS_*` 缩小做联调。

### 测试（完整权重）

`run_test_official_ppo.py` **默认 checkpoint**：若存在 `outputs/models_official/official_ppo_full_400k_final.zip` 则优先加载，否则 `official_ppo_10k_final.zip`。也可用：

```bash
export OFFICIAL_APEX_TEST_CHECKPOINT=/path/to/model.zip
python3 -u run_test_official_ppo.py --save_dir outputs/test_official_full
```

### 与官方 `multiprocess_s.py` 的差异（必读）

当前完整训练为 **SB3 PPO + 单环境 `AirSimDroneEnv`**，与仓库内 **Grounding DINO + SAM + VecNormalize** 的 `multiprocess_s.py` **推理栈不同**；若要复现该脚本行为，需单独对接权重 `vec_normalize_*.pkl` 与检测模型路径。见论文与 Supplement。

## 官方环境改动说明（本仓库内）

已对 `uav_search/train_code/uav_env_multi.py` 增加：

- 任务 JSON / `task_*.txt` **绝对路径**（不依赖 cwd）
- `OFFICIAL_APEX_SKIP_LAUNCH` 连接已有 AirSim
- `OFFICIAL_APEX_BASE_PORT`、`OFFICIAL_APEX_TASK_ID`、`OFFICIAL_APEX_ENV_ROOT`、`OFFICIAL_APEX_GRAPHICS_ADAPTER`
- `_compute_reward` 中稀疏/探索/吸引力等权重可通过 **`OFFICIAL_APEX_W_ATTRACTION`** 等环境变量覆盖（供两阶段课程）
- **`OFFICIAL_APEX_GRAPHICS_ADAPTER`** 默认 **`0`**（UE 内序号）；物理 GPU 用 **`CUDA_VISIBLE_DEVICES`** 与 nvidia-smi 对齐（见上文表格）

与论文 [arXiv:2602.00551](https://arxiv.org/pdf/2602.00551) 中 **RGB-D 反投影地图 + 6 动作异步飞控** 一致；**勿**与已废弃的 `_deprecated_custom_repro/` 权重混用。

## 测试（加载 10k 权重）

```bash
cd /path/to/apex
PYTHONUNBUFFERED=1 python3 -u run_test_official_ppo.py \
  --checkpoint outputs/models_official/official_ppo_10k_final.zip \
  --n_episodes 10 \
  --save_dir outputs/test_official
```

- **`test_results.json`**：每局步数、回报、`success`、整体 **SR**  
- **默认另存**（加 `--no_save_viz` 可关）  
  - `test_official/step_frames/`：`ep{N}_reset_rgb.png`、`ep{N}_step{XXXX}_rgb.png`  
  - `test_official/trajectories/`：`ep{N}_path.json` / `ep{N}_path.npz`（**完整 xyz**）；图：`traj_xy` 俯视、`traj_xz` 侧视高度、`traj_3d` 三维

**说明**：若之前只跑了 **3 局**，是因为命令行写了 `--n_episodes 3`；默认现为 **10 局**。
