#!/usr/bin/env python3
"""
官方 ``AirSimDroneEnv`` + Stable-Baselines3 PPO，默认可训练 10k 步（由环境变量改总步数）。

权重与 VecNormalize（若启用）写入项目根 ``outputs/models_official/``，
文件名前缀为 ``official_ppo_{总步数}_``（例如 20k → ``official_ppo_20000_final.zip``）。

环境变量（可选）：
  OFFICIAL_APEX_RESUME=           若设为 ``outputs/models_official/xxx.zip`` 的相对项目根路径或绝对路径，则从该权重 **续训**（见下）。
  OFFICIAL_APEX_TOTAL_STEPS=10000 **未设 RESUME**：总环境步数。**已设 RESUME**：本段 **追加** 的环境步数（例如 10k 权重再训 10000 → 累计约 20k）。
  OFFICIAL_APEX_RUN_PREFIX=       续训时 checkpoint/最终文件名前缀（默认 ``official_ppo_resume_{追加步数}``）
  OFFICIAL_APEX_CHECKPOINT_FREQ     存 checkpoint 间隔（默认=本段步数；续训时可设 5000）
  OFFICIAL_APEX_SKIP_LAUNCH=1   连接已运行的 AirSim（默认开）
  OFFICIAL_APEX_BASE_PORT=41460
  OFFICIAL_APEX_TASK_ID=4       rl_tasks.json 中任务 id（4=CabinLake Boat）
  OFFICIAL_APEX_GRAPHICS_ADAPTER=0  仅当 Python 自拉起 UE 时用 UE 内适配器序号（通常为 0，勿填 nvidia-smi 物理编号）

用法（物理 GPU 4）::
  cd <本仓库 apex 根目录>
  CUDA_VISIBLE_DEVICES=4 python3 run_train_ppo_10k.py

20k + 断 SSH（nohup）见项目根 ``scripts/run_official_train_20k_nohup.sh``。
"""
from __future__ import annotations

import os
import sys

# 在 import torch 之前：物理 GPU 用 nvidia-smi 上显示的编号（与 start_airsim 第二个参数一致）
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

APEX_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = APEX_ROOT
os.chdir(APEX_ROOT)
if APEX_ROOT not in sys.path:
    sys.path.insert(0, APEX_ROOT)

os.environ.setdefault("OFFICIAL_APEX_SKIP_LAUNCH", "1")
os.environ.setdefault("OFFICIAL_APEX_BASE_PORT", "41460")
os.environ.setdefault("OFFICIAL_APEX_TASK_ID", "4")
os.environ["OFFICIAL_APEX_ENV_ROOT"] = os.path.join(PROJECT, "data", "uavon", "envs", "TRAIN_ENVS")
# UE 渲染适配器：仅在 Python 自拉起 AirSim 时使用；与 shell 启动脚本一致应为「可见 GPU 内序号」通常为 0，
# 切勿写成 nvidia-smi 物理编号（易与 -GraphicsAdapter=4 等混淆）。
if "OFFICIAL_APEX_GRAPHICS_ADAPTER" not in os.environ:
    os.environ["OFFICIAL_APEX_GRAPHICS_ADAPTER"] = "0"

SAVE_DIR = os.path.join(PROJECT, "outputs", "models_official")
TB_DIR = os.path.join(PROJECT, "outputs", "tb_official")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from uav_search.train_code.uav_env_multi import AirSimDroneEnv

def _resolve_resume_path(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    if os.path.isfile(raw):
        return raw
    cand = os.path.join(PROJECT, raw.lstrip("/"))
    if os.path.isfile(cand):
        return cand
    cand2 = os.path.join(SAVE_DIR, os.path.basename(raw))
    if os.path.isfile(cand2):
        return cand2
    raise FileNotFoundError(f"OFFICIAL_APEX_RESUME 找不到权重: {raw}")


RESUME_PATH = _resolve_resume_path(os.environ.get("OFFICIAL_APEX_RESUME", ""))
# 续训：TOTAL_STEPS = 本段追加步数；全新：TOTAL_STEPS = 总步数
TOTAL_STEPS = int(os.environ.get("OFFICIAL_APEX_TOTAL_STEPS", "10000"))
# 须 <= 单轮 rollout；若总步数很小则自动缩小，避免 SB3 永远凑不齐第一批 n_steps
_N_STEPS = int(os.environ.get("OFFICIAL_APEX_N_STEPS", "1024"))
if _N_STEPS > TOTAL_STEPS:
    _N_STEPS = max(64, min(512, TOTAL_STEPS))
_BATCH = int(os.environ.get("OFFICIAL_APEX_BATCH_SIZE", "64"))
_BATCH = max(8, min(_BATCH, _N_STEPS))
while _N_STEPS % _BATCH != 0 and _BATCH > 8:
    _BATCH -= 1

# 与 TOTAL_STEPS 一致时只在训练结束存盘；设更小值可在长训中多次 checkpoint
_CKPT_FREQ = int(os.environ.get("OFFICIAL_APEX_CHECKPOINT_FREQ", str(max(TOTAL_STEPS, 1))))
_CKPT_FREQ = max(1, _CKPT_FREQ)
if RESUME_PATH:
    _RUN_PREFIX = os.environ.get("OFFICIAL_APEX_RUN_PREFIX", f"official_ppo_resume_{TOTAL_STEPS}")
else:
    _RUN_PREFIX = f"official_ppo_{TOTAL_STEPS}"


def main() -> None:
    print("[APEX] PROJECT_ROOT", PROJECT)
    print("[APEX] APEX_ROOT", APEX_ROOT)
    print("[APEX] SAVE_DIR", SAVE_DIR)
    if RESUME_PATH:
        print("[APEX] RESUME from", RESUME_PATH)
        print("[APEX] this segment timesteps (追加)", TOTAL_STEPS, "(n_steps/batch 以权重内为准，必要时与 env 对齐)")
    else:
        print("[APEX] TOTAL_STEPS", TOTAL_STEPS, "n_steps", _N_STEPS, "batch_size", _BATCH)
    print("[APEX] checkpoint every", _CKPT_FREQ, "steps | save prefix", _RUN_PREFIX)
    print("[APEX] CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if torch.cuda.is_available():
        print("[APEX] cuda:0 ->", torch.cuda.get_device_name(0))
    env = AirSimDroneEnv(worker_index=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cb = CheckpointCallback(
        save_freq=_CKPT_FREQ,
        save_path=SAVE_DIR,
        name_prefix=_RUN_PREFIX,
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    if RESUME_PATH:
        model = PPO.load(RESUME_PATH, env=env, device=device)
        model.tensorboard_log = TB_DIR
        # 若当前脚本算出的 n_steps 与存档一致可保持；SB3 已加载 policy 超参
        print("[APEX] loaded n_steps=", model.n_steps, "batch_size=", model.batch_size)
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=_N_STEPS,
            batch_size=_BATCH,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
            device=device,
            tensorboard_log=TB_DIR,
        )
    model.learn(total_timesteps=TOTAL_STEPS, callback=cb)
    final_path = os.path.join(SAVE_DIR, f"{_RUN_PREFIX}_final")
    model.save(final_path)
    print("[APEX] saved final:", final_path)
    env.close()


if __name__ == "__main__":
    main()
