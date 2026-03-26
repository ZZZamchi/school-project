#!/usr/bin/env python3
"""在 ``AirSimDroneEnv`` 上评测：``ppo`` 或 ``apex_vl``（Qwen3-VL + 结构化拓扑 facts；无纯视觉启发式策略）。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

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
if "OFFICIAL_APEX_GRAPHICS_ADAPTER" not in os.environ:
    os.environ["OFFICIAL_APEX_GRAPHICS_ADAPTER"] = "0"

import numpy as np

from uav_search.train_code.uav_env_multi import AirSimDroneEnv
from uav_search.topo_text_map.builder import TopoTextMapBuilder
from uav_search.topo_text_map.topo_nav_policy import (
    ACTION_NAMES_CN as TOPO_ACTION_NAMES_CN,
    flight_telemetry_pose_only,
)
from uav_search.topo_text_map.apex_vl_nav import apex_vl_topo_nav_decide
from uav_search.topo_text_map.ego_map_context import ego_map_context_for_policy


def _episode_success_from_info(info: dict) -> bool:
    if not info:
        return False
    rs = float(info.get("ep_reward_sparse", 0.0))
    return rs > 0.0


def _get_xyz(client) -> list[float]:
    st = client.getMultirotorState()
    p = st.kinematics_estimated.position
    return [float(p.x_val), float(p.y_val), float(p.z_val)]


def _path_length_meters(xyz: list[list[float]]) -> float:
    """世界坐标系下轨迹折线长度（米）。"""
    arr = np.asarray(xyz, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    d = np.diff(arr, axis=0)
    return float(np.sum(np.linalg.norm(d, axis=1)))


def _spl_episode(success: bool, path_len_m: float, shortest_m: float) -> float:
    """SPL：成功时 shortest / max(path, shortest)；失败为 0。最短路径用起点—目标欧氏距离近似。"""
    if not success or shortest_m <= 1e-6:
        return 0.0
    return float(shortest_m / max(path_len_m, shortest_m))


def _capture_rgb(client) -> np.ndarray:
    from uav_search.airsim_utils import get_images

    pil_img, _, _, _, _, _ = get_images(client)
    return np.asarray(pil_img.convert("RGB"))


def _save_rgb_frame(client, path: str) -> None:
    import cv2

    rgb = _capture_rgb(client)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _plot_traj_xy(
    path_png: str,
    xyz: list[list[float]],
    goal_xyz: np.ndarray | None = None,
    start_xyz: list[float] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(xyz, dtype=np.float64)
    plt.figure(figsize=(8, 8))
    plt.plot(arr[:, 0], arr[:, 1], "b-", linewidth=1.5, label="path")
    if goal_xyz is not None:
        plt.scatter([goal_xyz[0]], [goal_xyz[1]], c="red", s=80, marker="x", label="ref")
    if start_xyz is not None:
        plt.scatter([start_xyz[0]], [start_xyz[1]], c="green", s=60, marker="o", label="start")
    plt.xlabel("x (world)")
    plt.ylabel("y (world)")
    plt.axis("equal")
    plt.legend(loc="best")
    plt.title("Trajectory — XY")
    plt.tight_layout()
    plt.savefig(path_png, dpi=140)
    plt.close()


def _plot_traj_xz(
    path_png: str,
    xyz: list[list[float]],
    goal_xyz: np.ndarray | None = None,
    start_xyz: list[float] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(xyz, dtype=np.float64)
    plt.figure(figsize=(9, 5))
    plt.plot(arr[:, 0], arr[:, 2], "b-", linewidth=1.5, label="path")
    if goal_xyz is not None:
        plt.scatter([goal_xyz[0]], [goal_xyz[2]], c="red", s=80, marker="x", label="ref")
    if start_xyz is not None:
        plt.scatter([start_xyz[0]], [start_xyz[2]], c="green", s=60, marker="o", label="start")
    plt.xlabel("x (world)")
    plt.ylabel("z (world)")
    plt.legend(loc="best")
    plt.title("Trajectory — XZ")
    plt.tight_layout()
    plt.savefig(path_png, dpi=140)
    plt.close()


def _plot_traj_yz(
    path_png: str,
    xyz: list[list[float]],
    goal_xyz: np.ndarray | None = None,
    start_xyz: list[float] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(xyz, dtype=np.float64)
    plt.figure(figsize=(9, 5))
    plt.plot(arr[:, 1], arr[:, 2], "b-", linewidth=1.5, label="path")
    if goal_xyz is not None:
        plt.scatter([goal_xyz[1]], [goal_xyz[2]], c="red", s=80, marker="x", label="ref")
    if start_xyz is not None:
        plt.scatter([start_xyz[1]], [start_xyz[2]], c="green", s=60, marker="o", label="start")
    plt.xlabel("y (world)")
    plt.ylabel("z (world)")
    plt.legend(loc="best")
    plt.title("Trajectory — YZ")
    plt.tight_layout()
    plt.savefig(path_png, dpi=140)
    plt.close()


def _plot_traj_3d(
    path_png: str,
    xyz: list[list[float]],
    goal_xyz: np.ndarray | None = None,
    start_xyz: list[float] | None = None,
    subtitle: str = "",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    import matplotlib.pyplot as plt

    arr = np.asarray(xyz, dtype=np.float64)
    pts_list = [arr]
    if start_xyz is not None:
        pts_list.insert(0, np.asarray(start_xyz, dtype=np.float64).reshape(1, 3))
    if goal_xyz is not None:
        pts_list.append(goal_xyz.reshape(1, 3))
    pts = np.vstack(pts_list)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if len(arr) >= 2:
        segs = np.stack([arr[:-1], arr[1:]], axis=1)
        zmid = 0.5 * (segs[:, 0, 2] + segs[:, 1, 2])
        zmin, zmax = float(arr[:, 2].min()), float(arr[:, 2].max())
        if zmax - zmin < 1e-6:
            zmin, zmax = zmin - 1.0, zmax + 1.0
        norm = plt.Normalize(zmin, zmax)
        colors = plt.cm.coolwarm(norm(zmid))
        lc = Line3DCollection(segs, colors=colors, linewidths=2.2)
        ax.add_collection3d(lc)
    elif len(arr) == 1:
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], c="steelblue", s=40, label="path")

    if start_xyz is not None:
        ax.scatter(
            [start_xyz[0]],
            [start_xyz[1]],
            [start_xyz[2]],
            c="limegreen",
            s=120,
            marker="o",
            edgecolors="darkgreen",
            linewidths=1.0,
            label="start",
            zorder=10,
        )
    if len(arr) > 0:
        ax.scatter(
            [arr[-1, 0]],
            [arr[-1, 1]],
            [arr[-1, 2]],
            c="dodgerblue",
            s=90,
            marker="s",
            label="end",
            zorder=9,
        )
    if goal_xyz is not None:
        ax.scatter(
            [goal_xyz[0]],
            [goal_xyz[1]],
            [goal_xyz[2]],
            c="red",
            s=140,
            marker="x",
            linewidths=2.5,
            label="ref",
            zorder=11,
        )

    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = (mx - mn).max()
    if span < 1e-3:
        span = 10.0
    center = 0.5 * (mn + mx)
    half = 0.55 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    ax.set_xlabel("x (world)")
    ax.set_ylabel("y (world)")
    ax.set_zlabel("z (world)")
    _title = "Trajectory — 3D (world)"
    if subtitle:
        _title = f"{_title}\n{subtitle}"
    ax.set_title(_title, fontsize=11)
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()


def _subsample_indices(idx: np.ndarray, max_pts: int, seed: int) -> np.ndarray:
    if len(idx) <= max_pts:
        return idx
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(idx), max_pts, replace=False)
    return idx[pick]


def _plot_obstacle_map_3d(path_png: str, obstacle_map: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol = np.asarray(obstacle_map, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"obstacle_map 期望 (40,40,10)，得到 {vol.shape}")
    idx = np.argwhere(vol > 0.5)
    idx = _subsample_indices(idx, 14_000, seed=41)
    if len(idx) == 0:
        idx = np.array([[20, 20, 5]], dtype=np.int64)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        idx[:, 0],
        idx[:, 1],
        idx[:, 2],
        c="dimgray",
        s=8,
        alpha=0.35,
        depthshade=True,
    )
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_zlabel("grid z")
    ax.set_title(title)
    ax.set_xlim(0, vol.shape[0] - 1)
    ax.set_ylim(0, vol.shape[1] - 1)
    ax.set_zlim(0, vol.shape[2] - 1)
    try:
        ax.set_box_aspect((1, 1, float(vol.shape[2]) / float(vol.shape[0])))
    except Exception:
        pass
    ax.view_init(elev=22, azim=-55)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()


def _plot_attraction_map_3d(path_png: str, attraction_ch0: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol = np.asarray(attraction_ch0, dtype=np.float64)
    idx = np.argwhere(np.isfinite(vol) & (vol > 1e-6))
    if len(idx) == 0:
        idx = np.array([[20, 20, 5]], dtype=np.int64)
        vals = np.array([0.0])
    else:
        vals = vol[idx[:, 0], idx[:, 1], idx[:, 2]]
        if len(idx) > 14_000:
            order = np.argsort(-vals)[:14_000]
            idx = idx[order]
            vals = vals[order]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        idx[:, 0],
        idx[:, 1],
        idx[:, 2],
        c=vals,
        cmap="hot",
        s=10,
        alpha=0.55,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, shrink=0.55, label="attraction ch0")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_zlabel("grid z")
    ax.set_title(title)
    ax.set_xlim(0, vol.shape[0] - 1)
    ax.set_ylim(0, vol.shape[1] - 1)
    ax.set_zlim(0, vol.shape[2] - 1)
    try:
        ax.set_box_aspect((1, 1, float(vol.shape[2]) / float(vol.shape[0])))
    except Exception:
        pass
    ax.view_init(elev=22, azim=-55)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()


def _task_string_for_topo(env: AirSimDroneEnv) -> str:
    t = env.task_data[env.task_id]
    name = t.get("object_name", "").strip()
    desc = (t.get("description") or "").strip()
    if name and desc:
        return f"{name}。{desc}"
    return name or desc or "task"


def _task_meta_for_id(env: AirSimDroneEnv, task_id: int) -> dict:
    t = env.task_data[task_id]
    return {
        "map": t.get("map"),
        "object_name": t.get("object_name"),
        "task_description": t.get("description", ""),
        "difficulty": t.get("difficulty"),
    }


def _save_topo_json(path: str, builder: TopoTextMapBuilder, meta: dict) -> None:
    payload = {
        **meta,
        "topo_text_map": builder.to_json_dict(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _default_checkpoint_path() -> str:
    mdir = os.path.join(PROJECT, "outputs", "models_official")
    candidates = [
        os.path.join(mdir, "official_ppo_full_400k_final.zip"),
        os.path.join(mdir, "official_ppo_resume_20000_final.zip"),
        os.path.join(mdir, "official_ppo_resume_10000_final.zip"),
        os.path.join(mdir, "official_ppo_20000_final.zip"),
        os.path.join(mdir, "official_ppo_10000_final.zip"),
        os.path.join(mdir, "official_ppo_10k_final.zip"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return candidates[-1]


def main() -> None:
    os.environ.setdefault("OFFICIAL_APEX_DISABLE_TASK_ROTATION", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--n_episodes", type=int, default=None)
    ap.add_argument(
        "--task_ids",
        type=str,
        default="4,5,6,7",
        help="rl_tasks.json 中的 task_id（逗号分隔）",
    )
    ap.add_argument("--episodes_per_task", type=int, default=5)
    ap.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(PROJECT, "outputs", "test_official"),
    )
    ap.add_argument("--no_save_viz", action="store_true")
    ap.add_argument("--no_traj_2d", action="store_true")
    ap.add_argument("--no_traj_plots", action="store_true")
    ap.add_argument(
        "--no_topo",
        action="store_true",
        help="不保存 topo_maps/（apex_vl 仍维护拓扑，仅不写盘）",
    )
    ap.add_argument("--topo_grid", type=float, default=15.0)
    ap.add_argument(
        "--policy",
        type=str,
        choices=("ppo", "apex_vl"),
        default="apex_vl",
    )
    ap.add_argument(
        "--vl_max_new_tokens",
        type=int,
        default=128,
        help="apex_vl：Qwen3-VL 生成上限",
    )
    ap.add_argument(
        "--vl_temperature",
        type=float,
        default=0.2,
        help="apex_vl：采样温度；0 为贪心",
    )
    ap.add_argument(
        "--apex_vl_model",
        type=str,
        default=None,
        help="apex_vl：模型目录或 HF id（默认读 OFFICIAL_APEX_QWEN3_VL_DIR 或 Qwen/Qwen3-VL-8B-Instruct）",
    )
    ap.add_argument(
        "--oob_grid_margin",
        type=float,
        default=2.5,
        help="apex_vl：栅格安全边距（越小越晚触发边界避障，更多步可走 VLM 决策）",
    )
    args = ap.parse_args()

    if args.n_episodes is not None:
        task_id_list = [int(os.environ.get("OFFICIAL_APEX_TASK_ID", "4"))]
        episodes_per_task = args.n_episodes
    else:
        task_id_list = [int(x.strip()) for x in args.task_ids.split(",") if x.strip()]
        episodes_per_task = args.episodes_per_task

    multi_task = len(task_id_list) > 1
    save_viz = not args.no_save_viz
    os.makedirs(args.save_dir, exist_ok=True)

    ckpt: str | None = None
    model = None
    device = "cpu"

    if args.policy == "ppo":
        ckpt_arg = args.checkpoint
        if ckpt_arg is None:
            ckpt_arg = os.environ.get("OFFICIAL_APEX_TEST_CHECKPOINT")
        if ckpt_arg is None:
            ckpt_arg = _default_checkpoint_path()
        ckpt = ckpt_arg
        if not os.path.isabs(ckpt):
            if ckpt.startswith("outputs/"):
                ckpt = os.path.join(PROJECT, ckpt)
            else:
                ckpt = os.path.join(PROJECT, ckpt)
        if not os.path.isfile(ckpt):
            raise SystemExit(f"checkpoint 不存在: {ckpt}")
        import torch
        from stable_baselines3 import PPO

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PPO.load(ckpt, device=device)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    print("[APEX test] policy:", args.policy, flush=True)
    if ckpt is not None:
        print("[APEX test] checkpoint:", ckpt, flush=True)
        print("[APEX test] device:", device, flush=True)
    print(
        "[APEX test] task_ids:",
        task_id_list,
        "episodes_per_task:",
        episodes_per_task,
        "save_viz:",
        save_viz,
        "no_traj_plots:",
        args.no_traj_plots,
        "oob_grid_margin:",
        args.oob_grid_margin,
        flush=True,
    )

    env = AirSimDroneEnv(worker_index=0)
    rows = []
    t0 = time.time()
    global_idx = 0

    for tid in task_id_list:
        env.task_id = tid
        env.episode_id = 0
        root = os.path.join(args.save_dir, f"task_{tid}") if multi_task else args.save_dir
        traj_dir = os.path.join(root, "trajectories")
        frames_dir = os.path.join(root, "step_frames")
        topo_dir: str | None = None
        use_topo_policy = args.policy == "apex_vl"
        save_topo = save_viz and use_topo_policy and not args.no_topo
        if save_viz:
            os.makedirs(traj_dir, exist_ok=True)
            os.makedirs(frames_dir, exist_ok=True)
            if save_topo:
                topo_dir = os.path.join(root, "topo_maps")
                os.makedirs(topo_dir, exist_ok=True)

        for ep in range(episodes_per_task):
            obs, _ = env.reset()
            client = env.client
            if client is None:
                raise RuntimeError("env.client 为空")

            traj_xyz: list[list[float]] = []
            start_xyz = _get_xyz(client)
            traj_xyz.append(start_xyz.copy())
            goal_xyz = np.asarray(env.target_position, dtype=np.float64).reshape(3).copy()

            topo_builder: TopoTextMapBuilder | None = None
            if use_topo_policy:
                topo_builder = TopoTextMapBuilder(
                    task=_task_string_for_topo(env),
                    grid_size=args.topo_grid,
                    seed=tid * 1000 + ep + 17,
                )
                if topo_dir is not None:
                    _save_topo_json(
                        os.path.join(topo_dir, f"ep{ep}_reset_topo.json"),
                        topo_builder,
                        {
                            "task_id": tid,
                            "episode": ep,
                            "step_index": 0,
                            "phase": "reset",
                            "xyz": start_xyz,
                        },
                    )

            if save_viz:
                _save_rgb_frame(client, os.path.join(frames_dir, f"ep{ep}_reset_rgb.png"))

            ep_reward = 0.0
            steps = 0
            done = False
            last_info: dict = {}
            step_decisions_log: list[dict] = []
            while not done:
                if model is None:
                    env._update_uav_pose_from_airsim()
                map_ctx = None
                if args.policy == "apex_vl":
                    map_ctx = ego_map_context_for_policy(
                        env.attraction_map,
                        env.exploration_map,
                        env.obstacle_map,
                        env.uav_pose,
                    )
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                    action_int = int(np.asarray(action).reshape(-1)[0])
                    decision_meta: dict = {
                        "action_name_cn": TOPO_ACTION_NAMES_CN.get(action_int, str(action_int)),
                        "policy": "ppo",
                    }
                elif args.policy == "apex_vl":
                    if topo_builder is None:
                        raise RuntimeError("apex_vl 需要 topo_builder")
                    rgb_live = _capture_rgb(client)
                    action_int, decision = apex_vl_topo_nav_decide(
                        client,
                        topo_builder,
                        rgb_live,
                        grid_position=np.asarray(env.uav_pose["position"]),
                        task_text=_task_string_for_topo(env),
                        grid_margin=args.oob_grid_margin,
                        max_new_tokens=args.vl_max_new_tokens,
                        temperature=args.vl_temperature,
                        model_id_or_path=args.apex_vl_model,
                        map_context=map_ctx,
                    )
                    action = np.array([action_int], dtype=np.int64)
                    decision_meta = {
                        "policy": "apex_vl",
                        "action_id": action_int,
                        "action_name_cn": TOPO_ACTION_NAMES_CN.get(action_int, str(action_int)),
                        "decision": decision,
                    }
                else:
                    raise RuntimeError(f"未处理的 policy: {args.policy}")
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                steps += 1
                last_info = info if isinstance(info, dict) else {}
                done = bool(terminated or truncated)

                pos = _get_xyz(client)
                traj_xyz.append(pos)
                if topo_builder is not None:
                    topo_builder.step(pos[0], pos[1], pos[2], action_int)
                if topo_dir is not None and topo_builder is not None:
                    if model is not None:
                        fl = flight_telemetry_pose_only(client)
                    else:
                        fl = decision_meta.get("decision", {}).get("flight", {})
                    step_decisions_log.append(
                        {
                            "step": steps,
                            **decision_meta,
                            "flight_direction": fl,
                            "xyz_after_step": pos,
                        }
                    )
                    _save_topo_json(
                        os.path.join(topo_dir, f"ep{ep}_step{steps:04d}_topo.json"),
                        topo_builder,
                        {
                            "task_id": tid,
                            "episode": ep,
                            "step_index": steps,
                            "xyz": pos,
                            "action_id": action_int,
                            "action_name_cn": TOPO_ACTION_NAMES_CN.get(action_int, str(action_int)),
                            "flight_direction": fl,
                            "decision_detail": decision_meta,
                        },
                    )
                if save_viz:
                    _save_rgb_frame(client, os.path.join(frames_dir, f"ep{ep}_step{steps:04d}_rgb.png"))

            success = _episode_success_from_info(last_info)
            path_len_m = _path_length_meters(traj_xyz)
            start_np = np.asarray(start_xyz, dtype=np.float64).reshape(3)
            goal_np = np.asarray(goal_xyz, dtype=np.float64).reshape(3)
            shortest_m = float(np.linalg.norm(goal_np - start_np))
            spl = _spl_episode(success, path_len_m, shortest_m)

            if topo_dir is not None and step_decisions_log:
                sdp = os.path.join(topo_dir, f"ep{ep}_step_decisions.json")
                with open(sdp, "w", encoding="utf-8") as sf:
                    json.dump(
                        {
                            "task_id": tid,
                            "episode": ep,
                            "policy": args.policy,
                            "steps": step_decisions_log,
                        },
                        sf,
                        ensure_ascii=False,
                        indent=2,
                    )
                print("[APEX test] step log:", sdp, flush=True)

            if save_viz:
                jpath = os.path.join(traj_dir, f"ep{ep}_path.json")
                _tm = _task_meta_for_id(env, tid)
                path_obj: dict = {
                    "task_id": tid,
                    "episode": ep,
                    "object_name": _tm["object_name"],
                    "task_description": _tm["task_description"],
                    "map": _tm["map"],
                    "difficulty": _tm["difficulty"],
                    "start_xyz": start_xyz,
                    "goal_xyz": goal_xyz.reshape(-1).tolist(),
                    "trajectory_xyz": traj_xyz,
                    "path_length_m": round(path_len_m, 4),
                    "shortest_path_m": round(shortest_m, 4),
                    "spl": round(spl, 6),
                    "oob_grid_margin": args.oob_grid_margin,
                    "steps": steps,
                    "success": success,
                    "policy": args.policy,
                }
                with open(jpath, "w", encoding="utf-8") as jf:
                    json.dump(path_obj, jf, indent=2, ensure_ascii=False)
                np.savez_compressed(
                    os.path.join(traj_dir, f"ep{ep}_path.npz"),
                    xyz=np.asarray(traj_xyz, dtype=np.float64),
                )
                if not args.no_traj_plots:
                    try:
                        base = os.path.join(traj_dir, f"ep{ep}")
                        if not args.no_traj_2d:
                            _plot_traj_xy(f"{base}_traj_xy.png", traj_xyz, None, start_xyz=start_xyz)
                            _plot_traj_xz(f"{base}_traj_xz.png", traj_xyz, None, start_xyz=start_xyz)
                            _plot_traj_yz(f"{base}_traj_yz.png", traj_xyz, None, start_xyz=start_xyz)
                        sub3d = (
                            f"path={path_len_m:.1f}m | straight={shortest_m:.1f}m | SPL={spl:.3f} | ok={success}"
                        )
                        _plot_traj_3d(
                            f"{base}_traj_3d.png",
                            traj_xyz,
                            goal_np,
                            start_xyz=start_xyz,
                            subtitle=sub3d,
                        )
                        _plot_obstacle_map_3d(
                            f"{base}_obstacle_map_3d.png",
                            env.obstacle_map,
                            f"task {tid} ep{ep} — Obstacle (grid)",
                        )
                        _plot_attraction_map_3d(
                            f"{base}_attraction_map_3d.png",
                            env.attraction_map[:, :, :, 0],
                            f"task {tid} ep{ep} — Attraction ch0 (grid)",
                        )
                        np.savez_compressed(
                            f"{base}_maps.npz",
                            obstacle=env.obstacle_map,
                            attraction_ch0=env.attraction_map[:, :, :, 0],
                        )
                    except Exception as e:
                        print(f"[APEX test] plot skip task{tid} ep{ep}: {e}", flush=True)

            _row_meta = _task_meta_for_id(env, tid)
            rows.append(
                {
                    "global_index": global_idx,
                    "task_id": tid,
                    "episode": ep,
                    **_row_meta,
                    "steps": steps,
                    "return": round(ep_reward, 4),
                    "success": success,
                    "path_length_m": round(path_len_m, 4),
                    "shortest_path_m": round(shortest_m, 4),
                    "spl": round(spl, 6),
                    "info_keys": list(last_info.keys()) if last_info else [],
                    "ep_reward_sparse": last_info.get("ep_reward_sparse"),
                    "ep_step": last_info.get("ep_step"),
                }
            )
            global_idx += 1
            print(
                f"[APEX test] task {tid} ep {ep} steps={steps} success={success} SPL={spl:.3f} "
                f"path_len={path_len_m:.1f}m return={ep_reward:.2f}",
                flush=True,
            )

    env.close()
    sr = float(np.mean([1.0 if r["success"] else 0.0 for r in rows])) if rows else 0.0
    spls = [float(r["spl"]) for r in rows]
    mean_spl = float(np.mean(spls)) if spls else 0.0
    plens = [float(r["path_length_m"]) for r in rows]
    mean_path_m = float(np.mean(plens)) if plens else 0.0
    out = {
        "policy": args.policy,
        "checkpoint": ckpt,
        "task_ids": task_id_list,
        "tasks": {str(tid): _task_meta_for_id(env, tid) for tid in task_id_list},
        "episodes_per_task": episodes_per_task,
        "total_episodes": len(rows),
        "wall_time_sec": round(time.time() - t0, 2),
        "SR": sr,
        "mean_SPL": round(mean_spl, 6),
        "mean_path_length_m": round(mean_path_m, 4),
        "oob_grid_margin": args.oob_grid_margin,
        "map_fusion_hints": {
            "APEX_FUSE_W_EXPLORATION": os.environ.get("APEX_FUSE_W_EXPLORATION", "0.35"),
            "APEX_FUSE_W_ATTRACTION": os.environ.get("APEX_FUSE_W_ATTRACTION", "0.75"),
            "APEX_FUSE_W_OBSTACLE": os.environ.get("APEX_FUSE_W_OBSTACLE", "0.6"),
        },
        "save_viz": save_viz,
        "save_dir": args.save_dir,
        "multi_task_subdirs": multi_task,
        "traj_2d_projections": save_viz and not args.no_traj_2d and not args.no_traj_plots,
        "traj_3d": save_viz and not args.no_traj_plots,
        "no_traj_plots": bool(args.no_traj_plots),
        "episodes": rows,
    }
    out_path = os.path.join(args.save_dir, "test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(
        "[APEX test] SR =",
        sr,
        "mean_SPL =",
        mean_spl,
        "| results:",
        out_path,
        flush=True,
    )
    if save_viz:
        print("[APEX test] out:", args.save_dir, "| task_* subdirs:", multi_task, flush=True)


if __name__ == "__main__":
    main()
