#!/usr/bin/env python3
"""
检查本机 AirSim RPC 是否可达（不启动 UE，只连 MultirotorClient）。

默认 ``OFFICIAL_APEX_BASE_PORT`` 未设置时为 **41451**（与 ``AirSimDroneEnv`` 一致）。
``run_test_official_ppo.py`` 内会 ``setdefault(..., "41460")``，评测前请与该脚本一致或显式 export。

用法::
  cd <本仓库 apex 根目录>
  export OFFICIAL_APEX_BASE_PORT=41460   # 与评测脚本一致时
  python3 scripts/check_airsim.py

退出码: 0 连接成功；1 失败。
"""
from __future__ import annotations

import os
import sys

OFFICIAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if OFFICIAL not in sys.path:
    sys.path.insert(0, OFFICIAL)


def main() -> int:
    try:
        import airsim
    except ImportError:
        print("[check_airsim] 未安装 airsim: pip install airsim", file=sys.stderr)
        return 1

    base = int(os.environ.get("OFFICIAL_APEX_BASE_PORT", "41451"))
    worker = int(os.environ.get("OFFICIAL_APEX_WORKER_INDEX", "0"))
    port = base + worker
    print(f"[check_airsim] 尝试连接 127.0.0.1:{port} (OFFICIAL_APEX_BASE_PORT={base}, worker={worker})", flush=True)

    client = airsim.MultirotorClient(port=port)
    try:
        client.confirmConnection()
    except Exception as e:
        print(f"[check_airsim] 失败: {e}", file=sys.stderr)
        print(
            "[check_airsim] 请确认：1) UE/AirSim 已运行且 ApiServerPort 与端口一致；"
            "2) 或设置 OFFICIAL_APEX_SKIP_LAUNCH=1 连接已手动启动的实例。",
            file=sys.stderr,
        )
        return 1

    st = client.getMultirotorState()
    p = st.kinematics_estimated.position
    print(f"[check_airsim] OK — position: x={p.x_val:.2f} y={p.y_val:.2f} z={p.z_val:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
