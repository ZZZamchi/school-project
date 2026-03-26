#!/bin/bash
# 检测 AirSim 是否在指定端口可连（用于排查 41460 未就绪问题）
# 用法: ./scripts/check_airsim.sh [port]
set -e
cd "$(dirname "$0")/.."
PORT="${1:-41460}"
echo "检测 127.0.0.1:$PORT ..."
if command -v nc >/dev/null 2>&1; then
  if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then
    echo "端口 $PORT 可连接。"
    exit 0
  fi
fi
if command -v timeout >/dev/null 2>&1; then
  if timeout 2 bash -c "echo >/dev/tcp/127.0.0.1/$PORT" 2>/dev/null; then
    echo "端口 $PORT 可连接。"
    exit 0
  fi
fi
echo "端口 $PORT 无法连接。请先运行: ./scripts/start_airsim_single_scene.sh"
exit 1
