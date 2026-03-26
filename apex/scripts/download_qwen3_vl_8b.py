#!/usr/bin/env python3
"""下载 Qwen3-VL-8B-Instruct 到本地目录（供 OFFICIAL_APEX_QWEN3_VL_DIR 使用）。"""
from __future__ import annotations

import argparse
import os

APEX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = APEX_ROOT

DEFAULT_REPO = "Qwen/Qwen3-VL-8B-Instruct"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--local-dir",
        default=os.path.join(PROJECT, "models", "Qwen3-VL-8B-Instruct"),
        help="本地保存目录（默认 apex/models/Qwen3-VL-8B-Instruct）",
    )
    ap.add_argument("--repo", default=DEFAULT_REPO)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    os.makedirs(args.local_dir, exist_ok=True)
    print(f"[download] repo={args.repo} -> {args.local_dir}")
    snapshot_download(
        repo_id=args.repo,
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("[download] done. export OFFICIAL_APEX_QWEN3_VL_DIR=" + args.local_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
