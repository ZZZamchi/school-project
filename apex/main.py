#!/usr/bin/env python3
"""入口提示：训练请在本目录执行 `python3 -u run_train_ppo_10k.py`，见 docs/TRAIN_OFFICIAL_GPU4.md"""
import sys

if __name__ == "__main__":
    print(__doc__)
    print("\n训练: python3 -u run_train_ppo_10k.py\n")
    print("单测: python3 -m unittest discover -v tests\n")
    sys.exit(2)
