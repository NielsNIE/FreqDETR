#!/usr/bin/env python3
"""Print trainer CLI help without importing heavy training dependencies.

Usage: python scripts/print_trainer_help.py
"""
import argparse
import os

def print_help():
    ap = argparse.ArgumentParser(description="FreqDETR training entrypoint (help only)")
    ap.add_argument("--config", type=str, required=True,
                    help="Path to the model config YAML (e.g. configs/model/freqdetr_base.yaml)")
    ap.add_argument("--data", type=str, default="configs/dataset/corn_local.yaml",
                    help="Path to dataset/data YAML describing dataset root and annotation paths")
    ap.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)),
                    help="Local rank for distributed training. Set automatically by torchrun.")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (optional)")
    ap.print_help()

if __name__ == "__main__":
    print_help()
