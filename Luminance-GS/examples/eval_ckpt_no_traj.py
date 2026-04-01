#!/usr/bin/env python3
"""Evaluate a checkpoint with Luminance-GS metrics without trajectory rendering."""

from __future__ import annotations

import argparse
import os
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_examples", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--exp_name", required=True, choices=["low", "over_exp", "variance"])
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    # simple_trainer_ours parses CLI at import time via tyro; isolate argv first.
    orig_argv = sys.argv[:]
    sys.path.insert(0, args.repo_examples)
    sys.argv = [orig_argv[0]]
    import simple_trainer_ours as trainer  # type: ignore

    sys.argv = orig_argv

    cfg = trainer.Config(
        data_dir=args.data_dir,
        exp_name=args.exp_name,
        result_dir=args.result_dir,
        disable_viewer=True,
        ckpt=args.ckpt,
    )
    cfg.adjust_steps(cfg.steps_scaler)

    runner = trainer.Runner(cfg)
    ckpt = torch.load(args.ckpt, map_location=runner.device)
    for k in runner.splats.keys():
        runner.splats[k].data = ckpt["splats"][k]

    runner.eval(step=ckpt["step"])


if __name__ == "__main__":
    main()
