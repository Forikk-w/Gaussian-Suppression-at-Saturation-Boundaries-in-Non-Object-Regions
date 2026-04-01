#!/usr/bin/env python3
"""Run pruning evaluation tasks when free GPUs are available."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--repo_examples", required=True)
    parser.add_argument("--eval_script", required=True)
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--check_interval_sec", type=int, default=30)
    parser.add_argument("--gpu_mem_free_max", type=int, default=1000)
    parser.add_argument("--gpu_util_free_max", type=int, default=10)
    parser.add_argument("--log_jsonl", required=True)
    return parser.parse_args()


def run_cmd(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True).strip()


def get_free_gpus(mem_max: int, util_max: int) -> list[str]:
    g_out = run_cmd(
        "nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used "
        "--format=csv,noheader,nounits"
    )
    a_out = run_cmd("nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader || true")

    app_uuids = {x.strip() for x in a_out.splitlines() if x.strip()}
    free = []
    for line in g_out.splitlines():
        idx, uuid, util, mem = [x.strip() for x in line.split(",")]
        if uuid in app_uuids:
            continue
        if int(mem) <= mem_max and int(util) <= util_max:
            free.append(idx)
    return free


def scene_data_dir(scene: str) -> str:
    return f"/home/wd/workspace/repos/Luminance-GS/Luminance-GS/data/LOM_full/{scene}"


def prune_tag(prune_ratio: float) -> str:
    return f"prune_{int(round(prune_ratio * 100)):02d}"


def task_result_dir(eval_root: str, exp_dir: str, ratio: float) -> str:
    return f"{eval_root}/{exp_dir}/{prune_tag(ratio)}"


def append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    os.makedirs(args.eval_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_jsonl), exist_ok=True)

    df = pd.read_csv(args.manifest_csv)
    tasks = df[df["prune_ratio"] > 0].sort_values(["scene", "mode", "prune_ratio"]).to_dict("records")

    for t in tasks:
        scene = t["scene"]
        mode = t["mode"]
        exp_dir = t["exp_dir"]
        ratio = float(t["prune_ratio"])
        ckpt = t["ckpt_out"]
        result_dir = task_result_dir(args.eval_root, exp_dir, ratio)
        stats_path = f"{result_dir}/stats/val_step9999.json"

        if os.path.exists(stats_path):
            append_jsonl(
                args.log_jsonl,
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "event": "skip_existing",
                    "scene": scene,
                    "mode": mode,
                    "exp_dir": exp_dir,
                    "prune_ratio": ratio,
                    "result_dir": result_dir,
                    "stats_path": stats_path,
                },
            )
            continue

        while True:
            free = get_free_gpus(args.gpu_mem_free_max, args.gpu_util_free_max)
            append_jsonl(
                args.log_jsonl,
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "event": "gpu_check",
                    "free_gpus": free,
                    "scene": scene,
                    "mode": mode,
                    "prune_ratio": ratio,
                },
            )
            if free:
                gpu = free[0]
                break
            time.sleep(args.check_interval_sec)

        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu} python {args.eval_script} "
            f"--repo_examples {args.repo_examples} "
            f"--data_dir {scene_data_dir(scene)} "
            f"--exp_name {mode} "
            f"--result_dir {result_dir} "
            f"--ckpt {ckpt}"
        )

        start = time.time()
        append_jsonl(
            args.log_jsonl,
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "event": "task_start",
                "gpu": gpu,
                "scene": scene,
                "mode": mode,
                "exp_dir": exp_dir,
                "prune_ratio": ratio,
                "result_dir": result_dir,
                "ckpt": ckpt,
            },
        )
        subprocess.run(cmd, shell=True, check=True)
        elapsed = time.time() - start

        stats = {}
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)

        append_jsonl(
            args.log_jsonl,
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "event": "task_done",
                "gpu": gpu,
                "scene": scene,
                "mode": mode,
                "exp_dir": exp_dir,
                "prune_ratio": ratio,
                "result_dir": result_dir,
                "stats_path": stats_path,
                "elapsed_sec": elapsed,
                "stats": stats,
            },
        )


if __name__ == "__main__":
    main()
