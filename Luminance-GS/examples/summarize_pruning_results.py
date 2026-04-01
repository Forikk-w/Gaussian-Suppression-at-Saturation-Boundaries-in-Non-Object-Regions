#!/usr/bin/env python3
"""Summarize pruning metrics and draw prune-ratio vs PSNR curves."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--exp_root", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def prune_tag(r: float) -> str:
    return f"prune_{int(round(r * 100)):02d}"


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(f"{args.out_dir}/figures", exist_ok=True)
    os.makedirs(f"{args.out_dir}/tables", exist_ok=True)

    df = pd.read_csv(args.manifest_csv).sort_values(["scene", "mode", "prune_ratio"]).reset_index(drop=True)

    rows = []
    missing = []
    for _, r in df.iterrows():
        scene = r["scene"]
        mode = r["mode"]
        exp_dir = r["exp_dir"]
        prune_ratio = float(r["prune_ratio"])
        remain = int(r["remaining_count"])

        if prune_ratio == 0:
            stats_path = f"{args.exp_root}/{exp_dir}/stats/val_step9999.json"
        else:
            stats_path = f"{args.eval_root}/{exp_dir}/{prune_tag(prune_ratio)}/stats/val_step9999.json"

        if not os.path.exists(stats_path):
            missing.append(
                {
                    "scene": scene,
                    "mode": mode,
                    "exp_dir": exp_dir,
                    "prune_ratio": prune_ratio,
                    "expected_stats_path": stats_path,
                }
            )
            continue

        with open(stats_path, "r") as f:
            s = json.load(f)

        rows.append(
            {
                "scene": scene,
                "mode": mode,
                "exp_dir": exp_dir,
                "prune_ratio": prune_ratio,
                "remaining_count_manifest": remain,
                "num_GS_metric": int(s["num_GS"]),
                "psnr": float(s["psnr"]),
                "ssim": float(s["ssim"]),
                "lpips": float(s["lpips"]),
                "ellipse_time": float(s["ellipse_time"]),
                "stats_path": stats_path,
            }
        )

    result = pd.DataFrame(rows).sort_values(["scene", "mode", "prune_ratio"]).reset_index(drop=True)
    result.to_csv(f"{args.out_dir}/tables/pruning_metrics.csv", index=False)
    pd.DataFrame(missing).to_csv(f"{args.out_dir}/tables/missing_tasks.csv", index=False)

    # Curve: prune ratio vs PSNR (one line per scene+mode)
    fig = plt.figure(figsize=(9.2, 5.2), dpi=220)
    for (scene, mode), g in result.groupby(["scene", "mode"]):
        g = g.sort_values("prune_ratio")
        plt.plot(
            g["prune_ratio"],
            g["psnr"],
            marker="o",
            linewidth=1.8,
            label=f"{scene}-{mode}",
        )
    plt.xlabel("Pruning Ratio")
    plt.ylabel("PSNR")
    plt.title("Pruning Ratio vs PSNR")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    fig.savefig(f"{args.out_dir}/figures/pruning_ratio_vs_psnr.png")
    plt.close(fig)

    # Scene-level averaged curve (low / over_exp)
    fig = plt.figure(figsize=(8.2, 4.8), dpi=220)
    for mode, g in result.groupby("mode"):
        c = g.groupby("prune_ratio")["psnr"].mean().reset_index()
        plt.plot(c["prune_ratio"], c["psnr"], marker="o", linewidth=2.0, label=f"{mode} mean")
    plt.xlabel("Pruning Ratio")
    plt.ylabel("Mean PSNR (bike/buu/sofa)")
    plt.title("Mean Pruning Curve by Mode")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(f"{args.out_dir}/figures/pruning_ratio_vs_psnr_mode_mean.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
