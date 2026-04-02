import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.colmap import Dataset, Parser
from analyze_exp7_importance_pruning import (
    compute_batch_weights,
    load_json,
    load_splats,
    prepare_projection,
    resolve_scene_paths,
)

CURRENT_DIR = Path(__file__).resolve().parent
GSPATH = CURRENT_DIR.parent / "gsplat"
import sys

sys.path.append(str(GSPATH))

from cuda._wrapper import rasterize_to_indices_in_range  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze buu importance/coverage statistics.")
    parser.add_argument(
        "--repo-root",
        type=str,
        default="/data2/wd/workspace/repos/Luminance-GS/Luminance-GS",
    )
    parser.add_argument(
        "--experiment-root",
        type=str,
        default="/data2/wd/workspace/experiments/luminance_gs",
    )
    parser.add_argument(
        "--importance-csv",
        type=str,
        default="/data2/wd/workspace/experiments/luminance_gs_analysis/exp7_importance_pruning/exp7_importance_scores.csv",
    )
    parser.add_argument(
        "--analysis-root",
        type=str,
        default="/data2/wd/workspace/experiments/luminance_gs_analysis/exp7_buu_importance_analysis",
    )
    parser.add_argument("--scene", type=str, default="buu")
    parser.add_argument("--importance-batch-per-iter", type=int, default=64)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--small-threshold", type=float, default=1e-6)
    parser.add_argument("--tiny-threshold", type=float, default=0.004688415149489732)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_importance_scores(csv_path: Path, scene: str) -> np.ndarray:
    scores: Dict[int, float] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["scene"] != scene:
                continue
            scores[int(row["gaussian_idx"])] = float(row["importance_score"])
    if not scores:
        raise ValueError(f"No importance rows found for scene {scene}")
    arr = np.zeros(max(scores.keys()) + 1, dtype=np.float64)
    for idx, score in scores.items():
        arr[idx] = score
    return arr


@torch.no_grad()
def compute_coverage_counts(
    parser: Parser,
    scene_tensors: Dict[str, torch.Tensor],
    cfg: Dict,
    tile_size: int,
    batch_per_iter: int,
    device: torch.device,
) -> np.ndarray:
    trainset = Dataset(parser, split="train")
    num_gaussians = int(scene_tensors["means3d"].shape[0])
    coverage_counts = np.zeros(num_gaussians, dtype=np.int32)
    for image_idx in range(len(trainset)):
        data = trainset[image_idx]
        image = data["image"].numpy()
        height, width = image.shape[:2]
        projection = prepare_projection(
            scene_tensors,
            data["camtoworld"].to(device),
            data["K"].to(device),
            width,
            height,
            cfg,
            tile_size,
        )
        trans_flat = torch.ones(height * width, device=device, dtype=torch.float64)
        view_hits = torch.zeros(num_gaussians, device=device, dtype=torch.bool)
        step = 0
        while True:
            gs_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range(
                step,
                step + batch_per_iter,
                trans_flat.view(1, height, width).float(),
                projection["means2d"],
                projection["conics"],
                projection["opacities"],
                width,
                height,
                tile_size,
                projection["isect_offsets"],
                projection["flatten_ids"],
            )
            if gs_ids.numel() == 0:
                break
            gs_sorted, weights, (ray_unique, tail_trans) = compute_batch_weights(
                gs_ids,
                pixel_ids,
                camera_ids,
                projection["means2d"],
                projection["conics"],
                projection["opacities"],
                trans_flat,
                width,
                height,
            )
            positive = weights > 0
            if torch.any(positive):
                view_hits[torch.unique(gs_sorted[positive])] = True
            trans_flat[ray_unique] = trans_flat[ray_unique] * tail_trans
            step += batch_per_iter
        coverage_counts += view_hits.cpu().numpy().astype(np.int32)
    return coverage_counts


def plot_importance_histogram(scores: np.ndarray, small_threshold: float, output_path: Path) -> None:
    nonzero = scores[scores > 0]
    zero_count = int((scores == 0).sum())
    near_zero_count = int(((scores > 0) & (scores < small_threshold)).sum())
    other_count = int(scores.size - zero_count - near_zero_count)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(
        ["==0", f"(0,{small_threshold:g})", f">={small_threshold:g}"],
        [zero_count, near_zero_count, other_count],
        color=["#c44e52", "#dd8452", "#4c72b0"],
    )
    axes[0].set_title("Importance Buckets")
    axes[0].set_ylabel("Gaussian Count")
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].text(
        0.02,
        0.98,
        f"zero_ratio={zero_count / scores.size:.4f}\nnonzero_ratio={nonzero.size / scores.size:.4f}",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    if nonzero.size > 0:
        log_nonzero = np.log10(nonzero)
        bins = np.linspace(log_nonzero.min(), log_nonzero.max(), 60)
        axes[1].hist(log_nonzero, bins=bins, color="#4c72b0", alpha=0.9)
        axes[1].set_xlabel("log10(importance), nonzero only")
        axes[1].set_ylabel("Gaussian Count")
        axes[1].set_title("Nonzero Importance Distribution")
    else:
        axes[1].text(0.5, 0.5, "No nonzero scores", ha="center", va="center")
        axes[1].set_title("Nonzero Importance Distribution")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_coverage_histogram(coverage_counts: np.ndarray, output_path: Path) -> None:
    max_cov = int(coverage_counts.max())
    bins = np.arange(max_cov + 2) - 0.5
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(coverage_counts, bins=bins, color="#55a868", alpha=0.9, rwidth=0.9)
    ax.set_xlabel("Covered Training Views")
    ax.set_ylabel("Gaussian Count")
    ax.set_title("Per-Gaussian View Coverage")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_coverage_distribution_csv(coverage_counts: np.ndarray, output_path: Path) -> None:
    unique, counts = np.unique(coverage_counts, return_counts=True)
    total = coverage_counts.size
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["covered_views", "count", "ratio"])
        writer.writeheader()
        for cov, count in zip(unique, counts):
            writer.writerow(
                {
                    "covered_views": int(cov),
                    "count": int(count),
                    "ratio": float(count / total),
                }
            )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    experiment_root = Path(args.experiment_root)
    importance_csv = Path(args.importance_csv)
    analysis_root = Path(args.analysis_root)
    analysis_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    scene = args.scene
    tic = time.time()

    importance_scores = load_importance_scores(importance_csv, scene)
    data_dir, exp_dir, ckpt_path = resolve_scene_paths(repo_root, experiment_root, scene)
    cfg = load_json(exp_dir / "cfg.json")
    parser = Parser(str(data_dir), exp_name="over_exp", factor=1, normalize=False, test_every=8)
    splats = load_splats(ckpt_path, device)
    scales_mean = torch.exp(splats["scales"]).mean(dim=-1).cpu().numpy()

    scene_tensors = {
        "means3d": splats["means3d"],
        "quats": splats["quats"],
        "scales_exp": torch.exp(splats["scales"]),
        "opacity_sigmoid": torch.sigmoid(splats["opacities"]),
    }
    coverage_counts = compute_coverage_counts(
        parser,
        scene_tensors,
        cfg,
        args.tile_size,
        args.importance_batch_per_iter,
        device,
    )

    zero_mask = importance_scores == 0
    nonzero_mask = importance_scores > 0
    small_mask = (importance_scores > 0) & (importance_scores < args.small_threshold)
    tiny_mask = scales_mean < args.tiny_threshold
    normal_mask = ~tiny_mask

    summary = {
        "scene": scene,
        "num_gaussians": int(importance_scores.size),
        "importance_zero_count": int(zero_mask.sum()),
        "importance_zero_ratio": float(zero_mask.mean()),
        "importance_small_positive_count": int(small_mask.sum()),
        "importance_small_positive_ratio": float(small_mask.mean()),
        "nonzero_count": int(nonzero_mask.sum()),
        "nonzero_ratio": float(nonzero_mask.mean()),
        "nonzero_min": float(importance_scores[nonzero_mask].min()) if np.any(nonzero_mask) else 0.0,
        "nonzero_p1": float(np.quantile(importance_scores[nonzero_mask], 0.01)) if np.any(nonzero_mask) else 0.0,
        "nonzero_p10": float(np.quantile(importance_scores[nonzero_mask], 0.10)) if np.any(nonzero_mask) else 0.0,
        "nonzero_median": float(np.median(importance_scores[nonzero_mask])) if np.any(nonzero_mask) else 0.0,
        "nonzero_p90": float(np.quantile(importance_scores[nonzero_mask], 0.90)) if np.any(nonzero_mask) else 0.0,
        "nonzero_p99": float(np.quantile(importance_scores[nonzero_mask], 0.99)) if np.any(nonzero_mask) else 0.0,
        "nonzero_max": float(importance_scores[nonzero_mask].max()) if np.any(nonzero_mask) else 0.0,
        "covered_views_mean": float(coverage_counts.mean()),
        "covered_views_median": float(np.median(coverage_counts)),
        "covered_views_max": int(coverage_counts.max()),
        "covered_views_zero_count": int((coverage_counts == 0).sum()),
        "covered_views_zero_ratio": float((coverage_counts == 0).mean()),
        "tiny_threshold": float(args.tiny_threshold),
        "tiny_count": int(tiny_mask.sum()),
        "tiny_importance_zero_count": int((tiny_mask & zero_mask).sum()),
        "tiny_importance_zero_ratio": float((tiny_mask & zero_mask).sum() / tiny_mask.sum()) if tiny_mask.sum() else 0.0,
        "normal_count": int(normal_mask.sum()),
        "normal_importance_zero_count": int((normal_mask & zero_mask).sum()),
        "normal_importance_zero_ratio": float((normal_mask & zero_mask).sum() / normal_mask.sum()) if normal_mask.sum() else 0.0,
    }

    plot_importance_histogram(
        importance_scores,
        args.small_threshold,
        analysis_root / f"{scene}_importance_histogram.png",
    )
    plot_coverage_histogram(
        coverage_counts,
        analysis_root / f"{scene}_coverage_histogram.png",
    )
    write_coverage_distribution_csv(
        coverage_counts,
        analysis_root / f"{scene}_coverage_distribution.csv",
    )
    with (analysis_root / f"{scene}_importance_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[buu-importance-analysis] elapsed {time.time() - tic:.1f}s")


if __name__ == "__main__":
    main()
