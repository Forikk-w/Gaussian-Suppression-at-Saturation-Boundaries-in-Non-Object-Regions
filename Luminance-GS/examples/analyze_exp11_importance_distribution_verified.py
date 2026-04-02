import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
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
from run_exp9_pruning_curve_verified import verify_baseline_and_collect

CURRENT_DIR = Path(__file__).resolve().parent
GSPATH = CURRENT_DIR.parent / "gsplat"
import sys

sys.path.append(str(GSPATH))

from cuda._wrapper import rasterize_to_indices_in_range  # noqa: E402


SCENES = ["bike", "buu", "chair", "sofa"]
TINY_THRESHOLD = 0.004688
BASELINE_METRIC_TOL = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute verified rendering-importance distributions on over-exposure training views."
    )
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
        "--analysis-root",
        type=str,
        default="/data2/wd/workspace/experiments/luminance_gs_analysis/exp11_importance_distribution_verified",
    )
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--importance-batch-per-iter", type=int, default=64)
    parser.add_argument("--tiny-threshold", type=float, default=TINY_THRESHOLD)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


@torch.no_grad()
def accumulate_importance_and_coverage_for_view(
    importance: torch.Tensor,
    coverage_counts: np.ndarray,
    projection: Dict[str, torch.Tensor],
    tile_size: int,
    batch_per_iter: int,
    width: int,
    height: int,
) -> None:
    trans_flat = torch.ones(height * width, device=importance.device, dtype=torch.float64)
    view_hits = torch.zeros(importance.shape[0], device=importance.device, dtype=torch.bool)
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
        importance.index_add_(0, gs_sorted, weights)
        positive = weights > 0
        if torch.any(positive):
            view_hits[torch.unique(gs_sorted[positive])] = True
        trans_flat[ray_unique] = trans_flat[ray_unique] * tail_trans
        step += batch_per_iter

    coverage_counts += view_hits.cpu().numpy().astype(np.int32)


def summarize_scene(
    scene: str,
    importance_scores: np.ndarray,
    coverage_counts: np.ndarray,
    scale_values: np.ndarray,
    tiny_threshold: float,
    baseline_verification: Dict[str, object],
    num_train_views: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    num_gaussians = int(importance_scores.size)
    zero_mask = importance_scores == 0.0
    nonzero_mask = ~zero_mask
    nonzero_scores = importance_scores[nonzero_mask]

    tiny_mask = scale_values < tiny_threshold
    normal_mask = ~tiny_mask
    tiny_zero_ratio = float(zero_mask[tiny_mask].mean()) if tiny_mask.any() else 0.0
    normal_zero_ratio = float(zero_mask[normal_mask].mean()) if normal_mask.any() else 0.0

    if nonzero_scores.size > 0:
        quantiles = {
            "min": float(nonzero_scores.min()),
            "p1": float(np.percentile(nonzero_scores, 1)),
            "p10": float(np.percentile(nonzero_scores, 10)),
            "median": float(np.percentile(nonzero_scores, 50)),
            "p90": float(np.percentile(nonzero_scores, 90)),
            "p99": float(np.percentile(nonzero_scores, 99)),
            "max": float(nonzero_scores.max()),
        }
    else:
        quantiles = {
            "min": 0.0,
            "p1": 0.0,
            "p10": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

    cov_unique, cov_counts = np.unique(coverage_counts, return_counts=True)
    cov_distribution = {
        str(int(cov)): {
            "count": int(count),
            "ratio": float(count / num_gaussians),
        }
        for cov, count in zip(cov_unique, cov_counts)
    }

    summary_json = {
        "scene": scene,
        "num_gaussians": num_gaussians,
        "num_train_views": int(num_train_views),
        "tiny_threshold": float(tiny_threshold),
        "baseline_verification": baseline_verification,
        "importance_zero_count": int(zero_mask.sum()),
        "importance_zero_ratio": float(zero_mask.mean()),
        "nonzero_count": int(nonzero_mask.sum()),
        "nonzero_ratio": float(nonzero_mask.mean()),
        "nonzero_quantiles": quantiles,
        "covered_views_zero_count": int((coverage_counts == 0).sum()),
        "covered_views_zero_ratio": float((coverage_counts == 0).mean()),
        "covered_views_mean": float(coverage_counts.mean()),
        "covered_views_max": int(coverage_counts.max()),
        "covered_views_distribution": cov_distribution,
        "tiny_gaussian_count": int(tiny_mask.sum()),
        "normal_gaussian_count": int(normal_mask.sum()),
        "tiny_importance_zero_ratio": tiny_zero_ratio,
        "normal_importance_zero_ratio": normal_zero_ratio,
    }

    csv_row = {
        "scene": scene,
        "importance_zero_count": int(zero_mask.sum()),
        "importance_zero_ratio": float(zero_mask.mean()),
        "nonzero_count": int(nonzero_mask.sum()),
        "nonzero_ratio": float(nonzero_mask.mean()),
        "nonzero_p10": quantiles["p10"],
        "nonzero_median": quantiles["median"],
        "nonzero_p90": quantiles["p90"],
        "covered_views_zero_count": int((coverage_counts == 0).sum()),
        "covered_views_zero_ratio": float((coverage_counts == 0).mean()),
        "covered_views_max": int(coverage_counts.max()),
        "tiny_importance_zero_ratio": tiny_zero_ratio,
        "normal_importance_zero_ratio": normal_zero_ratio,
    }
    return summary_json, csv_row


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "scene",
        "importance_zero_count",
        "importance_zero_ratio",
        "nonzero_count",
        "nonzero_ratio",
        "nonzero_p10",
        "nonzero_median",
        "nonzero_p90",
        "covered_views_zero_count",
        "covered_views_zero_ratio",
        "covered_views_max",
        "tiny_importance_zero_ratio",
        "normal_importance_zero_ratio",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    experiment_root = Path(args.experiment_root)
    analysis_root = Path(args.analysis_root)
    baseline_root = analysis_root / "baseline_images"
    analysis_root.mkdir(parents=True, exist_ok=True)
    baseline_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    all_rows: List[Dict[str, object]] = []
    meta = {
        "scenes": args.scenes,
        "tiny_threshold": args.tiny_threshold,
        "tile_size": args.tile_size,
        "importance_batch_per_iter": args.importance_batch_per_iter,
        "baseline_metric_tol": BASELINE_METRIC_TOL,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for scene in args.scenes:
        tic = time.time()
        data_dir, exp_dir, ckpt_path = resolve_scene_paths(repo_root, experiment_root, scene)
        cfg = load_json(exp_dir / "cfg.json")
        parser = Parser(
            str(data_dir),
            exp_name="over_exp",
            factor=int(cfg.get("data_factor", 1)),
            normalize=True,
            test_every=int(cfg.get("test_every", 8)),
        )
        valset = Dataset(parser, split="val")
        trainset = Dataset(parser, split="train")
        splats = load_splats(ckpt_path, device)

        baseline_images, _, verification = verify_baseline_and_collect(
            scene,
            parser,
            valset,
            splats,
            cfg,
            exp_dir,
            device,
        )
        imageio.imwrite(baseline_root / f"{scene}_val00_baseline.png", baseline_images[0])

        num_gaussians = int(splats["means3d"].shape[0])
        importance = torch.zeros(num_gaussians, device=device, dtype=torch.float64)
        coverage_counts = np.zeros(num_gaussians, dtype=np.int32)
        scale_values = torch.exp(splats["scales"]).mean(dim=-1).detach().cpu().numpy()

        scene_tensors = {
            "means3d": splats["means3d"],
            "quats": splats["quats"],
            "scales_exp": torch.exp(splats["scales"]),
            "opacity_sigmoid": torch.sigmoid(splats["opacities"]),
        }

        for image_idx in range(len(trainset)):
            data = trainset[image_idx]
            height, width = data["image"].shape[:2]
            projection = prepare_projection(
                scene_tensors,
                data["camtoworld"].to(device),
                data["K"].to(device),
                width,
                height,
                cfg,
                args.tile_size,
            )
            accumulate_importance_and_coverage_for_view(
                importance,
                coverage_counts,
                projection,
                args.tile_size,
                args.importance_batch_per_iter,
                width,
                height,
            )

        importance_scores = importance.detach().cpu().numpy()
        summary_json, csv_row = summarize_scene(
            scene,
            importance_scores,
            coverage_counts,
            scale_values,
            args.tiny_threshold,
            verification,
            len(trainset),
        )
        with (analysis_root / f"importance_summary_{scene}.json").open("w") as f:
            json.dump(summary_json, f, indent=2)
        all_rows.append(csv_row)

        print(
            f"[exp11] {scene}: baseline_ok=True, train_views={len(trainset)}, "
            f"zero_ratio={csv_row['importance_zero_ratio']:.4f}, elapsed={time.time() - tic:.1f}s"
        )
        torch.cuda.empty_cache()

    write_csv(analysis_root / "importance_distribution_all.csv", all_rows)
    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with (analysis_root / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
