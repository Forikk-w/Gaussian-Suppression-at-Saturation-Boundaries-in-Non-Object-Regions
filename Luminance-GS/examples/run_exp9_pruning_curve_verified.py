import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets.colmap import Dataset, Parser
from analyze_exp7_importance_pruning import load_json, load_splats, resolve_scene_paths
from rendering_double import rasterization_dual


SCENES = ["bike", "buu", "chair", "sofa"]
PRUNE_RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
MASKED_OPACITY_LOGIT = -100.0
BASELINE_METRIC_TOL = 1e-5
CRITICAL_DELTA_PSNR_MIN = -0.01
CRITICAL_DELTA_SSIM_MIN = -1e-3
CRITICAL_DELTA_LPIPS_MAX = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verified pruning curve with baseline consistency checks.")
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
        default="/data2/wd/workspace/experiments/luminance_gs_analysis/exp9_pruning_curve",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_importance_scores(csv_path: Path) -> Dict[str, np.ndarray]:
    per_scene: Dict[str, Dict[int, float]] = {scene: {} for scene in SCENES}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene = row["scene"]
            if scene not in per_scene:
                continue
            per_scene[scene][int(row["gaussian_idx"])] = float(row["importance_score"])
    result: Dict[str, np.ndarray] = {}
    for scene, scores in per_scene.items():
        if not scores:
            raise ValueError(f"Missing importance scores for scene {scene}")
        arr = np.zeros(max(scores.keys()) + 1, dtype=np.float64)
        for idx, value in scores.items():
            arr[idx] = value
        result[scene] = arr
    return result


def build_keep_mask(importance_scores: np.ndarray, prune_ratio: float) -> np.ndarray:
    total = len(importance_scores)
    keep_count = int(math.floor(total * (1.0 - prune_ratio)))
    keep_mask = np.zeros(total, dtype=bool)
    if keep_count <= 0:
        return keep_mask

    nonzero_ids = np.flatnonzero(importance_scores > 0.0)
    zero_ids = np.flatnonzero(importance_scores == 0.0)
    if nonzero_ids.size > 0:
        order = np.argsort(-importance_scores[nonzero_ids], kind="stable")
        nonzero_sorted = nonzero_ids[order]
    else:
        nonzero_sorted = np.empty(0, dtype=np.int64)

    keep_nonzero = min(keep_count, nonzero_sorted.size)
    if keep_nonzero > 0:
        keep_mask[nonzero_sorted[:keep_nonzero]] = True

    remaining = keep_count - keep_nonzero
    if remaining > 0 and zero_ids.size > 0:
        keep_mask[zero_ids[:remaining]] = True
    return keep_mask


def build_masked_opacities(opacities: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    masked = opacities.clone()
    masked[~keep_mask] = MASKED_OPACITY_LOGIT
    return masked


@torch.no_grad()
def render_eval_like(
    splats: Dict[str, torch.Tensor],
    cfg: Dict,
    camtoworld: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    means = splats["means3d"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    colors_low = colors * splats["adjust_k"] + splats["adjust_b"]
    render_mode = "antialiased" if cfg.get("antialiased", False) else "classic"
    render_enh, _, _, _, _ = rasterization_dual(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        colors_low=colors_low,
        viewmats=torch.linalg.inv(camtoworld[None, ...]),
        Ks=K[None, ...],
        width=width,
        height=height,
        packed=bool(cfg.get("packed", False)),
        absgrad=bool(cfg.get("absgrad", False)),
        sparse_grad=bool(cfg.get("sparse_grad", False)),
        rasterize_mode=render_mode,
        sh_degree=int(cfg.get("sh_degree", 3)),
        near_plane=float(cfg.get("near_plane", 0.01)),
        far_plane=float(cfg.get("far_plane", 1e10)),
        render_mode="RGB+ED",
    )
    return torch.clamp(render_enh[..., :3], 0.0, 1.0)


def compute_metrics(preds: List[torch.Tensor], gts: List[torch.Tensor], device: torch.device) -> Dict[str, float]:
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    psnrs = []
    ssims = []
    lpipss = []
    for pred, gt in zip(preds, gts):
        psnrs.append(psnr_metric(pred, gt))
        ssims.append(ssim_metric(pred, gt))
        lpipss.append(lpips_metric(pred, gt))
    return {
        "psnr": float(torch.stack(psnrs).mean().item()),
        "ssim": float(torch.stack(ssims).mean().item()),
        "lpips": float(torch.stack(lpipss).mean().item()),
    }


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    labeled = image.copy()
    overlay = labeled.copy()
    cv2.rectangle(overlay, (8, 8), (250, 42), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.45, labeled, 0.55, 0, labeled)
    cv2.putText(
        labeled,
        text,
        (18, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def make_compare_image(left: np.ndarray, right: np.ndarray, prune_ratio: float) -> np.ndarray:
    left_labeled = add_label(left, "Baseline")
    right_labeled = add_label(right, f"Pruned (-{int(round(prune_ratio * 100))}%)")
    pad = np.full((left.shape[0], 6, 3), 255, dtype=np.uint8)
    line = np.full((left.shape[0], 2, 3), 180, dtype=np.uint8)
    return np.concatenate([left_labeled, pad, line, pad, right_labeled], axis=1)


def verify_baseline_and_collect(
    scene: str,
    parser: Parser,
    valset: Dataset,
    splats: Dict[str, torch.Tensor],
    cfg: Dict,
    exp_dir: Path,
    device: torch.device,
) -> Tuple[List[np.ndarray], List[torch.Tensor], Dict[str, object]]:
    baseline_images: List[np.ndarray] = []
    baseline_preds: List[torch.Tensor] = []
    gts: List[torch.Tensor] = []
    per_image = []
    for idx in range(len(valset)):
        data = valset[idx]
        height, width = data["image"].shape[:2]
        render = render_eval_like(
            splats,
            cfg,
            data["camtoworld"].to(device),
            data["K"].to(device),
            width,
            height,
        )
        render_u8 = (render.squeeze(0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        ref = imageio.imread(exp_dir / "renders" / f"val_{idx:04d}_enh.png")[..., :3]
        equal = bool(np.array_equal(render_u8, ref))
        diff = np.abs(render_u8.astype(np.int16) - ref.astype(np.int16))
        per_image.append(
            {
                "val_idx": idx,
                "equal": equal,
                "max_abs_diff": int(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "num_diff_pixels": int(np.any(diff > 0, axis=-1).sum()),
            }
        )
        if not equal:
            raise RuntimeError(f"Baseline image mismatch for {scene} val_{idx:04d}")
        baseline_images.append(render_u8)
        baseline_preds.append(render.permute(0, 3, 1, 2))
        gts.append((data["image"].to(device) / 255.0).permute(2, 0, 1).unsqueeze(0))

    metrics = compute_metrics(baseline_preds, gts, device)
    stats_path = exp_dir / "stats" / "val_step9999.json"
    ref_stats = json.load(stats_path.open("r"))
    metric_diffs = {k: metrics[k] - float(ref_stats[k]) for k in ["psnr", "ssim", "lpips"]}
    if any(abs(metric_diffs[k]) > BASELINE_METRIC_TOL for k in metric_diffs):
        raise RuntimeError(f"Baseline metrics mismatch for {scene}: {metric_diffs}")

    return baseline_images, gts, {
        "scene": scene,
        "num_val_views": len(valset),
        "per_image": per_image,
        "baseline_metrics": metrics,
        "reference_metrics": {k: float(ref_stats[k]) for k in ["psnr", "ssim", "lpips"]},
        "metric_diffs": metric_diffs,
        "baseline_match": True,
    }


def pick_critical_ratio(rows: List[Dict[str, float]]) -> float:
    critical = 0.0
    for row in rows:
        if (
            row["delta_psnr"] >= CRITICAL_DELTA_PSNR_MIN
            and row["delta_ssim"] >= CRITICAL_DELTA_SSIM_MIN
            and row["delta_lpips"] <= CRITICAL_DELTA_LPIPS_MAX
        ):
            critical = row["prune_ratio"]
    return critical


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    experiment_root = Path(args.experiment_root)
    importance_csv = Path(args.importance_csv)
    analysis_root = Path(args.analysis_root)
    analysis_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    importance_scores = load_importance_scores(importance_csv)
    all_rows: List[Dict[str, float]] = []
    verification_summary: Dict[str, object] = {}
    meta = {
        "scenes": SCENES,
        "prune_ratios": PRUNE_RATIOS,
        "importance_csv": str(importance_csv),
        "mask_mode": "keep gaussian count fixed; set pruned opacity logits to -100.0",
        "baseline_metric_tol": BASELINE_METRIC_TOL,
        "critical_rule": {
            "delta_psnr_min": CRITICAL_DELTA_PSNR_MIN,
            "delta_ssim_min": CRITICAL_DELTA_SSIM_MIN,
            "delta_lpips_max": CRITICAL_DELTA_LPIPS_MAX,
            "compare_val_idx": 0,
        },
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for scene in SCENES:
        tic = time.time()
        data_dir, exp_dir, ckpt_path = resolve_scene_paths(repo_root, experiment_root, scene)
        cfg = load_json(exp_dir / "cfg.json")
        parser = Parser(str(data_dir), exp_name="over_exp", factor=1, normalize=True, test_every=8)
        valset = Dataset(parser, split="val")
        splats = load_splats(ckpt_path, device)

        baseline_images, gts, verification = verify_baseline_and_collect(
            scene, parser, valset, splats, cfg, exp_dir, device
        )
        verification_summary[scene] = verification

        baseline_metrics = verification["baseline_metrics"]
        scene_rows: List[Dict[str, float]] = []
        original_opacities = splats["opacities"]
        first_view_pruned_images: Dict[float, np.ndarray] = {0.0: baseline_images[0]}

        for prune_ratio in PRUNE_RATIOS:
            if prune_ratio == 0.0:
                metrics = baseline_metrics
            else:
                keep_mask_np = build_keep_mask(importance_scores[scene], prune_ratio)
                keep_mask = torch.from_numpy(keep_mask_np).to(device=device, dtype=torch.bool)
                masked_opacities = build_masked_opacities(original_opacities, keep_mask)
                splats["opacities"] = masked_opacities
                preds = []
                first_render = None
                for idx in range(len(valset)):
                    data = valset[idx]
                    height, width = data["image"].shape[:2]
                    render = render_eval_like(
                        splats,
                        cfg,
                        data["camtoworld"].to(device),
                        data["K"].to(device),
                        width,
                        height,
                    )
                    preds.append(render.permute(0, 3, 1, 2))
                    if idx == 0:
                        first_render = (render.squeeze(0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                splats["opacities"] = original_opacities
                metrics = compute_metrics(preds, gts, device)
                first_view_pruned_images[prune_ratio] = first_render

            row = {
                "scene": scene,
                "prune_ratio": prune_ratio,
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
                "delta_psnr": metrics["psnr"] - baseline_metrics["psnr"],
                "delta_ssim": metrics["ssim"] - baseline_metrics["ssim"],
                "delta_lpips": metrics["lpips"] - baseline_metrics["lpips"],
            }
            scene_rows.append(row)
            all_rows.append(row)

        critical_ratio = pick_critical_ratio(scene_rows)
        compare = make_compare_image(
            baseline_images[0],
            first_view_pruned_images[critical_ratio],
            critical_ratio,
        )
        imageio.imwrite(analysis_root / f"{scene}_pruning_compare.png", compare)
        verification_summary[scene]["critical_ratio"] = critical_ratio

        print(
            f"[exp9-verified] {scene}: baseline_ok=True, critical_ratio={critical_ratio}, "
            f"val_views={len(valset)}, elapsed {time.time() - tic:.1f}s"
        )
        torch.cuda.empty_cache()

    csv_path = analysis_root / "exp9_pruning_curve.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene", "prune_ratio", "psnr", "ssim", "lpips", "delta_psnr", "delta_ssim", "delta_lpips"],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    with (analysis_root / "baseline_verification.json").open("w") as f:
        json.dump(verification_summary, f, indent=2)
    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with (analysis_root / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
