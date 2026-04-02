import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets.colmap import Dataset, Parser

CURRENT_DIR = Path(__file__).resolve().parent
GSPATH = CURRENT_DIR.parent / "gsplat"
import sys

sys.path.append(str(GSPATH))

from cuda._wrapper import (  # noqa: E402
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_indices_in_range,
)
from rendering_double import rasterization_dual  # noqa: E402


SCENES = ["bike", "buu", "chair", "sofa"]
THRESHOLDS = [245, 250, 252]
GROUPS = ["A", "B", "C"]
EDGE_BAND = 0
SAT_INTERIOR = 1
NORMAL = 2
CLASS_NAMES = {EDGE_BAND: "sat_boundary", SAT_INTERIOR: "sat_interior", NORMAL: "normal"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Importance-based pruning/source analysis.")
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
        default="/data2/wd/workspace/experiments/luminance_gs_analysis/exp7_importance_pruning",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=SCENES,
    )
    parser.add_argument("--sat-dilate-radius", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--importance-batch-per-iter", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def resolve_scene_paths(repo_root: Path, experiment_root: Path, scene: str) -> Tuple[Path, Path, Path]:
    data_dir = repo_root / "data" / "LOM_full" / scene
    candidates = [
        experiment_root / f"{scene}_over_exp",
        experiment_root / f"{scene}_over_exp_rerun",
    ]
    exp_dir = next((p for p in candidates if p.exists()), None)
    if exp_dir is None:
        raise FileNotFoundError(f"Missing experiment directory for scene {scene}")
    ckpt_path = exp_dir / "ckpts" / "ckpt_9999.pt"
    cfg_path = exp_dir / "cfg.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing cfg: {cfg_path}")
    return data_dir, exp_dir, ckpt_path


def load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def load_splats(ckpt_path: Path, device: torch.device) -> Dict[str, Tensor]:
    ckpt = torch.load(ckpt_path, map_location=device)
    return {k: v.detach().to(device) for k, v in ckpt["splats"].items()}


def split_groups(num_gaussians: int) -> Dict[str, np.ndarray]:
    cut_a = int(math.floor(num_gaussians * 0.30))
    cut_b = int(math.floor(num_gaussians * 0.50))
    return {
        "A": np.arange(0, cut_a, dtype=np.int64),
        "B": np.arange(cut_a, cut_b, dtype=np.int64),
        "C": np.arange(cut_b, num_gaussians, dtype=np.int64),
    }


def build_scene_tensors(splats: Dict[str, Tensor]) -> Dict[str, Tensor]:
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    return {
        "means3d": splats["means3d"],
        "quats": splats["quats"],
        "scales_exp": torch.exp(splats["scales"]),
        "scale_mean": torch.exp(splats["scales"]).mean(dim=-1),
        "opacity_sigmoid": torch.sigmoid(splats["opacities"]),
        "colors": colors,
        "colors_low": colors * splats["adjust_k"] + splats["adjust_b"],
    }


def prepare_projection(
    scene_tensors: Dict[str, Tensor],
    camtoworld: Tensor,
    K: Tensor,
    width: int,
    height: int,
    cfg: Dict,
    tile_size: int,
) -> Dict[str, Tensor]:
    viewmats = torch.linalg.inv(camtoworld[None, ...])
    Ks = K[None, ...]
    proj_results = fully_fused_projection(
        scene_tensors["means3d"],
        None,
        scene_tensors["quats"],
        scene_tensors["scales_exp"],
        viewmats,
        Ks,
        width,
        height,
        eps2d=0.3,
        packed=bool(cfg.get("packed", False)),
        near_plane=float(cfg.get("near_plane", 0.01)),
        far_plane=float(cfg.get("far_plane", 1e10)),
        radius_clip=0.0,
        sparse_grad=False,
        calc_compensations=bool(cfg.get("antialiased", False)),
    )
    if bool(cfg.get("packed", False)):
        raise RuntimeError("Packed mode is not supported in exp7 script.")

    radii, means2d, depths, conics, compensations = proj_results
    opacities = scene_tensors["opacity_sigmoid"].unsqueeze(0)
    if compensations is not None:
        opacities = opacities * compensations
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    _, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=False,
        n_cameras=1,
    )
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
    return {
        "radii": radii,
        "means2d": means2d,
        "conics": conics,
        "opacities": opacities,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
    }


def compute_batch_weights(
    gs_ids: Tensor,
    pixel_ids: Tensor,
    camera_ids: Tensor,
    means2d: Tensor,
    conics: Tensor,
    opacities: Tensor,
    trans_flat: Tensor,
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    pixel_x = pixel_ids % width
    pixel_y = pixel_ids // width
    pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1).to(means2d.dtype) + 0.5
    deltas = pixel_coords - means2d[camera_ids, gs_ids]
    conic_vals = conics[camera_ids, gs_ids]
    sigmas = (
        0.5
        * (conic_vals[:, 0] * deltas[:, 0] ** 2 + conic_vals[:, 2] * deltas[:, 1] ** 2)
        + conic_vals[:, 1] * deltas[:, 0] * deltas[:, 1]
    )
    alphas = torch.clamp_max(opacities[camera_ids, gs_ids] * torch.exp(-sigmas), 0.999)
    ray_indices = camera_ids * (height * width) + pixel_ids
    order = torch.argsort(ray_indices, stable=True)
    ray_sorted = ray_indices[order]
    gs_sorted = gs_ids[order]
    alpha_sorted = alphas[order].double()

    new_ray = torch.empty_like(ray_sorted, dtype=torch.bool)
    new_ray[0] = True
    new_ray[1:] = ray_sorted[1:] != ray_sorted[:-1]
    segment_ids = torch.cumsum(new_ray.to(torch.int64), dim=0) - 1
    start_idx = torch.nonzero(new_ray, as_tuple=False).squeeze(-1)
    prev_sum = torch.zeros_like(start_idx, dtype=torch.float64)

    log_one_minus = torch.log(torch.clamp(1.0 - alpha_sorted, min=1e-12))
    cumsum_log = torch.cumsum(log_one_minus, dim=0)
    if start_idx.numel() > 1:
        prev_sum[1:] = cumsum_log[start_idx[1:] - 1]
    base = prev_sum[segment_ids]
    exclusive_log = cumsum_log - log_one_minus - base
    local_trans = torch.exp(exclusive_log)
    weights = alpha_sorted * local_trans * trans_flat[ray_sorted]

    end_idx = torch.cat([start_idx[1:] - 1, start_idx.new_tensor([ray_sorted.numel() - 1])])
    ray_unique = ray_sorted[end_idx]
    tail_trans = torch.exp(cumsum_log[end_idx] - prev_sum)
    return gs_sorted, weights, (ray_unique, tail_trans)


@torch.no_grad()
def accumulate_importance_for_view(
    importance: Tensor,
    projection: Dict[str, Tensor],
    tile_size: int,
    batch_per_iter: int,
    width: int,
    height: int,
) -> None:
    trans_flat = torch.ones(height * width, device=importance.device, dtype=torch.float64)
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
        trans_flat[ray_unique] = trans_flat[ray_unique] * tail_trans
        step += batch_per_iter


def make_class_map(image: np.ndarray, threshold: int, dilate_radius: int) -> np.ndarray:
    max_rgb = image.max(axis=-1)
    sat_mask = max_rgb >= threshold
    kernel = np.ones((2 * dilate_radius + 1, 2 * dilate_radius + 1), dtype=np.uint8)
    dilated = cv2.dilate(sat_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    edge_band = np.logical_xor(dilated, sat_mask)
    class_map = np.full(max_rgb.shape, NORMAL, dtype=np.uint8)
    class_map[sat_mask] = SAT_INTERIOR
    class_map[edge_band] = EDGE_BAND
    return class_map


def update_vote_counts(
    vote_counts: Dict[int, np.ndarray],
    projection: Dict[str, Tensor],
    class_maps: Dict[int, np.ndarray],
    width: int,
    height: int,
) -> None:
    visible = projection["radii"][0] > 0
    if not torch.any(visible):
        return
    gaussian_ids = torch.where(visible)[0].cpu().numpy()
    means2d = projection["means2d"][0, visible]
    xs = torch.round(means2d[:, 0]).to(torch.int64)
    ys = torch.round(means2d[:, 1]).to(torch.int64)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if not torch.any(valid):
        return
    gaussian_ids = gaussian_ids[valid.cpu().numpy()]
    xs = xs[valid].cpu().numpy()
    ys = ys[valid].cpu().numpy()
    for threshold, class_map in class_maps.items():
        classes = class_map[ys, xs]
        vote_counts[threshold][gaussian_ids, classes] += 1


def summarize_source_analysis(
    scene: str,
    importance_scores: np.ndarray,
    opacity_values: np.ndarray,
    scale_values: np.ndarray,
    vote_counts: Dict[int, np.ndarray],
) -> List[Dict[str, object]]:
    sorted_idx = np.argsort(importance_scores, kind="stable")
    group_ranges = split_groups(len(sorted_idx))
    rows: List[Dict[str, object]] = []
    for threshold in THRESHOLDS:
        majority_classes = vote_counts[threshold].argmax(axis=1)
        no_vote = vote_counts[threshold].sum(axis=1) == 0
        majority_classes[no_vote] = NORMAL
        for group_name, index_range in group_ranges.items():
            group_ids = sorted_idx[index_range]
            group_classes = majority_classes[group_ids]
            num_gaussians = int(group_ids.size)
            boundary_count = int(np.sum(group_classes == EDGE_BAND))
            interior_count = int(np.sum(group_classes == SAT_INTERIOR))
            normal_count = int(np.sum(group_classes == NORMAL))
            rows.append(
                {
                    "scene": scene,
                    "threshold": threshold,
                    "group": group_name,
                    "num_gaussians": num_gaussians,
                    "sat_boundary_count": boundary_count,
                    "sat_boundary_ratio": boundary_count / num_gaussians if num_gaussians else 0.0,
                    "sat_interior_count": interior_count,
                    "sat_interior_ratio": interior_count / num_gaussians if num_gaussians else 0.0,
                    "normal_count": normal_count,
                    "normal_ratio": normal_count / num_gaussians if num_gaussians else 0.0,
                    "importance_mean": float(importance_scores[group_ids].mean()) if num_gaussians else 0.0,
                    "opacity_mean": float(opacity_values[group_ids].mean()) if num_gaussians else 0.0,
                    "scale_mean": float(scale_values[group_ids].mean()) if num_gaussians else 0.0,
                }
            )
    return rows


@torch.no_grad()
def render_enhanced(
    splats: Dict[str, Tensor],
    camtoworld: Tensor,
    K: Tensor,
    width: int,
    height: int,
    cfg: Dict,
) -> Tensor:
    render_mode = "antialiased" if cfg.get("antialiased", False) else "classic"
    render_enh, _, _, _, _ = rasterization_dual(
        means=splats["means3d"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=torch.cat([splats["sh0"], splats["shN"]], dim=1),
        colors_low=torch.cat([splats["sh0"], splats["shN"]], dim=1) * splats["adjust_k"] + splats["adjust_b"],
        viewmats=torch.linalg.inv(camtoworld[None, ...]),
        Ks=K[None, ...],
        width=width,
        height=height,
        near_plane=float(cfg.get("near_plane", 0.01)),
        far_plane=float(cfg.get("far_plane", 1e10)),
        packed=bool(cfg.get("packed", False)),
        absgrad=bool(cfg.get("absgrad", False)),
        sparse_grad=bool(cfg.get("sparse_grad", False)),
        rasterize_mode=render_mode,
        sh_degree=int(cfg.get("sh_degree", 3)),
        render_mode="RGB",
    )
    return torch.clamp(render_enh[..., :3], 0.0, 1.0)


@torch.no_grad()
def evaluate_pruning(
    parser: Parser,
    splats: Dict[str, Tensor],
    cfg: Dict,
    sorted_idx: np.ndarray,
    device: torch.device,
) -> List[Dict[str, object]]:
    valset = Dataset(parser, split="val")
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    group_ranges = split_groups(len(sorted_idx))
    keep_sets = {
        0.0: np.ones(len(sorted_idx), dtype=bool),
        0.3: np.ones(len(sorted_idx), dtype=bool),
        0.5: np.ones(len(sorted_idx), dtype=bool),
    }
    keep_sets[0.3][sorted_idx[group_ranges["A"]]] = False
    keep_sets[0.5][sorted_idx[np.concatenate([group_ranges["A"], group_ranges["B"]])]] = False

    rows: List[Dict[str, object]] = []
    baseline = None
    for prune_ratio in [0.0, 0.3, 0.5]:
        keep = torch.from_numpy(keep_sets[prune_ratio]).to(device=device, dtype=torch.bool)
        pruned_splats = {k: v[keep].contiguous() for k, v in splats.items()}
        psnrs: List[Tensor] = []
        ssims: List[Tensor] = []
        lpipss: List[Tensor] = []
        for i in range(len(valset)):
            data = valset[i]
            gt = (data["image"].to(device) / 255.0).permute(2, 0, 1).unsqueeze(0)
            render = render_enhanced(
                pruned_splats,
                data["camtoworld"].to(device),
                data["K"].to(device),
                gt.shape[-1],
                gt.shape[-2],
                cfg,
            )
            pred = render.permute(0, 3, 1, 2)
            psnrs.append(psnr_metric(pred, gt))
            ssims.append(ssim_metric(pred, gt))
            lpipss.append(lpips_metric(pred, gt))
        row = {
            "prune_ratio": prune_ratio,
            "psnr": float(torch.stack(psnrs).mean().item()),
            "ssim": float(torch.stack(ssims).mean().item()),
            "lpips": float(torch.stack(lpipss).mean().item()),
        }
        if baseline is None:
            baseline = row
        row["delta_psnr"] = row["psnr"] - baseline["psnr"]
        row["delta_ssim"] = row["ssim"] - baseline["ssim"]
        row["delta_lpips"] = row["lpips"] - baseline["lpips"]
        rows.append(row)
        torch.cuda.empty_cache()
    return rows


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_scene_importance_tensor(path: Path, importance_scores: np.ndarray) -> None:
    torch.save({"importance_scores": torch.from_numpy(importance_scores)}, path)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    experiment_root = Path(args.experiment_root)
    analysis_root = Path(args.analysis_root)
    analysis_root.mkdir(parents=True, exist_ok=True)
    (analysis_root / "importance_scores").mkdir(exist_ok=True)

    device = torch.device(args.device)
    all_importance_rows: List[Dict[str, object]] = []
    all_pruning_rows: List[Dict[str, object]] = []
    all_source_rows: List[Dict[str, object]] = []
    meta = {
        "scenes": args.scenes,
        "thresholds": THRESHOLDS,
        "sat_dilate_radius": args.sat_dilate_radius,
        "tile_size": args.tile_size,
        "importance_batch_per_iter": args.importance_batch_per_iter,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for scene in args.scenes:
        scene_tic = time.time()
        data_dir, exp_dir, ckpt_path = resolve_scene_paths(repo_root, experiment_root, scene)
        cfg = load_json(exp_dir / "cfg.json")
        parser = Parser(str(data_dir), exp_name="over_exp", factor=1, normalize=False, test_every=8)
        trainset = Dataset(parser, split="train")
        splats = load_splats(ckpt_path, device)
        scene_tensors = build_scene_tensors(splats)

        num_gaussians = int(scene_tensors["means3d"].shape[0])
        importance = torch.zeros(num_gaussians, device=device, dtype=torch.float64)
        vote_counts = {
            threshold: np.zeros((num_gaussians, 3), dtype=np.int32) for threshold in THRESHOLDS
        }

        for image_idx in range(len(trainset)):
            data = trainset[image_idx]
            image = data["image"].numpy().astype(np.uint8)
            height, width = image.shape[:2]
            class_maps = {
                threshold: make_class_map(image, threshold, args.sat_dilate_radius)
                for threshold in THRESHOLDS
            }
            projection = prepare_projection(
                scene_tensors,
                data["camtoworld"].to(device),
                data["K"].to(device),
                width,
                height,
                cfg,
                args.tile_size,
            )
            update_vote_counts(vote_counts, projection, class_maps, width, height)
            accumulate_importance_for_view(
                importance,
                projection,
                args.tile_size,
                args.importance_batch_per_iter,
                width,
                height,
            )

        importance_scores = importance.detach().cpu().numpy()
        opacity_values = scene_tensors["opacity_sigmoid"].detach().cpu().numpy()
        scale_values = scene_tensors["scale_mean"].detach().cpu().numpy()
        sorted_idx = np.argsort(importance_scores, kind="stable")

        for gaussian_idx, score in enumerate(importance_scores):
            all_importance_rows.append(
                {
                    "scene": scene,
                    "gaussian_idx": gaussian_idx,
                    "importance_score": float(score),
                }
            )

        save_scene_importance_tensor(
            analysis_root / "importance_scores" / f"{scene}_importance_scores.pt",
            importance_scores,
        )

        pruning_rows = evaluate_pruning(parser, splats, cfg, sorted_idx, device)
        for row in pruning_rows:
            row["scene"] = scene
            all_pruning_rows.append(row)

        all_source_rows.extend(
            summarize_source_analysis(
                scene,
                importance_scores,
                opacity_values,
                scale_values,
                vote_counts,
            )
        )

        print(
            f"[exp7] {scene}: {num_gaussians} GS, "
            f"{len(trainset)} train views, elapsed {time.time() - scene_tic:.1f}s"
        )
        torch.cuda.empty_cache()

    write_csv(
        analysis_root / "exp7_importance_scores.csv",
        ["scene", "gaussian_idx", "importance_score"],
        all_importance_rows,
    )
    write_csv(
        analysis_root / "exp7_pruning_metrics.csv",
        [
            "scene",
            "prune_ratio",
            "psnr",
            "ssim",
            "lpips",
            "delta_psnr",
            "delta_ssim",
            "delta_lpips",
        ],
        all_pruning_rows,
    )
    write_csv(
        analysis_root / "exp7_source_analysis.csv",
        [
            "scene",
            "threshold",
            "group",
            "num_gaussians",
            "sat_boundary_count",
            "sat_boundary_ratio",
            "sat_interior_count",
            "sat_interior_ratio",
            "normal_count",
            "normal_ratio",
            "importance_mean",
            "opacity_mean",
            "scale_mean",
        ],
        all_source_rows,
    )
    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with (analysis_root / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
