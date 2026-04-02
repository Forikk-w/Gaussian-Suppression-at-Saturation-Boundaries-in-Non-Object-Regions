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

from datasets.colmap import Dataset, Parser
from analyze_exp7_importance_pruning import load_json, load_splats

CURRENT_DIR = Path(__file__).resolve().parent
GSPATH = CURRENT_DIR.parent / "gsplat"
import sys

sys.path.append(str(GSPATH))

from cuda._wrapper import fully_fused_projection  # noqa: E402


SCENES = ["bike", "buu", "chair", "sofa"]
SAT_PIXEL_THRESHOLD = 250
SAT_DILATE_RADIUS = 3
ELLIPSE_SIGMA_CUTOFF = 4.5  # 3-sigma ellipse boundary under sigma=0.5*x^T C x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Gaussian projection density between over-exp and low-light checkpoints.")
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
        default="/data2/wd/workspace/experiments/luminance_gs_analysis/exp12_density_comparison",
    )
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--sat-pixel-threshold", type=int, default=SAT_PIXEL_THRESHOLD)
    parser.add_argument("--sat-dilate-radius", type=int, default=SAT_DILATE_RADIUS)
    parser.add_argument("--density-budget", type=int, default=6_000_000)
    parser.add_argument("--density-chunk-cap", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def resolve_scene_experiment_dirs(experiment_root: Path, scene: str) -> Tuple[Path, Path]:
    low_candidates = [
        experiment_root / f"{scene}_low",
        experiment_root / f"{scene}_low_rerun",
    ]
    over_candidates = [
        experiment_root / f"{scene}_over_exp",
        experiment_root / f"{scene}_over_exp_rerun",
    ]
    low_dir = next((p for p in low_candidates if p.exists()), None)
    over_dir = next((p for p in over_candidates if p.exists()), None)
    if low_dir is None:
        raise FileNotFoundError(f"Missing low-light experiment directory for {scene}")
    if over_dir is None:
        raise FileNotFoundError(f"Missing over-exp experiment directory for {scene}")
    return low_dir, over_dir


def build_parser(data_dir: Path, cfg: Dict, exp_name: str) -> Parser:
    return Parser(
        str(data_dir),
        exp_name=exp_name,
        factor=int(cfg.get("data_factor", 1)),
        normalize=True,
        test_every=int(cfg.get("test_every", 8)),
    )


def project_scene(
    splats: Dict[str, torch.Tensor],
    camtoworld: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    cfg: Dict,
) -> Dict[str, torch.Tensor]:
    proj_results = fully_fused_projection(
        splats["means3d"],
        None,
        splats["quats"],
        torch.exp(splats["scales"]),
        torch.linalg.inv(camtoworld[None, ...]),
        K[None, ...],
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
        raise RuntimeError("Packed mode is not supported for density comparison.")
    radii, means2d, _depths, conics, _compensations = proj_results
    return {
        "radii": radii[0],
        "means2d": means2d[0],
        "conics": conics[0],
    }


@torch.no_grad()
def compute_density_map(
    projection: Dict[str, torch.Tensor],
    width: int,
    height: int,
    density_budget: int,
    density_chunk_cap: int,
) -> torch.Tensor:
    radii = projection["radii"]
    means2d = projection["means2d"]
    conics = projection["conics"]
    visible_ids = torch.where(radii > 0)[0]
    density = torch.zeros(height * width, device=means2d.device, dtype=torch.float32)
    if visible_ids.numel() == 0:
        return density.view(height, width)

    visible_ids = visible_ids[torch.argsort(radii[visible_ids])]
    ptr = 0
    ones_cache: Dict[int, torch.Tensor] = {}
    while ptr < visible_ids.numel():
        remaining = visible_ids.numel() - ptr
        chunk_size = min(density_chunk_cap, remaining)
        while True:
            chunk_ids = visible_ids[ptr : ptr + chunk_size]
            max_radius = int(radii[chunk_ids].max().item())
            patch_area = (2 * max_radius + 1) ** 2
            if chunk_size == 1 or chunk_size * patch_area <= density_budget:
                break
            chunk_size = max(1, chunk_size // 2)
        chunk_ids = visible_ids[ptr : ptr + chunk_size]
        max_radius = int(radii[chunk_ids].max().item())

        offsets = torch.arange(-max_radius, max_radius + 1, device=means2d.device, dtype=torch.int64)
        grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")
        grid_x = grid_x.reshape(1, -1)
        grid_y = grid_y.reshape(1, -1)

        chunk_means = means2d[chunk_ids]
        chunk_conics = conics[chunk_ids]
        base_x = torch.floor(chunk_means[:, 0]).to(torch.int64).unsqueeze(1)
        base_y = torch.floor(chunk_means[:, 1]).to(torch.int64).unsqueeze(1)
        px = base_x + grid_x
        py = base_y + grid_y

        valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        dx = px.to(torch.float32) + 0.5 - chunk_means[:, 0:1]
        dy = py.to(torch.float32) + 0.5 - chunk_means[:, 1:2]
        sigma = (
            0.5
            * (chunk_conics[:, 0:1] * dx * dx + chunk_conics[:, 2:3] * dy * dy)
            + chunk_conics[:, 1:2] * dx * dy
        )
        inside = valid & (sigma <= ELLIPSE_SIGMA_CUTOFF)
        if inside.any():
            pixel_ids = (py[inside] * width + px[inside]).to(torch.int64)
            count = pixel_ids.numel()
            ones = ones_cache.get(count)
            if ones is None:
                ones = torch.ones(count, device=means2d.device, dtype=density.dtype)
                ones_cache[count] = ones
            density.index_add_(0, pixel_ids, ones)
        ptr += chunk_size

    return density.view(height, width)


def build_sat_mask(image: np.ndarray, threshold: int, dilate_radius: int) -> Tuple[np.ndarray, np.ndarray]:
    max_rgb = image.max(axis=-1)
    sat = max_rgb >= threshold
    kernel = np.ones((2 * dilate_radius + 1, 2 * dilate_radius + 1), dtype=np.uint8)
    sat_dilated = cv2.dilate(sat.astype(np.uint8), kernel, iterations=1).astype(bool)
    return sat, sat_dilated


def overlay_contours(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    canvas = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, 1, lineType=cv2.LINE_AA)
    return canvas


def make_delta_heatmap(delta_density: np.ndarray) -> np.ndarray:
    scale = float(np.percentile(np.abs(delta_density), 99.5))
    scale = max(scale, 1e-6)
    normalized = np.clip((delta_density / scale + 1.0) * 0.5, 0.0, 1.0)
    heat = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_COOLWARM if hasattr(cv2, "COLORMAP_COOLWARM") else cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return heat


def make_ratio_heatmap(density_ratio: np.ndarray) -> np.ndarray:
    log_ratio = np.log2(np.clip(density_ratio, 1e-6, None))
    scale = float(np.percentile(np.abs(log_ratio), 99.5))
    scale = max(scale, 1e-6)
    normalized = np.clip((log_ratio / scale + 1.0) * 0.5, 0.0, 1.0)
    heat = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_COOLWARM if hasattr(cv2, "COLORMAP_COOLWARM") else cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return heat


def add_text(image: np.ndarray, lines: List[str]) -> np.ndarray:
    canvas = image.copy()
    box_h = 26 + 22 * len(lines)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (8, 8), (310, box_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
    for idx, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (16, 28 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def make_triptych(
    over_train_img: np.ndarray,
    sat_mask: np.ndarray,
    delta_density: np.ndarray,
    density_ratio: np.ndarray,
    left_lines: List[str],
    middle_lines: List[str],
    right_lines: List[str],
) -> np.ndarray:
    left = add_text(overlay_contours(over_train_img, sat_mask), left_lines)
    middle = add_text(make_delta_heatmap(delta_density), middle_lines)
    right = add_text(overlay_contours(make_ratio_heatmap(density_ratio), sat_mask), right_lines)
    sep = np.full((left.shape[0], 10, 3), 255, dtype=np.uint8)
    line = np.full((left.shape[0], 2, 3), 180, dtype=np.uint8)
    return np.concatenate([left, sep, line, sep, middle, sep, line, sep, right], axis=1)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "scene",
        "val_idx",
        "mean_raw_sat_overexp",
        "mean_raw_sat_low",
        "mean_raw_normal_overexp",
        "mean_raw_normal_low",
        "sat_ratio",
        "normal_ratio",
        "total_gs_overexp",
        "total_gs_low",
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
    analysis_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    rows: List[Dict[str, object]] = []
    overview_images: List[np.ndarray] = []
    alignment_report: Dict[str, Dict[str, float]] = {}
    meta = {
        "scenes": args.scenes,
        "sat_pixel_threshold": args.sat_pixel_threshold,
        "sat_dilate_radius": args.sat_dilate_radius,
        "ellipse_sigma_cutoff": ELLIPSE_SIGMA_CUTOFF,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for scene in args.scenes:
        tic = time.time()
        data_dir = repo_root / "data" / "LOM_full" / scene
        low_dir, over_dir = resolve_scene_experiment_dirs(experiment_root, scene)
        low_cfg = load_json(low_dir / "cfg.json")
        over_cfg = load_json(over_dir / "cfg.json")
        parser_low = build_parser(data_dir, low_cfg, "low")
        parser_over = build_parser(data_dir, over_cfg, "over_exp")
        val_low = Dataset(parser_low, split="val")
        val_over = Dataset(parser_over, split="val")
        train_over = Dataset(parser_over, split="train")

        if len(val_low) != len(val_over):
            raise RuntimeError(f"Validation view count mismatch for {scene}: {len(val_low)} vs {len(val_over)}")

        low_splats = load_splats(low_dir / "ckpts" / "ckpt_9999.pt", device)
        over_splats = load_splats(over_dir / "ckpts" / "ckpt_9999.pt", device)
        total_gs_low = int(low_splats["means3d"].shape[0])
        total_gs_over = int(over_splats["means3d"].shape[0])

        scene_out = analysis_root / scene
        scene_out.mkdir(exist_ok=True)

        max_cam_diff = 0.0
        max_k_diff = 0.0
        train_camtoworlds = []
        train_images = []
        for train_idx in range(len(train_over)):
            item = train_over[train_idx]
            train_camtoworlds.append(item["camtoworld"].clone())
            train_images.append(item["image"].numpy().astype(np.uint8))
        train_centers = torch.stack([c[:3, 3] for c in train_camtoworlds], dim=0)
        nearest_train_records: List[Dict[str, object]] = []
        for idx in range(len(val_over)):
            data_over = val_over[idx]
            data_low = val_low[idx]
            max_cam_diff = max(max_cam_diff, float(torch.max(torch.abs(data_over["camtoworld"] - data_low["camtoworld"])).item()))
            max_k_diff = max(max_k_diff, float(torch.max(torch.abs(data_over["K"] - data_low["K"])).item()))

            over_img = data_over["image"].numpy().astype(np.uint8)
            if data_low["image"].shape[:2] != data_over["image"].shape[:2]:
                raise RuntimeError(f"Image shape mismatch at {scene} val{idx:02d}")
            height, width = over_img.shape[:2]

            proj_over = project_scene(
                over_splats,
                data_over["camtoworld"].to(device),
                data_over["K"].to(device),
                width,
                height,
                over_cfg,
            )
            proj_low = project_scene(
                low_splats,
                data_low["camtoworld"].to(device),
                data_low["K"].to(device),
                width,
                height,
                low_cfg,
            )

            density_over = compute_density_map(
                proj_over,
                width,
                height,
                args.density_budget,
                args.density_chunk_cap,
            ).detach().cpu().numpy()
            density_low = compute_density_map(
                proj_low,
                width,
                height,
                args.density_budget,
                args.density_chunk_cap,
            ).detach().cpu().numpy()

            delta_density = density_over - density_low
            density_ratio = density_over / (density_low + 1.0)

            val_center = data_over["camtoworld"][:3, 3]
            nearest_train_idx = int(torch.argmin(torch.norm(train_centers - val_center[None, :], dim=1)).item())
            nearest_train_distance = float(torch.norm(train_centers[nearest_train_idx] - val_center).item())
            nearest_train_img = train_images[nearest_train_idx]
            if nearest_train_img.shape[:2] != over_img.shape[:2]:
                nearest_train_img = cv2.resize(nearest_train_img, (width, height), interpolation=cv2.INTER_LINEAR)

            _sat_raw, sat_mask = build_sat_mask(
                nearest_train_img,
                args.sat_pixel_threshold,
                args.sat_dilate_radius,
            )
            normal_mask = ~sat_mask
            mean_raw_sat_overexp = float(density_over[sat_mask].mean()) if np.any(sat_mask) else 0.0
            mean_raw_sat_low = float(density_low[sat_mask].mean()) if np.any(sat_mask) else 0.0
            mean_raw_normal_overexp = float(density_over[normal_mask].mean()) if np.any(normal_mask) else 0.0
            mean_raw_normal_low = float(density_low[normal_mask].mean()) if np.any(normal_mask) else 0.0
            sat_ratio = float(mean_raw_sat_overexp / mean_raw_sat_low) if abs(mean_raw_sat_low) > 1e-12 else math.nan
            normal_ratio = float(mean_raw_normal_overexp / mean_raw_normal_low) if abs(mean_raw_normal_low) > 1e-12 else math.nan
            row = {
                "scene": scene,
                "val_idx": idx,
                "mean_raw_sat_overexp": mean_raw_sat_overexp,
                "mean_raw_sat_low": mean_raw_sat_low,
                "mean_raw_normal_overexp": mean_raw_normal_overexp,
                "mean_raw_normal_low": mean_raw_normal_low,
                "sat_ratio": sat_ratio,
                "normal_ratio": normal_ratio,
                "total_gs_overexp": total_gs_over,
                "total_gs_low": total_gs_low,
            }
            rows.append(row)
            nearest_train_records.append(
                {
                    "val_idx": idx,
                    "nearest_train_idx": nearest_train_idx,
                    "nearest_train_distance": nearest_train_distance,
                }
            )

            triptych = make_triptych(
                nearest_train_img,
                sat_mask,
                delta_density,
                density_ratio,
                [
                    f"{scene} val{idx:02d}",
                    f"nearest train={nearest_train_idx}",
                    f"cam_dist={nearest_train_distance:.4f}",
                ],
                [
                    f"sat over={mean_raw_sat_overexp:.2f}",
                    f"sat low={mean_raw_sat_low:.2f}",
                    f"normal over={mean_raw_normal_overexp:.2f}",
                    f"normal low={mean_raw_normal_low:.2f}",
                ],
                [
                    f"sat_ratio={sat_ratio:.3f}" if not math.isnan(sat_ratio) else "sat_ratio=nan",
                    f"normal_ratio={normal_ratio:.3f}" if not math.isnan(normal_ratio) else "normal_ratio=nan",
                ],
            )
            imageio.imwrite(scene_out / f"{scene}_val{idx:02d}_density_compare.png", triptych)
            if idx == 0:
                overview_images.append(triptych)

        alignment_report[scene] = {
            "max_abs_camtoworld_diff": max_cam_diff,
            "max_abs_K_diff": max_k_diff,
            "num_val_views": len(val_over),
            "low_dir": str(low_dir),
            "over_dir": str(over_dir),
            "nearest_train_records": nearest_train_records,
        }
        print(
            f"[exp12] {scene}: val_views={len(val_over)}, total_gs_over={total_gs_over}, "
            f"total_gs_low={total_gs_low}, cam_diff={max_cam_diff:.3e}, elapsed={time.time() - tic:.1f}s"
        )
        torch.cuda.empty_cache()

    if overview_images:
        overview = np.concatenate(overview_images, axis=1)
        imageio.imwrite(analysis_root / "all_scenes_density_overview.png", overview)

    write_csv(analysis_root / "density_comparison.csv", rows)
    meta["alignment_report"] = alignment_report
    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with (analysis_root / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
