#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from datasets.colmap import Parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project tiny-scale gaussians to over-exposed images and "
            "analyze whether they concentrate near saturation boundaries."
        )
    )
    parser.add_argument(
        "--exp_root",
        type=str,
        default="/home/wd/workspace/experiments/luminance_gs",
        help="Root directory of trained experiment outputs.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/wd/workspace/repos/Luminance-GS/Luminance-GS/data/LOM_full",
        help="Root directory of LOM_full datasets.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="/home/wd/workspace/experiments/luminance_gs_analysis/exp3_tiny_scale_projection_overexp",
        help="Output root for overlays and tables.",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="bike,buu,chair,shrub,sofa",
        help="Comma-separated scene names.",
    )
    parser.add_argument(
        "--ckpt_step",
        type=int,
        default=9999,
        help="Checkpoint step to load (ckpt_{step}.pt). Fallback to latest when missing.",
    )
    parser.add_argument(
        "--tiny_scale_threshold",
        type=float,
        default=None,
        help="Tiny scale threshold on mean(exp(scale_xyz)). If omitted, read from meta json or per-scene P10.",
    )
    parser.add_argument(
        "--tiny_scale_meta_json",
        type=str,
        default="/home/wd/workspace/experiments/luminance_gs_analysis/exp1_opacity_scale/tables/meta.json",
        help="Meta json with global tiny threshold from experiment 1.",
    )
    parser.add_argument(
        "--sat_threshold",
        type=int,
        default=250,
        help="Saturation threshold on max RGB channel [0,255].",
    )
    parser.add_argument(
        "--boundary_radius",
        type=int,
        default=3,
        help="Dilation radius (pixels) around saturation boundary.",
    )
    return parser.parse_args()


def read_global_tiny_threshold(meta_json: str) -> float:
    if not os.path.exists(meta_json):
        return None
    try:
        with open(meta_json, "r") as f:
            data = json.load(f)
        val = data.get("tiny_scale_threshold_value", None)
        return float(val) if val is not None else None
    except Exception:
        return None


def resolve_overexp_exp_dir(exp_root: str, scene: str) -> str:
    candidates = [
        os.path.join(exp_root, f"{scene}_over_exp"),
        os.path.join(exp_root, f"{scene}_over_exp_rerun"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"No over_exp experiment folder found for scene={scene}")


def resolve_checkpoint(exp_dir: str, ckpt_step: int) -> str:
    preferred = os.path.join(exp_dir, "ckpts", f"ckpt_{ckpt_step}.pt")
    if os.path.exists(preferred):
        return preferred
    all_ckpts = sorted(glob.glob(os.path.join(exp_dir, "ckpts", "ckpt_*.pt")))
    if not all_ckpts:
        raise FileNotFoundError(f"No checkpoints found under {exp_dir}/ckpts")
    return all_ckpts[-1]


def load_tiny_points(ckpt_path: str, tiny_threshold: float = None) -> Tuple[np.ndarray, Dict[str, float]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    splats = ckpt["splats"]
    means3d = splats["means3d"].detach().cpu().numpy().astype(np.float32)
    scales = splats["scales"].detach().cpu().numpy().astype(np.float32)
    scale_mean = np.exp(scales).mean(axis=1)

    if tiny_threshold is None:
        tiny_threshold = float(np.quantile(scale_mean, 0.10))
        threshold_source = "scene_p10"
    else:
        threshold_source = "global_or_manual"

    tiny_mask = scale_mean < tiny_threshold
    tiny_points = means3d[tiny_mask]
    stats = {
        "num_total": int(means3d.shape[0]),
        "num_tiny": int(tiny_mask.sum()),
        "tiny_ratio": float(tiny_mask.mean()),
        "tiny_threshold": float(tiny_threshold),
        "threshold_source": threshold_source,
    }
    return tiny_points, stats


def project_world_to_image(points_world: np.ndarray, camtoworld: np.ndarray, K: np.ndarray) -> np.ndarray:
    if points_world.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    worldtocam = np.linalg.inv(camtoworld)
    pts_cam = (worldtocam[:3, :3] @ points_world.T + worldtocam[:3, 3:4]).T
    z = pts_cam[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float32)
    pts_cam = pts_cam[valid]
    proj = (K @ pts_cam.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    return uv.astype(np.float32)


def build_saturation_masks(image_bgr: np.ndarray, sat_threshold: int, boundary_radius: int) -> Tuple[np.ndarray, np.ndarray]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sat_mask = (np.max(image_rgb, axis=2) >= sat_threshold).astype(np.uint8)
    sat_mask_u8 = (sat_mask * 255).astype(np.uint8)

    boundary = cv2.morphologyEx(
        sat_mask_u8,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    )
    if boundary_radius > 0:
        k = 2 * boundary_radius + 1
        boundary = cv2.dilate(boundary, np.ones((k, k), dtype=np.uint8), iterations=1)

    boundary_mask = boundary > 0
    sat_mask_bool = sat_mask_u8 > 0
    return sat_mask_bool, boundary_mask


def draw_overlay(
    image_bgr: np.ndarray,
    boundary_mask: np.ndarray,
    pts_xy: np.ndarray,
    title_lines: List[str],
) -> np.ndarray:
    out = image_bgr.copy()

    if boundary_mask.any():
        layer = out.copy()
        layer[boundary_mask] = (0, 255, 255)  # yellow boundary band
        out = cv2.addWeighted(out, 0.72, layer, 0.28, 0)

    h, w = out.shape[:2]
    if pts_xy.shape[0] > 0:
        x = np.rint(pts_xy[:, 0]).astype(np.int32)
        y = np.rint(pts_xy[:, 1]).astype(np.int32)
        inb = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x = x[inb]
        y = y[inb]
        if x.size > 0:
            p_mask = np.zeros((h, w), dtype=np.uint8)
            p_mask[y, x] = 255
            p_mask = cv2.dilate(p_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
            out[p_mask > 0] = (0, 0, 255)  # red tiny-gaussian projections

    y0 = 28
    for line in title_lines:
        cv2.putText(
            out,
            line,
            (14, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (14, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y0 += 26
    return out


def safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num / den)


def analyze_scene(
    scene: str,
    exp_root: str,
    data_root: str,
    out_root: str,
    ckpt_step: int,
    tiny_threshold_global: float,
    sat_threshold: int,
    boundary_radius: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    exp_dir = resolve_overexp_exp_dir(exp_root, scene)
    ckpt_path = resolve_checkpoint(exp_dir, ckpt_step)
    data_dir = os.path.join(data_root, scene)

    parser = Parser(
        data_dir=data_dir,
        exp_name="over_exp",
        factor=1,
        normalize=True,
        test_every=8,
    )

    tiny_points, tiny_stats = load_tiny_points(ckpt_path, tiny_threshold=tiny_threshold_global)

    scene_dir = os.path.join(out_root, scene)
    overlay_dir = os.path.join(scene_dir, "overlays")
    mask_dir = os.path.join(scene_dir, "masks")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    idx_start = parser.lentrain
    idx_end = parser.lentrain + parser.lenval
    rows: List[Dict[str, object]] = []

    total_proj = 0
    total_on_boundary = 0
    total_expected = 0.0

    for local_i, idx in enumerate(range(idx_start, idx_end)):
        image_path = parser.image_paths[idx]
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        camera_id = parser.camera_ids[idx]
        K = parser.Ks_dict[camera_id].copy()
        camtoworld = parser.camtoworlds[idx]

        pts_xy = project_world_to_image(tiny_points, camtoworld, K)

        h, w = image_bgr.shape[:2]
        x = np.rint(pts_xy[:, 0]).astype(np.int32)
        y = np.rint(pts_xy[:, 1]).astype(np.int32)
        inb = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x = x[inb]
        y = y[inb]
        pts_xy_inb = np.stack([x, y], axis=1).astype(np.float32) if x.size > 0 else np.empty((0, 2), dtype=np.float32)

        sat_mask, boundary_mask = build_saturation_masks(
            image_bgr=image_bgr,
            sat_threshold=sat_threshold,
            boundary_radius=boundary_radius,
        )

        n_proj = int(pts_xy_inb.shape[0])
        if n_proj > 0:
            on_boundary = int(boundary_mask[y, x].sum())
        else:
            on_boundary = 0
        boundary_area_ratio = float(boundary_mask.mean())
        on_boundary_ratio = safe_ratio(on_boundary, n_proj)
        enrich = safe_ratio(on_boundary_ratio, boundary_area_ratio)

        expected = float(n_proj * boundary_area_ratio)
        total_proj += n_proj
        total_on_boundary += on_boundary
        total_expected += expected

        name = parser.image_names[idx]
        title = [
            f"{scene} | {name}",
            f"tiny proj: {n_proj} | on boundary: {on_boundary} ({on_boundary_ratio:.2%})",
            f"boundary area: {boundary_area_ratio:.2%} | enrichment: {enrich:.2f}x",
        ]
        overlay = draw_overlay(
            image_bgr=image_bgr,
            boundary_mask=boundary_mask,
            pts_xy=pts_xy_inb,
            title_lines=title,
        )

        stem = f"val_{local_i:04d}"
        cv2.imwrite(os.path.join(overlay_dir, f"{stem}_overlay.png"), overlay)
        cv2.imwrite(os.path.join(mask_dir, f"{stem}_satmask.png"), (sat_mask.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(mask_dir, f"{stem}_boundary_band.png"), (boundary_mask.astype(np.uint8) * 255))

        p_mask = np.zeros((h, w), dtype=np.uint8)
        if n_proj > 0:
            p_mask[y, x] = 255
        cv2.imwrite(os.path.join(mask_dir, f"{stem}_tiny_points.png"), p_mask)

        rows.append(
            {
                "scene": scene,
                "image_index_in_val": local_i,
                "image_name": name,
                "image_path": image_path,
                "num_tiny_total_3d": tiny_stats["num_tiny"],
                "num_tiny_projected_in_image": n_proj,
                "num_tiny_on_sat_boundary_band": on_boundary,
                "ratio_tiny_on_sat_boundary_band": on_boundary_ratio,
                "sat_boundary_band_area_ratio": boundary_area_ratio,
                "enrichment_vs_area": enrich,
                "tiny_scale_threshold": tiny_stats["tiny_threshold"],
                "tiny_threshold_source": tiny_stats["threshold_source"],
                "sat_threshold": sat_threshold,
                "boundary_radius": boundary_radius,
                "ckpt_path": ckpt_path,
            }
        )

    scene_summary = {
        "scene": scene,
        "exp_dir": exp_dir,
        "ckpt_path": ckpt_path,
        "num_gaussian_total": tiny_stats["num_total"],
        "num_gaussian_tiny": tiny_stats["num_tiny"],
        "ratio_gaussian_tiny": tiny_stats["tiny_ratio"],
        "tiny_scale_threshold": tiny_stats["tiny_threshold"],
        "tiny_threshold_source": tiny_stats["threshold_source"],
        "num_val_images": parser.lenval,
        "num_tiny_projected_all_val": total_proj,
        "num_tiny_on_sat_boundary_all_val": total_on_boundary,
        "ratio_tiny_on_sat_boundary_all_val": safe_ratio(total_on_boundary, total_proj),
        "expected_on_boundary_by_area": total_expected,
        "enrichment_vs_area_all_val": safe_ratio(total_on_boundary, total_expected),
        "sat_threshold": sat_threshold,
        "boundary_radius": boundary_radius,
    }

    with open(os.path.join(scene_dir, "scene_summary.json"), "w") as f:
        json.dump(scene_summary, f, indent=2)
    return rows, scene_summary


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    os.makedirs(args.out_root, exist_ok=True)

    tiny_threshold = args.tiny_scale_threshold
    if tiny_threshold is None:
        tiny_threshold = read_global_tiny_threshold(args.tiny_scale_meta_json)

    all_rows: List[Dict[str, object]] = []
    scene_summaries: List[Dict[str, object]] = []

    for scene in scenes:
        rows, summary = analyze_scene(
            scene=scene,
            exp_root=args.exp_root,
            data_root=args.data_root,
            out_root=args.out_root,
            ckpt_step=args.ckpt_step,
            tiny_threshold_global=tiny_threshold,
            sat_threshold=args.sat_threshold,
            boundary_radius=args.boundary_radius,
        )
        all_rows.extend(rows)
        scene_summaries.append(summary)

    write_csv(os.path.join(args.out_root, "tables", "per_image_projection_stats.csv"), all_rows)
    write_csv(os.path.join(args.out_root, "tables", "scene_summary.csv"), scene_summaries)

    meta = {
        "scenes": scenes,
        "exp_root": args.exp_root,
        "data_root": args.data_root,
        "tiny_scale_threshold_input": args.tiny_scale_threshold,
        "tiny_scale_threshold_meta_json": args.tiny_scale_meta_json,
        "tiny_scale_threshold_effective": tiny_threshold,
        "sat_threshold": args.sat_threshold,
        "boundary_radius": args.boundary_radius,
        "notes": [
            "Overlay color: red=tiny-scale gaussian projection, yellow=saturation boundary band.",
            "enrichment_vs_area > 1 means tiny projections appear near boundary more than random area expectation.",
        ],
    }
    with open(os.path.join(args.out_root, "tables", "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    readme_path = os.path.join(args.out_root, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Tiny-Scale Gaussian Projection on Over-Exposure Images\n\n")
        f.write("- Red points: projections of tiny-scale gaussians.\n")
        f.write("- Yellow band: boundary band of saturated pixels in original over_exp image.\n")
        f.write("- `tables/scene_summary.csv` gives scene-level concentration metrics.\n")
        f.write("- `tables/per_image_projection_stats.csv` gives per-image metrics.\n")

    print(f"[Done] outputs: {args.out_root}")


if __name__ == "__main__":
    main()
