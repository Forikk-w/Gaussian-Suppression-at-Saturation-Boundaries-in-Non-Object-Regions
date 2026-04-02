import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from datasets.colmap import Dataset, Parser
from analyze_exp7_importance_pruning import (
    load_json,
    load_splats,
    render_enhanced,
    resolve_scene_paths,
)


SCENE_PRUNE_RATIOS = {
    "bike": 0.7,
    "buu": 0.9,
    "chair": 0.9,
    "sofa": 0.6,
}
MASKED_OPACITY_LOGIT = -100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render baseline vs critical-pruned comparisons.")
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
        default="/data2/wd/workspace/experiments/luminance_gs_analysis/exp9_pruning_comparisons",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_importance_scores(csv_path: Path) -> Dict[str, np.ndarray]:
    per_scene: Dict[str, Dict[int, float]] = {scene: {} for scene in SCENE_PRUNE_RATIOS}
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
        nonzero_sorted = nonzero_ids[np.argsort(-importance_scores[nonzero_ids], kind="stable")]
    else:
        nonzero_sorted = np.empty(0, dtype=np.int64)

    keep_nonzero = min(keep_count, nonzero_sorted.size)
    if keep_nonzero > 0:
        keep_mask[nonzero_sorted[:keep_nonzero]] = True

    remaining = keep_count - keep_nonzero
    if remaining > 0 and zero_ids.size > 0:
        keep_mask[zero_ids[:remaining]] = True

    return keep_mask


def render_image(splats: Dict[str, torch.Tensor], cfg: Dict, data: Dict[str, torch.Tensor], device: torch.device) -> np.ndarray:
    gt_h, gt_w = data["image"].shape[:2]
    render = render_enhanced(
        splats,
        data["camtoworld"].to(device),
        data["K"].to(device),
        gt_w,
        gt_h,
        cfg,
    )
    image = (render.squeeze(0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return image


def build_masked_opacities(opacities: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    masked = opacities.clone()
    masked[~keep_mask] = MASKED_OPACITY_LOGIT
    return masked


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    labeled = image.copy()
    overlay = labeled.copy()
    cv2.rectangle(overlay, (8, 8), (220, 42), (0, 0, 0), thickness=-1)
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


def make_summary_grid(scene_first_images: Dict[str, np.ndarray]) -> np.ndarray:
    ordered_scenes = ["bike", "buu", "chair", "sofa"]
    tiles = []
    for scene in ordered_scenes:
        image = scene_first_images[scene].copy()
        overlay = image.copy()
        cv2.rectangle(overlay, (8, 8), (180, 42), (0, 0, 0), thickness=-1)
        cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)
        cv2.putText(
            image,
            scene,
            (18, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(image)

    top = np.concatenate([tiles[0], np.full((tiles[0].shape[0], 16, 3), 255, np.uint8), tiles[1]], axis=1)
    bottom = np.concatenate([tiles[2], np.full((tiles[2].shape[0], 16, 3), 255, np.uint8), tiles[3]], axis=1)
    width = max(top.shape[1], bottom.shape[1])
    if top.shape[1] < width:
        pad = np.full((top.shape[0], width - top.shape[1], 3), 255, dtype=np.uint8)
        top = np.concatenate([top, pad], axis=1)
    if bottom.shape[1] < width:
        pad = np.full((bottom.shape[0], width - bottom.shape[1], 3), 255, dtype=np.uint8)
        bottom = np.concatenate([bottom, pad], axis=1)
    spacer = np.full((20, width, 3), 255, dtype=np.uint8)
    return np.concatenate([top, spacer, bottom], axis=0)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    experiment_root = Path(args.experiment_root)
    importance_csv = Path(args.importance_csv)
    analysis_root = Path(args.analysis_root)
    analysis_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    importance_scores = load_importance_scores(importance_csv)
    meta = {
        "scene_prune_ratios": SCENE_PRUNE_RATIOS,
        "importance_csv": str(importance_csv),
        "mask_mode": "keep gaussian count fixed; set pruned opacity logits to -100.0",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    first_compare_images: Dict[str, np.ndarray] = {}

    for scene, prune_ratio in SCENE_PRUNE_RATIOS.items():
        tic = time.time()
        scene_dir = analysis_root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        data_dir, exp_dir, ckpt_path = resolve_scene_paths(repo_root, experiment_root, scene)
        cfg = load_json(exp_dir / "cfg.json")
        parser = Parser(str(data_dir), exp_name="over_exp", factor=1, normalize=True, test_every=8)
        valset = Dataset(parser, split="val")
        splats = load_splats(ckpt_path, device)

        keep_mask_np = build_keep_mask(importance_scores[scene], prune_ratio)
        keep_mask = torch.from_numpy(keep_mask_np).to(device=device, dtype=torch.bool)
        original_opacities = splats["opacities"]
        masked_opacities = build_masked_opacities(original_opacities, keep_mask)

        for idx in range(len(valset)):
            data = valset[idx]
            splats["opacities"] = original_opacities
            baseline = render_image(splats, cfg, data, device)
            splats["opacities"] = masked_opacities
            pruned = render_image(splats, cfg, data, device)
            splats["opacities"] = original_opacities
            compare = make_compare_image(baseline, pruned, prune_ratio)
            out_path = scene_dir / f"{scene}_val{idx:02d}_compare.png"
            imageio.imwrite(out_path, compare)
            if idx == 0:
                first_compare_images[scene] = compare

        splats["opacities"] = original_opacities

        print(
            f"[exp9] {scene}: val_views={len(valset)}, prune_ratio={prune_ratio}, "
            f"kept={int(keep_mask_np.sum())}, elapsed {time.time() - tic:.1f}s"
        )
        torch.cuda.empty_cache()

    summary_image = make_summary_grid(first_compare_images)
    imageio.imwrite(analysis_root / "all_scenes_compare.png", summary_image)
    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with (analysis_root / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
