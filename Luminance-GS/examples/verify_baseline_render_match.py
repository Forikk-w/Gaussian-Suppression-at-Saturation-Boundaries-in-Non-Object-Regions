import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from datasets.colmap import Dataset, Parser
from analyze_exp7_importance_pruning import load_json, load_splats, render_enhanced, resolve_scene_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify baseline render matches saved eval render.")
    parser.add_argument("--scene", type=str, default="bike")
    parser.add_argument("--val-idx", type=int, default=0)
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
        "--output-dir",
        type=str,
        default="/data2/wd/workspace/experiments/luminance_gs_analysis/exp9_baseline_check",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    experiment_root = Path(args.experiment_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    data_dir, exp_dir, ckpt_path = resolve_scene_paths(repo_root, experiment_root, args.scene)
    cfg = load_json(exp_dir / "cfg.json")
    parser = Parser(str(data_dir), exp_name="over_exp", factor=1, normalize=True, test_every=8)
    valset = Dataset(parser, split="val")
    splats = load_splats(ckpt_path, device)
    data = valset[args.val_idx]

    height, width = data["image"].shape[:2]
    render = render_enhanced(
        splats,
        data["camtoworld"].to(device),
        data["K"].to(device),
        width,
        height,
        cfg,
    )
    baseline = (render.squeeze(0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    ref_path = exp_dir / "renders" / f"val_{args.val_idx:04d}_enh.png"
    reference = imageio.imread(ref_path)[..., :3]

    diff = baseline.astype(np.int16) - reference.astype(np.int16)
    absdiff = np.abs(diff)
    out_path = output_dir / f"{args.scene}_val{args.val_idx:02d}_baseline_render.png"
    imageio.imwrite(out_path, baseline)

    stats = {
        "scene": args.scene,
        "val_idx": args.val_idx,
        "baseline_path": str(out_path),
        "reference_path": str(ref_path),
        "render_shape": list(baseline.shape),
        "reference_shape": list(reference.shape),
        "equal": bool(np.array_equal(baseline, reference)),
        "max_abs_diff": int(absdiff.max()),
        "mean_abs_diff": float(absdiff.mean()),
        "num_diff_pixels": int(np.any(absdiff > 0, axis=-1).sum()),
    }
    with (output_dir / f"{args.scene}_val{args.val_idx:02d}_baseline_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
