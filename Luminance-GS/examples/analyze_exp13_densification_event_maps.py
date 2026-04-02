import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np

from datasets.colmap import Dataset, Parser


SCENES = ["bike", "buu", "chair", "sofa"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze densification event maps with boundary/interior/normal masks.")
    parser.add_argument(
        "--repo-root",
        type=str,
        default="/data2/wd/workspace/repos/Luminance-GS/Luminance-GS",
    )
    parser.add_argument(
        "--analysis-root",
        type=str,
        default="/data2/wd/workspace/experiments/luminance_gs_analysis/exp13_densification_event_maps",
    )
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--sat-pixel-threshold", type=int, default=250)
    parser.add_argument("--sat-dilate-radius", type=int, default=3)
    parser.add_argument("--sat-mask-mean-threshold", type=float, default=0.3)
    return parser.parse_args()


def build_parser(data_dir: Path, exp_name: str) -> Parser:
    return Parser(str(data_dir), exp_name=exp_name, factor=1, normalize=True, test_every=8)


def load_event_bundle(event_dir: Path) -> Dict[str, object]:
    return {
        "total": np.load(event_dir / "event_map_total.npy"),
        "clone": np.load(event_dir / "event_map_clone.npy"),
        "split": np.load(event_dir / "event_map_split.npy"),
        "meta": json.load((event_dir / "event_map_meta.json").open("r")),
    }


def build_masks_for_image(image: np.ndarray, threshold: int, dilate_radius: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_rgb = image.max(axis=-1)
    sat_raw = max_rgb >= threshold
    kernel = np.ones((2 * dilate_radius + 1, 2 * dilate_radius + 1), dtype=np.uint8)
    sat_dilated = cv2.dilate(sat_raw.astype(np.uint8), kernel, iterations=1).astype(bool)
    sat_boundary = np.logical_xor(sat_dilated, sat_raw)
    sat_interior = sat_raw
    normal = ~sat_dilated
    return sat_boundary, sat_interior, normal


def build_representative_masks(
    images: List[np.ndarray],
    threshold: int,
    dilate_radius: int,
    mean_threshold: float,
) -> Dict[str, np.ndarray]:
    per_image = [build_masks_for_image(image, threshold, dilate_radius) for image in images]
    boundary_mean = np.mean(np.stack([item[0].astype(np.float32) for item in per_image], axis=0), axis=0)
    interior_mean = np.mean(np.stack([item[1].astype(np.float32) for item in per_image], axis=0), axis=0)
    normal_mean = np.mean(np.stack([item[2].astype(np.float32) for item in per_image], axis=0), axis=0)

    mean_stack = np.stack([boundary_mean, interior_mean, normal_mean], axis=0)
    max_mean = mean_stack.max(axis=0)
    labels = mean_stack.argmax(axis=0)

    boundary = (labels == 0) & (max_mean >= mean_threshold)
    interior = (labels == 1) & (max_mean >= mean_threshold)
    normal = (labels == 2) & (max_mean >= mean_threshold)

    # Threshold 0.3 guarantees at least one class per pixel in practice because the three class means sum to 1.
    fallback = ~(boundary | interior | normal)
    if np.any(fallback):
        normal = normal | fallback

    return {
        "boundary": boundary,
        "interior": interior,
        "normal": normal,
        "boundary_mean": boundary_mean,
        "interior_mean": interior_mean,
        "normal_mean": normal_mean,
    }


def safe_region_ratio(event_map: np.ndarray, region_mask: np.ndarray, total_events: int) -> float:
    if total_events <= 0:
        return 0.0
    return float(event_map[region_mask].sum() / float(total_events))


def safe_mean(array: np.ndarray, region_mask: np.ndarray) -> Optional[float]:
    if not np.any(region_mask):
        return None
    return float(array[region_mask].mean())


def safe_enrichment(over_ratio: float, low_ratio: float) -> Optional[float]:
    if low_ratio <= 0.0:
        return None
    return float(over_ratio / low_ratio)


def overlay_region_legend(image: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    h = canvas.shape[0]
    y0 = h - 86
    overlay = canvas.copy()
    cv2.rectangle(overlay, (8, y0), (250, h - 8), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
    cv2.rectangle(canvas, (16, y0 + 12), (28, y0 + 24), (255, 140, 0), thickness=-1)
    cv2.putText(canvas, "interior", (36, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (16, y0 + 32), (28, y0 + 44), (220, 40, 40), thickness=-1)
    cv2.putText(canvas, "boundary", (36, y0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (16, y0 + 52), (28, y0 + 64), (60, 200, 60), thickness=-1)
    cv2.putText(canvas, "normal edge", (36, y0 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def overlay_regions(image: np.ndarray, boundary: np.ndarray, interior: np.ndarray, normal: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    overlay = canvas.copy()
    overlay[interior] = np.array([255, 140, 0], dtype=np.uint8)
    overlay[boundary] = np.array([220, 40, 40], dtype=np.uint8)
    canvas = cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0.0)

    sat_dilated = boundary | interior
    interior_contours, _ = cv2.findContours(interior.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_contours, _ = cv2.findContours(boundary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dilated_contours, _ = cv2.findContours(sat_dilated.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, interior_contours, -1, (255, 180, 40), 1, lineType=cv2.LINE_AA)
    cv2.drawContours(canvas, boundary_contours, -1, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.drawContours(canvas, dilated_contours, -1, (60, 200, 60), 1, lineType=cv2.LINE_AA)
    return overlay_region_legend(canvas)


def add_text(image: np.ndarray, lines: List[str], box_width: int = 360) -> np.ndarray:
    canvas = image.copy()
    box_h = 24 + 20 * len(lines)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (8, 8), (box_width, box_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
    for idx, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (16, 26 + idx * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def make_event_heatmap(event_map: np.ndarray, scale: float) -> np.ndarray:
    values = np.log1p(event_map.astype(np.float32))
    scale = max(scale, 1e-6)
    normalized = np.clip(values / scale, 0.0, 1.0)
    heat = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def make_delta_heatmap(delta_map: np.ndarray, scale: float) -> np.ndarray:
    scale = max(scale, 1e-6)
    normalized = np.clip((delta_map / scale + 1.0) * 0.5, 0.0, 1.0)
    cmap = cv2.COLORMAP_COOLWARM if hasattr(cv2, "COLORMAP_COOLWARM") else cv2.COLORMAP_JET
    heat = cv2.applyColorMap((normalized * 255).astype(np.uint8), cmap)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def compose_five_panel(
    over_image: np.ndarray,
    over_map: np.ndarray,
    low_map: np.ndarray,
    delta_map: np.ndarray,
    boundary: np.ndarray,
    interior: np.ndarray,
    normal: np.ndarray,
    stats: Dict[str, object],
) -> np.ndarray:
    event_scale = float(
        max(
            np.percentile(np.log1p(over_map.astype(np.float32)), 99.5),
            np.percentile(np.log1p(low_map.astype(np.float32)), 99.5),
        )
    )
    delta_scale = float(np.percentile(np.abs(delta_map.astype(np.float32)), 99.5))

    panel1 = add_text(
        overlay_regions(over_image, boundary, interior, normal),
        [
            f"{stats['scene']} over_exp train0",
            f"boundary={stats['boundary_area_ratio']:.3f}",
            f"interior={stats['interior_area_ratio']:.3f}",
            f"normal={stats['normal_area_ratio']:.3f}",
        ],
    )
    panel2 = add_text(
        make_event_heatmap(over_map, event_scale),
        [
            f"over total={stats['total_events_over']}",
            f"boundary={stats['over_boundary_event_ratio']:.3f}",
            f"interior={stats['over_interior_event_ratio']:.3f}",
            f"normal={stats['over_normal_event_ratio']:.3f}",
        ],
    )
    panel3 = add_text(
        make_event_heatmap(low_map, event_scale),
        [
            f"low total={stats['total_events_low']}",
            f"boundary={stats['low_boundary_event_ratio']:.3f}",
            f"interior={stats['low_interior_event_ratio']:.3f}",
            f"normal={stats['low_normal_event_ratio']:.3f}",
        ],
    )
    panel4 = add_text(
        make_delta_heatmap(delta_map, delta_scale),
        [
            f"delta boundary={stats['mean_delta_boundary']:.2f}" if stats["mean_delta_boundary"] is not None else "delta boundary=null",
            f"delta interior={stats['mean_delta_interior']:.2f}" if stats["mean_delta_interior"] is not None else "delta interior=null",
            f"delta normal={stats['mean_delta_normal']:.2f}" if stats["mean_delta_normal"] is not None else "delta normal=null",
        ],
    )
    panel5 = add_text(
        overlay_regions(make_delta_heatmap(delta_map, delta_scale), boundary, interior, normal),
        [
            f"enrich boundary={stats['enrichment_boundary']:.3f}" if stats["enrichment_boundary"] is not None else "enrich boundary=null",
            f"enrich interior={stats['enrichment_interior']:.3f}" if stats["enrichment_interior"] is not None else "enrich interior=null",
            f"enrich normal={stats['enrichment_normal']:.3f}" if stats["enrichment_normal"] is not None else "enrich normal=null",
        ],
    )

    h = panel1.shape[0]
    sep = np.full((h, 8, 3), 255, dtype=np.uint8)
    line = np.full((h, 2, 3), 180, dtype=np.uint8)
    return np.concatenate(
        [panel1, sep, line, sep, panel2, sep, line, sep, panel3, sep, line, sep, panel4, sep, line, sep, panel5],
        axis=1,
    )


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "scene",
        "boundary_area_ratio",
        "interior_area_ratio",
        "normal_area_ratio",
        "over_boundary_event_ratio",
        "over_interior_event_ratio",
        "over_normal_event_ratio",
        "low_boundary_event_ratio",
        "low_interior_event_ratio",
        "low_normal_event_ratio",
        "enrichment_boundary",
        "enrichment_interior",
        "enrichment_normal",
        "mean_delta_boundary",
        "mean_delta_interior",
        "mean_delta_normal",
        "total_events_over",
        "total_events_low",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    analysis_root = Path(args.analysis_root)
    event_root = analysis_root / "densification_event_maps"

    rows: List[Dict[str, object]] = []
    panels = []
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "analysis_root": str(analysis_root),
        "version": "v2_boundary_interior_normal",
        "sat_pixel_threshold": args.sat_pixel_threshold,
        "sat_dilate_radius": args.sat_dilate_radius,
        "sat_mask_mean_threshold": args.sat_mask_mean_threshold,
        "region_definition": {
            "boundary": "sat_dilated XOR sat_raw",
            "interior": "sat_raw",
            "normal": "NOT sat_dilated",
            "representative_mask_rule": "per-image three-class masks are averaged, thresholded at 0.3, then resolved to a disjoint partition by argmax over class means",
        },
        "scenes": [],
    }

    for scene in args.scenes:
        data_dir = repo_root / "data" / "LOM_full" / scene
        train_over = Dataset(build_parser(data_dir, "over_exp"), split="train")

        over_bundle = load_event_bundle(event_root / f"{scene}_over_exp")
        low_bundle = load_event_bundle(event_root / f"{scene}_low")
        over_map = over_bundle["total"].astype(np.float32)
        low_map = low_bundle["total"].astype(np.float32)
        delta_map = over_map - low_map

        over_images = [train_over[i]["image"].numpy().astype(np.uint8) for i in range(len(train_over))]
        masks = build_representative_masks(
            over_images,
            threshold=args.sat_pixel_threshold,
            dilate_radius=args.sat_dilate_radius,
            mean_threshold=args.sat_mask_mean_threshold,
        )

        boundary = masks["boundary"]
        interior = masks["interior"]
        normal = masks["normal"]

        boundary_area_ratio = float(boundary.mean())
        interior_area_ratio = float(interior.mean())
        normal_area_ratio = float(normal.mean())

        total_events_over = int(over_bundle["meta"]["total_events"])
        total_events_low = int(low_bundle["meta"]["total_events"])

        over_boundary_event_ratio = safe_region_ratio(over_map, boundary, total_events_over)
        over_interior_event_ratio = safe_region_ratio(over_map, interior, total_events_over)
        over_normal_event_ratio = safe_region_ratio(over_map, normal, total_events_over)
        low_boundary_event_ratio = safe_region_ratio(low_map, boundary, total_events_low)
        low_interior_event_ratio = safe_region_ratio(low_map, interior, total_events_low)
        low_normal_event_ratio = safe_region_ratio(low_map, normal, total_events_low)

        enrichment_boundary = safe_enrichment(over_boundary_event_ratio, low_boundary_event_ratio)
        enrichment_interior = safe_enrichment(over_interior_event_ratio, low_interior_event_ratio)
        enrichment_normal = safe_enrichment(over_normal_event_ratio, low_normal_event_ratio)

        mean_delta_boundary = safe_mean(delta_map, boundary)
        mean_delta_interior = safe_mean(delta_map, interior)
        mean_delta_normal = safe_mean(delta_map, normal)

        row = {
            "scene": scene,
            "boundary_area_ratio": boundary_area_ratio,
            "interior_area_ratio": interior_area_ratio,
            "normal_area_ratio": normal_area_ratio,
            "over_boundary_event_ratio": over_boundary_event_ratio,
            "over_interior_event_ratio": over_interior_event_ratio,
            "over_normal_event_ratio": over_normal_event_ratio,
            "low_boundary_event_ratio": low_boundary_event_ratio,
            "low_interior_event_ratio": low_interior_event_ratio,
            "low_normal_event_ratio": low_normal_event_ratio,
            "enrichment_boundary": enrichment_boundary,
            "enrichment_interior": enrichment_interior,
            "enrichment_normal": enrichment_normal,
            "mean_delta_boundary": mean_delta_boundary,
            "mean_delta_interior": mean_delta_interior,
            "mean_delta_normal": mean_delta_normal,
            "total_events_over": total_events_over,
            "total_events_low": total_events_low,
        }
        rows.append(row)

        panel = compose_five_panel(
            over_image=over_images[0],
            over_map=over_map,
            low_map=low_map,
            delta_map=delta_map,
            boundary=boundary,
            interior=interior,
            normal=normal,
            stats=row,
        )
        imageio.imwrite(analysis_root / f"{scene}_densification_compare_v2.png", panel)
        panels.append(panel)

        report["scenes"].append(
            {
                "scene": scene,
                "boundary_area_ratio": boundary_area_ratio,
                "interior_area_ratio": interior_area_ratio,
                "normal_area_ratio": normal_area_ratio,
                "over_boundary_event_ratio": over_boundary_event_ratio,
                "over_interior_event_ratio": over_interior_event_ratio,
                "over_normal_event_ratio": over_normal_event_ratio,
                "low_boundary_event_ratio": low_boundary_event_ratio,
                "low_interior_event_ratio": low_interior_event_ratio,
                "low_normal_event_ratio": low_normal_event_ratio,
                "enrichment_boundary": enrichment_boundary,
                "enrichment_interior": enrichment_interior,
                "enrichment_normal": enrichment_normal,
                "mean_delta_boundary": mean_delta_boundary,
                "mean_delta_interior": mean_delta_interior,
                "mean_delta_normal": mean_delta_normal,
                "mask_pixel_counts": {
                    "boundary": int(boundary.sum()),
                    "interior": int(interior.sum()),
                    "normal": int(normal.sum()),
                },
                "mask_mean_ranges": {
                    "boundary_max": float(masks["boundary_mean"].max()),
                    "interior_max": float(masks["interior_mean"].max()),
                    "normal_max": float(masks["normal_mean"].max()),
                },
                "conditions": {
                    "over_exp": over_bundle["meta"],
                    "low": low_bundle["meta"],
                },
                "figure_v2": f"{scene}_densification_compare_v2.png",
            }
        )

        print(
            f"[exp13-v2] {scene}: "
            f"over_boundary={over_boundary_event_ratio:.4f}, low_boundary={low_boundary_event_ratio:.4f}, "
            f"enrich_boundary={enrichment_boundary if enrichment_boundary is not None else 'null'}"
        )

    if panels:
        max_h = max(panel.shape[0] for panel in panels)
        max_w = max(panel.shape[1] for panel in panels)
        padded = []
        for panel in panels:
            canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
            canvas[: panel.shape[0], : panel.shape[1]] = panel
            padded.append(canvas)
        top = np.concatenate(padded[:2], axis=1)
        bottom = np.concatenate(padded[2:], axis=1)
        imageio.imwrite(analysis_root / "all_scenes_overview_v2.png", np.concatenate([top, bottom], axis=0))

    write_csv(analysis_root / "densification_analysis_v2.csv", rows)
    with (analysis_root / "densification_report_v2.json").open("w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
