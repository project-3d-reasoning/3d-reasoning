#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.evaluation.eval_vggt_first_frame_points import (  # noqa: E402
    build_unique_sequences,
    load_sequence_inputs,
    transform_points,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay GT point reprojections on RGB images.")
    parser.add_argument(
        "--ann-path",
        type=Path,
        default=REPO_ROOT / "data/train/scanrefer_train_32frames.json",
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/media")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--sequence-offset", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--dedupe-sequences", action="store_true", default=True)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "outputs/visualizations/gt_reprojection_overlay_train_seq0_f4.png",
    )
    return parser.parse_args()


def project_cam_points(points_cam: np.ndarray, intrinsics: np.ndarray):
    z = points_cam[..., 2]
    u = points_cam[..., 0] * intrinsics[0, 0] / z + intrinsics[0, 2]
    v = points_cam[..., 1] * intrinsics[1, 1] / z + intrinsics[1, 2]
    return u, v


def make_overlay(rgb: np.ndarray, u: np.ndarray, v: np.ndarray, depth: np.ndarray, valid: np.ndarray, alpha: float = 0.8):
    overlay = rgb.copy()
    if not valid.any():
        return overlay

    flat_u = u[valid]
    flat_v = v[valid]
    flat_d = depth[valid]

    pixel_valid = np.isfinite(flat_u) & np.isfinite(flat_v) & np.isfinite(flat_d) & (flat_d > 0)
    flat_u = flat_u[pixel_valid]
    flat_v = flat_v[pixel_valid]
    flat_d = flat_d[pixel_valid]
    if flat_u.size == 0:
        return overlay

    ui = np.rint(flat_u).astype(np.int32)
    vi = np.rint(flat_v).astype(np.int32)
    h, w = rgb.shape[:2]
    in_bounds = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui = ui[in_bounds]
    vi = vi[in_bounds]
    flat_d = flat_d[in_bounds]
    if ui.size == 0:
        return overlay

    d_lo, d_hi = np.percentile(flat_d, [5, 95])
    d_hi = max(d_hi, d_lo + 1e-6)
    colors = plt.cm.turbo(np.clip((flat_d - d_lo) / (d_hi - d_lo), 0.0, 1.0))[:, :3]
    overlay[vi, ui] = (1.0 - alpha) * overlay[vi, ui] + alpha * colors
    return overlay


def make_error_overlay(rgb: np.ndarray, err_px: np.ndarray, valid: np.ndarray, vmax: float = 1.0):
    heat = rgb.copy()
    if valid.any():
        cmap = plt.cm.inferno(np.clip(err_px / vmax, 0.0, 1.0))[:, :, :3]
        heat[valid] = 0.3 * heat[valid] + 0.7 * cmap[valid]
    return heat


def format_err_title(name: str, err: np.ndarray) -> str:
    if err.size == 0:
        return f"{name}\nno valid points"
    return f"{name}\nmean={err.mean():.4f}px max={err.max():.4f}px"


def main():
    args = parse_args()
    with args.ann_path.open() as f:
        samples = json.load(f)

    sequences = build_unique_sequences(samples, args.dedupe_sequences)
    if args.sequence_offset > 0:
        sequences = sequences[args.sequence_offset :]
    if not sequences:
        raise ValueError("No sequences available.")

    image_rel_paths = sequences[0][: args.max_frames]
    images, gt_points_current, gt_points_first, valid_masks, _, cam_from_first, gt_intrinsics = load_sequence_inputs(
        image_rel_paths, args.data_root, args.target_size
    )
    rgb_images = images.permute(0, 2, 3, 1).numpy().clip(0.0, 1.0)

    num_frames = len(image_rel_paths)
    fig, axes = plt.subplots(num_frames, 4, figsize=(16, 4 * num_frames), dpi=180)
    if num_frames == 1:
        axes = axes[None, :]

    for frame_idx in range(num_frames):
        rgb = rgb_images[frame_idx]
        valid = valid_masks[frame_idx]
        intrinsics = gt_intrinsics[frame_idx]

        cur_points = gt_points_current[frame_idx]
        cur_u, cur_v = project_cam_points(cur_points, intrinsics)

        first_points = gt_points_first[frame_idx]
        first_back_to_cur = transform_points(first_points, cam_from_first[frame_idx])
        first_u, first_v = project_cam_points(first_back_to_cur, intrinsics)

        grid_u, grid_v = np.meshgrid(
            np.arange(rgb.shape[1], dtype=np.float32),
            np.arange(rgb.shape[0], dtype=np.float32),
        )
        cur_err = np.sqrt((cur_u[valid] - grid_u[valid]) ** 2 + (cur_v[valid] - grid_v[valid]) ** 2)
        first_err_map = np.full(valid.shape, np.nan, dtype=np.float32)
        first_err_map[valid] = np.sqrt((first_u[valid] - grid_u[valid]) ** 2 + (first_v[valid] - grid_v[valid]) ** 2)
        first_err = first_err_map[valid]

        cur_overlay = make_overlay(rgb, cur_u, cur_v, cur_points[..., 2], valid)
        first_overlay = make_overlay(rgb, first_u, first_v, first_back_to_cur[..., 2], valid)
        first_heat = make_error_overlay(rgb, np.nan_to_num(first_err_map, nan=0.0), valid, vmax=0.05)

        axes[frame_idx, 0].imshow(rgb)
        axes[frame_idx, 0].set_title(f"Frame {frame_idx} RGB", fontsize=9)
        axes[frame_idx, 1].imshow(cur_overlay)
        axes[frame_idx, 1].set_title(format_err_title("GT current -> current", cur_err), fontsize=9)
        axes[frame_idx, 2].imshow(first_overlay)
        axes[frame_idx, 2].set_title(format_err_title("GT first -> current", first_err), fontsize=9)
        axes[frame_idx, 3].imshow(first_heat)
        axes[frame_idx, 3].set_title("GT first -> current\npixel error heat", fontsize=9)

        for col in range(4):
            axes[frame_idx, col].set_xticks([])
            axes[frame_idx, col].set_yticks([])

    scene_name = Path(image_rel_paths[0]).parent.name
    fig.suptitle(
        f"GT point reprojection overlays\nann={args.ann_path.name} scene={scene_name} frames={num_frames}",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, bbox_inches="tight")
    print(f"Saved overlay visualization to {args.output_path}")


if __name__ == "__main__":
    main()
