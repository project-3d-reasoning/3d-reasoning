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
    load_model,
    load_sequence_inputs,
    predict_world_points,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize VGGT depth alignment against ScanRefer GT points.")
    parser.add_argument(
        "--ann-path",
        type=Path,
        default=REPO_ROOT / "data/evaluation/scanrefer_first_frame/scanrefer_val_32frames.json",
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/media")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/data7t-root/huggingface/hub/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9"
        ),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--sequence-offset", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--dedupe-sequences", action="store_true", default=True)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "outputs/visualizations/vggt_depth_alignment_scene0011_00_seq0.png",
    )
    return parser.parse_args()


def sample_flat_indices(valid_mask: np.ndarray, max_points: int) -> np.ndarray:
    flat_valid = np.flatnonzero(valid_mask.reshape(-1))
    if flat_valid.size <= max_points:
        return flat_valid
    return flat_valid[np.linspace(0, flat_valid.size - 1, num=max_points, dtype=int)]


def compute_plot_limits(point_sets, valid_sets):
    xs = []
    zs = []
    for points, valid in zip(point_sets, valid_sets):
        if valid.any():
            xs.append(points[..., 0][valid])
            zs.append(points[..., 2][valid])
    if not xs:
        return (-1.0, 1.0), (-1.0, 1.0)

    x = np.concatenate(xs)
    z = np.concatenate(zs)
    x_lo, x_hi = np.percentile(x, [1, 99])
    z_lo, z_hi = np.percentile(z, [1, 99])
    x_pad = max((x_hi - x_lo) * 0.05, 0.1)
    z_pad = max((z_hi - z_lo) * 0.05, 0.1)
    return (float(x_lo - x_pad), float(x_hi + x_pad)), (float(z_lo - z_pad), float(z_hi + z_pad))


def plot_topdown(ax, points: np.ndarray, valid: np.ndarray, colors: np.ndarray, max_points: int, xlim, zlim, title: str):
    flat_points = points.reshape(-1, 3)
    flat_colors = colors.reshape(-1, 3)
    idx = sample_flat_indices(valid, max_points)
    if idx.size > 0:
        ax.scatter(flat_points[idx, 0], flat_points[idx, 2], c=flat_colors[idx], s=0.4, linewidths=0, alpha=0.9)
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def metric_title(prefix: str, errors: np.ndarray) -> str:
    if errors.size == 0:
        return f"{prefix}\nno valid points"
    mean_err = float(errors.mean())
    acc_05 = float((errors <= 0.5).mean())
    return f"{prefix}\nmean={mean_err:.2f}m  acc@0.5={acc_05:.1%}"


def main():
    args = parse_args()

    with args.ann_path.open() as f:
        samples = json.load(f)

    sequences = build_unique_sequences(samples, args.dedupe_sequences)
    if args.sequence_offset > 0:
        sequences = sequences[args.sequence_offset :]
    if not sequences:
        raise ValueError("No sequences available for visualization.")

    image_rel_paths = sequences[0][: args.max_frames]
    model = load_model(args.model_path, args.device, "depth_cam")

    images, _, gt_points_first, valid_masks, _, cam_from_first, gt_intrinsics = load_sequence_inputs(
        image_rel_paths, args.data_root, args.target_size
    )
    pred_depth_cam = predict_world_points(
        model,
        images,
        args.device,
        "depth_cam",
        gt_extrinsics=cam_from_first,
        gt_intrinsics=gt_intrinsics,
    )
    pred_depth_gtcam = predict_world_points(
        model,
        images,
        args.device,
        "depth_gtcam",
        gt_extrinsics=cam_from_first,
        gt_intrinsics=gt_intrinsics,
    )

    num_frames = gt_points_first.shape[0]
    rgb_images = images.permute(0, 2, 3, 1).numpy().clip(0.0, 1.0)

    all_valid_cam = valid_masks & np.isfinite(pred_depth_cam).all(axis=-1) & np.isfinite(gt_points_first).all(axis=-1)
    all_valid_gtcam = valid_masks & np.isfinite(pred_depth_gtcam).all(axis=-1) & np.isfinite(gt_points_first).all(axis=-1)

    xlim, zlim = compute_plot_limits(
        [gt_points_first, pred_depth_cam, pred_depth_gtcam],
        [valid_masks, all_valid_cam, all_valid_gtcam],
    )

    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="black")
    fig, axes = plt.subplots(num_frames, 6, figsize=(18, 4 * num_frames), dpi=180)
    if num_frames == 1:
        axes = axes[None, :]

    scene_name = Path(image_rel_paths[0]).parent.name

    for frame_idx in range(num_frames):
        rgb = rgb_images[frame_idx]
        gt = gt_points_first[frame_idx]
        pred_cam = pred_depth_cam[frame_idx]
        pred_gtcam = pred_depth_gtcam[frame_idx]
        valid_gt = valid_masks[frame_idx]
        valid_cam = all_valid_cam[frame_idx]
        valid_gtcam = all_valid_gtcam[frame_idx]

        err_cam = np.full(valid_gt.shape, np.nan, dtype=np.float32)
        err_gtcam = np.full(valid_gt.shape, np.nan, dtype=np.float32)
        if valid_cam.any():
            err_cam[valid_cam] = np.linalg.norm(pred_cam[valid_cam] - gt[valid_cam], axis=-1)
        if valid_gtcam.any():
            err_gtcam[valid_gtcam] = np.linalg.norm(pred_gtcam[valid_gtcam] - gt[valid_gtcam], axis=-1)

        axes[frame_idx, 0].imshow(rgb)
        axes[frame_idx, 0].set_xticks([])
        axes[frame_idx, 0].set_yticks([])
        axes[frame_idx, 0].set_title(f"Frame {frame_idx} RGB", fontsize=9)

        plot_topdown(
            axes[frame_idx, 1],
            gt,
            valid_gt,
            rgb,
            args.max_points,
            xlim,
            zlim,
            "GT first-frame",
        )
        plot_topdown(
            axes[frame_idx, 2],
            pred_cam,
            valid_cam,
            rgb,
            args.max_points,
            xlim,
            zlim,
            metric_title("Pred depth + pred cam", err_cam[valid_cam]),
        )
        plot_topdown(
            axes[frame_idx, 3],
            pred_gtcam,
            valid_gtcam,
            rgb,
            args.max_points,
            xlim,
            zlim,
            metric_title("Pred depth + GT cam", err_gtcam[valid_gtcam]),
        )

        im_cam = axes[frame_idx, 4].imshow(err_cam, cmap=cmap, vmin=0.0, vmax=1.0)
        axes[frame_idx, 4].set_xticks([])
        axes[frame_idx, 4].set_yticks([])
        axes[frame_idx, 4].set_title("Error map\npred depth + pred cam", fontsize=9)

        im_gtcam = axes[frame_idx, 5].imshow(err_gtcam, cmap=cmap, vmin=0.0, vmax=1.0)
        axes[frame_idx, 5].set_xticks([])
        axes[frame_idx, 5].set_yticks([])
        axes[frame_idx, 5].set_title("Error map\npred depth + GT cam", fontsize=9)

    fig.colorbar(im_cam, ax=axes[:, 4], fraction=0.02, pad=0.01, label="L2 error (m)")
    fig.colorbar(im_gtcam, ax=axes[:, 5], fraction=0.02, pad=0.01, label="L2 error (m)")
    fig.suptitle(
        f"VGGT depth alignment vs GT first-frame points\nscene={scene_name}  frames={num_frames}",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, bbox_inches="tight")
    print(f"Saved visualization to {args.output_path}")


if __name__ == "__main__":
    main()
