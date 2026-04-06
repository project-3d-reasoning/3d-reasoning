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
)


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-view GT depth/pose consistency check.")
    parser.add_argument(
        "--ann-path",
        type=Path,
        default=REPO_ROOT / "data/train/scanrefer_train_32frames.json",
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/media")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--sequence-offset", type=int, default=0)
    parser.add_argument("--source-frame", type=int, default=0)
    parser.add_argument("--target-frame", type=int, default=1)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "outputs/visualizations/gt_cross_view_consistency_train_seq0_f0_to_f1.png",
    )
    return parser.parse_args()


def transform_points_h(points: np.ndarray, extrinsic_3x4: np.ndarray) -> np.ndarray:
    ones = np.ones((*points.shape[:-1], 1), dtype=np.float32)
    points_h = np.concatenate([points, ones], axis=-1)
    return points_h @ extrinsic_3x4.T


def render_depth_from_points(points_cam: np.ndarray, intrinsics: np.ndarray, image_hw):
    h, w = image_hw
    z = points_cam[..., 2]
    valid = np.isfinite(points_cam).all(axis=-1) & (z > 1e-6)
    pts = points_cam[valid]
    if pts.size == 0:
        return np.full((h, w), np.nan, dtype=np.float32), np.zeros((h, w), dtype=bool)

    u = pts[:, 0] * intrinsics[0, 0] / pts[:, 2] + intrinsics[0, 2]
    v = pts[:, 1] * intrinsics[1, 1] / pts[:, 2] + intrinsics[1, 2]

    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    in_bounds = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui = ui[in_bounds]
    vi = vi[in_bounds]
    z = pts[in_bounds, 2]
    if ui.size == 0:
        return np.full((h, w), np.nan, dtype=np.float32), np.zeros((h, w), dtype=bool)

    order = np.argsort(z)
    ui = ui[order]
    vi = vi[order]
    z = z[order]

    rendered = np.full((h, w), np.nan, dtype=np.float32)
    filled = np.zeros((h, w), dtype=bool)
    rendered[vi, ui] = z
    filled[vi, ui] = True
    return rendered, filled


def make_depth_vis(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vis = np.zeros(depth.shape + (3,), dtype=np.float32)
    if valid.any():
        vals = depth[valid]
        lo, hi = np.percentile(vals, [2, 98])
        hi = max(hi, lo + 1e-6)
        colors = plt.cm.viridis(np.clip((depth - lo) / (hi - lo), 0.0, 1.0))[:, :, :3]
        vis[valid] = colors[valid]
    return vis


def make_error_overlay(rgb: np.ndarray, err_map: np.ndarray, valid: np.ndarray, vmax: float = 0.2) -> np.ndarray:
    out = rgb.copy()
    colors = plt.cm.inferno(np.clip(err_map / vmax, 0.0, 1.0))[:, :, :3]
    out[valid] = 0.3 * out[valid] + 0.7 * colors[valid]
    return out


def main():
    args = parse_args()

    with args.ann_path.open() as f:
        samples = json.load(f)
    sequences = build_unique_sequences(samples, True)
    if args.sequence_offset > 0:
        sequences = sequences[args.sequence_offset :]
    if not sequences:
        raise ValueError("No sequences available.")

    image_rel_paths = sequences[0][: max(args.source_frame, args.target_frame) + 1]
    images, gt_points_current, gt_points_first, valid_masks, _, cam_from_first, gt_intrinsics = load_sequence_inputs(
        image_rel_paths, args.data_root, args.target_size
    )

    src = args.source_frame
    tgt = args.target_frame
    rgb_tgt = images[tgt].permute(1, 2, 0).numpy().clip(0.0, 1.0)
    tgt_depth = gt_points_current[tgt][..., 2]
    tgt_valid = valid_masks[tgt]

    src_points_first = gt_points_first[src]
    src_valid = valid_masks[src]
    src_points_tgt_cam = transform_points_h(src_points_first[src_valid], cam_from_first[tgt])
    rendered_depth, rendered_valid = render_depth_from_points(src_points_tgt_cam, gt_intrinsics[tgt], tgt_depth.shape)

    overlap_valid = rendered_valid & tgt_valid & np.isfinite(rendered_depth)
    depth_err = np.abs(rendered_depth[overlap_valid] - tgt_depth[overlap_valid])
    err_map = np.zeros_like(tgt_depth, dtype=np.float32)
    err_map[overlap_valid] = depth_err

    overlap_ratio = float(overlap_valid.sum()) / max(int(tgt_valid.sum()), 1)
    summary = {
        "source_frame": src,
        "target_frame": tgt,
        "overlap_ratio_vs_target_valid": overlap_ratio,
        "depth_abs_mean_m": float(depth_err.mean()) if depth_err.size else None,
        "depth_abs_median_m": float(np.median(depth_err)) if depth_err.size else None,
        "depth_acc@0.05m": float((depth_err <= 0.05).mean()) if depth_err.size else None,
        "depth_acc@0.10m": float((depth_err <= 0.10).mean()) if depth_err.size else None,
        "depth_acc@0.20m": float((depth_err <= 0.20).mean()) if depth_err.size else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=180)
    axes[0].imshow(rgb_tgt)
    axes[0].set_title(f"Target RGB\nframe {tgt}")

    axes[1].imshow(make_depth_vis(tgt_depth, tgt_valid))
    axes[1].set_title("Target GT depth")

    axes[2].imshow(make_depth_vis(rendered_depth, rendered_valid))
    axes[2].set_title(f"Rendered depth\nframe {src} -> frame {tgt}")

    axes[3].imshow(make_error_overlay(rgb_tgt, err_map, overlap_valid))
    if depth_err.size:
        axes[3].set_title(
            f"Depth error on overlap\nmean={depth_err.mean():.3f}m  acc@0.1={((depth_err <= 0.1).mean()):.1%}"
        )
    else:
        axes[3].set_title("Depth error on overlap\nno overlap")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    scene = Path(image_rel_paths[0]).parent.name
    fig.suptitle(f"GT cross-view consistency\nscene={scene} source={src} target={tgt}", fontsize=14, y=1.02)
    fig.tight_layout()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, bbox_inches="tight")
    print(f"Saved cross-view visualization to {args.output_path}")


if __name__ == "__main__":
    main()
