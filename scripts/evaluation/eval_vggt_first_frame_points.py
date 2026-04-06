#!/usr/bin/env python3
import argparse
import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_vl.model.vggt.models.vggt import VGGT  # noqa: E402
from qwen_vl.model.vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate VGGT pixel-wise 3D points against ScanRefer first-frame coordinates."
    )
    parser.add_argument(
        "--ann-path",
        type=Path,
        default=REPO_ROOT / "data/evaluation/scanrefer_first_frame/scanrefer_val_32frames.json",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data/media",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/data7t-root/huggingface/hub/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9"
        ),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.50],
        help="Euclidean distance thresholds in meters.",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=1,
        help="How many unique 32-frame sequences to evaluate. Default keeps CPU smoke tests practical.",
    )
    parser.add_argument(
        "--sequence-offset",
        type=int,
        default=0,
        help="Skip this many unique sequences before evaluation.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Optionally truncate each sequence to the first N frames.",
    )
    parser.add_argument(
        "--dedupe-sequences",
        action="store_true",
        help="Evaluate unique image sequences only.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--comparison-mode",
        type=str,
        default="pred_current_to_first_vs_gt_first",
        choices=[
            "pred_first_vs_gt_first",
            "pred_current_vs_gt_current",
            "pred_current_to_first_vs_gt_first",
        ],
        help=(
            "How to compare predicted points against GT. "
            "Use this to test whether VGGT outputs first-frame coordinates or per-frame camera coordinates."
        ),
    )
    parser.add_argument(
        "--prediction-source",
        type=str,
        default="point",
        choices=["point", "depth_cam", "depth_gtcam", "depth_gtpose", "depth_gtintrinsics"],
        help=(
            "How to build predicted 3D points. "
            "`depth_cam` uses predicted depth + predicted camera; "
            "`depth_gtcam` uses predicted depth + GT pose/intrinsics; "
            "`depth_gtpose` uses predicted depth + GT pose + predicted intrinsics; "
            "`depth_gtintrinsics` uses predicted depth + predicted pose + GT intrinsics."
        ),
    )
    return parser.parse_args()


def read_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32)


def get_preprocess_params(width: int, height: int, target_size: int, patch_multiple: int = 14):
    new_width = target_size
    new_height = int(round((height * (new_width / width)) / patch_multiple) * patch_multiple)
    crop_top = 0
    crop_bottom = new_height
    if new_height > target_size:
        crop_top = (new_height - target_size) // 2
        crop_bottom = crop_top + target_size
    out_height = crop_bottom - crop_top
    return {
        "orig_width": width,
        "orig_height": height,
        "new_width": new_width,
        "new_height": new_height,
        "crop_top": crop_top,
        "crop_bottom": crop_bottom,
        "out_height": out_height,
    }


def preprocess_rgb(path: Path, target_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    params = get_preprocess_params(width, height, target_size)
    image = image.resize((params["new_width"], params["new_height"]), Image.Resampling.BICUBIC)
    if params["new_height"] > target_size:
        image = image.crop((0, params["crop_top"], params["new_width"], params["crop_bottom"]))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def preprocess_depth_and_intrinsics(depth: np.ndarray, intrinsic: np.ndarray, target_size: int):
    height, width = depth.shape
    params = get_preprocess_params(width, height, target_size)

    # Keep GT discrete while matching resized image pixels.
    depth_img = Image.fromarray(depth)
    depth_img = depth_img.resize((params["new_width"], params["new_height"]), Image.Resampling.NEAREST)
    depth_resized = np.asarray(depth_img, dtype=np.float32)
    if params["new_height"] > target_size:
        depth_resized = depth_resized[params["crop_top"] : params["crop_bottom"], :]

    scale_x = params["new_width"] / width
    scale_y = params["new_height"] / height
    intrinsic = intrinsic.copy()
    intrinsic[0, 0] *= scale_x
    intrinsic[1, 1] *= scale_y
    intrinsic[0, 2] *= scale_x
    intrinsic[1, 2] *= scale_y
    if params["new_height"] > target_size:
        intrinsic[1, 2] -= params["crop_top"]

    return depth_resized, intrinsic


def unproject_depth_to_cam(depth_m: np.ndarray, intrinsic: np.ndarray):
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    points = np.stack([x, y, z], axis=-1)
    valid = depth_m > 1e-6
    return points, valid


def invert_extrinsic_3x4(extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3:4]
    rotation_t = rotation.T
    inv_translation = -rotation_t @ translation
    inv = np.zeros((4, 4), dtype=np.float32)
    inv[:3, :3] = rotation_t
    inv[:3, 3:4] = inv_translation
    inv[3, 3] = 1.0
    return inv


def depth_to_world_coords_points(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray):
    cam_coords_points, point_mask = unproject_depth_to_cam(depth_map, intrinsic)
    cam_to_world = invert_extrinsic_3x4(extrinsic)
    world_coords_points = transform_points(cam_coords_points, cam_to_world)
    return world_coords_points, point_mask


def unproject_depth_map_to_point_map_local(depth_map: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray):
    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1),
            extrinsics_cam[frame_idx],
            intrinsics_cam[frame_idx],
        )
        world_points_list.append(cur_world_points)
    return np.stack(world_points_list, axis=0)


def transform_points(points: np.ndarray, transform: np.ndarray):
    ones = np.ones((*points.shape[:2], 1), dtype=np.float32)
    points_h = np.concatenate([points, ones], axis=-1)
    transformed = points_h @ transform.T
    return transformed[..., :3]


def load_sequence_inputs(image_rel_paths, data_root: Path, target_size: int):
    rgb_tensors = []
    gt_points_current_frame = []
    gt_points_first_frame = []
    valid_masks = []
    cam_to_first_transforms = []
    cam_from_first_extrinsics = []
    gt_intrinsics_processed = []

    first_pose = None

    for rel_path in image_rel_paths:
        rgb_path = data_root / rel_path
        scene_dir = rgb_path.parent
        stem = rgb_path.stem

        depth_path = scene_dir / f"{stem}.png"
        pose_path = scene_dir / f"{stem}.txt"
        depth_intrinsic_path = scene_dir / "depth_intrinsic.txt"

        rgb_tensors.append(preprocess_rgb(rgb_path, target_size))

        depth_raw = np.asarray(Image.open(depth_path), dtype=np.float32)
        # ScanNet depth png is stored in millimeters.
        depth_m = depth_raw / 1000.0
        depth_intrinsic = read_matrix(depth_intrinsic_path)[:3, :3]
        depth_m, depth_intrinsic = preprocess_depth_and_intrinsics(depth_m, depth_intrinsic, target_size)
        cam_points, valid = unproject_depth_to_cam(depth_m, depth_intrinsic)
        gt_intrinsics_processed.append(depth_intrinsic.astype(np.float32))

        pose = read_matrix(pose_path)
        if first_pose is None:
            first_pose = pose
            first_pose_inv = np.linalg.inv(first_pose).astype(np.float32)

        cam_to_first = (first_pose_inv @ pose).astype(np.float32)
        cam_from_first = (np.linalg.inv(pose) @ first_pose).astype(np.float32)
        gt_points_first = transform_points(cam_points, cam_to_first)

        gt_points_current_frame.append(cam_points)
        gt_points_first_frame.append(gt_points_first)
        valid_masks.append(valid)
        cam_to_first_transforms.append(cam_to_first)
        cam_from_first_extrinsics.append(cam_from_first[:3])

    images = torch.stack(rgb_tensors, dim=0)
    gt_points_current = np.stack(gt_points_current_frame, axis=0)
    gt_points = np.stack(gt_points_first_frame, axis=0)
    valid_masks = np.stack(valid_masks, axis=0)
    cam_to_first = np.stack(cam_to_first_transforms, axis=0)
    cam_from_first = np.stack(cam_from_first_extrinsics, axis=0)
    gt_intrinsics = np.stack(gt_intrinsics_processed, axis=0)
    return images, gt_points_current, gt_points, valid_masks, cam_to_first, cam_from_first, gt_intrinsics


def init_stats(thresholds, num_frames):
    def make_bucket():
        bucket = {
            "count": 0,
            "sum_error": 0.0,
            "sum_sq_error": 0.0,
        }
        for thr in thresholds:
            bucket[f"acc@{thr:.2f}m"] = 0
        return bucket

    overall = make_bucket()
    by_frame = [make_bucket() for _ in range(num_frames)]
    return overall, by_frame


def update_stats(bucket, errors: np.ndarray, thresholds):
    bucket["count"] += int(errors.size)
    bucket["sum_error"] += float(errors.sum())
    bucket["sum_sq_error"] += float(np.square(errors).sum())
    for thr in thresholds:
        bucket[f"acc@{thr:.2f}m"] += int((errors <= thr).sum())


def finalize_bucket(bucket, thresholds):
    count = max(bucket["count"], 1)
    result = {
        "count": bucket["count"],
        "mean_l2_m": bucket["sum_error"] / count,
        "rmse_m": math.sqrt(bucket["sum_sq_error"] / count),
    }
    for thr in thresholds:
        result[f"acc@{thr:.2f}m"] = bucket[f"acc@{thr:.2f}m"] / count
    return result


def build_unique_sequences(samples, dedupe_sequences):
    if not dedupe_sequences:
        return [sample["images"] for sample in samples]
    ordered = OrderedDict()
    for sample in samples:
        key = tuple(sample["images"])
        ordered.setdefault(key, sample["images"])
    return list(ordered.values())


def load_model(model_path: Path, device: str, prediction_source: str):
    enable_camera = prediction_source in {"depth_cam", "depth_gtpose", "depth_gtintrinsics"}
    enable_depth = prediction_source in {"depth_cam", "depth_gtcam", "depth_gtpose", "depth_gtintrinsics"}
    enable_point = prediction_source == "point"
    model = VGGT.from_pretrained(
        str(model_path),
        enable_camera=enable_camera,
        enable_point=enable_point,
        enable_depth=enable_depth,
        enable_track=False,
    )
    model.eval()
    model.to(device)
    return model


def predict_world_points_point(model: VGGT, images: torch.Tensor, device: str):
    with torch.inference_mode():
        batch = images.unsqueeze(0).to(device=device, dtype=torch.float32)
        aggregated_tokens_list, patch_start_idx = model.aggregator(batch)
        pts3d, _ = model.point_head(aggregated_tokens_list, images=batch, patch_start_idx=patch_start_idx)
        return pts3d[0].detach().cpu().numpy().astype(np.float32)


def predict_world_points_depth_cam(model: VGGT, images: torch.Tensor, device: str):
    with torch.inference_mode():
        batch = images.unsqueeze(0).to(device=device, dtype=torch.float32)
        predictions = model(batch)
        depth = predictions["depth"][0].detach().cpu()
        pose_enc = predictions["pose_enc"].detach().cpu()

        height, width = depth.shape[1], depth.shape[2]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc,
            image_size_hw=(height, width),
            pose_encoding_type="absT_quaR_FoV",
            build_intrinsics=True,
        )
        world_points = unproject_depth_map_to_point_map_local(
            depth.numpy(),
            extrinsics[0].numpy(),
            intrinsics[0].numpy(),
        )
        return world_points.astype(np.float32)


def predict_world_points_depth_gtcam(
    model: VGGT,
    images: torch.Tensor,
    device: str,
    gt_extrinsics: np.ndarray,
    gt_intrinsics: np.ndarray,
):
    with torch.inference_mode():
        batch = images.unsqueeze(0).to(device=device, dtype=torch.float32)
        predictions = model(batch)
        depth = predictions["depth"][0].detach().cpu()
        world_points = unproject_depth_map_to_point_map_local(depth.numpy(), gt_extrinsics, gt_intrinsics)
        return world_points.astype(np.float32)


def predict_world_points_depth_gtpose(
    model: VGGT,
    images: torch.Tensor,
    device: str,
    gt_extrinsics: np.ndarray,
):
    with torch.inference_mode():
        batch = images.unsqueeze(0).to(device=device, dtype=torch.float32)
        predictions = model(batch)
        depth = predictions["depth"][0].detach().cpu()
        pose_enc = predictions["pose_enc"].detach().cpu()

        height, width = depth.shape[1], depth.shape[2]
        _, intrinsics = pose_encoding_to_extri_intri(
            pose_enc,
            image_size_hw=(height, width),
            pose_encoding_type="absT_quaR_FoV",
            build_intrinsics=True,
        )
        world_points = unproject_depth_map_to_point_map_local(
            depth.numpy(),
            gt_extrinsics,
            intrinsics[0].numpy(),
        )
        return world_points.astype(np.float32)


def predict_world_points_depth_gtintrinsics(
    model: VGGT,
    images: torch.Tensor,
    device: str,
    gt_intrinsics: np.ndarray,
):
    with torch.inference_mode():
        batch = images.unsqueeze(0).to(device=device, dtype=torch.float32)
        predictions = model(batch)
        depth = predictions["depth"][0].detach().cpu()
        pose_enc = predictions["pose_enc"].detach().cpu()

        height, width = depth.shape[1], depth.shape[2]
        extrinsics, _ = pose_encoding_to_extri_intri(
            pose_enc,
            image_size_hw=(height, width),
            pose_encoding_type="absT_quaR_FoV",
            build_intrinsics=True,
        )
        world_points = unproject_depth_map_to_point_map_local(
            depth.numpy(),
            extrinsics[0].numpy(),
            gt_intrinsics,
        )
        return world_points.astype(np.float32)


def predict_world_points(
    model: VGGT,
    images: torch.Tensor,
    device: str,
    prediction_source: str,
    gt_extrinsics: np.ndarray | None = None,
    gt_intrinsics: np.ndarray | None = None,
):
    if prediction_source == "point":
        return predict_world_points_point(model, images, device)
    if prediction_source == "depth_cam":
        return predict_world_points_depth_cam(model, images, device)
    if prediction_source == "depth_gtcam":
        return predict_world_points_depth_gtcam(model, images, device, gt_extrinsics, gt_intrinsics)
    if prediction_source == "depth_gtpose":
        return predict_world_points_depth_gtpose(model, images, device, gt_extrinsics)
    if prediction_source == "depth_gtintrinsics":
        return predict_world_points_depth_gtintrinsics(model, images, device, gt_intrinsics)
    raise ValueError(f"Unsupported prediction_source: {prediction_source}")


def transform_pred_points(pred_points: np.ndarray, cam_to_first: np.ndarray, comparison_mode: str):
    if comparison_mode in {"pred_first_vs_gt_first", "pred_current_vs_gt_current"}:
        return pred_points
    if comparison_mode == "pred_current_to_first_vs_gt_first":
        transformed = []
        for frame_idx in range(pred_points.shape[0]):
            transformed.append(transform_points(pred_points[frame_idx], cam_to_first[frame_idx]))
        return np.stack(transformed, axis=0)
    raise ValueError(f"Unsupported comparison_mode: {comparison_mode}")


def main():
    args = parse_args()
    torch.set_num_threads(args.num_threads)

    with args.ann_path.open() as f:
        samples = json.load(f)

    sequences = build_unique_sequences(samples, args.dedupe_sequences)
    if args.sequence_offset > 0:
        sequences = sequences[args.sequence_offset :]
    if args.max_seqs > 0:
        sequences = sequences[: args.max_seqs]

    if not sequences:
        raise ValueError("No sequences selected for evaluation.")

    model = load_model(args.model_path, args.device, args.prediction_source)

    overall_stats = None
    frame_stats = None
    sequence_reports = []
    eval_start = time.time()

    for seq_idx, image_rel_paths in enumerate(sequences):
        if args.max_frames > 0:
            image_rel_paths = image_rel_paths[: args.max_frames]

        seq_start = time.time()
        images, gt_points_current, gt_points_first, valid_masks, cam_to_first, cam_from_first, gt_intrinsics = load_sequence_inputs(
            image_rel_paths, args.data_root, args.target_size
        )
        pred_points = predict_world_points(
            model,
            images,
            args.device,
            args.prediction_source,
            gt_extrinsics=cam_from_first,
            gt_intrinsics=gt_intrinsics,
        )
        pred_points_eval = transform_pred_points(pred_points, cam_to_first, args.comparison_mode)

        if args.comparison_mode == "pred_current_vs_gt_current":
            gt_points_eval = gt_points_current
        else:
            gt_points_eval = gt_points_first

        if pred_points_eval.shape != gt_points_eval.shape:
            raise ValueError(
                f"Prediction/GT shape mismatch: pred={pred_points_eval.shape}, gt={gt_points_eval.shape}"
            )

        num_frames = pred_points_eval.shape[0]
        if overall_stats is None:
            overall_stats, frame_stats = init_stats(args.thresholds, num_frames)

        seq_bucket, seq_frame_buckets = init_stats(args.thresholds, num_frames)

        for frame_idx in range(num_frames):
            pred = pred_points_eval[frame_idx]
            gt = gt_points_eval[frame_idx]
            valid = valid_masks[frame_idx] & np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
            if not valid.any():
                continue

            errors = np.linalg.norm(pred[valid] - gt[valid], axis=-1)
            update_stats(seq_bucket, errors, args.thresholds)
            update_stats(seq_frame_buckets[frame_idx], errors, args.thresholds)
            update_stats(overall_stats, errors, args.thresholds)
            update_stats(frame_stats[frame_idx], errors, args.thresholds)

        seq_report = {
            "sequence_index": seq_idx,
            "first_image": image_rel_paths[0],
            "num_frames": num_frames,
            "runtime_sec": time.time() - seq_start,
            "overall": finalize_bucket(seq_bucket, args.thresholds),
            "per_frame": [finalize_bucket(bucket, args.thresholds) for bucket in seq_frame_buckets],
        }
        sequence_reports.append(seq_report)
        print(json.dumps(seq_report, ensure_ascii=False))

    summary = {
        "ann_path": str(args.ann_path),
        "model_path": str(args.model_path),
        "device": args.device,
        "comparison_mode": args.comparison_mode,
        "prediction_source": args.prediction_source,
        "num_sequences_evaluated": len(sequences),
        "num_frames_per_sequence": len(frame_stats),
        "thresholds_m": args.thresholds,
        "elapsed_sec": time.time() - eval_start,
        "overall": finalize_bucket(overall_stats, args.thresholds),
        "per_frame": [finalize_bucket(bucket, args.thresholds) for bucket in frame_stats],
        "sequence_reports": sequence_reports,
    }

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
