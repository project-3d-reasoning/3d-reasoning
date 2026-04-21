#!/usr/bin/env python3
"""
Coordinate-channel sensitivity probe for VGGT pre-merge geometry tokens.

Pipeline:
1) Sample a subset (default 10%) of training entries.
2) Build per-patch 3D coordinate targets in the first-frame camera coordinate system,
   using depth maps + camera pose matrices.
3) Extract VGGT pre-merge tokens and train a linear probe (C -> 3) with Euclidean loss.
4) Select top-20% channels by probe weight norm (coordinate-sensitive channels).
5) Add Gaussian noise on those channels and tune sigma for ~30% Euclidean-loss increase.
6) Save sigma + channel indices + metrics + per-step training loss to a log json file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qwen_vl.data.utils import load_and_preprocess_images
from qwen_vl.model.vggt.models.vggt import VGGT

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _iter_json_array_items(annotation_path: str) -> Iterator[dict]:
    try:
        import ijson  # type: ignore
    except Exception:
        ijson = None

    if ijson is not None:
        with open(annotation_path, "rb") as f:
            for item in ijson.items(f, "item"):
                yield item
    else:
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected json array in {annotation_path}")
        for item in data:
            yield item


def _count_json_array_items(annotation_path: str) -> int:
    try:
        import ijson  # type: ignore
    except Exception:
        ijson = None

    if ijson is not None:
        count = 0
        with open(annotation_path, "rb") as f:
            for _ in ijson.items(f, "item"):
                count += 1
        return count

    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected json array in {annotation_path}")
    return len(data)


def _iter_jsonl_items(annotation_path: str) -> Iterator[dict]:
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _count_jsonl_items(annotation_path: str) -> int:
    count = 0
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def sample_entries(
    annotation_path: str,
    sample_ratio: float,
    seed: int,
    max_entries: int,
) -> Tuple[List[dict], int]:
    is_jsonl = annotation_path.endswith(".jsonl")
    total = _count_jsonl_items(annotation_path) if is_jsonl else _count_json_array_items(annotation_path)
    if total <= 0:
        return [], 0

    target = max(1, int(total * sample_ratio))
    if max_entries > 0:
        target = min(target, max_entries)
    target = min(target, total)

    rng = random.Random(seed)
    selected_ids = set(rng.sample(range(total), target))
    entries: List[dict] = []

    iterator = _iter_jsonl_items(annotation_path) if is_jsonl else _iter_json_array_items(annotation_path)
    for idx, item in enumerate(iterator):
        if idx in selected_ids:
            entries.append(item)
    return entries, total


def resolve_image_path(image_ref: str, media_root: str, entry_data_path: Optional[str]) -> str:
    if os.path.isabs(image_ref):
        return image_ref
    if entry_data_path and not os.path.isabs(entry_data_path):
        candidate = os.path.join(media_root, entry_data_path, image_ref)
        if os.path.exists(candidate):
            return candidate
    if entry_data_path and os.path.isabs(entry_data_path):
        candidate = os.path.join(entry_data_path, image_ref)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(media_root, image_ref)


def pick_frames_keep_first(frame_paths: Sequence[str], max_frames_per_entry: int, seed: int) -> List[str]:
    n = len(frame_paths)
    if max_frames_per_entry <= 0 or n <= max_frames_per_entry:
        return list(frame_paths)
    if max_frames_per_entry == 1:
        return [frame_paths[0]]

    rng = random.Random(seed)
    remain_idx = list(range(1, n))
    chosen = sorted(rng.sample(remain_idx, max_frames_per_entry - 1))
    idx = [0] + chosen
    return [frame_paths[i] for i in idx]


def load_scene_calib(scene_dir: str) -> Optional[Dict[str, float]]:
    intr_path = os.path.join(scene_dir, "intrinsic.txt")
    depth_intr_path = os.path.join(scene_dir, "depth_intrinsic.txt")
    if not (os.path.exists(intr_path) and os.path.exists(depth_intr_path)):
        return None

    intr = np.loadtxt(intr_path, dtype=np.float32)
    dintr = np.loadtxt(depth_intr_path, dtype=np.float32)
    if intr.shape != (4, 4) or dintr.shape != (4, 4):
        return None
    return {
        "fx_rgb": float(intr[0, 0]),
        "fy_rgb": float(intr[1, 1]),
        "cx_rgb": float(intr[0, 2]),
        "cy_rgb": float(intr[1, 2]),
        "fx_d": float(dintr[0, 0]),
        "fy_d": float(dintr[1, 1]),
        "cx_d": float(dintr[0, 2]),
        "cy_d": float(dintr[1, 2]),
    }


def load_pose_from_txt(rgb_path: str) -> Optional[np.ndarray]:
    pose_path = os.path.splitext(rgb_path)[0] + ".txt"
    if not os.path.exists(pose_path):
        return None
    try:
        pose = np.loadtxt(pose_path, dtype=np.float32)
    except Exception:
        return None
    if pose.shape != (4, 4):
        return None
    return pose


def compute_resize_params_for_rgb(rgb_w: int, rgb_h: int, target_size: int, patch_size: int) -> Tuple[float, int, int]:
    scale = target_size / float(rgb_w)
    resized_h = int(round((rgb_h * scale) / patch_size) * patch_size)
    crop_top = 0
    if resized_h > target_size:
        crop_top = (resized_h - target_size) // 2
    return scale, resized_h, crop_top


def compute_patch_coord_targets_in_first_cam(
    rgb_path: str,
    out_h: int,
    out_w: int,
    patch_size: int,
    depth_scale: float,
    calib_cache: Dict[str, Optional[Dict[str, float]]],
    t_i_to_0: np.ndarray,
    use_intrinsics: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    depth_path = os.path.splitext(rgb_path)[0] + ".png"
    if not os.path.exists(depth_path):
        return torch.empty(0, 3), torch.empty(0, dtype=torch.bool)

    scene_dir = os.path.dirname(rgb_path)
    if scene_dir not in calib_cache:
        calib_cache[scene_dir] = load_scene_calib(scene_dir)
    calib = calib_cache[scene_dir]
    if calib is None:
        return torch.empty(0, 3), torch.empty(0, dtype=torch.bool)

    rgb_img = Image.open(rgb_path)
    rgb_w, rgb_h = rgb_img.size
    rgb_img.close()

    depth_img = Image.open(depth_path)
    depth_np = np.array(depth_img, dtype=np.float32) / depth_scale
    depth_img.close()
    if depth_np.ndim != 2:
        return torch.empty(0, 3), torch.empty(0, dtype=torch.bool)
    d_h, d_w = depth_np.shape

    scale, resized_h, crop_top = compute_resize_params_for_rgb(
        rgb_w=rgb_w,
        rgb_h=rgb_h,
        target_size=out_w,
        patch_size=patch_size,
    )
    expected_out_h = min(resized_h, out_w)
    if expected_out_h != out_h:
        # Fallback: if image resize/crop mismatch, skip to keep geometry consistent.
        return torch.empty(0, 3), torch.empty(0, dtype=torch.bool)

    y = torch.arange(out_h, dtype=torch.float32)
    x = torch.arange(out_w, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Processed RGB pixel -> original RGB pixel.
    yy_in_resized = yy + crop_top
    x_rgb = xx / scale
    y_rgb = yy_in_resized / scale

    if use_intrinsics:
        x_d = (x_rgb - calib["cx_rgb"]) / calib["fx_rgb"] * calib["fx_d"] + calib["cx_d"]
        y_d = (y_rgb - calib["cy_rgb"]) / calib["fy_rgb"] * calib["fy_d"] + calib["cy_d"]
    else:
        x_d = x_rgb * (float(d_w) / float(rgb_w))
        y_d = y_rgb * (float(d_h) / float(rgb_h))

    # Sample depth at mapped depth coordinates.
    grid_x = 2.0 * x_d / max(d_w - 1, 1) - 1.0
    grid_y = 2.0 * y_d / max(d_h - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1,H,W,2]
    depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)  # [1,1,Hd,Wd]
    z = F.grid_sample(
        depth_t,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]
    valid = z > 0.0

    # Camera-i coordinates from depth intrinsics.
    x_cam = (x_d - calib["cx_d"]) / calib["fx_d"] * z
    y_cam = (y_d - calib["cy_d"]) / calib["fy_d"] * z
    p_cam_i = torch.stack([x_cam, y_cam, z], dim=-1)  # [H,W,3]

    # Transform camera-i -> camera-0.
    t = torch.from_numpy(t_i_to_0.astype(np.float32))
    r = t[:3, :3]
    tt = t[:3, 3]
    p_cam0 = p_cam_i.reshape(-1, 3).matmul(r.t()) + tt
    p_cam0 = p_cam0.reshape(out_h, out_w, 3)

    ph = out_h // patch_size
    pw = out_w // patch_size
    if ph <= 0 or pw <= 0:
        return torch.empty(0, 3), torch.empty(0, dtype=torch.bool)

    p_cam0 = p_cam0[: ph * patch_size, : pw * patch_size]
    valid = valid[: ph * patch_size, : pw * patch_size]

    # [ph, patch, pw, patch, 3]
    p_blocks = p_cam0.view(ph, patch_size, pw, patch_size, 3)
    v_blocks = valid.view(ph, patch_size, pw, patch_size)

    cnt = v_blocks.sum(dim=(1, 3)).to(torch.float32)  # [ph,pw]
    cnt_safe = torch.clamp(cnt, min=1.0).unsqueeze(-1)
    sum_xyz = p_blocks.sum(dim=(1, 3))  # [ph,pw,3]
    mean_xyz = sum_xyz / cnt_safe
    patch_valid = cnt > 0

    return mean_xyz.reshape(-1, 3), patch_valid.reshape(-1)


def extract_premerge_tokens(vggt_model: VGGT, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    with torch.no_grad():
        if device.type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                aggregated_tokens_list, patch_start_idx = vggt_model.aggregator(images[None])
        else:
            aggregated_tokens_list, patch_start_idx = vggt_model.aggregator(images[None])
    feat = aggregated_tokens_list[-2][0, :, patch_start_idx:]  # [S,P,C]
    return feat.float().cpu()


def train_coord_probe_with_logs(
    x_train: torch.Tensor,  # [N,C]
    y_train: torch.Tensor,  # [N,3]
    ridge_lambda: float,
    steps: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, float]], Dict[str, float]]:
    if steps <= 0:
        raise ValueError("steps must be > 0 for coordinate probe.")

    x_mean = x_train.mean(dim=0)
    x_std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    y_mean = y_train.mean(dim=0)
    y_std = y_train.std(dim=0, unbiased=False).clamp_min(1e-6)

    c = x_train.shape[1]
    w_n = torch.zeros(c, 3, dtype=torch.float32, device=device, requires_grad=True)
    b_n = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([w_n, b_n], lr=lr, weight_decay=0.0)

    x_mean_dev = x_mean.to(device)
    x_std_dev = x_std.to(device)
    y_mean_dev = y_mean.to(device)
    y_std_dev = y_std.to(device)

    n_train = x_train.shape[0]
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    step_losses: List[Dict[str, float]] = []
    for step in range(1, steps + 1):
        if batch_size <= 0 or batch_size >= n_train:
            idx = torch.arange(n_train)
        else:
            idx = torch.randint(0, n_train, size=(batch_size,), generator=g)

        xb = x_train[idx].to(device, non_blocking=True)
        yb = y_train[idx].to(device, non_blocking=True)
        xb = (xb - x_mean_dev) / x_std_dev
        yb = (yb - y_mean_dev) / y_std_dev

        pred = xb.matmul(w_n) + b_n  # [B,3]
        l2 = torch.norm(pred - yb, dim=-1).mean()
        reg = ridge_lambda * torch.mean(w_n ** 2)
        loss = l2 + reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step_losses.append(
            {
                "step": step,
                "train_loss_total": float(loss.detach().cpu().item()),
                "train_loss_l2": float(l2.detach().cpu().item()),
                "train_loss_reg": float(reg.detach().cpu().item()),
            }
        )

    # De-normalize to original coordinate space:
    # y = ((x-xm)/xs @ Wn + bn) * ys + ym
    # => W = diag(1/xs) @ Wn @ diag(ys)
    # => b = ym + bn*ys - xm @ W
    w_n_cpu = w_n.detach().cpu()
    b_n_cpu = b_n.detach().cpu()
    scale_in = 1.0 / x_std.unsqueeze(-1)  # [C,1]
    scale_out = y_std.unsqueeze(0)  # [1,3]
    w = w_n_cpu * scale_in * scale_out  # [C,3]
    b = y_mean + b_n_cpu * y_std - x_mean.matmul(w)  # [3]

    norm_stats = {
        "x_std_min": float(x_std.min().item()),
        "x_std_max": float(x_std.max().item()),
        "y_std_min": float(y_std.min().item()),
        "y_std_max": float(y_std.max().item()),
    }
    return w, b, step_losses, norm_stats


def eval_coord_probe(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    pred = x.matmul(w) + b
    diff = pred - y
    l2 = torch.norm(diff, dim=-1).mean().item()
    mse_xyz = torch.mean(diff ** 2).item()
    rmse_xyz = math.sqrt(max(mse_xyz, 0.0))
    mae_xyz = torch.mean(torch.abs(diff)).item()
    return {"l2_mean": l2, "rmse_xyz": rmse_xyz, "mae_xyz": mae_xyz}


def evaluate_with_noise(
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    channel_idx: torch.Tensor,
    sigma: float,
    repeats: int,
    seed: int,
) -> Dict[str, float]:
    acc = {"l2_mean": 0.0, "rmse_xyz": 0.0, "mae_xyz": 0.0}
    for rep in range(repeats):
        g = torch.Generator(device=x_val.device)
        g.manual_seed(seed + rep + int(sigma * 1000))
        x_noisy = x_val.clone()
        noise = torch.randn(
            x_noisy.shape[0],
            channel_idx.numel(),
            generator=g,
            dtype=x_noisy.dtype,
            device=x_noisy.device,
        ) * sigma
        x_noisy[:, channel_idx] = x_noisy[:, channel_idx] + noise
        cur = eval_coord_probe(x_noisy, y_val, w, b)
        for k in acc:
            acc[k] += cur[k]
    for k in acc:
        acc[k] /= repeats
    return acc


def tune_sigma_for_l2(
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    channel_idx: torch.Tensor,
    baseline_metrics: Dict[str, float],
    target_drop_ratio: float,
    sigma_min: float,
    sigma_max: float,
    repeats: int,
    search_steps: int,
    seed: int,
) -> Tuple[float, Dict[str, float], float, List[Dict[str, float]]]:
    trace: List[Dict[str, float]] = []
    base = baseline_metrics["l2_mean"]
    target = base * (1.0 + target_drop_ratio)

    def score_fn(sigma: float) -> float:
        return evaluate_with_noise(x_val, y_val, w, b, channel_idx, sigma, repeats, seed)["l2_mean"]

    upper = sigma_max
    upper_score = score_fn(upper)
    while upper_score < target and upper < 100.0:
        upper *= 2.0
        upper_score = score_fn(upper)
    lo, hi = sigma_min, upper

    best_sigma = hi
    best_gap = float("inf")
    best_metric = evaluate_with_noise(x_val, y_val, w, b, channel_idx, best_sigma, repeats, seed)
    for _ in range(search_steps):
        mid = (lo + hi) * 0.5
        m = evaluate_with_noise(x_val, y_val, w, b, channel_idx, mid, repeats, seed)
        cur = m["l2_mean"]
        gap = abs(cur - target)
        trace.append({"sigma": mid, "criterion_score": cur})
        if gap < best_gap:
            best_gap = gap
            best_sigma = mid
            best_metric = m
        if cur < target:
            lo = mid
        else:
            hi = mid
    return best_sigma, best_metric, target, trace


def collect_probe_dataset(
    entries: Sequence[dict],
    media_root: str,
    vggt_model: VGGT,
    device: torch.device,
    target_size: int,
    patch_size: int,
    depth_scale: float,
    patches_per_frame: int,
    max_patch_samples: int,
    max_frames_per_entry: int,
    frames_per_forward: int,
    seed: int,
    use_intrinsics: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    calib_cache: Dict[str, Optional[Dict[str, float]]] = {}

    x_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []

    used_entries = 0
    used_frames = 0
    skipped_frames = 0
    total_patches = 0

    for eidx, entry in enumerate(tqdm(entries, desc="Collecting patch samples")):
        image_refs = entry.get("images", entry.get("image"))
        if image_refs is None:
            continue
        if isinstance(image_refs, str):
            image_refs = [image_refs]
        if not isinstance(image_refs, list) or len(image_refs) == 0:
            continue

        entry_data_path = entry.get("data_path")
        frame_paths = [
            resolve_image_path(str(ref), media_root=media_root, entry_data_path=entry_data_path)
            for ref in image_refs
        ]
        frame_paths = [p for p in frame_paths if os.path.exists(p)]
        if not frame_paths:
            continue

        # keep first frame as reference frame when subsampling
        frame_paths = pick_frames_keep_first(frame_paths, max_frames_per_entry=max_frames_per_entry, seed=seed + eidx)
        if not frame_paths:
            continue

        ref_pose = load_pose_from_txt(frame_paths[0])
        if ref_pose is None:
            continue
        try:
            w_to_c0 = np.linalg.inv(ref_pose)
        except Exception:
            continue

        # Build per-frame transform cam_i -> cam_0
        t_i_to_0_list: List[np.ndarray] = []
        valid_paths: List[str] = []
        ref_scene_dir = os.path.dirname(frame_paths[0])
        for fp in frame_paths:
            if os.path.dirname(fp) != ref_scene_dir:
                continue
            pose_i = load_pose_from_txt(fp)
            if pose_i is None:
                continue
            t_i_to_0_list.append((w_to_c0 @ pose_i).astype(np.float32))
            valid_paths.append(fp)

        if not valid_paths:
            continue

        used_entries += 1
        frame_paths = valid_paths

        for start in range(0, len(frame_paths), max(1, frames_per_forward)):
            chunk_paths = frame_paths[start : start + max(1, frames_per_forward)]
            chunk_t = t_i_to_0_list[start : start + max(1, frames_per_forward)]

            try:
                images_t = load_and_preprocess_images(chunk_paths, mode="crop", target_size=target_size)
            except Exception:
                skipped_frames += len(chunk_paths)
                continue

            try:
                token_chunk = extract_premerge_tokens(vggt_model, images_t, device=device)  # [S,P,C]
            except Exception:
                skipped_frames += len(chunk_paths)
                continue

            out_h, out_w = int(images_t.shape[-2]), int(images_t.shape[-1])
            for local_i, rgb_path in enumerate(chunk_paths):
                used_frames += 1
                if local_i >= token_chunk.shape[0]:
                    skipped_frames += 1
                    continue

                coord_patch, valid_patch = compute_patch_coord_targets_in_first_cam(
                    rgb_path=rgb_path,
                    out_h=out_h,
                    out_w=out_w,
                    patch_size=patch_size,
                    depth_scale=depth_scale,
                    calib_cache=calib_cache,
                    t_i_to_0=chunk_t[local_i],
                    use_intrinsics=use_intrinsics,
                )
                if coord_patch.numel() == 0:
                    skipped_frames += 1
                    continue

                feat = token_chunk[local_i]
                p = min(feat.shape[0], coord_patch.shape[0], valid_patch.shape[0])
                feat = feat[:p]
                coord_patch = coord_patch[:p]
                valid_patch = valid_patch[:p]

                valid_idx = torch.where(valid_patch)[0]
                if valid_idx.numel() == 0:
                    skipped_frames += 1
                    continue

                if patches_per_frame > 0 and valid_idx.numel() > patches_per_frame:
                    perm = torch.randperm(valid_idx.numel())[:patches_per_frame]
                    valid_idx = valid_idx[perm]

                remaining = max_patch_samples - total_patches if max_patch_samples > 0 else valid_idx.numel()
                if remaining <= 0:
                    break
                if valid_idx.numel() > remaining:
                    perm = torch.randperm(valid_idx.numel())[:remaining]
                    valid_idx = valid_idx[perm]

                x_list.append(feat[valid_idx])
                y_list.append(coord_patch[valid_idx])
                total_patches += valid_idx.numel()

            if max_patch_samples > 0 and total_patches >= max_patch_samples:
                break
        if max_patch_samples > 0 and total_patches >= max_patch_samples:
            break

    if not x_list:
        raise RuntimeError("No valid (token, coord) patch samples collected.")

    x = torch.cat(x_list, dim=0).to(torch.float32)
    y = torch.cat(y_list, dim=0).to(torch.float32)
    stats = {
        "used_entries": used_entries,
        "used_frames": used_frames,
        "skipped_frames": skipped_frames,
        "total_patch_samples": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
    }
    return x, y, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coordinate sensitivity probe for VGGT pre-merge geometry tokens.")
    parser.add_argument("--annotation_path", type=str, required=True, help="Training annotation json/jsonl path.")
    parser.add_argument("--media_root", type=str, default="data/media", help="Root for relative image paths.")
    parser.add_argument("--sample_ratio", type=float, default=0.10, help="Entry sampling ratio from training set.")
    parser.add_argument("--max_entries", type=int, default=0, help="Cap sampled entries (0 means no cap).")
    parser.add_argument("--max_collect_steps", type=int, default=1000, help="Max sampled entries to process during patch collection (0 means no cap).")
    parser.add_argument("--max_frames_per_entry", type=int, default=0, help="Cap frames per entry (0 means all).")
    parser.add_argument("--frames_per_forward", type=int, default=8, help="Frames per VGGT forward for memory.")
    parser.add_argument("--patches_per_frame", type=int, default=512, help="Random patch cap per frame.")
    parser.add_argument("--max_patch_samples", type=int, default=0, help="Total patch sample cap (0 means no cap).")
    parser.add_argument("--target_size", type=int, default=518, help="Preprocess target width/size.")
    parser.add_argument("--patch_size", type=int, default=14, help="VGGT patch size.")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Depth png scale to meters.")
    parser.add_argument("--vggt_model_path", type=str, default="facebook/VGGT-1B", help="HF id or local path.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--ridge_lambda", type=float, default=1e-3, help="Ridge penalty for probe.")
    parser.add_argument("--probe_train_steps", type=int, default=1000, help="Linear probe optimizer steps.")
    parser.add_argument("--probe_lr", type=float, default=2e-2, help="Linear probe learning rate.")
    parser.add_argument("--probe_batch_size", type=int, default=8192, help="Linear probe batch size (<=0 for full batch).")
    parser.add_argument("--channel_ratio", type=float, default=0.2, help="Top-k channel ratio by weight norm.")
    parser.add_argument("--target_drop_ratio", type=float, default=0.3, help="Target Euclidean-loss increase ratio.")
    parser.add_argument("--sigma_min", type=float, default=0.0, help="Sigma search lower bound.")
    parser.add_argument("--sigma_max", type=float, default=2.0, help="Sigma search initial upper bound.")
    parser.add_argument("--sigma_search_steps", type=int, default=16, help="Binary search steps for sigma.")
    parser.add_argument("--noise_eval_repeats", type=int, default=3, help="Repeats for noisy metric estimate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu.")
    parser.add_argument("--use_intrinsics", action="store_true", help="Use intrinsic/depth_intrinsic for rgb-depth alignment.")
    parser.add_argument("--log_path", type=str, default="", help="Output log json path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")
    device = torch.device(args.device)

    sampled_entries, total_entries = sample_entries(
        annotation_path=args.annotation_path,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
        max_entries=args.max_entries,
    )
    if not sampled_entries:
        raise RuntimeError("No sampled entries found.")

    print(f"[INFO] total entries={total_entries}, sampled={len(sampled_entries)}")
    if args.max_collect_steps > 0:
        sampled_entries = sampled_entries[: args.max_collect_steps]
        print(f"[INFO] applying collection step cap: max_collect_steps={args.max_collect_steps}, effective_entries={len(sampled_entries)}")

    print(f"[INFO] loading VGGT from: {args.vggt_model_path}")
    vggt_model = VGGT.from_pretrained(
        args.vggt_model_path,
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=False,
    )
    vggt_model = vggt_model.to(device)
    vggt_model.eval()

    x, y, collect_stats = collect_probe_dataset(
        entries=sampled_entries,
        media_root=args.media_root,
        vggt_model=vggt_model,
        device=device,
        target_size=args.target_size,
        patch_size=args.patch_size,
        depth_scale=args.depth_scale,
        patches_per_frame=args.patches_per_frame,
        max_patch_samples=args.max_patch_samples,
        max_frames_per_entry=args.max_frames_per_entry,
        frames_per_forward=args.frames_per_forward,
        seed=args.seed,
        use_intrinsics=args.use_intrinsics,
    )
    print(f"[INFO] collected patch samples: {x.shape[0]} with feature_dim={x.shape[1]}")

    n = x.shape[0]
    val_n = max(1, int(n * args.val_ratio))
    perm = torch.randperm(n)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    if train_idx.numel() == 0:
        raise RuntimeError("Not enough samples for train split.")

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    w, b, probe_step_losses, probe_norm_stats = train_coord_probe_with_logs(
        x_train=x_train,
        y_train=y_train,
        ridge_lambda=args.ridge_lambda,
        steps=args.probe_train_steps,
        lr=args.probe_lr,
        batch_size=args.probe_batch_size,
        seed=args.seed,
        device=device,
    )
    train_metrics = eval_coord_probe(x_train, y_train, w, b)
    val_metrics = eval_coord_probe(x_val, y_val, w, b)

    # Channel sensitivity by weight norm across xyz heads.
    channel_importance = torch.norm(w, dim=1)  # [C]
    c = channel_importance.numel()
    k = max(1, int(c * args.channel_ratio))
    topk = torch.topk(channel_importance, k=k, largest=True)
    coord_related_channels = topk.indices

    best_sigma, noisy_metrics, target_value, sigma_trace = tune_sigma_for_l2(
        x_val=x_val,
        y_val=y_val,
        w=w,
        b=b,
        channel_idx=coord_related_channels,
        baseline_metrics=val_metrics,
        target_drop_ratio=args.target_drop_ratio,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        repeats=args.noise_eval_repeats,
        search_steps=args.sigma_search_steps,
        seed=args.seed,
    )
    realized_drop = (noisy_metrics["l2_mean"] - val_metrics["l2_mean"]) / max(abs(val_metrics["l2_mean"]), 1e-8)

    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_path.strip()
    if not log_path:
        out_dir = ROOT_DIR / "logs" / "coord_probe"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(out_dir / f"coord_probe_{now}.json")
    else:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    log_payload = {
        "timestamp_utc": now,
        "annotation_path": args.annotation_path,
        "media_root": args.media_root,
        "sample_ratio": args.sample_ratio,
        "total_entries": total_entries,
        "sampled_entries": len(sampled_entries),
        "collection_stats": collect_stats,
        "split": {
            "train_samples": int(x_train.shape[0]),
            "val_samples": int(x_val.shape[0]),
            "val_ratio": args.val_ratio,
        },
        "probe": {
            "ridge_lambda": args.ridge_lambda,
            "train_steps": args.probe_train_steps,
            "train_lr": args.probe_lr,
            "train_batch_size": args.probe_batch_size,
            "normalization": probe_norm_stats,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_loss_steps": probe_step_losses,
        },
        "coord_related_channels": {
            "channel_ratio": args.channel_ratio,
            "count": int(coord_related_channels.numel()),
            "indices": coord_related_channels.tolist(),
            "weight_norm_topk": topk.values.tolist(),
        },
        "noise_tuning": {
            "criterion": "l2_increase",
            "target_drop_ratio": args.target_drop_ratio,
            "criterion_target_value": target_value,
            "sigma_found": best_sigma,
            "noisy_metrics_at_sigma": noisy_metrics,
            "realized_relative_drop": realized_drop,
            "eval_repeats": args.noise_eval_repeats,
            "trace": sigma_trace,
        },
        "config": vars(args),
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=2)

    print("[RESULT] coord-related channels:", coord_related_channels.numel())
    print(f"[RESULT] sigma={best_sigma:.6f}, criterion=l2_increase, realized_drop={realized_drop:.4f}")
    print(f"[RESULT] log saved to: {log_path}")


if __name__ == "__main__":
    main()

