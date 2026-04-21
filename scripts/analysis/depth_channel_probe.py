#!/usr/bin/env python3
"""
Depth-channel sensitivity probe for VGGT pre-merge geometry tokens.

Pipeline:
1) Sample 10% entries from a training annotation file.
2) Extract VGGT pre-merge geometry tokens (aggregated_tokens_list[-2], patch tokens only).
3) Train a linear probe to predict per-patch mean depth.
4) Pick top-20% channels by absolute probe weight (depth-related channels).
5) Add Gaussian noise to those channels and tune sigma for ~30% probe degradation.
6) Save sigma + channel indices + metrics to a log json file.
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

# Make "src/" importable when running this script from repo root.
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

    if is_jsonl:
        total = _count_jsonl_items(annotation_path)
    else:
        total = _count_json_array_items(annotation_path)
    if total <= 0:
        return [], 0

    target = max(1, int(total * sample_ratio))
    if max_entries > 0:
        target = min(target, max_entries)
    target = min(target, total)

    rng = random.Random(seed)
    selected_ids = set(rng.sample(range(total), target))
    entries: List[dict] = []

    if is_jsonl:
        iterator = _iter_jsonl_items(annotation_path)
    else:
        iterator = _iter_json_array_items(annotation_path)

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


def pick_frames(frame_paths: Sequence[str], max_frames_per_entry: int, seed: int) -> List[str]:
    if max_frames_per_entry <= 0 or len(frame_paths) <= max_frames_per_entry:
        return list(frame_paths)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(frame_paths)), max_frames_per_entry))
    return [frame_paths[i] for i in idx]


def load_intrinsics(scene_dir: str) -> Optional[Dict[str, float]]:
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


def compute_resize_params_for_rgb(
    rgb_w: int,
    rgb_h: int,
    target_size: int,
    patch_size: int,
) -> Tuple[float, int, int]:
    # Matches qwen_vl.data.utils.load_and_preprocess_images(mode="crop")
    scale = target_size / float(rgb_w)
    resized_h = int(round((rgb_h * scale) / patch_size) * patch_size)
    crop_top = 0
    if resized_h > target_size:
        crop_top = (resized_h - target_size) // 2
    return scale, resized_h, crop_top


def compute_patch_depth_targets(
    rgb_path: str,
    out_h: int,
    out_w: int,
    patch_size: int,
    depth_scale: float,
    intr_cache: Dict[str, Optional[Dict[str, float]]],
    use_intrinsics: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    depth_path = os.path.splitext(rgb_path)[0] + ".png"
    if not os.path.exists(depth_path):
        return torch.empty(0), torch.empty(0, dtype=torch.bool)

    scene_dir = os.path.dirname(rgb_path)
    if scene_dir not in intr_cache:
        intr_cache[scene_dir] = load_intrinsics(scene_dir)
    intr = intr_cache[scene_dir]

    rgb_img = Image.open(rgb_path)
    rgb_w, rgb_h = rgb_img.size
    rgb_img.close()

    depth_img = Image.open(depth_path)
    depth_np = np.array(depth_img, dtype=np.float32) / depth_scale
    depth_img.close()
    if depth_np.ndim != 2:
        return torch.empty(0), torch.empty(0, dtype=torch.bool)
    d_h, d_w = depth_np.shape

    scale, resized_h, crop_top = compute_resize_params_for_rgb(
        rgb_w=rgb_w,
        rgb_h=rgb_h,
        target_size=out_w,  # width is fixed to target_size in crop mode
        patch_size=patch_size,
    )
    # If resized_h <= target_size there is no crop, and out_h should match resized_h.
    # If resized_h > target_size then out_h is target_size.
    expected_out_h = min(resized_h, out_w)
    if expected_out_h != out_h:
        # Fallback to simple resize-based alignment when shape does not match expectation.
        depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
        aligned = F.interpolate(depth_t, size=(out_h, out_w), mode="bilinear", align_corners=False)[0, 0]
    else:
        # Build coordinate map in processed rgb space -> original rgb space.
        y = torch.arange(out_h, dtype=torch.float32)
        x = torch.arange(out_w, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        yy_in_resized = yy + crop_top
        x_rgb = xx / scale
        y_rgb = yy_in_resized / scale

        if use_intrinsics and intr is not None:
            x_d = (x_rgb - intr["cx_rgb"]) / intr["fx_rgb"] * intr["fx_d"] + intr["cx_d"]
            y_d = (y_rgb - intr["cy_rgb"]) / intr["fy_rgb"] * intr["fy_d"] + intr["cy_d"]
        else:
            # Fallback: scale by resolution ratio.
            x_d = x_rgb * (float(d_w) / float(rgb_w))
            y_d = y_rgb * (float(d_h) / float(rgb_h))

        grid_x = 2.0 * x_d / max(d_w - 1, 1) - 1.0
        grid_y = 2.0 * y_d / max(d_h - 1, 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]

        depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)  # [1,1,Hd,Wd]
        aligned = F.grid_sample(
            depth_t,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0]

    valid = aligned > 0.0
    ph = out_h // patch_size
    pw = out_w // patch_size
    if ph <= 0 or pw <= 0:
        return torch.empty(0), torch.empty(0, dtype=torch.bool)

    aligned = aligned[: ph * patch_size, : pw * patch_size]
    valid = valid[: ph * patch_size, : pw * patch_size]

    aligned_blocks = aligned.view(ph, patch_size, pw, patch_size)
    valid_blocks = valid.view(ph, patch_size, pw, patch_size)

    sum_depth = aligned_blocks.sum(dim=(1, 3))
    cnt_depth = valid_blocks.sum(dim=(1, 3)).to(torch.float32)
    mean_depth = sum_depth / torch.clamp(cnt_depth, min=1.0)
    patch_valid = cnt_depth > 0

    return mean_depth.reshape(-1), patch_valid.reshape(-1)


def extract_premerge_tokens(vggt_model: VGGT, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    # images: [S, 3, H, W]
    images = images.to(device, non_blocking=True)
    with torch.no_grad():
        if device.type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                aggregated_tokens_list, patch_start_idx = vggt_model.aggregator(images[None])
        else:
            aggregated_tokens_list, patch_start_idx = vggt_model.aggregator(images[None])

    # Pre-merge VGGT geometry tokens: [S, P, C]
    feat = aggregated_tokens_list[-2][0, :, patch_start_idx:]
    return feat.float().cpu()


def fit_linear_probe_ridge(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    ridge_lambda: float,
) -> Tuple[torch.Tensor, float]:
    # Closed-form ridge: w = (X^T X + lambda I)^-1 X^T y
    # with a bias term (not regularized).
    n, c = x_train.shape
    ones = torch.ones(n, 1, dtype=x_train.dtype)
    x_aug = torch.cat([x_train, ones], dim=1)  # [N, C+1]

    xtx = x_aug.t().matmul(x_aug)
    xty = x_aug.t().matmul(y_train)

    reg = torch.eye(c + 1, dtype=x_train.dtype)
    reg[c, c] = 0.0  # do not regularize bias
    xtx = xtx + ridge_lambda * reg

    w_aug = torch.linalg.solve(xtx, xty)
    weight = w_aug[:c]
    bias = float(w_aug[c].item())
    return weight, bias


def train_linear_probe_with_logs(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    ridge_lambda: float,
    steps: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, float, List[Dict[str, float]], Dict[str, float]]:
    if steps <= 0:
        w, b = fit_linear_probe_ridge(x_train, y_train, ridge_lambda=ridge_lambda)
        return w, b, [], {}

    # Standardize for stable optimization.
    x_mean = x_train.mean(dim=0)
    x_std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    y_mean = y_train.mean()
    y_std = y_train.std(unbiased=False).clamp_min(1e-6)

    c = x_train.shape[1]
    w_n = torch.zeros(c, dtype=torch.float32, device=device, requires_grad=True)
    b_n = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
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

        pred = xb.matmul(w_n) + b_n
        mse = torch.mean((pred - yb) ** 2)
        reg = ridge_lambda * torch.mean(w_n ** 2)
        loss = mse + reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step_losses.append(
            {
                "step": step,
                "train_loss_total": float(loss.detach().cpu().item()),
                "train_loss_mse": float(mse.detach().cpu().item()),
                "train_loss_reg": float(reg.detach().cpu().item()),
            }
        )

    # Convert normalized probe back to original scale:
    # y = ((x - x_mean)/x_std @ w_n + b_n) * y_std + y_mean
    # => y = x @ (w_n * y_std/x_std) + (y_mean + y_std*b_n - x_mean @ (w_n * y_std/x_std))
    w_n_cpu = w_n.detach().cpu()
    b_n_cpu = b_n.detach().cpu()
    scale = (y_std / x_std).to(torch.float32)
    w = w_n_cpu * scale
    b = float((y_mean + y_std * b_n_cpu - torch.sum(x_mean * w)).item())

    norm_stats = {
        "x_std_min": float(x_std.min().item()),
        "x_std_max": float(x_std.max().item()),
        "y_std": float(y_std.item()),
    }
    return w, b, step_losses, norm_stats


def eval_probe(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: float) -> Dict[str, float]:
    pred = x.matmul(weight) + bias
    mse = torch.mean((pred - y) ** 2).item()
    rmse = math.sqrt(max(mse, 0.0))
    y_mean = torch.mean(y)
    sst = torch.sum((y - y_mean) ** 2).item()
    sse = torch.sum((pred - y) ** 2).item()
    r2 = 1.0 - (sse / sst) if sst > 1e-12 else 0.0
    mae = torch.mean(torch.abs(pred - y)).item()
    return {"rmse": rmse, "r2": r2, "mae": mae}


def evaluate_with_noise(
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    weight: torch.Tensor,
    bias: float,
    channel_idx: torch.Tensor,
    sigma: float,
    repeats: int,
    seed: int,
) -> Dict[str, float]:
    metric_acc = {"rmse": 0.0, "r2": 0.0, "mae": 0.0}
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
        cur = eval_probe(x_noisy, y_val, weight, bias)
        for k in metric_acc:
            metric_acc[k] += cur[k]
    for k in metric_acc:
        metric_acc[k] /= repeats
    return metric_acc


def tune_sigma(
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    weight: torch.Tensor,
    bias: float,
    channel_idx: torch.Tensor,
    baseline_metrics: Dict[str, float],
    target_drop_ratio: float,
    sigma_min: float,
    sigma_max: float,
    repeats: int,
    search_steps: int,
    seed: int,
) -> Tuple[float, Dict[str, float], str, float, List[Dict[str, float]]]:
    trace: List[Dict[str, float]] = []
    baseline_r2 = baseline_metrics["r2"]
    baseline_rmse = baseline_metrics["rmse"]

    if baseline_r2 > 0:
        criterion = "r2_drop"
        target_value = baseline_r2 * (1.0 - target_drop_ratio)

        def score_fn(sigma: float) -> float:
            return evaluate_with_noise(
                x_val, y_val, weight, bias, channel_idx, sigma, repeats=repeats, seed=seed
            )["r2"]

        # Ensure upper bound can cross target.
        upper = sigma_max
        upper_score = score_fn(upper)
        while upper_score > target_value and upper < 100.0:
            upper *= 2.0
            upper_score = score_fn(upper)
        lo, hi = sigma_min, upper

        best_sigma = hi
        best_gap = float("inf")
        best_metric = evaluate_with_noise(x_val, y_val, weight, bias, channel_idx, best_sigma, repeats, seed)
        for _ in range(search_steps):
            mid = (lo + hi) * 0.5
            m = evaluate_with_noise(x_val, y_val, weight, bias, channel_idx, mid, repeats, seed)
            cur = m["r2"]
            gap = abs(cur - target_value)
            trace.append({"sigma": mid, "criterion_score": cur})
            if gap < best_gap:
                best_gap = gap
                best_sigma = mid
                best_metric = m
            if cur > target_value:
                lo = mid
            else:
                hi = mid
        return best_sigma, best_metric, criterion, target_value, trace

    criterion = "rmse_increase"
    target_value = baseline_rmse * (1.0 + target_drop_ratio)

    def score_fn(sigma: float) -> float:
        return evaluate_with_noise(
            x_val, y_val, weight, bias, channel_idx, sigma, repeats=repeats, seed=seed
        )["rmse"]

    upper = sigma_max
    upper_score = score_fn(upper)
    while upper_score < target_value and upper < 100.0:
        upper *= 2.0
        upper_score = score_fn(upper)
    lo, hi = sigma_min, upper

    best_sigma = hi
    best_gap = float("inf")
    best_metric = evaluate_with_noise(x_val, y_val, weight, bias, channel_idx, best_sigma, repeats, seed)
    for _ in range(search_steps):
        mid = (lo + hi) * 0.5
        m = evaluate_with_noise(x_val, y_val, weight, bias, channel_idx, mid, repeats, seed)
        cur = m["rmse"]
        gap = abs(cur - target_value)
        trace.append({"sigma": mid, "criterion_score": cur})
        if gap < best_gap:
            best_gap = gap
            best_sigma = mid
            best_metric = m
        if cur < target_value:
            lo = mid
        else:
            hi = mid
    return best_sigma, best_metric, criterion, target_value, trace


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
    intr_cache: Dict[str, Optional[Dict[str, float]]] = {}

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

        frame_paths = pick_frames(frame_paths, max_frames_per_entry=max_frames_per_entry, seed=seed + eidx)
        if not frame_paths:
            continue

        used_entries += 1

        # Chunk frames to reduce memory pressure.
        for start in range(0, len(frame_paths), max(1, frames_per_forward)):
            chunk_paths = frame_paths[start : start + max(1, frames_per_forward)]
            try:
                images_t = load_and_preprocess_images(chunk_paths, mode="crop", target_size=target_size)
            except Exception:
                skipped_frames += len(chunk_paths)
                continue

            try:
                token_chunk = extract_premerge_tokens(vggt_model, images_t, device=device)  # [S, P, C]
            except Exception:
                skipped_frames += len(chunk_paths)
                continue

            out_h, out_w = int(images_t.shape[-2]), int(images_t.shape[-1])
            for local_i, rgb_path in enumerate(chunk_paths):
                used_frames += 1
                if local_i >= token_chunk.shape[0]:
                    skipped_frames += 1
                    continue

                depth_patch, valid_patch = compute_patch_depth_targets(
                    rgb_path=rgb_path,
                    out_h=out_h,
                    out_w=out_w,
                    patch_size=patch_size,
                    depth_scale=depth_scale,
                    intr_cache=intr_cache,
                    use_intrinsics=use_intrinsics,
                )
                if depth_patch.numel() == 0:
                    skipped_frames += 1
                    continue

                feat = token_chunk[local_i]  # [P, C]
                p = min(feat.shape[0], depth_patch.shape[0], valid_patch.shape[0])
                feat = feat[:p]
                depth_patch = depth_patch[:p]
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
                y_list.append(depth_patch[valid_idx])
                total_patches += valid_idx.numel()

            if max_patch_samples > 0 and total_patches >= max_patch_samples:
                break
        if max_patch_samples > 0 and total_patches >= max_patch_samples:
            break

    if not x_list:
        raise RuntimeError("No valid (token, depth) patch samples collected. Check data paths and files.")

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
    parser = argparse.ArgumentParser(description="Depth sensitivity probe for VGGT pre-merge geometry tokens.")
    parser.add_argument("--annotation_path", type=str, required=True, help="Training annotation json/jsonl path.")
    parser.add_argument("--media_root", type=str, default="data/media", help="Root for relative image paths.")
    parser.add_argument("--sample_ratio", type=float, default=0.10, help="Entry sampling ratio from training set.")
    parser.add_argument("--max_entries", type=int, default=0, help="Cap sampled entries (0 means no cap).")
    parser.add_argument("--max_frames_per_entry", type=int, default=0, help="Cap frames per entry (0 means all).")
    parser.add_argument("--frames_per_forward", type=int, default=8, help="Frames per VGGT forward for memory.")
    parser.add_argument("--patches_per_frame", type=int, default=512, help="Random patch cap per frame.")
    parser.add_argument("--max_patch_samples", type=int, default=300000, help="Total patch sample cap.")
    parser.add_argument("--target_size", type=int, default=518, help="Preprocess target width/size.")
    parser.add_argument("--patch_size", type=int, default=14, help="VGGT patch size.")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Depth png scale to meters.")
    parser.add_argument("--vggt_model_path", type=str, default="facebook/VGGT-1B", help="HF id or local path.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--ridge_lambda", type=float, default=1e-3, help="Ridge penalty for probe.")
    parser.add_argument("--probe_train_steps", type=int, default=1000, help="Linear probe optimizer steps.")
    parser.add_argument("--probe_lr", type=float, default=2e-2, help="Linear probe learning rate.")
    parser.add_argument("--probe_batch_size", type=int, default=8192, help="Linear probe batch size (<=0 for full batch).")
    parser.add_argument("--channel_ratio", type=float, default=0.2, help="Top-k channel ratio by |probe weight|.")
    parser.add_argument("--target_drop_ratio", type=float, default=0.3, help="Target performance drop ratio.")
    parser.add_argument("--sigma_min", type=float, default=0.0, help="Sigma search lower bound.")
    parser.add_argument("--sigma_max", type=float, default=2.0, help="Sigma search initial upper bound.")
    parser.add_argument("--sigma_search_steps", type=int, default=16, help="Binary search steps for sigma.")
    parser.add_argument("--noise_eval_repeats", type=int, default=3, help="Repeats for noisy metric estimate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu.")
    parser.add_argument("--use_intrinsics", action="store_true", help="Use intrinsic/depth_intrinsic for alignment.")
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

    # Train/val split
    n = x.shape[0]
    val_n = max(1, int(n * args.val_ratio))
    perm = torch.randperm(n)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    if train_idx.numel() == 0:
        raise RuntimeError("Not enough samples for train split.")

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    weight, bias, probe_step_losses, probe_norm_stats = train_linear_probe_with_logs(
        x_train=x_train,
        y_train=y_train,
        ridge_lambda=args.ridge_lambda,
        steps=args.probe_train_steps,
        lr=args.probe_lr,
        batch_size=args.probe_batch_size,
        seed=args.seed,
        device=device,
    )
    train_metrics = eval_probe(x_train, y_train, weight, bias)
    val_metrics = eval_probe(x_val, y_val, weight, bias)

    # Depth-related channels: top-k by absolute probe weight.
    c = weight.numel()
    k = max(1, int(c * args.channel_ratio))
    abs_w = torch.abs(weight)
    topk = torch.topk(abs_w, k=k, largest=True)
    depth_related_channels = topk.indices

    best_sigma, noisy_metrics, criterion, target_value, sigma_trace = tune_sigma(
        x_val=x_val,
        y_val=y_val,
        weight=weight,
        bias=bias,
        channel_idx=depth_related_channels,
        baseline_metrics=val_metrics,
        target_drop_ratio=args.target_drop_ratio,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        repeats=args.noise_eval_repeats,
        search_steps=args.sigma_search_steps,
        seed=args.seed,
    )

    if criterion == "r2_drop":
        realized_drop = (val_metrics["r2"] - noisy_metrics["r2"]) / max(abs(val_metrics["r2"]), 1e-8)
    else:
        realized_drop = (noisy_metrics["rmse"] - val_metrics["rmse"]) / max(abs(val_metrics["rmse"]), 1e-8)

    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_path.strip()
    if not log_path:
        out_dir = ROOT_DIR / "logs" / "depth_probe"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(out_dir / f"depth_probe_{now}.json")
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
        "depth_related_channels": {
            "channel_ratio": args.channel_ratio,
            "count": int(depth_related_channels.numel()),
            "indices": depth_related_channels.tolist(),
            "abs_weight_topk": topk.values.tolist(),
        },
        "noise_tuning": {
            "criterion": criterion,
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

    print("[RESULT] depth-related channels:", depth_related_channels.numel())
    print(f"[RESULT] sigma={best_sigma:.6f}, criterion={criterion}, realized_drop={realized_drop:.4f}")
    print(f"[RESULT] log saved to: {log_path}")


if __name__ == "__main__":
    main()
