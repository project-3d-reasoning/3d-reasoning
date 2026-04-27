#!/usr/bin/env python3
"""
Train a linear probe from VGGT pre-merge tokens to corresponding 2D visual
patch features, then rerun with a different sampling seed and compare channel
overlap.

Pipeline:
1) Sample a subset of entries with `--seed`.
2) Extract VGGT pre-merge tokens and corresponding Qwen2.5-VL 2D patch
   features (merged `feature_2d` by default, or pre-merge patch features via
   `--target_feature_level premerge`).
3) Train a linear probe for `--probe_train_steps` with MSE loss.
4) Select top `--channel_ratio` input channels by aggregated probe weight.
5) Resample with `--resample_seed` and repeat.
6) Save both runs + overlap statistics to one json log.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    import decord
except Exception:
    decord = None


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


QWEN_DEFAULT_MODEL_PATH = (
    "/data7t-root/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/"
    "snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
)


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
            if line:
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


def resolve_media_path(ref: str, media_root: str, entry_data_path: Optional[str]) -> str:
    if os.path.isabs(ref):
        return ref
    if entry_data_path and os.path.isabs(entry_data_path):
        candidate = os.path.join(entry_data_path, ref)
        if os.path.exists(candidate):
            return candidate
    if entry_data_path and not os.path.isabs(entry_data_path):
        candidate = os.path.join(media_root, entry_data_path, ref)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(media_root, ref)


def pick_items(items: Sequence[Union[str, Image.Image]], max_items: int, seed: int) -> List[Union[str, Image.Image]]:
    if max_items <= 0 or len(items) <= max_items:
        return list(items)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(items)), max_items))
    return [items[i] for i in idx]


def load_video_frames(video_path: str, max_frames: int, seed: int) -> List[Image.Image]:
    if os.path.isdir(video_path):
        frame_files = [
            os.path.join(video_path, name)
            for name in sorted(os.listdir(video_path))
            if os.path.isfile(os.path.join(video_path, name))
        ]
        frame_files = pick_items(frame_files, max_frames, seed=seed)
        return [Image.open(path).convert("RGB") for path in frame_files]

    if decord is None:
        raise RuntimeError("decord is required to decode video files.")

    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    if total_frames <= 0:
        return []
    if max_frames > 0 and total_frames > max_frames:
        frame_idx = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        frame_idx = np.arange(total_frames)
    return [Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in frame_idx.tolist()]


def resolve_entry_frames(
    entry: dict,
    media_root: str,
    max_frames_per_entry: int,
    seed: int,
) -> List[Union[str, Image.Image]]:
    entry_data_path = entry.get("data_path")
    image_refs = entry.get("images", entry.get("image"))
    if image_refs is not None:
        if isinstance(image_refs, str):
            image_refs = [image_refs]
        if not isinstance(image_refs, list):
            return []
        frame_paths = [
            resolve_media_path(str(ref), media_root=media_root, entry_data_path=entry_data_path)
            for ref in image_refs
        ]
        frame_paths = [path for path in frame_paths if os.path.exists(path)]
        return pick_items(frame_paths, max_frames_per_entry, seed=seed)

    video_ref = entry.get("video")
    if isinstance(video_ref, str):
        video_path = resolve_media_path(video_ref, media_root=media_root, entry_data_path=entry_data_path)
        if not os.path.exists(video_path):
            return []
        return load_video_frames(video_path, max_frames=max_frames_per_entry, seed=seed)

    return []


def maybe_load_dtype(dtype_name: str) -> object:
    if dtype_name.lower() == "auto":
        return "auto"
    if dtype_name.lower() == "bfloat16":
        return torch.bfloat16
    if dtype_name.lower() == "float16":
        return torch.float16
    if dtype_name.lower() == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def build_model_inputs(
    frames: Sequence[Union[str, Image.Image]],
    processor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    patch_size = int(processor.image_processor.patch_size)
    merge_size = int(processor.image_processor.merge_size)

    geometry_encoder_inputs: List[torch.Tensor] = []
    image_inputs: List[torch.Tensor] = []

    for frame in frames:
        processed = load_and_preprocess_images([frame])[0]
        geometry_encoder_inputs.append(processed)

        _, height, width = processed.shape
        if (width // patch_size) % merge_size > 0:
            width = width - ((width // patch_size) % merge_size) * patch_size
        if (height // patch_size) % merge_size > 0:
            height = height - ((height // patch_size) % merge_size) * patch_size
        image_inputs.append(processed[:, :height, :width])

    vision_inputs = processor.image_processor(
        image_inputs,
        return_tensors="pt",
        do_rescale=False,
    )
    return (
        vision_inputs["pixel_values"],
        vision_inputs["image_grid_thw"],
        torch.stack(geometry_encoder_inputs, dim=0),
    )


def extract_premerge_vggt_tokens(vggt_model: VGGT, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    with torch.no_grad():
        if device.type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(device=device)[0] >= 8 else torch.float16
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                aggregated_tokens_list, patch_start_idx = vggt_model.aggregator(images[None])
        else:
            aggregated_tokens_list, patch_start_idx = vggt_model.aggregator(images[None])
    return aggregated_tokens_list[-2][0, :, patch_start_idx:].float().cpu()


def extract_premerge_qwen_features(
    visual_model,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    pixel_values = pixel_values.to(device=device, dtype=next(visual_model.parameters()).dtype)
    image_grid_thw = image_grid_thw.to(device=device)

    with torch.no_grad():
        hidden_states = visual_model.patch_embed(pixel_values)
        rotary_pos_emb = visual_model.rot_pos_emb(image_grid_thw)
        window_index, cu_window_seqlens = visual_model.get_window_index(image_grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        merge_unit = visual_model.spatial_merge_unit

        hidden_states = hidden_states.reshape(seq_len // merge_unit, merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // merge_unit, merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2],
            image_grid_thw[:, 0],
        ).cumsum(
            dim=0,
            dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(visual_model.blocks):
            cu_seqlens_now = cu_seqlens if layer_num in visual_model.fullatt_block_indexes else cu_window_seqlens
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
            )

        hidden_states = hidden_states.reshape(seq_len // merge_unit, merge_unit, -1)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

    return hidden_states.float().cpu()


def extract_merged_qwen_features(
    visual_model,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    pixel_values = pixel_values.to(device=device, dtype=next(visual_model.parameters()).dtype)
    image_grid_thw = image_grid_thw.to(device=device)

    with torch.no_grad():
        hidden_states = visual_model(pixel_values, grid_thw=image_grid_thw)

    return hidden_states.float().cpu()


def split_flat_premerge_features(flat_features: torch.Tensor, image_grid_thw: torch.Tensor) -> List[torch.Tensor]:
    features: List[torch.Tensor] = []
    cursor = 0
    for row in image_grid_thw.tolist():
        t, h, w = [int(v) for v in row]
        n_tokens = t * h * w
        cur = flat_features[cursor : cursor + n_tokens].view(t, h, w, -1)
        cursor += n_tokens
        for ti in range(t):
            features.append(cur[ti].reshape(h * w, -1))
    return features


def split_flat_merged_features(
    flat_features: torch.Tensor,
    image_grid_thw: torch.Tensor,
    merge_size: int,
) -> List[torch.Tensor]:
    features: List[torch.Tensor] = []
    cursor = 0
    for row in image_grid_thw.tolist():
        t, h, w = [int(v) for v in row]
        merged_h = h // merge_size
        merged_w = w // merge_size
        n_tokens = t * merged_h * merged_w
        cur = flat_features[cursor : cursor + n_tokens].view(t, merged_h, merged_w, -1)
        cursor += n_tokens
        for ti in range(t):
            features.append(cur[ti].reshape(merged_h * merged_w, -1))
    return features


def regroup_vggt_tokens(
    tokens: torch.Tensor,
    height_patches: int,
    width_patches: int,
    target_h: int,
    target_w: int,
    merge_size: int,
) -> torch.Tensor:
    cropped = tokens[: height_patches * width_patches].reshape(height_patches, width_patches, -1)
    cropped = cropped[:target_h, :target_w, :]
    cropped = cropped.reshape(
        target_h // merge_size,
        merge_size,
        target_w // merge_size,
        merge_size,
        -1,
    )
    cropped = cropped.permute(0, 2, 1, 3, 4).contiguous()
    return cropped.reshape(target_h * target_w, -1)


def merge_grouped_vggt_tokens(tokens: torch.Tensor, merge_size: int) -> torch.Tensor:
    merge_unit = merge_size * merge_size
    if tokens.shape[0] % merge_unit != 0:
        raise ValueError(
            f"Grouped VGGT token count {tokens.shape[0]} is not divisible by merge unit {merge_unit}."
        )
    return tokens.view(-1, merge_unit, tokens.shape[-1]).mean(dim=1)


def fit_linear_probe_ridge_multi(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    ridge_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n, c = x_train.shape
    d = y_train.shape[1]
    ones = torch.ones(n, 1, dtype=x_train.dtype)
    x_aug = torch.cat([x_train, ones], dim=1)

    xtx = x_aug.t().matmul(x_aug)
    xty = x_aug.t().matmul(y_train)

    reg = torch.eye(c + 1, dtype=x_train.dtype)
    reg[c, c] = 0.0
    xtx = xtx + ridge_lambda * reg

    w_aug = torch.linalg.solve(xtx, xty)
    return w_aug[:c], w_aug[c]


def train_linear_probe_with_logs_multi(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    ridge_lambda: float,
    steps: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, float]], Dict[str, float]]:
    if steps <= 0:
        weight, bias = fit_linear_probe_ridge_multi(x_train, y_train, ridge_lambda=ridge_lambda)
        importance = torch.mean(torch.abs(weight), dim=1)
        return weight, bias, importance, [], {}

    x_mean = x_train.mean(dim=0)
    x_std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    y_mean = y_train.mean(dim=0)
    y_std = y_train.std(dim=0, unbiased=False).clamp_min(1e-6)

    c = x_train.shape[1]
    d = y_train.shape[1]
    weight_n = torch.zeros(c, d, dtype=torch.float32, device=device, requires_grad=True)
    bias_n = torch.zeros(d, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([weight_n, bias_n], lr=lr, weight_decay=0.0)

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

        pred = xb.matmul(weight_n) + bias_n
        mse = torch.mean((pred - yb) ** 2)
        reg = ridge_lambda * torch.mean(weight_n ** 2)
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

    weight_n_cpu = weight_n.detach().cpu()
    bias_n_cpu = bias_n.detach().cpu()
    weight = weight_n_cpu * (y_std.unsqueeze(0) / x_std.unsqueeze(1))
    bias = y_mean + y_std * bias_n_cpu - x_mean.matmul(weight)
    importance = torch.mean(torch.abs(weight_n_cpu), dim=1)
    norm_stats = {
        "x_std_min": float(x_std.min().item()),
        "x_std_max": float(x_std.max().item()),
        "y_std_min": float(y_std.min().item()),
        "y_std_max": float(y_std.max().item()),
    }
    return weight, bias, importance, step_losses, norm_stats


def eval_probe_multi(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> Dict[str, float]:
    pred = x.matmul(weight) + bias
    mse = torch.mean((pred - y) ** 2).item()
    rmse = math.sqrt(max(mse, 0.0))
    mae = torch.mean(torch.abs(pred - y)).item()
    return {"mse": mse, "rmse": rmse, "mae": mae}


def collect_probe_dataset(
    entries: Sequence[dict],
    media_root: str,
    vggt_model: VGGT,
    qwen_visual,
    processor,
    qwen_device: torch.device,
    vggt_device: torch.device,
    patches_per_frame: int,
    max_patch_samples: int,
    max_frames_per_entry: int,
    frames_per_forward: int,
    seed: int,
    target_feature_level: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    merge_size = int(processor.image_processor.merge_size)
    patch_size = int(processor.image_processor.patch_size)
    patch_rng = torch.Generator(device="cpu")
    patch_rng.manual_seed(seed)

    x_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    used_entries = 0
    used_frames = 0
    skipped_frames = 0
    total_patches = 0
    use_parallel_extract = qwen_device != vggt_device

    def run_feature_extract(
        pixel_values_cur: torch.Tensor,
        image_grid_thw_cur: torch.Tensor,
        geometry_encoder_inputs_cur: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not use_parallel_extract:
            vggt_chunk_cur = extract_premerge_vggt_tokens(vggt_model, geometry_encoder_inputs_cur, device=vggt_device)
            if target_feature_level == "premerge":
                qwen_chunk_cur = extract_premerge_qwen_features(
                    qwen_visual,
                    pixel_values=pixel_values_cur,
                    image_grid_thw=image_grid_thw_cur,
                    device=qwen_device,
                )
            else:
                qwen_chunk_cur = extract_merged_qwen_features(
                    qwen_visual,
                    pixel_values=pixel_values_cur,
                    image_grid_thw=image_grid_thw_cur,
                    device=qwen_device,
                )
            return vggt_chunk_cur, qwen_chunk_cur

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_vggt = executor.submit(
                extract_premerge_vggt_tokens,
                vggt_model,
                geometry_encoder_inputs_cur,
                vggt_device,
            )
            if target_feature_level == "premerge":
                future_qwen = executor.submit(
                    extract_premerge_qwen_features,
                    qwen_visual,
                    pixel_values_cur,
                    image_grid_thw_cur,
                    qwen_device,
                )
            else:
                future_qwen = executor.submit(
                    extract_merged_qwen_features,
                    qwen_visual,
                    pixel_values_cur,
                    image_grid_thw_cur,
                    qwen_device,
                )
            return future_vggt.result(), future_qwen.result()

    for eidx, entry in enumerate(tqdm(entries, desc="Collecting 2D feature probe samples")):
        frame_sources = resolve_entry_frames(
            entry=entry,
            media_root=media_root,
            max_frames_per_entry=max_frames_per_entry,
            seed=seed + eidx,
        )
        if not frame_sources:
            continue

        used_entries += 1

        for start in range(0, len(frame_sources), max(1, frames_per_forward)):
            chunk_frames = frame_sources[start : start + max(1, frames_per_forward)]
            try:
                pixel_values, image_grid_thw, geometry_encoder_inputs = build_model_inputs(chunk_frames, processor)
            except Exception:
                skipped_frames += len(chunk_frames)
                continue

            try:
                vggt_chunk, qwen_chunk = run_feature_extract(
                    pixel_values_cur=pixel_values,
                    image_grid_thw_cur=image_grid_thw,
                    geometry_encoder_inputs_cur=geometry_encoder_inputs,
                )
            except Exception:
                skipped_frames += len(chunk_frames)
                continue

            if target_feature_level == "premerge":
                qwen_by_frame = split_flat_premerge_features(qwen_chunk, image_grid_thw)
            else:
                qwen_by_frame = split_flat_merged_features(qwen_chunk, image_grid_thw, merge_size=merge_size)
            out_h_full = int(geometry_encoder_inputs.shape[-2] // patch_size)
            out_w_full = int(geometry_encoder_inputs.shape[-1] // patch_size)

            for local_i, qwen_feat in enumerate(qwen_by_frame):
                used_frames += 1
                if local_i >= vggt_chunk.shape[0]:
                    skipped_frames += 1
                    continue

                q_h = int(image_grid_thw[local_i, 1].item())
                q_w = int(image_grid_thw[local_i, 2].item())
                vggt_feat = regroup_vggt_tokens(
                    vggt_chunk[local_i],
                    height_patches=out_h_full,
                    width_patches=out_w_full,
                    target_h=q_h,
                    target_w=q_w,
                    merge_size=merge_size,
                )
                if target_feature_level == "merged":
                    vggt_feat = merge_grouped_vggt_tokens(vggt_feat, merge_size=merge_size)

                n_patch = min(vggt_feat.shape[0], qwen_feat.shape[0])
                if n_patch <= 0:
                    skipped_frames += 1
                    continue
                vggt_feat = vggt_feat[:n_patch]
                qwen_feat = qwen_feat[:n_patch]

                if patches_per_frame > 0 and n_patch > patches_per_frame:
                    idx = torch.randperm(n_patch, generator=patch_rng)[:patches_per_frame]
                else:
                    idx = torch.arange(n_patch)

                remaining = max_patch_samples - total_patches if max_patch_samples > 0 else idx.numel()
                if remaining <= 0:
                    break
                if idx.numel() > remaining:
                    perm = torch.randperm(idx.numel(), generator=patch_rng)[:remaining]
                    idx = idx[perm]

                x_list.append(vggt_feat[idx])
                y_list.append(qwen_feat[idx])
                total_patches += idx.numel()

            if max_patch_samples > 0 and total_patches >= max_patch_samples:
                break

        if max_patch_samples > 0 and total_patches >= max_patch_samples:
            break

    if not x_list:
        raise RuntimeError("No valid (VGGT, 2D feature) patch samples collected.")

    x = torch.cat(x_list, dim=0).to(torch.float32)
    y = torch.cat(y_list, dim=0).to(torch.float32)
    stats = {
        "used_entries": used_entries,
        "used_frames": used_frames,
        "skipped_frames": skipped_frames,
        "total_patch_samples": int(x.shape[0]),
        "feature_dim_in": int(x.shape[1]),
        "feature_dim_out": int(y.shape[1]),
    }
    return x, y, stats


def run_probe(
    *,
    annotation_path: str,
    media_root: str,
    sample_ratio: float,
    seed: int,
    max_entries: int,
    max_collect_steps: int,
    max_frames_per_entry: int,
    frames_per_forward: int,
    patches_per_frame: int,
    max_patch_samples: int,
    val_ratio: float,
    ridge_lambda: float,
    probe_train_steps: int,
    probe_lr: float,
    probe_batch_size: int,
    channel_ratio: float,
    target_feature_level: str,
    probe_device: torch.device,
    qwen_device: torch.device,
    vggt_device: torch.device,
    vggt_model: VGGT,
    qwen_visual,
    processor,
) -> Dict[str, object]:
    set_seed(seed)
    sampled_entries, total_entries = sample_entries(
        annotation_path=annotation_path,
        sample_ratio=sample_ratio,
        seed=seed,
        max_entries=max_entries,
    )
    if not sampled_entries:
        raise RuntimeError("No sampled entries found.")

    sampled_entries_before_cap = len(sampled_entries)
    if max_collect_steps > 0:
        sampled_entries = sampled_entries[:max_collect_steps]

    x, y, collect_stats = collect_probe_dataset(
        entries=sampled_entries,
        media_root=media_root,
        vggt_model=vggt_model,
        qwen_visual=qwen_visual,
        processor=processor,
        qwen_device=qwen_device,
        vggt_device=vggt_device,
        patches_per_frame=patches_per_frame,
        max_patch_samples=max_patch_samples,
        max_frames_per_entry=max_frames_per_entry,
        frames_per_forward=frames_per_forward,
        seed=seed,
        target_feature_level=target_feature_level,
    )

    n = x.shape[0]
    val_n = max(1, int(n * val_ratio))
    split_rng = torch.Generator(device="cpu")
    split_rng.manual_seed(seed + 100003)
    perm = torch.randperm(n, generator=split_rng)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    if train_idx.numel() == 0:
        raise RuntimeError("Not enough samples for train split.")

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    weight, bias, importance, probe_step_losses, probe_norm_stats = train_linear_probe_with_logs_multi(
        x_train=x_train,
        y_train=y_train,
        ridge_lambda=ridge_lambda,
        steps=probe_train_steps,
        lr=probe_lr,
        batch_size=probe_batch_size,
        seed=seed,
        device=probe_device,
    )
    train_metrics = eval_probe_multi(x_train, y_train, weight, bias)
    val_metrics = eval_probe_multi(x_val, y_val, weight, bias)

    c = importance.numel()
    k = max(1, int(c * channel_ratio))
    topk = torch.topk(importance, k=k, largest=True)
    sensitive_channels = topk.indices

    payload = {
        "seed": seed,
        "total_entries": total_entries,
        "sampled_entries_before_collect_cap": sampled_entries_before_cap,
        "sampled_entries_used_for_collection": len(sampled_entries),
        "collection_stats": collect_stats,
        "split": {
            "train_samples": int(x_train.shape[0]),
            "val_samples": int(x_val.shape[0]),
            "val_ratio": val_ratio,
        },
        "probe": {
            "ridge_lambda": ridge_lambda,
            "train_steps": probe_train_steps,
            "train_lr": probe_lr,
            "train_batch_size": probe_batch_size,
            "normalization": probe_norm_stats,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_loss_steps": probe_step_losses,
        },
        "feature_2d_sensitive_channels": {
            "channel_ratio": channel_ratio,
            "count": int(sensitive_channels.numel()),
            "importance_aggregation": "mean_abs_weight_normalized",
            "indices": sensitive_channels.tolist(),
            "importance_topk": topk.values.tolist(),
        },
        "target_feature_level": target_feature_level,
    }

    del x, y, x_train, y_train, x_val, y_val, weight, bias, importance
    gc.collect()
    for dev in {probe_device, qwen_device, vggt_device}:
        if dev.type == "cuda":
            with torch.cuda.device(dev):
                torch.cuda.empty_cache()

    return payload


def compare_channel_sets(run_a: Dict[str, object], run_b: Dict[str, object]) -> Dict[str, object]:
    channels_a = [int(x) for x in run_a["feature_2d_sensitive_channels"]["indices"]]
    channels_b = [int(x) for x in run_b["feature_2d_sensitive_channels"]["indices"]]

    set_a = set(channels_a)
    set_b = set(channels_b)
    overlap = sorted(set_a & set_b)
    a_only = sorted(set_a - set_b)
    b_only = sorted(set_b - set_a)
    union_count = len(set_a | set_b)

    return {
        "seed_a": run_a["seed"],
        "seed_b": run_b["seed"],
        "exact_same_set": set_a == set_b,
        "exact_same_order": channels_a == channels_b,
        "channel_count_a": len(channels_a),
        "channel_count_b": len(channels_b),
        "overlap_count": len(overlap),
        "overlap_ratio_vs_a": len(overlap) / max(len(channels_a), 1),
        "overlap_ratio_vs_b": len(overlap) / max(len(channels_b), 1),
        "jaccard": len(overlap) / max(union_count, 1),
        "overlap_channels": overlap,
        "seed_a_only_channels": a_only,
        "seed_b_only_channels": b_only,
        "val_metrics_a": run_a["probe"]["val_metrics"],
        "val_metrics_b": run_b["probe"]["val_metrics"],
    }


def infer_qwen_visual_hidden_size(qwen_visual) -> Optional[int]:
    if hasattr(qwen_visual, "embed_dim"):
        return int(qwen_visual.embed_dim)
    if hasattr(qwen_visual, "patch_embed") and hasattr(qwen_visual.patch_embed, "embed_dim"):
        return int(qwen_visual.patch_embed.embed_dim)
    if hasattr(qwen_visual, "config") and hasattr(qwen_visual.config, "hidden_size"):
        return int(qwen_visual.config.hidden_size)
    return None


def resolve_runtime_devices(args: argparse.Namespace) -> Tuple[torch.device, torch.device, torch.device]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")

    probe_device = torch.device(args.probe_device or args.device)
    qwen_device = torch.device(args.qwen_device or args.device)
    vggt_device = torch.device(args.vggt_device or args.device)

    if args.auto_split_devices and probe_device.type == "cuda":
        if not args.qwen_device:
            qwen_device = torch.device("cuda:0")
        if not args.vggt_device:
            if torch.cuda.device_count() < 2:
                raise RuntimeError("--auto_split_devices requires at least 2 visible CUDA devices.")
            vggt_device = torch.device("cuda:1")
        if not args.probe_device:
            probe_device = qwen_device

    for name, dev in (
        ("probe_device", probe_device),
        ("qwen_device", qwen_device),
        ("vggt_device", vggt_device),
    ):
        if dev.type == "cuda" and dev.index is not None and dev.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"{name}={dev} is not visible. Visible CUDA device count={torch.cuda.device_count()}."
            )

    return probe_device, qwen_device, vggt_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 2D-feature linear probe twice with different seeds and compare sensitive channel overlap."
    )
    parser.add_argument("--annotation_path", type=str, required=True, help="Training annotation json/jsonl path.")
    parser.add_argument("--media_root", type=str, default="data/media", help="Root for relative image paths.")
    parser.add_argument("--sample_ratio", type=float, default=0.10, help="Entry sampling ratio from training set.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the first sampled subset and probe.")
    parser.add_argument("--resample_seed", type=int, default=43, help="Seed for the second sampled subset and probe.")
    parser.add_argument("--max_entries", type=int, default=0, help="Cap sampled entries (0 means no cap).")
    parser.add_argument(
        "--max_collect_steps",
        type=int,
        default=1000,
        help="Cap sampled entries processed during feature collection (0 means no cap).",
    )
    parser.add_argument("--max_frames_per_entry", type=int, default=8, help="Cap frames per entry.")
    parser.add_argument("--frames_per_forward", type=int, default=8, help="Frames per forward pass.")
    parser.add_argument("--patches_per_frame", type=int, default=256, help="Random patch cap per frame.")
    parser.add_argument(
        "--max_patch_samples",
        type=int,
        default=120000,
        help="Total patch sample cap for each run (0 means no cap).",
    )
    parser.add_argument("--qwen_model_path", type=str, default=QWEN_DEFAULT_MODEL_PATH, help="Qwen model path.")
    parser.add_argument("--vggt_model_path", type=str, default="facebook/VGGT-1B", help="VGGT model path.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--ridge_lambda", type=float, default=1e-3, help="Ridge penalty for probe training.")
    parser.add_argument("--probe_train_steps", type=int, default=1000, help="Linear probe optimizer steps.")
    parser.add_argument("--probe_lr", type=float, default=2e-3, help="Linear probe learning rate.")
    parser.add_argument("--probe_batch_size", type=int, default=4096, help="Linear probe batch size.")
    parser.add_argument("--channel_ratio", type=float, default=0.2, help="Top-k channel ratio.")
    parser.add_argument(
        "--target_feature_level",
        type=str,
        default="merged",
        choices=["merged", "premerge"],
        help="Target 2D feature level: merged repo `feature_2d` or raw pre-merge patch feature.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu.")
    parser.add_argument("--qwen_device", type=str, default="", help="Optional device for Qwen visual tower.")
    parser.add_argument("--vggt_device", type=str, default="", help="Optional device for VGGT encoder.")
    parser.add_argument("--probe_device", type=str, default="", help="Optional device for probe training.")
    parser.add_argument(
        "--auto_split_devices",
        action="store_true",
        help="If multiple CUDA devices are visible, place Qwen on cuda:0 and VGGT on cuda:1.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Qwen/VGGT load dtype.",
    )
    parser.add_argument(
        "--disable_flash_attention_2",
        action="store_true",
        help="Disable FlashAttention 2 for the Qwen visual tower.",
    )
    parser.add_argument("--log_path", type=str, default="", help="Output log json path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed == args.resample_seed:
        raise ValueError("--resample_seed must differ from --seed.")

    set_seed(args.seed)
    probe_device, qwen_device, vggt_device = resolve_runtime_devices(args)

    qwen_load_kwargs = {"torch_dtype": maybe_load_dtype(args.dtype)}
    if not args.disable_flash_attention_2:
        qwen_load_kwargs["attn_implementation"] = "flash_attention_2"

    print(f"[INFO] loading Qwen model from: {args.qwen_model_path}")
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_model_path,
        **qwen_load_kwargs,
    )
    qwen_visual = qwen_model.visual.to(qwen_device)
    qwen_visual.eval()
    processor = AutoProcessor.from_pretrained(args.qwen_model_path, padding_side="left")
    del qwen_model
    gc.collect()

    print(f"[INFO] loading VGGT from: {args.vggt_model_path}")
    vggt_model = VGGT.from_pretrained(
        args.vggt_model_path,
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=False,
    )
    vggt_model = vggt_model.to(vggt_device)
    vggt_model.eval()
    print(
        f"[INFO] runtime devices: probe={probe_device}, qwen={qwen_device}, vggt={vggt_device}"
    )

    shared_kwargs = {
        "annotation_path": args.annotation_path,
        "media_root": args.media_root,
        "sample_ratio": args.sample_ratio,
        "max_entries": args.max_entries,
        "max_collect_steps": args.max_collect_steps,
        "max_frames_per_entry": args.max_frames_per_entry,
        "frames_per_forward": args.frames_per_forward,
        "patches_per_frame": args.patches_per_frame,
        "max_patch_samples": args.max_patch_samples,
        "val_ratio": args.val_ratio,
        "ridge_lambda": args.ridge_lambda,
        "probe_train_steps": args.probe_train_steps,
        "probe_lr": args.probe_lr,
        "probe_batch_size": args.probe_batch_size,
        "channel_ratio": args.channel_ratio,
        "target_feature_level": args.target_feature_level,
        "probe_device": probe_device,
        "qwen_device": qwen_device,
        "vggt_device": vggt_device,
        "vggt_model": vggt_model,
        "qwen_visual": qwen_visual,
        "processor": processor,
    }

    print(f"[INFO] running baseline seed={args.seed}")
    run_a = run_probe(seed=args.seed, **shared_kwargs)
    print(
        f"[RESULT] seed={args.seed} sensitive_channels={run_a['feature_2d_sensitive_channels']['count']} "
        f"val_mse={run_a['probe']['val_metrics']['mse']:.6f}"
    )

    print(f"[INFO] running resample seed={args.resample_seed}")
    run_b = run_probe(seed=args.resample_seed, **shared_kwargs)
    print(
        f"[RESULT] seed={args.resample_seed} sensitive_channels={run_b['feature_2d_sensitive_channels']['count']} "
        f"val_mse={run_b['probe']['val_metrics']['mse']:.6f}"
    )

    comparison = compare_channel_sets(run_a, run_b)

    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_path.strip()
    if not log_path:
        out_dir = ROOT_DIR / "logs" / "feature_2d_probe_compare"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(out_dir / f"feature_2d_probe_compare_seed{args.seed}_vs_{args.resample_seed}_{now}.json")
    else:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    qwen_hidden_size = infer_qwen_visual_hidden_size(qwen_visual)
    payload = {
        "timestamp_utc": now,
        "config": vars(args),
        "models": {
            "qwen_model_path": args.qwen_model_path,
            "vggt_model_path": args.vggt_model_path,
            "probe_device": str(probe_device),
            "qwen_device": str(qwen_device),
            "vggt_device": str(vggt_device),
            "qwen_visual_hidden_size": qwen_hidden_size,
            "qwen_patch_size": int(qwen_visual.patch_size),
            "qwen_merge_size": int(qwen_visual.spatial_merge_size),
        },
        "runs": {
            f"seed_{args.seed}": run_a,
            f"seed_{args.resample_seed}": run_b,
        },
        "comparison": comparison,
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[RESULT] exact_same_set:", comparison["exact_same_set"])
    print("[RESULT] exact_same_order:", comparison["exact_same_order"])
    print("[RESULT] overlap_count:", comparison["overlap_count"])
    print("[RESULT] overlap_ratio_vs_seed_a:", f"{comparison['overlap_ratio_vs_a']:.4f}")
    print("[RESULT] overlap_ratio_vs_seed_b:", f"{comparison['overlap_ratio_vs_b']:.4f}")
    print("[RESULT] jaccard:", f"{comparison['jaccard']:.4f}")
    print("[RESULT] log saved to:", log_path)


if __name__ == "__main__":
    main()
