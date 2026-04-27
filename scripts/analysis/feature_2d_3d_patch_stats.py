#!/usr/bin/env python3
"""
Collect per-location feature statistics for `feature_2d` and `feature_3d`
on the VSI-Bench test split.

The script samples videos from the VSI-Bench test set, runs the checkpoint's
2D visual encoder and 3D geometry branch, and accumulates mean/std for the
same frame slot, merged patch position, and channel across samples.

Saved artifacts:
- `*_stats.pt`: full tensors and compact derived summaries
- `*_summary.json`: human-readable run metadata and scalar summaries
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoProcessor

try:
    import decord
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install decord before running this script.") from exc

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install datasets before running this script.") from exc

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qwen_vl.data.utils import load_and_preprocess_images
from qwen_vl.model.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGenerationWithVGGT,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_vsibench_spec() -> Tuple[str, str]:
    from lmms_eval.tasks.vsibench import utils as vsibench_utils

    return vsibench_utils.dataset_path, resolve_vsibench_cache_dir(vsibench_utils.cache_dir)


def resolve_vsibench_cache_dir(preferred_cache_dir: Optional[str] = None) -> str:
    candidates: List[str] = []

    if preferred_cache_dir:
        candidates.append(os.path.expanduser(preferred_cache_dir))

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        candidates.append(os.path.join(os.path.expanduser(hf_home), "vsibench"))

    candidates.extend(
        [
            "/data7t-root/huggingface/vsibench",
            os.path.expanduser("~/.cache/huggingface/vsibench"),
        ]
    )

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isdir(candidate) and any(os.path.isdir(os.path.join(candidate, name)) for name in ("scannet", "scannetpp", "arkitscenes")):
            return candidate

    return os.path.expanduser(preferred_cache_dir) if preferred_cache_dir else candidates[0]


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def maybe_load_dtype(dtype_name: str) -> object:
    if dtype_name.lower() == "auto":
        return "auto"
    return resolve_torch_dtype(dtype_name)


def load_vsibench_test_split(dataset_path: str, cache_dir: str):
    load_kwargs = {"split": "test", "cache_dir": cache_dir}
    if not os.path.isdir(dataset_path):
        load_kwargs["token"] = True
    try:
        return load_dataset(dataset_path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("token", None)
        return load_dataset(dataset_path, **load_kwargs)


def build_sampling_records(dataset, sample_unit: str) -> List[dict]:
    if sample_unit == "doc":
        records = []
        for doc_idx, doc in enumerate(dataset):
            record = dict(doc)
            record["_doc_index"] = int(doc_idx)
            record["_sample_key"] = f"doc::{doc_idx}"
            records.append(record)
        return records

    if sample_unit != "unique_video":
        raise ValueError(f"Unsupported sample_unit: {sample_unit}")

    by_video: Dict[Tuple[str, str], dict] = {}
    for doc_idx, doc in enumerate(dataset):
        key = (str(doc["dataset"]), str(doc["scene_name"]))
        if key not in by_video:
            record = {
                "dataset": key[0],
                "scene_name": key[1],
                "_doc_index": int(doc_idx),
                "_sample_key": f"{key[0]}::{key[1]}",
                "_num_docs": 1,
            }
            by_video[key] = record
        else:
            by_video[key]["_num_docs"] += 1
    return list(by_video.values())


def sample_records(records: Sequence[dict], sample_limit: int, seed: int) -> List[dict]:
    if sample_limit <= 0 or sample_limit >= len(records):
        return list(records)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(records)), sample_limit))
    return [records[i] for i in chosen]


def resolve_video_path(cache_dir: str, dataset_name: str, scene_name: str) -> str:
    candidate_roots = [cache_dir]
    auto_cache_dir = resolve_vsibench_cache_dir(cache_dir)
    if auto_cache_dir != cache_dir:
        candidate_roots.append(auto_cache_dir)

    for root in candidate_roots:
        video_path = os.path.join(root, dataset_name, f"{scene_name}.mp4")
        if os.path.exists(video_path):
            return video_path

    tried = [os.path.join(root, dataset_name, f"{scene_name}.mp4") for root in candidate_roots]
    raise FileNotFoundError(f"Missing video. Tried: {tried}")


def sample_video_frames(
    video_path: str,
    max_num_frames: int,
    allow_short_videos: bool,
) -> Tuple[Optional[List[Image.Image]], int, List[int]]:
    vr = decord.VideoReader(video_path)
    num_frames = len(vr)

    if num_frames < max_num_frames and not allow_short_videos:
        return None, num_frames, []

    if num_frames <= max_num_frames:
        frame_indices = np.arange(num_frames)
    else:
        frame_indices = np.linspace(0, num_frames - 1, max_num_frames).astype(int)

    frames: List[Image.Image] = []
    for frame_idx in frame_indices.tolist():
        image = Image.fromarray(vr[frame_idx].asnumpy()).convert("RGB")
        setattr(image, "_vgllm_source_path", f"{video_path}#frame_{frame_idx}")
        frames.append(image)

    return frames, num_frames, frame_indices.tolist()


def build_model_inputs(
    frames: Sequence[Image.Image],
    processor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    patch_size = int(processor.image_processor.patch_size)
    merge_size = int(processor.image_processor.merge_size)

    geometry_encoder_inputs: List[torch.Tensor] = []
    image_inputs: List[torch.Tensor] = []

    for frame in frames:
        processed_image = load_and_preprocess_images([frame])[0]
        geometry_encoder_inputs.append(processed_image)

        _, height, width = processed_image.shape
        if (width // patch_size) % merge_size > 0:
            width = width - ((width // patch_size) % merge_size) * patch_size
        if (height // patch_size) % merge_size > 0:
            height = height - ((height // patch_size) % merge_size) * patch_size
        image_inputs.append(processed_image[:, :height, :width])

    # Use the image processor directly. The full Qwen processor expects text
    # and can crash on image-only inputs with a None-related error.
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


def flatten_signature(image_grid_thw: torch.Tensor, merge_size: int) -> Tuple[Tuple[int, int, int], ...]:
    rows = []
    for row in image_grid_thw.tolist():
        t, h, w = [int(v) for v in row]
        rows.append((t, h // merge_size, w // merge_size))
    return tuple(rows)


def signature_to_dict(signature: Tuple[Tuple[int, int, int], ...]) -> dict:
    return {
        "per_entry_merged_thw": [list(x) for x in signature],
        "num_entries": len(signature),
        "total_frames": int(sum(x[0] for x in signature)),
        "unique_merged_hw": sorted({(x[1], x[2]) for x in signature}),
    }


def split_flat_visual_embeddings(
    image_embeds: torch.Tensor,
    image_grid_thw: torch.Tensor,
    merge_size: int,
) -> torch.Tensor:
    per_frame: List[torch.Tensor] = []
    cursor = 0
    shapes = set()

    for row in image_grid_thw.tolist():
        t, h, w = [int(v) for v in row]
        merged_h = h // merge_size
        merged_w = w // merge_size
        num_tokens = t * merged_h * merged_w
        cur = image_embeds[cursor : cursor + num_tokens]
        cur = cur.view(t, merged_h, merged_w, -1)
        cursor += num_tokens
        for ti in range(t):
            per_frame.append(cur[ti])
            shapes.add((merged_h, merged_w))

    if len(shapes) != 1:
        raise ValueError(f"Found different merged feature shapes inside one sample: {sorted(shapes)}")

    return torch.stack(per_frame, dim=0)


def extract_sample_features(
    model,
    processor,
    frames: Sequence[Image.Image],
    device: torch.device,
) -> Tuple[Tuple[Tuple[int, int, int], ...], torch.Tensor, torch.Tensor, Dict[str, float]]:
    pixel_values, image_grid_thw, geometry_encoder_inputs = build_model_inputs(frames, processor)
    merge_size = int(model.visual.spatial_merge_size)

    signature = flatten_signature(image_grid_thw, merge_size=merge_size)

    pixel_values = pixel_values.to(device=device, dtype=model.visual.dtype)
    image_grid_thw = image_grid_thw.to(device=device)
    geometry_encoder_inputs = geometry_encoder_inputs.to(device=device)

    with torch.no_grad():
        flat_image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        feature_2d = split_flat_visual_embeddings(
            flat_image_embeds,
            image_grid_thw=image_grid_thw,
            merge_size=merge_size,
        )

        n_image, _, height, width = geometry_encoder_inputs.shape
        h_patch = height // model.geometry_encoder.patch_size
        w_patch = width // model.geometry_encoder.patch_size

        geometry_tokens = model.geometry_encoder.encode(geometry_encoder_inputs).to(flat_image_embeds.dtype)
        geometry_tokens = geometry_tokens.reshape(n_image, h_patch, w_patch, -1)
        feature_3d = model.geometry_merger(geometry_tokens)

        if feature_2d.shape != feature_3d.shape:
            raise ValueError(
                "feature_2d and feature_3d shape mismatch: "
                f"{tuple(feature_2d.shape)} vs {tuple(feature_3d.shape)}"
            )

        sample_metrics = {
            "mean_norm_ratio_3d_to_2d": float(
                (
                    feature_3d.float().norm(dim=-1)
                    / feature_2d.float().norm(dim=-1).clamp_min(1e-6)
                ).mean().item()
            ),
            "mean_abs_feature_3d": float(feature_3d.float().abs().mean().item()),
        }
        try:
            fused = model.feature_fusion(feature_2d, feature_3d)
            sample_metrics["mean_cosine_fused_vs_2d"] = float(
                F.cosine_similarity(fused.float(), feature_2d.float(), dim=-1).mean().item()
            )
            sample_metrics["mean_abs_fused_minus_2d"] = float((fused.float() - feature_2d.float()).abs().mean().item())
        except Exception:
            # Keep the main 2D/3D statistics pipeline alive even if auxiliary
            # fused-feature diagnostics fail for a checkpoint.
            sample_metrics["mean_cosine_fused_vs_2d"] = None
            sample_metrics["mean_abs_fused_minus_2d"] = None

    return signature, feature_2d.detach(), feature_3d.detach(), sample_metrics


@dataclass
class RunningMoments:
    mean: torch.Tensor
    m2: torch.Tensor
    count: int = 0

    @classmethod
    def zeros(cls, shape: Sequence[int], dtype: torch.dtype = torch.float32) -> "RunningMoments":
        return cls(
            mean=torch.zeros(shape, dtype=dtype, device="cpu"),
            m2=torch.zeros(shape, dtype=dtype, device="cpu"),
            count=0,
        )

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().to(device="cpu", dtype=self.mean.dtype)
        self.count += 1
        if self.count == 1:
            self.mean.copy_(x)
            return
        delta = x - self.mean
        self.mean.add_(delta / float(self.count))
        delta2 = x - self.mean
        self.m2.add_(delta * delta2)

    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.count == 0:
            raise RuntimeError("No samples were accumulated.")
        if self.count == 1:
            std = torch.zeros_like(self.mean)
        else:
            std = torch.sqrt((self.m2 / float(self.count - 1)).clamp_min(0))
        return self.mean.clone(), std


@dataclass
class SignatureGroup:
    signature: Tuple[Tuple[int, int, int], ...]
    feature_2d_stats: RunningMoments
    feature_3d_stats: RunningMoments
    kept_records: List[dict]
    sample_metric_history: List[Dict[str, float]]

    @classmethod
    def create(
        cls,
        signature: Tuple[Tuple[int, int, int], ...],
        feature_shape: Sequence[int],
    ) -> "SignatureGroup":
        return cls(
            signature=signature,
            feature_2d_stats=RunningMoments.zeros(feature_shape, dtype=torch.float32),
            feature_3d_stats=RunningMoments.zeros(feature_shape, dtype=torch.float32),
            kept_records=[],
            sample_metric_history=[],
        )


def tensor_quantiles(x: torch.Tensor, probs: Sequence[float]) -> Dict[str, float]:
    x = x.detach().float().reshape(-1)
    q = torch.quantile(x, torch.tensor(probs, dtype=x.dtype))
    return {f"q{int(p * 100):02d}": float(v.item()) for p, v in zip(probs, q)}


def topk_indices(values: torch.Tensor, k: int, largest: bool = True) -> List[int]:
    k = min(int(k), values.numel())
    if k <= 0:
        return []
    topk = torch.topk(values, k=k, largest=largest)
    return [int(i) for i in topk.indices.tolist()]


def cast_for_save(x: torch.Tensor, dtype_name: str) -> torch.Tensor:
    if dtype_name == "float16":
        return x.half()
    if dtype_name == "bfloat16":
        return x.bfloat16()
    if dtype_name == "float32":
        return x.float()
    raise ValueError(f"Unsupported save dtype: {dtype_name}")


def parse_args() -> argparse.Namespace:
    dataset_path_default, cache_dir_default = default_vsibench_spec()

    parser = argparse.ArgumentParser(description="Collect per-location 2D/3D feature statistics on VSI-Bench.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint to analyze.")
    parser.add_argument("--output_dir", type=str, default="logs/feature_patch_stats", help="Output directory.")
    parser.add_argument("--dataset_path", type=str, default=dataset_path_default, help="VSI-Bench dataset path/name.")
    parser.add_argument("--cache_dir", type=str, default=cache_dir_default, help="VSI-Bench cache dir.")
    parser.add_argument(
        "--sample_unit",
        type=str,
        default="unique_video",
        choices=["unique_video", "doc"],
        help="Sample unique videos by default to avoid repeated questions on the same video.",
    )
    parser.add_argument("--sample_limit", type=int, default=256, help="Number of sampled videos/docs. Use <=0 for all.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_num_frames", type=int, default=32, help="Number of uniformly sampled frames per video.")
    parser.add_argument(
        "--allow_short_videos",
        action="store_true",
        help="If set, videos with fewer than max_num_frames are kept with all available frames.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Saved tensor dtype.",
    )
    parser.add_argument(
        "--disable_flash_attention_2",
        action="store_true",
        help="Disable FlashAttention 2 loading. By default the script enables it to match eval-time setup.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Raise immediately on the first sample failure for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("This script currently requires CUDA.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = f"vsibench_patch_stats_{timestamp}"
    summary_path = output_dir / f"{run_name}_summary.json"
    stats_path = output_dir / f"{run_name}_stats.pt"
    error_log_path = output_dir / f"{run_name}_errors.log"

    args.cache_dir = resolve_vsibench_cache_dir(args.cache_dir)
    print(f"[INFO] loading VSI-Bench test split from dataset_path={args.dataset_path}")
    print(f"[INFO] using cache_dir={args.cache_dir}")
    dataset = load_vsibench_test_split(args.dataset_path, args.cache_dir)
    sampling_records = build_sampling_records(dataset, sample_unit=args.sample_unit)
    sampled_records = sample_records(sampling_records, sample_limit=args.sample_limit, seed=args.seed)
    print(
        f"[INFO] dataset docs={len(dataset)}, sampling_records={len(sampling_records)}, "
        f"sampled={len(sampled_records)}, sample_unit={args.sample_unit}"
    )

    print(f"[INFO] loading checkpoint from {args.checkpoint_path}")
    config = AutoConfig.from_pretrained(args.checkpoint_path)
    load_kwargs = {
        "config": config,
        "torch_dtype": maybe_load_dtype(args.dtype),
    }
    if not args.disable_flash_attention_2:
        load_kwargs["attn_implementation"] = "flash_attention_2"
        print("[INFO] attn_implementation=flash_attention_2")
    model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
        args.checkpoint_path,
        **load_kwargs,
    ).eval()
    device = torch.device(args.device)
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(args.checkpoint_path, padding_side="left")

    signature_groups: Dict[Tuple[Tuple[int, int, int], ...], SignatureGroup] = {}
    skipped_missing = 0
    skipped_short = 0
    skipped_errors = 0

    progress = tqdm(sampled_records, desc="Collecting VSI-Bench features")
    for record in progress:
        dataset_name = str(record["dataset"])
        scene_name = str(record["scene_name"])

        try:
            video_path = resolve_video_path(args.cache_dir, dataset_name, scene_name)
        except FileNotFoundError as exc:
            skipped_missing += 1
            print(f"[WARN] {exc}")
            continue

        frames, total_video_frames, sampled_frame_indices = sample_video_frames(
            video_path=video_path,
            max_num_frames=args.max_num_frames,
            allow_short_videos=args.allow_short_videos,
        )
        if frames is None:
            skipped_short += 1
            continue

        try:
            signature, feature_2d, feature_3d, sample_metrics = extract_sample_features(
                model=model,
                processor=processor,
                frames=frames,
                device=device,
            )
        except Exception as exc:
            skipped_errors += 1
            tb = traceback.format_exc()
            msg = f"[WARN] failed on {dataset_name}/{scene_name}: {type(exc).__name__}: {exc}"
            tqdm.write(msg)
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
                f.write(tb + "\n")
            if skipped_errors <= 3:
                tqdm.write(tb)
            if args.stop_on_error:
                raise
            continue

        if signature not in signature_groups:
            feature_shape = tuple(feature_2d.shape)
            signature_groups[signature] = SignatureGroup.create(signature=signature, feature_shape=feature_shape)
            print(f"[INFO] discovered signature={signature_to_dict(signature)}")
            print(f"[INFO] feature shape={feature_shape}")

        group = signature_groups[signature]
        group.feature_2d_stats.update(feature_2d.float())
        group.feature_3d_stats.update(feature_3d.float())
        group.sample_metric_history.append(sample_metrics)
        group.kept_records.append(
            {
                "dataset": dataset_name,
                "scene_name": scene_name,
                "video_path": video_path,
                "doc_index": int(record["_doc_index"]),
                "sample_key": str(record["_sample_key"]),
                "num_docs_for_video": int(record.get("_num_docs", 1)),
                "num_video_frames": int(total_video_frames),
                "sampled_frame_indices": [int(x) for x in sampled_frame_indices],
            }
        )
        total_success = sum(len(cur.kept_records) for cur in signature_groups.values())
        progress.set_postfix(kept=total_success, sig=len(signature_groups), err=skipped_errors)

    if not signature_groups:
        raise RuntimeError(
            "No valid samples were kept. "
            f"skipped_missing={skipped_missing}, skipped_short={skipped_short}, skipped_errors={skipped_errors}. "
            f"Please check cache_dir / frame count / model environment. "
            f"Sample failures are logged to: {error_log_path}"
        )

    target_signature, target_group = max(
        signature_groups.items(),
        key=lambda item: len(item[1].kept_records),
    )
    feature_2d_mean, feature_2d_std = target_group.feature_2d_stats.finalize()
    feature_3d_mean, feature_3d_std = target_group.feature_3d_stats.finalize()
    kept_records = target_group.kept_records
    sample_metric_history = target_group.sample_metric_history
    grouped_signature_counts = [
        {
            "signature": signature_to_dict(signature),
            "count": len(group.kept_records),
        }
        for signature, group in sorted(
            signature_groups.items(),
            key=lambda item: len(item[1].kept_records),
            reverse=True,
        )
    ]
    posthoc_dropped_other_signature_count = sum(
        len(group.kept_records) for signature, group in signature_groups.items() if signature != target_signature
    )
    print(f"[INFO] selected dominant signature={signature_to_dict(target_signature)}")
    print(
        f"[INFO] dominant signature kept={len(kept_records)}, "
        f"other_signature_samples={posthoc_dropped_other_signature_count}, "
        f"num_signatures={len(signature_groups)}"
    )

    feature_2d_std_patch = feature_2d_std.mean(dim=-1)
    feature_3d_std_patch = feature_3d_std.mean(dim=-1)
    feature_2d_std_channel = feature_2d_std.mean(dim=(0, 1, 2))
    feature_3d_std_channel = feature_3d_std.mean(dim=(0, 1, 2))
    std_ratio_patch = feature_3d_std_patch / feature_2d_std_patch.clamp_min(1e-6)
    std_ratio_channel = feature_3d_std_channel / feature_2d_std_channel.clamp_min(1e-6)

    stats_payload = {
        "checkpoint_path": args.checkpoint_path,
        "dataset_path": args.dataset_path,
        "cache_dir": args.cache_dir,
        "sample_unit": args.sample_unit,
        "target_signature": signature_to_dict(target_signature),
        "num_kept_samples": len(kept_records),
        "signature_group_counts": grouped_signature_counts,
        "feature_2d_mean": cast_for_save(feature_2d_mean, args.save_dtype),
        "feature_2d_std": cast_for_save(feature_2d_std, args.save_dtype),
        "feature_3d_mean": cast_for_save(feature_3d_mean, args.save_dtype),
        "feature_3d_std": cast_for_save(feature_3d_std, args.save_dtype),
        "feature_2d_std_patch": cast_for_save(feature_2d_std_patch, args.save_dtype),
        "feature_3d_std_patch": cast_for_save(feature_3d_std_patch, args.save_dtype),
        "feature_2d_std_channel": cast_for_save(feature_2d_std_channel, args.save_dtype),
        "feature_3d_std_channel": cast_for_save(feature_3d_std_channel, args.save_dtype),
        "std_ratio_patch": cast_for_save(std_ratio_patch, args.save_dtype),
        "std_ratio_channel": cast_for_save(std_ratio_channel, args.save_dtype),
        "kept_records": kept_records,
    }
    torch.save(stats_payload, stats_path)

    sample_metric_summary = {}
    if sample_metric_history:
        for key in sample_metric_history[0].keys():
            values = [float(x[key]) for x in sample_metric_history if x.get(key) is not None]
            if not values:
                sample_metric_summary[key] = None
                continue
            sample_metric_summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    summary = {
        "run_name": run_name,
        "checkpoint_path": args.checkpoint_path,
        "dataset_path": args.dataset_path,
        "cache_dir": args.cache_dir,
        "sample_unit": args.sample_unit,
        "sample_limit": int(args.sample_limit),
        "seed": int(args.seed),
        "max_num_frames": int(args.max_num_frames),
        "allow_short_videos": bool(args.allow_short_videos),
        "device": args.device,
        "dtype": args.dtype,
        "save_dtype": args.save_dtype,
        "dataset_doc_count": int(len(dataset)),
        "sampling_record_count": int(len(sampling_records)),
        "sampled_record_count": int(len(sampled_records)),
        "total_successful_samples_before_signature_selection": int(
            sum(len(group.kept_records) for group in signature_groups.values())
        ),
        "kept_count": int(len(kept_records)),
        "skipped_missing_video_count": int(skipped_missing),
        "skipped_short_video_count": int(skipped_short),
        "skipped_error_count": int(skipped_errors),
        "num_signature_groups": int(len(signature_groups)),
        "posthoc_dropped_other_signature_count": int(posthoc_dropped_other_signature_count),
        "target_signature": signature_to_dict(target_signature),
        "signature_group_counts": grouped_signature_counts,
        "feature_shape": list(feature_2d_mean.shape),
        "feature_2d_mean_abs_mean": float(feature_2d_mean.abs().mean().item()),
        "feature_3d_mean_abs_mean": float(feature_3d_mean.abs().mean().item()),
        "feature_2d_std_global_mean": float(feature_2d_std.mean().item()),
        "feature_3d_std_global_mean": float(feature_3d_std.mean().item()),
        "feature_2d_std_patch_quantiles": tensor_quantiles(feature_2d_std_patch, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]),
        "feature_3d_std_patch_quantiles": tensor_quantiles(feature_3d_std_patch, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]),
        "feature_2d_std_channel_quantiles": tensor_quantiles(feature_2d_std_channel, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]),
        "feature_3d_std_channel_quantiles": tensor_quantiles(feature_3d_std_channel, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]),
        "std_ratio_patch_quantiles": tensor_quantiles(std_ratio_patch, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]),
        "std_ratio_channel_quantiles": tensor_quantiles(std_ratio_channel, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]),
        "top_feature_2d_std_channels": topk_indices(feature_2d_std_channel, k=32, largest=True),
        "top_feature_3d_std_channels": topk_indices(feature_3d_std_channel, k=32, largest=True),
        "top_std_ratio_channels": topk_indices(std_ratio_channel, k=32, largest=True),
        "sample_metric_summary": sample_metric_summary,
        "stats_path": str(stats_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[RESULT] kept samples: {len(kept_records)}")
    print(f"[RESULT] stats saved to: {stats_path}")
    print(f"[RESULT] summary saved to: {summary_path}")
    print(f"[RESULT] error log saved to: {error_log_path}")


if __name__ == "__main__":
    main()
