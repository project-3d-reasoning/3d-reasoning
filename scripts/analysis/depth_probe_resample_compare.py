#!/usr/bin/env python3
"""
Rerun the depth probe on a newly resampled subset and compare channel overlap.

This script:
1. Loads a baseline depth probe log.
2. Reuses the same probe configuration by default.
3. Overrides the sampling seed (and optional selected args) to resample entries.
4. Launches a fresh depth probe run.
5. Compares depth-related channels with the baseline log and saves a comparison json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parents[2]
DEPTH_PROBE_SCRIPT = ROOT_DIR / "scripts" / "analysis" / "depth_channel_probe.py"

CONFIG_KEYS = [
    "annotation_path",
    "media_root",
    "sample_ratio",
    "max_entries",
    "max_collect_steps",
    "max_frames_per_entry",
    "frames_per_forward",
    "patches_per_frame",
    "max_patch_samples",
    "target_size",
    "patch_size",
    "depth_scale",
    "vggt_model_path",
    "val_ratio",
    "ridge_lambda",
    "probe_train_steps",
    "probe_lr",
    "probe_batch_size",
    "channel_ratio",
    "target_drop_ratio",
    "sigma_min",
    "sigma_max",
    "sigma_search_steps",
    "noise_eval_repeats",
    "device",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun depth probe with a new sample seed and compare channel overlap."
    )
    parser.add_argument(
        "--baseline_log_path",
        type=str,
        default="logs/depth_probe/depth_probe_20260421_053625.json",
        help="Baseline depth probe log json path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="New random seed for resampling and probe training.",
    )
    parser.add_argument(
        "--probe_output_dir",
        type=str,
        default="logs/depth_probe_resample",
        help="Directory to save the new probe log.",
    )
    parser.add_argument(
        "--compare_output_dir",
        type=str,
        default="logs/depth_probe_compare",
        help="Directory to save the overlap comparison json.",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=-1.0,
        help="Override sample ratio. Negative keeps baseline config.",
    )
    parser.add_argument(
        "--max_collect_steps",
        type=int,
        default=-1,
        help="Override max_collect_steps. Negative keeps baseline config.",
    )
    parser.add_argument(
        "--probe_train_steps",
        type=int,
        default=-1,
        help="Override probe_train_steps. Negative keeps baseline config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Override device. Empty keeps baseline config.",
    )
    parser.add_argument(
        "--vggt_model_path",
        type=str,
        default="",
        help="Override VGGT model path. Empty keeps baseline config.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_probe_config(baseline_config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(baseline_config)
    config["seed"] = args.seed
    if args.sample_ratio >= 0:
        config["sample_ratio"] = args.sample_ratio
    if args.max_collect_steps >= 0:
        config["max_collect_steps"] = args.max_collect_steps
    if args.probe_train_steps >= 0:
        config["probe_train_steps"] = args.probe_train_steps
    if args.device:
        config["device"] = args.device
    if args.vggt_model_path:
        config["vggt_model_path"] = args.vggt_model_path
    return config


def build_probe_command(config: Dict[str, Any], log_path: Path, use_intrinsics: bool) -> List[str]:
    cmd = [sys.executable, str(DEPTH_PROBE_SCRIPT)]
    for key in CONFIG_KEYS:
        value = config[key]
        cmd.extend([f"--{key}", str(value)])
    if use_intrinsics:
        cmd.append("--use_intrinsics")
    cmd.extend(["--seed", str(config["seed"])])
    cmd.extend(["--log_path", str(log_path)])
    return cmd


def compare_channel_sets(
    baseline_log: Dict[str, Any],
    new_log: Dict[str, Any],
    baseline_log_path: Path,
    new_log_path: Path,
) -> Dict[str, Any]:
    baseline_channels = [int(x) for x in baseline_log["depth_related_channels"]["indices"]]
    new_channels = [int(x) for x in new_log["depth_related_channels"]["indices"]]

    baseline_set = set(baseline_channels)
    new_set = set(new_channels)
    overlap = sorted(baseline_set & new_set)
    baseline_only = sorted(baseline_set - new_set)
    new_only = sorted(new_set - baseline_set)
    union_count = len(baseline_set | new_set)

    overlap_count = len(overlap)
    baseline_count = len(baseline_channels)
    new_count = len(new_channels)

    return {
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "baseline_log_path": str(baseline_log_path),
        "new_log_path": str(new_log_path),
        "baseline_seed": baseline_log.get("config", {}).get("seed"),
        "new_seed": new_log.get("config", {}).get("seed"),
        "baseline_channel_count": baseline_count,
        "new_channel_count": new_count,
        "exact_same_set": baseline_set == new_set,
        "exact_same_order": baseline_channels == new_channels,
        "overlap_count": overlap_count,
        "overlap_ratio_vs_baseline": overlap_count / max(baseline_count, 1),
        "overlap_ratio_vs_new": overlap_count / max(new_count, 1),
        "jaccard": overlap_count / max(union_count, 1),
        "overlap_channels": overlap,
        "baseline_only_channels": baseline_only,
        "new_only_channels": new_only,
        "baseline_probe_val_metrics": baseline_log.get("probe", {}).get("val_metrics", {}),
        "new_probe_val_metrics": new_log.get("probe", {}).get("val_metrics", {}),
    }


def main() -> None:
    args = parse_args()
    baseline_log_path = (ROOT_DIR / args.baseline_log_path).resolve() if not Path(args.baseline_log_path).is_absolute() else Path(args.baseline_log_path)
    if not baseline_log_path.exists():
        raise FileNotFoundError(f"Baseline log not found: {baseline_log_path}")

    baseline_log = load_json(baseline_log_path)
    baseline_config = baseline_log.get("config", {})
    if not baseline_config:
        raise RuntimeError(f"Baseline log does not contain config: {baseline_log_path}")

    probe_config = make_probe_config(baseline_config, args)
    use_intrinsics = bool(probe_config.get("use_intrinsics", False))

    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    probe_output_dir = ROOT_DIR / args.probe_output_dir
    compare_output_dir = ROOT_DIR / args.compare_output_dir
    probe_output_dir.mkdir(parents=True, exist_ok=True)
    compare_output_dir.mkdir(parents=True, exist_ok=True)

    new_log_path = probe_output_dir / f"depth_probe_seed{args.seed}_{now}.json"
    compare_log_path = compare_output_dir / f"depth_probe_compare_seed{args.seed}_{now}.json"

    cmd = build_probe_command(probe_config, new_log_path, use_intrinsics=use_intrinsics)
    print("[INFO] baseline log:", baseline_log_path)
    print("[INFO] new seed:", args.seed)
    print("[INFO] new probe log:", new_log_path)
    print("[INFO] compare log:", compare_log_path)
    print("[INFO] running command:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)

    new_log = load_json(new_log_path)
    compare_payload = compare_channel_sets(
        baseline_log=baseline_log,
        new_log=new_log,
        baseline_log_path=baseline_log_path,
        new_log_path=new_log_path,
    )

    with open(compare_log_path, "w", encoding="utf-8") as f:
        json.dump(compare_payload, f, indent=2)

    print("[RESULT] exact_same_set:", compare_payload["exact_same_set"])
    print("[RESULT] exact_same_order:", compare_payload["exact_same_order"])
    print("[RESULT] overlap_count:", compare_payload["overlap_count"])
    print("[RESULT] overlap_ratio_vs_baseline:", f"{compare_payload['overlap_ratio_vs_baseline']:.4f}")
    print("[RESULT] overlap_ratio_vs_new:", f"{compare_payload['overlap_ratio_vs_new']:.4f}")
    print("[RESULT] jaccard:", f"{compare_payload['jaccard']:.4f}")
    print("[RESULT] compare log saved to:", compare_log_path)


if __name__ == "__main__":
    main()
