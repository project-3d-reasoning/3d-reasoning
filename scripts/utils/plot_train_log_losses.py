#!/usr/bin/env python3
"""Plot loss curves from a training log into a single PNG figure.

This script parses log lines that contain Python-dict style metrics, e.g.
{'loss': 9.18, 'loss_ce': 0.55, 'loss_shared': 7.71, 'epoch': 0.53}

Example:
    python3 scripts/utils/plot_train_log_losses.py \
        /data7t-root/r1/dmgg/VG-LLM/3b-hsic/train.log \
        --output /data7t-root/r1/dmgg/VG-LLM/3b-hsic/train_losses.png
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "matplotlib is required for this script. "
        "On this machine, try running it with "
        "`/data2/miniconda3/envs/vgllm/bin/python`."
    ) from exc


MetricRecord = Dict[str, float]


PLOT_GROUPS = [
    ("Primary Losses", ["loss", "loss_ce", "loss_ce_raw"]),
    ("Aux Losses", ["loss_shared", "loss_ortho", "loss_recon"]),
    ("Weighted Aux Losses", ["loss_shared_weighted", "loss_ortho_weighted", "loss_recon_weighted"]),
    ("Target Ratios", ["loss_shared_ratio_target", "loss_ortho_ratio_target", "loss_recon_ratio_target"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path, help="Path to train.log")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <log_dir>/train_loss_curves.png",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.0,
        help="Optional EMA smoothing factor in [0, 1). Set 0 to disable smoothing.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output figure DPI.",
    )
    return parser.parse_args()


def _extract_metric_dict(line: str) -> MetricRecord | None:
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end <= start:
        return None

    payload = line[start : end + 1]
    try:
        data = ast.literal_eval(payload)
    except (ValueError, SyntaxError):
        return None

    if not isinstance(data, dict):
        return None

    metrics: MetricRecord = {}
    for key, value in data.items():
        if isinstance(value, (int, float)):
            metrics[str(key)] = float(value)
    if not any(key.startswith("loss") for key in metrics):
        return None
    return metrics


def load_metric_records(log_path: Path) -> List[MetricRecord]:
    records: List[MetricRecord] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            metrics = _extract_metric_dict(line)
            if metrics is not None:
                records.append(metrics)
    return records


def compute_ema(values: Sequence[float], alpha: float) -> List[float]:
    if not values:
        return []
    if alpha <= 0.0:
        return list(values)

    smoothed = [float(values[0])]
    for value in values[1:]:
        smoothed.append(alpha * smoothed[-1] + (1.0 - alpha) * float(value))
    return smoothed


def select_x_axis(records: Sequence[MetricRecord]) -> tuple[List[float], str]:
    epochs = [record.get("epoch") for record in records]
    if epochs and all(epoch is not None for epoch in epochs):
        unique_epochs = len({round(float(epoch), 6) for epoch in epochs})
        if unique_epochs > 1:
            return [float(epoch) for epoch in epochs], "Epoch"
    return [float(index + 1) for index in range(len(records))], "Log Index"


def collect_metric_series(records: Sequence[MetricRecord], metric_name: str) -> tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for index, record in enumerate(records):
        value = record.get(metric_name)
        if value is None:
            continue
        xs.append(index)
        ys.append(float(value))
    return xs, ys


def _index_to_axis_values(indices: Iterable[int], axis_values: Sequence[float]) -> List[float]:
    return [float(axis_values[index]) for index in indices]


def plot_groups(
    records: Sequence[MetricRecord],
    output_path: Path,
    ema_alpha: float,
    dpi: int,
) -> None:
    if not records:
        raise ValueError("No metric records were parsed from the log file.")

    axis_values, xlabel = select_x_axis(records)
    figure, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    axes = axes.flatten()

    for axis, (title, metric_names) in zip(axes, PLOT_GROUPS):
        plotted_any = False
        for metric_name in metric_names:
            indices, values = collect_metric_series(records, metric_name)
            if not values:
                continue
            x_values = _index_to_axis_values(indices, axis_values)
            if ema_alpha > 0.0:
                axis.plot(x_values, values, alpha=0.22, linewidth=1.0, label=f"{metric_name} raw")
                axis.plot(
                    x_values,
                    compute_ema(values, ema_alpha),
                    linewidth=2.0,
                    label=f"{metric_name} ema",
                )
            else:
                axis.plot(x_values, values, linewidth=1.8, label=metric_name)
            plotted_any = True

        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Value")
        axis.grid(True, alpha=0.25)
        if plotted_any:
            axis.legend(fontsize=9)
        else:
            axis.text(0.5, 0.5, "No matching metrics found", ha="center", va="center", transform=axis.transAxes)

    figure.suptitle(
        f"Training Loss Curves\n{output_path.parent.name}/{output_path.name}",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {args.log_path}")
    if not (0.0 <= args.ema_alpha < 1.0):
        raise ValueError("--ema-alpha must be in [0, 1).")

    output_path = args.output
    if output_path is None:
        output_path = args.log_path.parent / "train_loss_curves.png"

    records = load_metric_records(args.log_path)
    plot_groups(records, output_path=output_path, ema_alpha=args.ema_alpha, dpi=args.dpi)
    print(f"Saved loss plot to: {output_path}")
    print(f"Parsed metric records: {len(records)}")


if __name__ == "__main__":
    main()
