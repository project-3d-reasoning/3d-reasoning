# Depth Channel Probe Script

This folder provides a utility script:

- `scripts/analysis/depth_channel_probe.py`
- `scripts/analysis/run_depth_channel_probe.sh`

It implements the requested pipeline:

1. Sample 10% of a training annotation set.
2. Extract pre-merge VGGT geometry tokens.
3. Train a linear probe for patch mean depth.
4. Select top-20% channels by absolute probe weight.
5. Add Gaussian noise on those channels, tune `sigma` for ~30% probe degradation.
6. Save `sigma`, channel indices, metrics, and per-step probe training loss into a json log file.

## Example

```bash
python3 scripts/analysis/depth_channel_probe.py \
  --annotation_path data/train/scanrefer_train_32frames.json \
  --media_root data/media \
  --sample_ratio 0.1 \
  --vggt_model_path facebook/VGGT-1B \
  --use_intrinsics
```

Or use the one-command launcher:

```bash
bash scripts/analysis/run_depth_channel_probe.sh
```

## Notes

- The script expects ScanNet-style frame layout where each RGB file has a depth file with the same basename:
  - `xxx.jpg` -> `xxx.png`
- Intrinsics are read from scene directory:
  - `intrinsic.txt`
  - `depth_intrinsic.txt`
- If memory is tight, reduce:
  - `--frames_per_forward`
  - `--patches_per_frame`
  - `--max_patch_samples`
- Probe training defaults:
  - `--probe_train_steps 1000`
  - `--probe_batch_size 8192`
