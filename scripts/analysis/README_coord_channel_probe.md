# Coordinate Channel Probe Script

Files:

- `scripts/analysis/coord_channel_probe.py`
- `scripts/analysis/run_coord_channel_probe.sh`

This probe follows the same style as the depth-channel probe, but uses coordinate targets:

1. Sample training entries (default 10%).
2. For each frame, use depth + pose matrix to convert each pixel into **first-frame camera coordinates**.
3. Average per patch to form a 3D target `(x, y, z)`.
4. Train a linear probe on VGGT pre-merge tokens to predict patch-average coordinates.
5. Pick top-20% channels by probe weight norm.
6. Add Gaussian noise and tune `sigma` so Euclidean loss increases by ~30%.
7. Save metrics, per-step probe loss, channel indices, and tuned sigma to a json log.

## Quick Run

```bash
bash scripts/analysis/run_coord_channel_probe.sh
```

## Notes

- The script expects ScanNet-style files per frame:
  - RGB: `xxxx.jpg`
  - Depth: `xxxx.png`
  - Pose: `xxxx.txt` (4x4)
- Scene intrinsics read from:
  - `intrinsic.txt`
  - `depth_intrinsic.txt`
- Defaults:
  - `MAX_COLLECT_STEPS=1000`
  - `PROBE_TRAIN_STEPS=1000`
  - `MAX_PATCH_SAMPLES=0` (no cap)

