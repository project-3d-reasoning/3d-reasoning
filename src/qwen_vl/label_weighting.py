"""Utilities for token-aligned label weighting."""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch


LABEL_WEIGHT_MASK_SUFFIX = ".label_weight_masks.pt"
SCAN2CAP_LEXICON_FILENAME = "scan2cap_span_lexicon.json"

LABEL_WEIGHT_CODE_IGNORE = 0
LABEL_WEIGHT_CODE_DEFAULT = 1
LABEL_WEIGHT_CODE_SCANREFER_FRAME = 2
LABEL_WEIGHT_CODE_SCANREFER_BBOX = 3
LABEL_WEIGHT_CODE_SCAN2CAP_CATEGORY = 4
LABEL_WEIGHT_CODE_SCAN2CAP_ATTRIBUTE = 5
LABEL_WEIGHT_CODE_SCAN2CAP_RELATION = 6
LABEL_WEIGHT_CODE_SCANNET_DET_LABEL = 7
LABEL_WEIGHT_CODE_SCANNET_DET_BBOX = 8

LABEL_WEIGHT_CODE_TO_NAME: Dict[int, str] = {
    LABEL_WEIGHT_CODE_IGNORE: "ignore",
    LABEL_WEIGHT_CODE_DEFAULT: "default_supervised",
    LABEL_WEIGHT_CODE_SCANREFER_FRAME: "scanrefer_frame",
    LABEL_WEIGHT_CODE_SCANREFER_BBOX: "scanrefer_bbox",
    LABEL_WEIGHT_CODE_SCAN2CAP_CATEGORY: "scan2cap_category",
    LABEL_WEIGHT_CODE_SCAN2CAP_ATTRIBUTE: "scan2cap_attribute",
    LABEL_WEIGHT_CODE_SCAN2CAP_RELATION: "scan2cap_relation",
    LABEL_WEIGHT_CODE_SCANNET_DET_LABEL: "scannet_det_label",
    LABEL_WEIGHT_CODE_SCANNET_DET_BBOX: "scannet_det_bbox",
}

LABEL_WEIGHT_NAME_TO_CODE: Dict[str, int] = {
    name: code for code, name in LABEL_WEIGHT_CODE_TO_NAME.items()
}

LABEL_WEIGHT_MAX_CODE = max(LABEL_WEIGHT_CODE_TO_NAME)


def get_label_weight_mask_path(annotation_path: str, masks_dir: str) -> str:
    base_name = os.path.splitext(os.path.basename(annotation_path))[0]
    return os.path.join(masks_dir, f"{base_name}{LABEL_WEIGHT_MASK_SUFFIX}")


class LabelWeightMaskStore:
    """Compact storage for per-sample assistant token codes."""

    def __init__(self, path: str):
        payload = torch.load(path, map_location="cpu")
        self.path = path
        self.flat_codes = payload["flat_codes"].to(torch.uint8).cpu()
        self.offsets = payload["offsets"].to(torch.long).cpu()
        self.code_to_name = payload.get("code_to_name", dict(LABEL_WEIGHT_CODE_TO_NAME))
        if self.offsets.ndim != 1 or self.offsets.numel() == 0:
            raise ValueError(f"Invalid offsets tensor in label weight mask file: {path}")
        if self.offsets[-1].item() != self.flat_codes.numel():
            raise ValueError(
                f"Mask store offsets do not match flat code size in {path}: "
                f"{self.offsets[-1].item()} vs {self.flat_codes.numel()}"
            )

    def __len__(self) -> int:
        return max(int(self.offsets.numel()) - 1, 0)

    def get_codes(self, sample_idx: int) -> torch.Tensor:
        if sample_idx < 0 or sample_idx >= len(self):
            raise IndexError(f"Sample index {sample_idx} is out of range for {self.path}")
        start = int(self.offsets[sample_idx].item())
        end = int(self.offsets[sample_idx + 1].item())
        return self.flat_codes[start:end]


def build_full_label_weight_codes(
    labels: torch.Tensor,
    assistant_codes: Optional[torch.Tensor],
    ignore_index: int = -100,
) -> torch.Tensor:
    """Expand assistant-content codes into a full sequence aligned with labels."""

    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.long)
    full_codes = torch.zeros_like(labels, dtype=torch.uint8)

    supervised_positions = torch.nonzero(labels != ignore_index, as_tuple=False).squeeze(-1)
    if supervised_positions.numel() == 0:
        return full_codes

    full_codes[supervised_positions] = LABEL_WEIGHT_CODE_DEFAULT
    if assistant_codes is None:
        return full_codes

    if not isinstance(assistant_codes, torch.Tensor):
        assistant_codes = torch.as_tensor(assistant_codes, dtype=torch.uint8)
    else:
        assistant_codes = assistant_codes.to(dtype=torch.uint8)

    num_content_tokens = min(int(supervised_positions.numel()), int(assistant_codes.numel()))
    if num_content_tokens <= 0:
        return full_codes

    content_positions = supervised_positions[:num_content_tokens]
    content_codes = assistant_codes[:num_content_tokens].to(device=full_codes.device)
    full_codes[content_positions] = torch.where(
        content_codes > 0,
        content_codes,
        full_codes[content_positions],
    )
    return full_codes


def get_label_weight_value_table(config, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Map compact token codes to runtime scalar weights."""

    table = torch.ones(LABEL_WEIGHT_MAX_CODE + 1, device=device, dtype=dtype)
    table[LABEL_WEIGHT_CODE_IGNORE] = 0.0
    table[LABEL_WEIGHT_CODE_DEFAULT] = float(getattr(config, "label_weight_default", 1.0))
    table[LABEL_WEIGHT_CODE_SCANREFER_FRAME] = float(
        getattr(config, "label_weight_scanrefer_frame", 1.0)
    )
    table[LABEL_WEIGHT_CODE_SCANREFER_BBOX] = float(
        getattr(config, "label_weight_scanrefer_bbox", 1.0)
    )
    table[LABEL_WEIGHT_CODE_SCAN2CAP_CATEGORY] = float(
        getattr(config, "label_weight_scan2cap_category", 1.0)
    )
    table[LABEL_WEIGHT_CODE_SCAN2CAP_ATTRIBUTE] = float(
        getattr(config, "label_weight_scan2cap_attribute", 1.0)
    )
    table[LABEL_WEIGHT_CODE_SCAN2CAP_RELATION] = float(
        getattr(config, "label_weight_scan2cap_relation", 1.0)
    )
    table[LABEL_WEIGHT_CODE_SCANNET_DET_LABEL] = float(
        getattr(config, "label_weight_scannet_det_label", 1.0)
    )
    table[LABEL_WEIGHT_CODE_SCANNET_DET_BBOX] = float(
        getattr(config, "label_weight_scannet_det_bbox", 1.0)
    )
    return table
