"""VGGT geometry encoder implementation."""

import torch
import torch.nn as nn
from typing import Dict

from .base import BaseGeometryEncoder, GeometryEncoderConfig


class VGGTEncoder(BaseGeometryEncoder):
    """VGGT geometry encoder wrapper."""
    
    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)

        # Lazy import to avoid circular dependencies
        from ..vggt.models.vggt import VGGT

        self.enable_point_head = bool(config.encoder_kwargs.get("enable_point", False))

        # Initialize VGGT model
        self.vggt = VGGT(
            enable_camera=False,
            enable_point=self.enable_point_head,
            enable_depth=False,
            enable_track=False,
        )

        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False

        self.reference_frame = config.reference_frame    
        self.patch_size = 14
        
    
    def _forward_impl(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the VGGT aggregator and optional dense geometry heads once."""
        grad_enabled = self.training and any(param.requires_grad for param in self.vggt.parameters())
        if grad_enabled:
            self.vggt.train()
        else:
            self.vggt.eval()

        # Apply reference frame transformation
        images = self._apply_reference_frame_transform(images)
        
        # Determine dtype for mixed precision
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.set_grad_enabled(grad_enabled):
            with torch.cuda.amp.autocast(dtype=dtype):
                # Get aggregated tokens from VGGT
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images[None])
                features = aggregated_tokens_list[-2][0, :, patch_start_idx:]

                outputs = {"features": features}
                if self.enable_point_head and self.vggt.point_head is not None:
                    pts3d, pts3d_conf = self.vggt.point_head(
                        aggregated_tokens_list,
                        images=images[None],
                        patch_start_idx=patch_start_idx,
                    )
                    outputs["world_points"] = pts3d[0]
                    outputs["world_points_conf"] = pts3d_conf[0]

        outputs["features"] = self._apply_inverse_reference_frame_transform(outputs["features"])
        if "world_points" in outputs:
            outputs["world_points"] = self._apply_inverse_reference_frame_transform(outputs["world_points"])
        if "world_points_conf" in outputs:
            outputs["world_points_conf"] = self._apply_inverse_reference_frame_transform(outputs["world_points_conf"])
        return outputs

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using VGGT."""
        return self._forward_impl(images)["features"]

    def encode_with_aux(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode images and return optional dense world point predictions."""
        return self._forward_impl(images)
    
    def get_feature_dim(self) -> int:
        """Get VGGT feature dimension."""
        return 2048  # VGGT feature dimension
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for compatibility."""
        return self.encode(images)
    
    def _apply_reference_frame_transform(self, images: torch.Tensor) -> torch.Tensor:
        """Apply reference frame transformation if needed."""
        if self.reference_frame != "first":
            return torch.flip(images, dims=(0,))
        return images
    
    def _apply_inverse_reference_frame_transform(self, features: torch.Tensor) -> torch.Tensor:
        """Apply inverse reference frame transformation if needed."""
        if self.reference_frame != "first":
            return torch.flip(features, dims=(0,))
        return features

    
    def load_model(self, model_path: str) -> None:
        """Load pretrained VGGT model."""
        from ..vggt.models.vggt import VGGT
        self.vggt = VGGT.from_pretrained(
            model_path,
            enable_camera=False,
            enable_point=self.enable_point_head,
            enable_depth=False,
            enable_track=False,
        )

        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False
