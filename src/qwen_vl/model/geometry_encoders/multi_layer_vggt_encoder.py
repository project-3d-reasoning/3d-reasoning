"""Multi-layer VGGT geometry encoder implementation."""

import torch
import torch.nn as nn
from typing import Optional, List
from .base import BaseGeometryEncoder, GeometryEncoderConfig


class MultiLayerVGGTEncoder(BaseGeometryEncoder):
    """VGGT geometry encoder with multi-layer feature fusion.

    Extracts features from multiple layers of VGGT aggregator and fuses them
    to capture both low-level geometric details and high-level semantic information.
    """

    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)

        # Lazy import to avoid circular dependencies
        from ..vggt.models.vggt import VGGT

        # Initialize VGGT model
        self.vggt = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)

        # Multi-layer configuration
        self.use_multi_layer = getattr(config, 'use_multi_layer', True)
        self.layer_indices = getattr(config, 'layer_indices', [-3, -2, -1])  # Last 3 layers by default
        self.layer_fusion_method = getattr(config, 'layer_fusion_method', 'weighted')  # 'weighted', 'concat', 'mean'

        # Feature dimensions
        self.single_layer_dim = 2048  # VGGT feature dimension per layer

        if self.use_multi_layer and self.layer_fusion_method == 'concat':
            self.output_dim = self.single_layer_dim * len(self.layer_indices)
        else:
            self.output_dim = self.single_layer_dim

        # Layer fusion weights (learnable)
        if self.use_multi_layer and self.layer_fusion_method == 'weighted':
            self.layer_weights = nn.Parameter(
                torch.ones(len(self.layer_indices)) / len(self.layer_indices)
            )

        # Optional projection layer for concatenated features
        if self.use_multi_layer and self.layer_fusion_method == 'concat':
            self.projection = nn.Sequential(
                nn.LayerNorm(self.output_dim),
                nn.Linear(self.output_dim, self.single_layer_dim),
                nn.GELU(),
                nn.LayerNorm(self.single_layer_dim),
            )

        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False

        self.reference_frame = config.reference_frame
        self.patch_size = 14


    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using multi-layer VGGT features."""
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

                if self.use_multi_layer:
                    # Extract features from multiple layers
                    multi_layer_features = []
                    for idx in self.layer_indices:
                        layer_features = aggregated_tokens_list[idx][0, :, patch_start_idx:]
                        multi_layer_features.append(layer_features)

                    # Fuse multi-layer features
                    features = self._fuse_layers(multi_layer_features)
                else:
                    # Original single-layer behavior
                    features = aggregated_tokens_list[-2][0, :, patch_start_idx:]

        # Apply inverse reference frame transformation
        features = self._apply_inverse_reference_frame_transform(features)

        return features

    def _fuse_layers(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse features from multiple layers.

        Args:
            layer_features: List of feature tensors from different layers

        Returns:
            Fused feature tensor
        """
        if self.layer_fusion_method == 'mean':
            # Simple average
            return torch.stack(layer_features, dim=0).mean(dim=0)

        elif self.layer_fusion_method == 'weighted':
            # Learnable weighted sum
            weights = torch.softmax(self.layer_weights, dim=0)
            stacked = torch.stack(layer_features, dim=0)  # [num_layers, seq_len, dim]
            return (stacked * weights.view(-1, 1, 1)).sum(dim=0)

        elif self.layer_fusion_method == 'concat':
            # Concatenate and project
            concat_features = torch.cat(layer_features, dim=-1)  # [seq_len, num_layers * dim]
            return self.projection(concat_features)

        else:
            raise ValueError(f"Unknown layer fusion method: {self.layer_fusion_method}")

    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.output_dim

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
        self.vggt = VGGT.from_pretrained(model_path, enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)

        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False
