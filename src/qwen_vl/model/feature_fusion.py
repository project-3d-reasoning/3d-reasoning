"""Feature fusion modules for combining 2D and 3D features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class FeatureFusionConfig:
    """Configuration for feature fusion."""
    fusion_method: str = "add"  # "add", "concat", "gated", "weighted", "cross_attention"
    hidden_size: int = 3584
    num_heads: int = 8
    dropout: float = 0.1
    num_layers: int = 1


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block with position encoding, MLP and residual connections."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Layer norms
        self.norm1_query = nn.LayerNorm(hidden_size)
        self.norm1_key = nn.LayerNorm(hidden_size)
        self.norm1_value = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def get_2d_sincos_pos_embed(self, height: int, width: int, embed_dim: int, device: torch.device) -> torch.Tensor:
        """
        Generate 2D sinusoidal position embeddings.
        
        Args:
            height: Height of the grid
            width: Width of the grid  
            embed_dim: Embedding dimension
            device: Device to create tensor on
            
        Returns:
            pos_embed: Position embeddings of shape [height*width, embed_dim]
        """
        # Generate grid coordinates
        grid_h = torch.arange(height, dtype=torch.float32, device=device)
        grid_w = torch.arange(width, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=0)  # [2, height, width]
        
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        return pos_embed
        
    def get_2d_sincos_pos_embed_from_grid(self, embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
        """
        Generate 2D sinusoidal position embeddings from grid.
        
        Args:
            embed_dim: Embedding dimension
            grid: Grid coordinates of shape [2, height, width]
            
        Returns:
            pos_embed: Position embeddings of shape [height*width, embed_dim]
        """
        assert embed_dim % 2 == 0
        
        # Use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # [height*width, embed_dim//2]
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # [height*width, embed_dim//2]
        
        emb = torch.cat([emb_h, emb_w], dim=1)  # [height*width, embed_dim]
        return emb
        
    def get_1d_sincos_pos_embed_from_grid(self, embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
        """
        Generate 1D sinusoidal position embeddings.
        
        Args:
            embed_dim: Embedding dimension
            pos: Position tensor of shape [height, width]
            
        Returns:
            emb: Position embeddings of shape [height*width, embed_dim]
        """
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # [embed_dim//2]
        
        pos = pos.flatten()
        out = torch.einsum('m,d->md', pos, omega)  # [height*width, embed_dim//2], outer product
        
        emb_sin = torch.sin(out)  # [height*width, embed_dim//2]
        emb_cos = torch.cos(out)  # [height*width, embed_dim//2]
        
        emb = torch.cat([emb_sin, emb_cos], dim=1)  # [height*width, embed_dim]
        return emb
        
    def forward(self, features_2d: torch.Tensor, features_3d: torch.Tensor, h_grid: int, w_grid: int) -> torch.Tensor:
        # Normalize features
        query = self.norm1_query(features_2d)
        key = self.norm1_key(features_3d)
        value = self.norm1_value(features_3d)
        
        # Add batch dimension if needed
        if query.dim() == 2:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Generate 2D position embeddings
        pos_embed = self.get_2d_sincos_pos_embed(h_grid, w_grid, self.hidden_size, query.device).to(query.dtype)  # [h_grid*w_grid, hidden_size]

        # Add position embeddings to query and key
        # Assuming features are organized as [batch_size, h_grid*w_grid, hidden_size]
        query = query + pos_embed.unsqueeze(0)  # Broadcast across batch dimension
        key = key + pos_embed.unsqueeze(0)
            
        # Cross-attention: 2D features as query, 3D features as key/value
        attn_output, _ = self.cross_attention(query, key, value)
        
        if squeeze_output:
            attn_output = attn_output.squeeze(0)
            
        # First residual connection
        x = features_2d + attn_output
        
        # MLP with second residual connection
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output
        
        return x


class CoordinatePositionalEncoding(nn.Module):
    """Patch-level sinusoidal encoding from first-frame 3D coordinates."""

    def __init__(
        self,
        output_dim: int,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        sub_patch_divisions_per_patch: int = 2,
        base: float = 10000.0,
        coord_scale: float = 10.0,
    ):
        super().__init__()
        if output_dim % 2 != 0:
            raise ValueError("output_dim must be even for sin/cos coordinate encoding")
        if patch_size % sub_patch_divisions_per_patch != 0:
            raise ValueError(
                "patch_size must be divisible by sub_patch_divisions_per_patch for hierarchical coord pooling"
            )

        self.output_dim = output_dim
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.pixel_merge_size = patch_size * spatial_merge_size
        self.sub_patch_divisions_per_patch = sub_patch_divisions_per_patch
        self.sub_patch_size = patch_size // sub_patch_divisions_per_patch
        self.subdivisions_per_axis = spatial_merge_size * sub_patch_divisions_per_patch
        self.subgroup_count = self.subdivisions_per_axis ** 2
        self.base = base
        self.coord_scale = coord_scale

        self.axis_pair_dims = self._build_axis_pair_dims(output_dim)
        total_pairs = output_dim // 2
        subgroup_base_pairs = total_pairs // self.subgroup_count
        subgroup_remainder = total_pairs % self.subgroup_count
        self.subgroup_axis_pair_dims = [
            self._build_axis_pair_dims(
                (subgroup_base_pairs + (1 if subgroup_idx < subgroup_remainder else 0)) * 2
            )
            for subgroup_idx in range(self.subgroup_count)
        ]
        self.axis_embed_dims = [pair_dim * 2 for pair_dim in self.axis_pair_dims]

    def _build_axis_pair_dims(self, output_dim: int) -> list[int]:
        if output_dim % 2 != 0:
            raise ValueError("output_dim must be even for sin/cos coordinate encoding")
        total_pairs = output_dim // 2
        base_pairs = total_pairs // 3
        remainder = total_pairs % 3
        return [
            base_pairs + (1 if axis_idx < remainder else 0)
            for axis_idx in range(3)
        ]

    def _encode_axis(self, values: torch.Tensor, pair_dim: int) -> torch.Tensor:
        if pair_dim == 0:
            return values.new_zeros(values.shape + (0,))
        omega = torch.arange(pair_dim, dtype=torch.float32, device=values.device)
        omega = 1.0 / (self.base ** (omega / pair_dim))
        out = values.float().unsqueeze(-1) * omega
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return emb.to(values.dtype)

    def encode_points(self, coord_points: torch.Tensor, axis_pair_dims: Optional[list[int]] = None) -> torch.Tensor:
        if coord_points.shape[-1] != 3:
            raise ValueError(f"coord_points must end with 3 values, got shape {tuple(coord_points.shape)}")

        if axis_pair_dims is None:
            axis_pair_dims = self.axis_pair_dims

        scaled_points = coord_points / self.coord_scale
        x_embed = self._encode_axis(scaled_points[..., 0], axis_pair_dims[0])
        y_embed = self._encode_axis(scaled_points[..., 1], axis_pair_dims[1])
        z_embed = self._encode_axis(scaled_points[..., 2], axis_pair_dims[2])
        return torch.cat([x_embed, y_embed, z_embed], dim=-1)

    def forward(self, coord_points: torch.Tensor, coord_mask: torch.Tensor) -> torch.Tensor:
        if coord_points.dim() == 3:
            coord_points = coord_points.unsqueeze(0)
        if coord_mask.dim() == 2:
            coord_mask = coord_mask.unsqueeze(0)

        _, height, width, _ = coord_points.shape
        valid_height = (height // self.pixel_merge_size) * self.pixel_merge_size
        valid_width = (width // self.pixel_merge_size) * self.pixel_merge_size
        coord_points = coord_points[:, :valid_height, :valid_width, :]
        coord_mask = coord_mask[:, :valid_height, :valid_width].to(coord_points.dtype)

        coord_points = coord_points.permute(0, 3, 1, 2)
        coord_mask = coord_mask.unsqueeze(1)
        kernel = self.sub_patch_size
        area = kernel * kernel

        pooled_sum = F.avg_pool2d(coord_points * coord_mask, kernel_size=kernel, stride=kernel) * area
        pooled_count = F.avg_pool2d(coord_mask, kernel_size=kernel, stride=kernel) * area
        pooled_points = pooled_sum / pooled_count.clamp_min(1.0)

        batch_size = pooled_points.shape[0]
        h_grid = valid_height // self.pixel_merge_size
        w_grid = valid_width // self.pixel_merge_size
        pooled_points = pooled_points.permute(0, 2, 3, 1).reshape(
            batch_size,
            h_grid,
            self.subdivisions_per_axis,
            w_grid,
            self.subdivisions_per_axis,
            3,
        )
        pooled_points = pooled_points.permute(0, 1, 3, 2, 4, 5).reshape(
            batch_size,
            h_grid,
            w_grid,
            self.subgroup_count,
            3,
        )

        patch_valid = (pooled_count.squeeze(1) > 0).reshape(
            batch_size,
            h_grid,
            self.subdivisions_per_axis,
            w_grid,
            self.subdivisions_per_axis,
        )
        patch_valid = patch_valid.permute(0, 1, 3, 2, 4).reshape(
            batch_size,
            h_grid,
            w_grid,
            self.subgroup_count,
            1,
        )

        coord_embed_chunks = []
        for subgroup_idx, axis_pair_dims in enumerate(self.subgroup_axis_pair_dims):
            subgroup_embed = self.encode_points(
                pooled_points[..., subgroup_idx, :],
                axis_pair_dims=axis_pair_dims,
            )
            subgroup_embed = subgroup_embed * patch_valid[..., subgroup_idx, :].to(subgroup_embed.dtype)
            coord_embed_chunks.append(subgroup_embed)

        coord_embed = torch.cat(coord_embed_chunks, dim=-1)
        return coord_embed


class FeatureFusionModule(nn.Module):
    """Enhanced feature fusion module with multiple fusion strategies."""
    
    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_method = config.fusion_method
        self.hidden_size = config.hidden_size
        
        self._build_fusion_layers()
    
    def _build_fusion_layers(self):
        """Build fusion layers based on method."""
        if self.config.fusion_method == "concat":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
            
        elif self.config.fusion_method == "cross_attention":
            self.cross_attn_blocks = nn.ModuleList([
                CrossAttentionBlock(
                    self.hidden_size, 
                    self.config.num_heads, 
                    self.config.dropout
                ) 
                for _ in range(self.config.num_layers)
            ])

        elif self.config.fusion_method == "gated":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.gate_projection = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.Sigmoid()
            )
            
        elif self.config.fusion_method == "weighted":
            self.weight_2d = nn.Parameter(torch.tensor(0.5))
            self.weight_3d = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self,
        features_2d: torch.Tensor,
        features_3d: torch.Tensor,
        coord_pe: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse 2D and 3D features.
        
        Args:
            features_2d: 2D image features
            features_3d: 3D geometry features
        Returns:
            Fused features
        """

        _, h_grid, w_grid, _ = features_3d.shape
        if self.fusion_method == "add":
            fused = features_2d + features_3d
            if coord_pe is not None:
                fused = fused + coord_pe
            return fused
            
        elif self.fusion_method == "concat":
            features_2d = self.norm1(features_2d)
            features_3d = self.norm2(features_3d)
            concat_features = torch.cat([features_2d, features_3d], dim=-1)
            return self.projection(concat_features)
            
        elif self.fusion_method == "cross_attention":
            features_2d = features_2d.view(features_2d.size(0), -1, self.hidden_size)  # Flatten spatial dimensions
            features_3d = features_3d.view(features_3d.size(0), -1, self.hidden_size)
            x = features_2d
            for block in self.cross_attn_blocks:
                x = block(x, features_3d, h_grid, w_grid)
            return x
            
        elif self.fusion_method == "gated":
            features_2d = self.norm1(features_2d)
            features_3d = self.norm2(features_3d)
            concat_features = torch.cat([features_2d, features_3d], dim=-1)
            gate = self.gate_projection(concat_features)
            return gate * features_2d + (1 - gate) * features_3d
            
        elif self.fusion_method == "weighted":
            # Normalize weights to sum to 1
            weight_sum = self.weight_2d + self.weight_3d
            norm_weight_2d = self.weight_2d / weight_sum
            norm_weight_3d = self.weight_3d / weight_sum
            return norm_weight_2d * features_2d + norm_weight_3d * features_3d
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


class GeometryFeatureMerger(nn.Module):
    """Unified merger for geometry features from different encoders.
    
    Supports different merger types:
    - "mlp": MLP-based feature transformation with spatial merging
    - "avg": Average pooling across spatial merge dimensions
    - "attention": Attention-based merger (not implemented yet)
    """
    
    def __init__(self, output_dim: int, hidden_dim: int, context_dim: int, 
                 spatial_merge_size: int = 2, merger_type: str = "mlp"):
        super().__init__()
        self.merger_type = merger_type
        self.input_dim = context_dim * (spatial_merge_size ** 2)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.merge_size = spatial_merge_size
        
        if merger_type == "mlp":
            # Import here to avoid circular import
            try:
                from .modeling_qwen2_5_vl import Qwen2RMSNorm
            except ImportError:
                # Fallback to standard LayerNorm if Qwen2RMSNorm not available
                Qwen2RMSNorm = nn.LayerNorm
                
            self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif merger_type == "avg":
            self.mlp = nn.Sequential(
                nn.Linear(context_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif merger_type == "attention":
            # Add attention-based merger for future extensibility
            raise NotImplementedError("Attention merger not implemented yet")
        else:
            raise ValueError(f"Unknown merger type: {merger_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the merger."""

        n_image, h_patch, w_patch, dim = x.shape
        x = x[:, :h_patch // self.merge_size * self.merge_size, :w_patch // self.merge_size*self.merge_size , :]
        x = x.reshape(n_image, h_patch // self.merge_size, self.merge_size, w_patch // self.merge_size, self.merge_size, dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        if self.merger_type == "mlp":
            x = self.mlp(self.ln_q(x).view(-1, self.input_dim))
        elif self.merger_type == "avg":
            # Average pooling across spatial merge dimensions
            x = x.mean(dim=(3, 4))  # Average over the merge_size dimensions
            x = x.view(-1, dim)  # Flatten for projection
            x = self.mlp(x)
        else:
            raise NotImplementedError(f"Merger type {self.merger_type} not implemented")
        x = x.reshape(n_image, h_patch // self.merge_size, w_patch // self.merge_size, -1)
        return x
