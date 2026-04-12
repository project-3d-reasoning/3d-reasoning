"""Feature fusion modules for combining 2D and 3D features."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FeatureFusionConfig:
    """Configuration for feature fusion."""
    fusion_method: str = "add"  # "add", "concat", "gated", "weighted", "cross_attention"
    hidden_size: int = 3584
    num_heads: int = 8
    dropout: float = 0.1
    num_layers: int = 1
    use_hsic_fusion: bool = False
    backprop_hsic_loss: bool = True
    hsic_loss_weight: float = 0.0
    hsic_rbf_sigma_2d: float = 1.0
    hsic_rbf_sigma_3d: float = 1.0
    unique_3d_hsic_max_samples: int = -1
    hsic_projector_hidden_size: Optional[int] = None
    hsic_projector_dropout: float = 0.1


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
        if self.config.use_hsic_fusion:
            projector_hidden_size = self.config.hsic_projector_hidden_size or self.hidden_size
            self.feature_3d_projector = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, projector_hidden_size),
                nn.GELU(),
                nn.Dropout(self.config.hsic_projector_dropout),
                nn.Linear(projector_hidden_size, self.hidden_size),
            )

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

    @staticmethod
    def _reshape_features_for_hsic(features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 2:
            return features.unsqueeze(0)
        return features.reshape(features.shape[0], -1, features.shape[-1])

    @staticmethod
    def _pairwise_squared_distances(features: torch.Tensor) -> torch.Tensor:
        features_norm = (features ** 2).sum(dim=-1, keepdim=True)
        distances = features_norm + features_norm.transpose(-2, -1) - 2.0 * torch.matmul(features, features.transpose(-2, -1))
        return distances.clamp_min(0.0)

    @staticmethod
    def _center_kernel(kernel: torch.Tensor) -> torch.Tensor:
        row_mean = kernel.mean(dim=-1, keepdim=True)
        col_mean = kernel.mean(dim=-2, keepdim=True)
        total_mean = kernel.mean(dim=(-2, -1), keepdim=True)
        return kernel - row_mean - col_mean + total_mean

    def _resolve_rbf_sigma(self, distances: torch.Tensor, sigma: float) -> float:
        sigma = float(sigma)
        if sigma != -1:
            return max(sigma, 1e-6)

        # Median heuristic on pairwise Euclidean distances, excluding diagonal zeros.
        non_diagonal_mask = ~torch.eye(distances.shape[-1], dtype=torch.bool, device=distances.device).unsqueeze(0)
        valid_distances = distances.masked_select(non_diagonal_mask & (distances > 0))
        if valid_distances.numel() == 0:
            return 1.0

        sigma_estimate = valid_distances.median().sqrt().item()
        return max(sigma_estimate, 1e-6)

    def _compute_rbf_kernel(self, features: torch.Tensor, sigma: float) -> torch.Tensor:
        distances = self._pairwise_squared_distances(features.float())
        sigma = self._resolve_rbf_sigma(distances, sigma)
        gamma = 1.0 / (2.0 * sigma * sigma)
        return torch.exp(-gamma * distances)

    @staticmethod
    def _compute_feature_stats(features: torch.Tensor, prefix: str) -> Dict[str, torch.Tensor]:
        detached_features = features.detach().float()
        if detached_features.numel() == 0:
            zero = detached_features.new_zeros(())
            return {
                f"{prefix}_mean": zero,
                f"{prefix}_std": zero,
            }

        return {
            f"{prefix}_mean": detached_features.mean(),
            f"{prefix}_std": detached_features.std(unbiased=False),
        }

    @staticmethod
    def _subsample_features_for_hsic(
        features_2d: torch.Tensor,
        features_3d: torch.Tensor,
        max_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if max_samples <= 0 or features_2d.shape[1] <= max_samples:
            return features_2d, features_3d

        sampled_features_2d = []
        sampled_features_3d = []
        for feature_2d, feature_3d in zip(features_2d, features_3d):
            sample_indices = torch.randperm(feature_2d.shape[0], device=feature_2d.device)[:max_samples]
            sampled_features_2d.append(feature_2d.index_select(0, sample_indices))
            sampled_features_3d.append(feature_3d.index_select(0, sample_indices))

        return torch.stack(sampled_features_2d, dim=0), torch.stack(sampled_features_3d, dim=0)

    def _compute_rbf_hsic(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> torch.Tensor:
        features_2d = self._reshape_features_for_hsic(features_2d)
        features_3d = self._reshape_features_for_hsic(features_3d)

        if features_2d.shape != features_3d.shape:
            raise ValueError(
                "features_2d and features_3d must have the same shape for HSIC computation, "
                f"got {features_2d.shape} and {features_3d.shape}."
            )

        features_2d, features_3d = self._subsample_features_for_hsic(
            features_2d,
            features_3d,
            int(self.config.unique_3d_hsic_max_samples),
        )

        num_samples = features_2d.shape[1]
        if num_samples < 2:
            return features_2d.new_zeros(())

        kernel_2d = self._compute_rbf_kernel(features_2d, self.config.hsic_rbf_sigma_2d)
        kernel_3d = self._compute_rbf_kernel(features_3d, self.config.hsic_rbf_sigma_3d)
        centered_kernel_2d = self._center_kernel(kernel_2d)
        centered_kernel_3d = self._center_kernel(kernel_3d)

        denom = float((num_samples - 1) ** 2)
        hsic = (centered_kernel_2d * centered_kernel_3d).sum(dim=(-2, -1)) / denom
        return hsic.mean()

    def forward(
        self,
        features_2d: torch.Tensor,
        features_3d: torch.Tensor,
        compute_aux_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Fuse 2D and 3D features.
        
        Args:
            features_2d: 2D image features
            features_3d: 3D geometry features
        Returns:
            Fused features, optional raw HSIC loss, optional weighted HSIC loss,
            and optional logging stats for fused inputs
        """
        hsic_loss_raw = None
        hsic_loss_weighted = None
        aux_stats = None
        if self.config.use_hsic_fusion:
            if compute_aux_loss:
                if self.config.backprop_hsic_loss:
                    hsic_loss_raw = self._compute_rbf_hsic(features_2d, features_3d)
                else:
                    with torch.no_grad():
                        hsic_loss_raw = self._compute_rbf_hsic(features_2d, features_3d)
                hsic_loss_weighted = self.config.hsic_loss_weight * hsic_loss_raw

            feature_3d_projected = self.feature_3d_projector(features_3d)
            if compute_aux_loss:
                aux_stats = self._compute_feature_stats(features_2d, "feature_2d")
                aux_stats.update(self._compute_feature_stats(feature_3d_projected, "unique_3d_projected"))

            return features_2d + feature_3d_projected, hsic_loss_raw, hsic_loss_weighted, aux_stats

        _, h_grid, w_grid, _ = features_3d.shape
        if self.fusion_method == "add":
            return features_2d + features_3d, None, None, None
            
        elif self.fusion_method == "concat":
            features_2d = self.norm1(features_2d)
            features_3d = self.norm2(features_3d)
            concat_features = torch.cat([features_2d, features_3d], dim=-1)
            return self.projection(concat_features), None, None, None
            
        elif self.fusion_method == "cross_attention":
            features_2d = features_2d.view(features_2d.size(0), -1, self.hidden_size)  # Flatten spatial dimensions
            features_3d = features_3d.view(features_3d.size(0), -1, self.hidden_size)
            x = features_2d
            for block in self.cross_attn_blocks:
                x = block(x, features_3d, h_grid, w_grid)
            return x, None, None, None
            
        elif self.fusion_method == "gated":
            features_2d = self.norm1(features_2d)
            features_3d = self.norm2(features_3d)
            concat_features = torch.cat([features_2d, features_3d], dim=-1)
            gate = self.gate_projection(concat_features)
            return gate * features_2d + (1 - gate) * features_3d, None, None, None
            
        elif self.fusion_method == "weighted":
            # Normalize weights to sum to 1
            weight_sum = self.weight_2d + self.weight_3d
            norm_weight_2d = self.weight_2d / weight_sum
            norm_weight_3d = self.weight_3d / weight_sum
            return norm_weight_2d * features_2d + norm_weight_3d * features_3d, None, None, None
            
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
