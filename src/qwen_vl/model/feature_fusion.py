"""Feature fusion modules for combining 2D and 3D features."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vggt.layers.rope import RotaryPositionEmbedding2D


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
    
    def forward(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> torch.Tensor:
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
            return features_2d + features_3d
            
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


class LearnableQueryPrefixEncoder(nn.Module):
    """Extract a fixed number of prefix tokens from a variable-length memory."""

    def __init__(self, hidden_size: int, num_queries: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        head_dim = hidden_size // num_heads
        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by 4 to apply 2D RoPE cleanly in cross-attention."
            )

        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dropout = dropout
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_size) / math.sqrt(hidden_size))
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.rope = RotaryPositionEmbedding2D()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)

    def forward(self, memory: Optional[torch.Tensor], memory_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if memory is not None and memory.dim() != 3:
            raise ValueError(f"Expected memory to have shape [batch, seq, hidden], but got {tuple(memory.shape)}")
        if memory_positions is not None:
            if memory is None:
                raise ValueError("memory_positions was provided without memory")
            if memory_positions.shape[:2] != memory.shape[:2] or memory_positions.shape[-1] != 2:
                raise ValueError(
                    f"Expected memory_positions to have shape [batch, seq, 2], but got {tuple(memory_positions.shape)}"
                )

        if memory is None:
            batch_size = 1
            device = self.query_tokens.device
            dtype = self.query_tokens.dtype
        else:
            batch_size = memory.shape[0]
            device = memory.device
            dtype = memory.dtype

        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        if memory is None or memory.shape[1] == 0:
            x = queries
        else:
            query_states = self._reshape_heads(self.q_proj(self.query_norm(queries)))
            key_states = self._reshape_heads(self.k_proj(self.memory_norm(memory)))
            value_states = self._reshape_heads(self.v_proj(self.memory_norm(memory)))

            if memory_positions is not None:
                memory_positions = memory_positions.to(device=device)
                query_positions = torch.zeros(
                    batch_size,
                    queries.shape[1],
                    2,
                    device=device,
                    dtype=memory_positions.dtype,
                )
                query_states = self.rope(query_states, query_positions)
                key_states = self.rope(key_states, memory_positions)

            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                dropout_p=self.attn_dropout if self.training else 0.0,
            )
            attn_output = self.out_proj(self._merge_heads(attn_output))
            x = queries + attn_output

        x = x + self.mlp(self.output_norm(x))
        return x


def _sample_features_for_hsic(x: torch.Tensor, y: torch.Tensor, max_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    if max_samples > 0 and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0], device=x.device)[:max_samples]
        x = x.index_select(0, indices)
        y = y.index_select(0, indices)
    return x, y


def _pairwise_squared_distances(features: torch.Tensor) -> torch.Tensor:
    features_norm = (features ** 2).sum(dim=-1, keepdim=True)
    distances = features_norm + features_norm.transpose(-2, -1) - 2.0 * torch.matmul(features, features.transpose(-2, -1))
    return distances.clamp_min(0.0)


def _center_kernel(kernel: torch.Tensor) -> torch.Tensor:
    row_mean = kernel.mean(dim=-1, keepdim=True)
    col_mean = kernel.mean(dim=-2, keepdim=True)
    total_mean = kernel.mean(dim=(-2, -1), keepdim=True)
    return kernel - row_mean - col_mean + total_mean


def _resolve_rbf_sigma(distances: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = float(sigma)
    if sigma != -1:
        return distances.new_tensor(max(sigma, 1e-6))

    if distances.shape[-1] <= 1:
        return distances.new_tensor(1.0)

    with torch.no_grad():
        non_diagonal_mask = ~torch.eye(distances.shape[-1], dtype=torch.bool, device=distances.device)
        positive = distances.masked_select(non_diagonal_mask & (distances > 0))
        if positive.numel() == 0:
            return distances.new_tensor(1.0)
        sigma_estimate = positive.median().sqrt().clamp_min(1e-6)
    return sigma_estimate


def _rbf_kernel(distances: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma_tensor = _resolve_rbf_sigma(distances, sigma)
    sigma_sq = sigma_tensor.pow(2).clamp_min(1e-6)
    return torch.exp(-distances / (2.0 * sigma_sq))


def compute_rbf_hsic_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma_x: float = -1.0,
    sigma_y: float = -1.0,
    max_samples: int = 256,
) -> torch.Tensor:
    """Biased RBF-HSIC estimate used as a dependence penalty."""

    if x.shape != y.shape:
        raise ValueError(f"HSIC expects matching feature shapes, got {tuple(x.shape)} and {tuple(y.shape)}")

    if x.ndim != 2:
        raise ValueError(f"HSIC expects rank-2 tensors, got rank {x.ndim}")

    if x.shape[0] <= 1:
        return x.new_zeros(())

    x = x.float()
    y = y.float()
    x, y = _sample_features_for_hsic(x, y, max_samples)
    num_samples = x.shape[0]
    if num_samples <= 1:
        return x.new_zeros(())

    distances_x = _pairwise_squared_distances(x)
    distances_y = _pairwise_squared_distances(y)
    kernel_x = _center_kernel(_rbf_kernel(distances_x, sigma_x))
    kernel_y = _center_kernel(_rbf_kernel(distances_y, sigma_y))
    hsic = (kernel_x * kernel_y).sum() / ((num_samples - 1) ** 2)
    return hsic.mean().clamp_min(0.0)


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
        self.context_dim = context_dim
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

    def _group_spatial_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Group neighboring patches into merger windows without applying the projection MLP."""
        n_image, h_patch, w_patch, dim = x.shape
        h_patch = h_patch // self.merge_size * self.merge_size
        w_patch = w_patch // self.merge_size * self.merge_size
        x = x[:, :h_patch, :w_patch, :]
        x = x.reshape(
            n_image,
            h_patch // self.merge_size,
            self.merge_size,
            w_patch // self.merge_size,
            self.merge_size,
            dim,
        )
        return x.permute(0, 1, 3, 2, 4, 5).contiguous()

    def merge_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return merged geometry tokens before the merger MLP projection."""
        grouped = self._group_spatial_tokens(x)
        n_image, h_merge, w_merge, _, _, _ = grouped.shape
        return grouped.reshape(n_image, h_merge, w_merge, self.input_dim)

    def project_flat_merged_tokens(self, merged_tokens: torch.Tensor) -> torch.Tensor:
        """Project flattened merged tokens to the target hidden size."""
        if merged_tokens.numel() == 0:
            return merged_tokens.new_empty((0, self.output_dim))

        if self.merger_type == "mlp":
            grouped = merged_tokens.reshape(-1, self.merge_size, self.merge_size, self.context_dim)
            x = self.mlp(self.ln_q(grouped).reshape(-1, self.input_dim))
        elif self.merger_type == "avg":
            grouped = merged_tokens.reshape(-1, self.merge_size, self.merge_size, self.context_dim)
            x = grouped.mean(dim=(1, 2))
            x = self.mlp(x)
        else:
            raise NotImplementedError(f"Merger type {self.merger_type} not implemented")

        return x

    def project_merged_tokens(self, merged_tokens: torch.Tensor) -> torch.Tensor:
        """Project merged tokens laid out on the spatial grid."""
        n_image, h_merge, w_merge, _ = merged_tokens.shape
        x = self.project_flat_merged_tokens(merged_tokens.reshape(-1, self.input_dim))
        x = x.reshape(n_image, h_merge, w_merge, -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the merger."""
        merged_tokens = self.merge_tokens(x)
        return self.project_merged_tokens(merged_tokens)


class LASTViTSparseProjector(nn.Module):
    """Select stable geometry tokens in LAST-ViT style and project only the top-n selected ones."""

    def __init__(self, top_k: int = 1, top_n: int = 32, eps: float = 1e-6):
        super().__init__()
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")
        if top_n <= 0:
            raise ValueError(f"top_n must be > 0, but got {top_n}")

        self.top_k = int(top_k)
        self.top_n = int(top_n)
        self.eps = float(eps)

    def gaussian_kernel_1d(self, kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coords = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=device, dtype=dtype)
        kernel = torch.exp(-0.5 * (coords / max(sigma, self.eps)) ** 2)
        kernel = kernel / kernel.max().clamp_min(self.eps)
        return kernel

    def _compute_stability_scores(self, merged_tokens: torch.Tensor) -> torch.Tensor:
        flat_tokens = merged_tokens.reshape(merged_tokens.shape[0], -1, merged_tokens.shape[-1]).float()
        channel_dim = flat_tokens.shape[-1]
        kernel = self.gaussian_kernel_1d(
            kernel_size=channel_dim,
            sigma=channel_dim ** 0.5,
            device=flat_tokens.device,
            dtype=flat_tokens.dtype,
        ).view(1, 1, -1)

        filtered = torch.fft.fft(flat_tokens, dim=-1)
        filtered = torch.fft.fftshift(filtered, dim=-1)
        filtered = filtered * kernel
        filtered = torch.fft.ifftshift(filtered, dim=-1)
        filtered = torch.fft.ifft(filtered, dim=-1).real
        return flat_tokens / (filtered - flat_tokens).abs().clamp_min(self.eps)

    def forward(
        self,
        merged_tokens: torch.Tensor,
        projector: GeometryFeatureMerger,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if merged_tokens.dim() != 4:
            raise ValueError(f"Expected merged_tokens to have shape [batch, h, w, dim], got {tuple(merged_tokens.shape)}")

        batch_size, h_merge, w_merge, channel_dim = merged_tokens.shape
        token_count = h_merge * w_merge
        if token_count == 0:
            sparse_features = merged_tokens.new_zeros(batch_size, h_merge, w_merge, projector.output_dim)
            return sparse_features, {
                "cls_token": merged_tokens.new_zeros(batch_size, channel_dim),
                "selection_counts": merged_tokens.new_zeros(batch_size, h_merge, w_merge, dtype=torch.int64),
                "selected_mask": merged_tokens.new_zeros(batch_size, h_merge, w_merge, dtype=torch.bool),
            }

        flat_tokens = merged_tokens.reshape(batch_size, token_count, channel_dim)
        stability_scores = self._compute_stability_scores(merged_tokens)

        k = min(self.top_k, token_count)
        n = min(self.top_n, token_count)

        _, indices = torch.topk(stability_scores, k=k, dim=1, largest=True)
        selected_for_cls = torch.gather(flat_tokens, 1, indices.to(flat_tokens.device))
        cls_token = selected_for_cls.mean(dim=1)

        selection_counts = torch.zeros(batch_size, token_count, device=flat_tokens.device, dtype=torch.int64)
        flat_indices = indices.permute(0, 2, 1).reshape(batch_size, -1)
        selection_counts.scatter_add_(
            1,
            flat_indices,
            torch.ones_like(flat_indices, dtype=selection_counts.dtype),
        )

        _, top_n_indices = torch.topk(selection_counts, k=n, dim=1, largest=True)
        selected_mask = torch.zeros(batch_size, token_count, device=flat_tokens.device, dtype=torch.bool)
        selected_mask.scatter_(1, top_n_indices, True)

        projected_selected = projector.project_flat_merged_tokens(flat_tokens[selected_mask])
        sparse_features = flat_tokens.new_zeros(batch_size, token_count, projector.output_dim)
        sparse_features[selected_mask] = projected_selected.to(sparse_features.dtype)
        sparse_features = sparse_features.reshape(batch_size, h_merge, w_merge, projector.output_dim)

        return sparse_features, {
            "cls_token": cls_token,
            "selection_counts": selection_counts.reshape(batch_size, h_merge, w_merge),
            "selected_mask": selected_mask.reshape(batch_size, h_merge, w_merge),
        }
