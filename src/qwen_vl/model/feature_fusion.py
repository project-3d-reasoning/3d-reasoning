"""Feature fusion modules for combining 2D and 3D features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class FeatureFusionConfig:
    """Configuration for feature fusion."""
    fusion_method: str = "add"  # "add", "concat", "gated", "weighted", "cross_attention", "decompose_add", "decompose_concat", "nrsr_add", "nrsr_concat", "knn_concat"
    hidden_size: int = 3584
    num_heads: int = 8
    dropout: float = 0.1
    num_layers: int = 1
    decompose_hidden_size: Optional[int] = None
    nrsr_hidden_size: Optional[int] = None
    align_mode: str = "cosine"  # "cosine"(default) or "infonce"
    align_temperature: float = 0.07
    ortho_mode: str = "cosine"  # "cosine"(default) or "mine" (mine mode uses vCLUB)
    mine_hidden_size: Optional[int] = None  # Hidden size for q_theta in vCLUB
    knn_k: int = 9  # Number of nearest neighbors from other frames for knn_concat; self token is added separately.
    knn_min_valid_ratio: float = 0.25  # Minimum confidence mass ratio inside the patch center window to keep a patch/token valid.
    knn_pos_mlp_hidden_size: Optional[int] = None  # Hidden width for the relative position MLP in knn_concat.


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


class KNNTransformerCrossAttentionBlock(nn.Module):
    """Transformer-style cross-attention block for per-token KNN memory."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_dropout = nn.Dropout(dropout)

        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        memory_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.query_norm(hidden_states).unsqueeze(1)
        memory = self.memory_norm(memory_states)
        key_padding_mask = None if memory_valid is None else ~memory_valid

        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=memory,
            value=memory,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        hidden_states = hidden_states + self.out_dropout(attn_output.squeeze(1))

        mlp_output = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = hidden_states + mlp_output
        return hidden_states, attn_weights.squeeze(2)


class FeatureFusionModule(nn.Module):
    """Enhanced feature fusion module with multiple fusion strategies."""
    
    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_method = str(config.fusion_method).lower()
        self.config.fusion_method = self.fusion_method
        self.hidden_size = config.hidden_size
        self.align_mode = config.align_mode.lower()
        if self.align_mode == "cos":
            self.align_mode = "cosine"
        if self.align_mode not in {"cosine", "infonce"}:
            raise ValueError(
                f"Unknown align_mode: {config.align_mode}. Supported values: cosine, infonce."
            )
        self.align_temperature = max(float(config.align_temperature), 1e-6)
        self.ortho_mode = config.ortho_mode.lower()
        if self.ortho_mode == "cos":
            self.ortho_mode = "cosine"
        if self.ortho_mode not in {"cosine", "mine"}:
            raise ValueError(
                f"Unknown ortho_mode: {config.ortho_mode}. Supported values: cosine, mine."
            )
        
        self._build_fusion_layers()

    def _compute_align_loss(self, shared_3d_f: torch.Tensor, features_2d_f: torch.Tensor) -> torch.Tensor:
        if self.align_mode == "infonce":
            # Flatten all patches in the current batch so negatives are other patches from the same batch.
            shared_3d_flat = shared_3d_f.reshape(-1, shared_3d_f.shape[-1])
            features_2d_flat = features_2d_f.reshape(-1, features_2d_f.shape[-1])
            if shared_3d_flat.numel() == 0:
                return shared_3d_f.new_zeros(())

            shared_3d_flat = F.normalize(shared_3d_flat, dim=-1, eps=1e-6)
            features_2d_flat = F.normalize(features_2d_flat, dim=-1, eps=1e-6)
            logits = torch.matmul(shared_3d_flat, features_2d_flat.transpose(0, 1)) / self.align_temperature
            targets = torch.arange(shared_3d_flat.shape[0], device=shared_3d_flat.device)
            return F.cross_entropy(logits, targets)

        return 1.0 - F.cosine_similarity(shared_3d_f, features_2d_f, dim=-1, eps=1e-6).mean()

    def _build_decompose_mapper(self, hidden_dim: int) -> nn.Module:
        """Build lightweight mapper for shared/unique 3D feature decomposition."""
        return nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.hidden_size),
        )
    
    def _build_fusion_layers(self):
        """Build fusion layers based on method."""
        if self.fusion_method == "concat":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
            
        elif self.fusion_method == "cross_attention":
            self.cross_attn_blocks = nn.ModuleList([
                CrossAttentionBlock(
                    self.hidden_size, 
                    self.config.num_heads, 
                    self.config.dropout
                ) 
                for _ in range(self.config.num_layers)
            ])

        elif self.fusion_method == "gated":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.gate_projection = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.Sigmoid()
            )
            
        elif self.fusion_method == "weighted":
            self.weight_2d = nn.Parameter(torch.tensor(0.5))
            self.weight_3d = nn.Parameter(torch.tensor(0.5))

        elif self.fusion_method == "knn_concat":
            if self.hidden_size % self.config.num_heads != 0:
                raise ValueError(
                    f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.config.num_heads}) "
                    "for knn_concat."
                )
            self.knn_k = max(int(self.config.knn_k), 0)
            knn_num_layers = max(int(self.config.num_layers), 1)
            knn_pos_hidden_size = self.config.knn_pos_mlp_hidden_size or self.hidden_size
            self.knn_pos_mlp = nn.Sequential(
                nn.Linear(5, knn_pos_hidden_size),
                nn.GELU(),
                nn.Linear(knn_pos_hidden_size, self.hidden_size),
            )
            self.knn_cross_attn_blocks = nn.ModuleList([
                KNNTransformerCrossAttentionBlock(
                    self.hidden_size,
                    self.config.num_heads,
                    self.config.dropout,
                )
                for _ in range(knn_num_layers)
            ])

        elif self.fusion_method in {"decompose_add", "decompose_concat"}:
            if self.ortho_mode == "mine":
                mine_hidden = self.config.mine_hidden_size or self.hidden_size
                self.mine_statistics_network = nn.Sequential(
                    nn.LayerNorm(self.hidden_size),
                    nn.Linear(self.hidden_size, mine_hidden),
                    nn.GELU(),
                    nn.Linear(mine_hidden, self.hidden_size),
                    nn.GELU(),
                    nn.Linear(mine_hidden, self.hidden_size),
                )
                # Keep q_theta out of the main optimizer; we train it in a dedicated step.
                self.set_mine_network_trainable(False)
            if self.fusion_method == "decompose_concat":
                self.decompose_norm_2d = nn.LayerNorm(self.hidden_size)
                self.decompose_norm_unique = nn.LayerNorm(self.hidden_size)
                self.decompose_projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
                # Seed the concat projection with a decompose_add-like prior during post_init.
                self.decompose_projection._fusion_identity_block_scales = (2.0, 1.0)
                self.decompose_projection._fusion_identity_noise_scale = 0.01
        elif self.fusion_method in {"nrsr_add", "nrsr_concat"}:
            nrsr_hidden = self.config.nrsr_hidden_size or self.hidden_size
            self.nrsr_encoder = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, nrsr_hidden),
                nn.GELU(),
                nn.Linear(nrsr_hidden, self.hidden_size * 2),
            )
            if self.fusion_method == "nrsr_concat":
                self.nrsr_projection = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def set_mine_network_trainable(self, trainable: bool) -> None:
        if self.ortho_mode != "mine" or not hasattr(self, "mine_statistics_network"):
            return
        for p in self.mine_statistics_network.parameters():
            p.requires_grad_(trainable)

    def get_mine_parameters(self):
        if self.ortho_mode != "mine" or not hasattr(self, "mine_statistics_network"):
            return []
        return list(self.mine_statistics_network.parameters())

    def _predict_t2d_mean_from_unique(self, unique_flat: torch.Tensor) -> torch.Tensor:
        if self.ortho_mode != "mine" or not hasattr(self, "mine_statistics_network"):
            raise ValueError("q_theta is only available when ortho_mode='mine'.")

        unique_flat = unique_flat.float()
        device_type = "cuda" if unique_flat.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            self.mine_statistics_network.to(torch.float32)
            pred_mean = self.mine_statistics_network(unique_flat)
        return pred_mean

    def _compute_vclub_bound_flat(
        self, unique_flat: torch.Tensor, features_2d_flat: torch.Tensor
    ) -> torch.Tensor:
        if unique_flat.numel() == 0:
            return unique_flat.new_zeros(())

        if unique_flat.shape[0] != features_2d_flat.shape[0]:
            raise ValueError(
                f"Feature pair size mismatch for vCLUB objective: "
                f"{unique_flat.shape[0]} vs {features_2d_flat.shape[0]}"
            )

        features_2d_flat = features_2d_flat.float()
        pred_mean = self._predict_t2d_mean_from_unique(unique_flat)

        # log q(T_i | F_i), assuming fixed isotropic sigma=1.
        positive_log_prob = -0.5 * ((features_2d_flat - pred_mean) ** 2).sum(dim=-1)

        # log q(T_j | F_i) for all pairs (i, j), computed with matrix formulation.
        pred_norm = (pred_mean ** 2).sum(dim=-1, keepdim=True)  # [N, 1]
        t_norm = (features_2d_flat ** 2).sum(dim=-1).unsqueeze(0)  # [1, N]
        cross = torch.matmul(pred_mean, features_2d_flat.transpose(0, 1))  # [N, N]
        pairwise_log_prob = -0.5 * (pred_norm + t_norm - 2.0 * cross)
        negative_log_prob = pairwise_log_prob.mean(dim=1)

        vclub_bound = (positive_log_prob - negative_log_prob).mean()/4096
        # Theoretical lower bound is 0 for independent variables; clamp numeric noise.
        return torch.clamp(vclub_bound, min=0.0)

    def _compute_vclub_q_loss_flat(
        self, unique_flat: torch.Tensor, features_2d_flat: torch.Tensor
    ) -> torch.Tensor:
        if unique_flat.shape[0] != features_2d_flat.shape[0]:
            raise ValueError(
                f"Feature pair size mismatch for q_theta update: "
                f"{unique_flat.shape[0]} vs {features_2d_flat.shape[0]}"
            )
        features_2d_flat = features_2d_flat.float()
        pred_mean = self._predict_t2d_mean_from_unique(unique_flat)
        return F.mse_loss(pred_mean, features_2d_flat)

    def compute_mine_objective_from_cache(
        self, unique_flat: torch.Tensor, features_2d_flat: torch.Tensor
    ) -> torch.Tensor:
        if self.ortho_mode != "mine":
            raise ValueError("vCLUB objective is only available when ortho_mode='mine'.")
        return self._compute_vclub_bound_flat(unique_flat, features_2d_flat)

    def compute_mine_q_loss_from_cache(
        self, unique_flat: torch.Tensor, features_2d_flat: torch.Tensor
    ) -> torch.Tensor:
        if self.ortho_mode != "mine":
            raise ValueError("q_theta loss is only available when ortho_mode='mine'.")
        return self._compute_vclub_q_loss_flat(unique_flat, features_2d_flat)

    def _build_relative_position_encoding(
        self,
        query_positions: torch.Tensor,
        neighbor_positions: torch.Tensor,
        query_frames: torch.Tensor,
        neighbor_frames: torch.Tensor,
        neighbor_valid: torch.Tensor,
    ) -> torch.Tensor:
        delta = neighbor_positions - query_positions.unsqueeze(1)
        distance = torch.norm(delta, dim=-1, keepdim=True)
        delta_frame = (neighbor_frames - query_frames.unsqueeze(1)).to(delta.dtype).unsqueeze(-1)
        rel_inputs = torch.cat([delta, distance, delta_frame], dim=-1)
        rel_inputs = torch.where(
            neighbor_valid.unsqueeze(-1),
            rel_inputs,
            torch.zeros_like(rel_inputs),
        )
        rel_inputs = rel_inputs.to(self.knn_pos_mlp[0].weight.dtype)
        return self.knn_pos_mlp(rel_inputs)

    def _cross_frame_knn(
        self,
        positions: torch.Tensor,
        frame_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        k_other: int,
        chunk_size: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = positions.device
        num_tokens = positions.shape[0]
        if num_tokens == 0 or k_other <= 0:
            empty_idx = torch.empty((num_tokens, k_other), dtype=torch.long, device=device)
            empty_valid = torch.empty((num_tokens, k_other), dtype=torch.bool, device=device)
            return empty_idx, empty_valid

        frame_ids = frame_ids.long()
        valid_mask = valid_mask.bool()
        knn_idx = torch.zeros((num_tokens, k_other), dtype=torch.long, device=device)
        knn_valid = torch.zeros((num_tokens, k_other), dtype=torch.bool, device=device)
        valid_token_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        num_valid = valid_token_idx.shape[0]
        if num_valid == 0:
            return knn_idx, knn_valid

        with torch.no_grad():
            positions_valid = positions.detach().float().index_select(0, valid_token_idx)
            frame_ids_valid = frame_ids.index_select(0, valid_token_idx)
            inf = torch.tensor(float("inf"), device=device, dtype=positions_valid.dtype)
            topk_k = min(k_other, num_valid)
            knn_idx_chunks = []
            knn_valid_chunks = []

            for start in range(0, num_valid, chunk_size):
                end = min(start + chunk_size, num_valid)
                query_positions = positions_valid[start:end]
                query_frames = frame_ids_valid[start:end]

                pairwise_dist = torch.cdist(query_positions, positions_valid)
                cross_frame_mask = query_frames.unsqueeze(1) != frame_ids_valid.unsqueeze(0)
                pairwise_dist = pairwise_dist.masked_fill(~cross_frame_mask, inf)

                topk_vals, topk_local_idx = torch.topk(pairwise_dist, k=topk_k, dim=-1, largest=False)
                topk_valid_chunk = torch.isfinite(topk_vals)
                topk_idx_chunk = valid_token_idx[topk_local_idx]

                if topk_k < k_other:
                    pad_count = k_other - topk_k
                    pad_idx = torch.zeros((end - start, pad_count), dtype=torch.long, device=device)
                    pad_valid = torch.zeros((end - start, pad_count), dtype=torch.bool, device=device)
                    topk_idx_chunk = torch.cat([topk_idx_chunk, pad_idx], dim=-1)
                    topk_valid_chunk = torch.cat([topk_valid_chunk, pad_valid], dim=-1)

                knn_idx_chunks.append(topk_idx_chunk)
                knn_valid_chunks.append(topk_valid_chunk)

        knn_idx[valid_token_idx] = torch.cat(knn_idx_chunks, dim=0)
        knn_valid[valid_token_idx] = torch.cat(knn_valid_chunks, dim=0)
        return knn_idx, knn_valid

    def _knn_cross_attention(
        self,
        features_2d: torch.Tensor,
        features_3d: torch.Tensor,
        patch_world: torch.Tensor,
        patch_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_frames, h_grid, w_grid, _ = features_2d.shape
        num_tokens = num_frames * h_grid * w_grid
        if num_tokens == 0:
            return features_2d

        features_2d_flat = features_2d.reshape(num_tokens, self.hidden_size)
        features_3d_flat = features_3d.reshape(num_tokens, self.hidden_size)
        patch_world_flat = patch_world.reshape(num_tokens, 3)
        if patch_valid is None:
            patch_valid_flat = torch.ones(num_tokens, dtype=torch.bool, device=features_2d.device)
        else:
            patch_valid_flat = patch_valid.reshape(num_tokens).bool()

        frame_ids = (
            torch.arange(num_frames, device=features_2d.device)
            .view(num_frames, 1, 1)
            .expand(num_frames, h_grid, w_grid)
            .reshape(num_tokens)
        )

        knn_other_idx, knn_other_valid = self._cross_frame_knn(
            patch_world_flat,
            frame_ids,
            patch_valid_flat,
            k_other=self.knn_k,
        )
        self_idx = torch.arange(num_tokens, device=features_2d.device).unsqueeze(1)
        self_valid = torch.ones((num_tokens, 1), dtype=torch.bool, device=features_2d.device)
        neighbor_idx = torch.cat([self_idx, knn_other_idx], dim=1)
        neighbor_valid = torch.cat([self_valid, knn_other_valid], dim=1)

        neighbor_features = features_3d_flat[neighbor_idx]
        neighbor_positions = patch_world_flat[neighbor_idx]
        neighbor_frames = frame_ids[neighbor_idx]
        rel_pos = self._build_relative_position_encoding(
            query_positions=patch_world_flat,
            neighbor_positions=neighbor_positions,
            query_frames=frame_ids,
            neighbor_frames=neighbor_frames,
            neighbor_valid=neighbor_valid,
        )
        rel_pos = rel_pos.to(neighbor_features.dtype)
        memory_states = neighbor_features + rel_pos

        hidden_states = features_2d_flat
        for block in self.knn_cross_attn_blocks:
            hidden_states, _ = block(hidden_states, memory_states, neighbor_valid)
        return hidden_states.reshape(num_frames, h_grid, w_grid, self.hidden_size)

    def forward(
        self,
        features_2d: torch.Tensor,
        features_3d: torch.Tensor,
        patch_world: Optional[torch.Tensor] = None,
        patch_valid: Optional[torch.Tensor] = None,
        return_aux_losses: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
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

        elif self.fusion_method == "knn_concat":
            if patch_world is None:
                raise ValueError("patch_world must be provided for knn_concat fusion.")
            return self._knn_cross_attention(features_2d, features_3d, patch_world, patch_valid)

        elif self.fusion_method in {"nrsr_add", "nrsr_concat"}:
            stats = self.nrsr_encoder(features_3d)
            mu, log_var = torch.chunk(stats, 2, dim=-1)

            if self.training:
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                unique_3d = mu + eps * std
            else:
                unique_3d = mu

            if self.fusion_method == "nrsr_concat":
                fused = self.nrsr_projection(torch.cat([features_2d, unique_3d], dim=-1))
            else:
                fused = features_2d + unique_3d

            if not return_aux_losses:
                return fused

            mu_f = mu.float()
            log_var_f = log_var.float()
            kl_term = 1.0 + log_var_f - mu_f.pow(2) - log_var_f.exp()
            kl_loss = (-0.5 * kl_term.sum(dim=-1)).mean()
            aux_losses = {
                "loss_nrsr_kl": kl_loss,
            }
            return fused, aux_losses

        elif self.fusion_method in {"decompose_add", "decompose_concat"}:
            unique_3d = features_3d - features_2d

            if self.fusion_method == "decompose_concat":
                # Match the regular concat branch and stabilize the projection inputs.
                features_2d_proj = self.decompose_norm_2d(features_2d)
                unique_3d_proj = self.decompose_norm_unique(unique_3d)
                fused = self.decompose_projection(torch.cat([features_2d_proj, unique_3d_proj], dim=-1))
            else:
                fused = 2*features_2d + unique_3d

            if not return_aux_losses:
                return fused

            unique_3d_f = unique_3d.float()
            features_2d_f = features_2d.float()
            if self.ortho_mode == "mine":
                unique_flat = unique_3d_f.reshape(-1, unique_3d_f.shape[-1])
                features_2d_flat = features_2d_f.reshape(-1, features_2d_f.shape[-1])
                ortho_loss = self._compute_vclub_bound_flat(unique_flat, features_2d_flat)
            else:
                ortho_loss = F.cosine_similarity(unique_3d_f, features_2d_f, dim=-1, eps=1e-6).abs().mean()
            aux_losses = {
                "loss_ortho": ortho_loss,
            }
            if self.ortho_mode == "mine":
                aux_losses["mine_unique_features"] = unique_flat.detach()
                aux_losses["mine_2d_features"] = features_2d_flat.detach()
            return fused, aux_losses
            
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
