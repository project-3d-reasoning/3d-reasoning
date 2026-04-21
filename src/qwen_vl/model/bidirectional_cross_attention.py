"""
双向交叉注意力融合模块 (Bidirectional Cross-Attention Fusion)

让2D和3D特征双向增强，而非单向门控
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BidirectionalCrossAttention(nn.Module):
    """
    双向交叉注意力融合 (Bidirectional Cross-Attention Fusion)

    理论基础:
    - 2D特征: 语义丰富，但缺乏几何信息
    - 3D特征: 几何精确，但语义可能较弱
    - 双向交叉注意力让两者互补

    技术实现:
    - 2D -> 3D: 2D特征作为Query，查询3D几何信息
    - 3D -> 2D: 3D特征作为Query，查询2D语义信息
    - 门控融合双向增强后的特征
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_residual = use_residual

        # Layer Norms
        self.norm_2d = nn.LayerNorm(hidden_size)
        self.norm_3d = nn.LayerNorm(hidden_size)

        # 2D -> 3D 交叉注意力 (2D查询3D几何信息)
        self.cross_attn_2d_to_3d = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 3D -> 2D 交叉注意力 (3D查询2D语义信息)
        self.cross_attn_3d_to_2d = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        feat_2d: torch.Tensor,
        feat_3d: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feat_2d: [B, H, W, D] 2D视觉特征
            feat_3d: [B, H, W, D] 3D几何特征

        Returns:
            融合后的特征 [B, H, W, D]
        """
        B, H, W, D = feat_2d.shape
        N = H * W

        # 展平空间维度
        feat_2d_flat = feat_2d.view(B, N, D)  # [B, N, D]
        feat_3d_flat = feat_3d.view(B, N, D)

        # Normalize
        feat_2d_norm = self.norm_2d(feat_2d_flat)
        feat_3d_norm = self.norm_3d(feat_3d_flat)

        # 2D增强: 2D作为Query，从3D获取几何信息
        enhanced_2d, attn_weights_2d = self.cross_attn_2d_to_3d(
            query=feat_2d_norm,
            key=feat_3d_norm,
            value=feat_3d_norm
        )  # [B, N, D]

        # 3D增强: 3D作为Query，从2D获取语义信息
        enhanced_3d, attn_weights_3d = self.cross_attn_3d_to_2d(
            query=feat_3d_norm,
            key=feat_2d_norm,
            value=feat_2d_norm
        )  # [B, N, D]

        # 门控融合双向增强特征
        combined = torch.cat([enhanced_2d, enhanced_3d], dim=-1)  # [B, N, 2D]
        gate = self.gate(combined)  # [B, N, D]

        # 融合原始特征和增强特征
        output = gate * feat_2d_flat + (1 - gate) * feat_3d_flat

        # 残差连接
        if self.use_residual:
            output = output + 0.5 * (enhanced_2d + enhanced_3d)

        # 输出投影
        output = self.output_proj(output)

        return output.view(B, H, W, D)


class BidirectionalCrossAttentionV2(nn.Module):
    """
    双向交叉注意力融合 V2 (增强版)

    增加以下特性:
    - 多层交叉注意力
    - 位置感知注意力
    - 自适应融合权重
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 多层交叉注意力
        self.layers = nn.ModuleList([
            BidirectionalCrossAttentionLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(
        self,
        feat_2d: torch.Tensor,
        feat_3d: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feat_2d: [B, H, W, D] 2D视觉特征
            feat_3d: [B, H, W, D] 3D几何特征

        Returns:
            融合后的特征 [B, H, W, D]
        """
        x_2d = feat_2d
        x_3d = feat_3d

        for layer in self.layers:
            x_2d, x_3d = layer(x_2d, x_3d)

        # 最终融合
        output = self.final_fusion(x_2d + x_3d)

        return output


class BidirectionalCrossAttentionLayer(nn.Module):
    """单层双向交叉注意力"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm_2d = nn.LayerNorm(hidden_size)
        self.norm_3d = nn.LayerNorm(hidden_size)

        self.cross_attn_2d_to_3d = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.cross_attn_3d_to_2d = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ffn_2d = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

        self.ffn_3d = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

        self.norm_ffn_2d = nn.LayerNorm(hidden_size)
        self.norm_ffn_3d = nn.LayerNorm(hidden_size)

    def forward(
        self,
        feat_2d: torch.Tensor,
        feat_3d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_2d: [B, H, W, D]
            feat_3d: [B, H, W, D]

        Returns:
            (updated_2d, updated_3d)
        """
        B, H, W, D = feat_2d.shape
        N = H * W

        feat_2d_flat = feat_2d.view(B, N, D)
        feat_3d_flat = feat_3d.view(B, N, D)

        # Cross attention
        enhanced_2d, _ = self.cross_attn_2d_to_3d(
            self.norm_2d(feat_2d_flat),
            self.norm_3d(feat_3d_flat),
            self.norm_3d(feat_3d_flat)
        )

        enhanced_3d, _ = self.cross_attn_3d_to_2d(
            self.norm_3d(feat_3d_flat),
            self.norm_2d(feat_2d_flat),
            self.norm_2d(feat_2d_flat)
        )

        # Residual
        feat_2d_flat = feat_2d_flat + enhanced_2d
        feat_3d_flat = feat_3d_flat + enhanced_3d

        # FFN
        feat_2d_flat = feat_2d_flat + self.ffn_2d(self.norm_ffn_2d(feat_2d_flat))
        feat_3d_flat = feat_3d_flat + self.ffn_3d(self.norm_ffn_3d(feat_3d_flat))

        return feat_2d_flat.view(B, H, W, D), feat_3d_flat.view(B, H, W, D)


# 工厂函数
def create_bidirectional_cross_attention(
    version: str = "v1",
    hidden_size: int = 3584,
    num_heads: int = 8,
    num_layers: int = 2,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    创建双向交叉注意力模块

    Args:
        version: "v1" (单层) 或 "v2" (多层)
        hidden_size: 特征维度
        num_heads: 注意力头数
        num_layers: 层数 (仅v2)
        dropout: dropout率

    Returns:
        对应的双向交叉注意力模块
    """
    if version == "v1":
        return BidirectionalCrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
    elif version == "v2":
        return BidirectionalCrossAttentionV2(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown version: {version}")
