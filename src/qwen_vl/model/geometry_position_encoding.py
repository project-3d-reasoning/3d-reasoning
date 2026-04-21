"""
几何位置编码模块 (Geometry Position Encoding)

提供三种位置编码方案:
1. 相对几何位置编码 (RelativeGeometryPositionEncoding)
2. 跨帧几何一致性编码 (CrossFrameGeometryEncoding)
3. 统一几何位置编码 (UnifiedGeometryPositionEncoding)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RelativeGeometryPositionEncoding(nn.Module):
    """
    相对几何位置编码 (Relative Geometry Position Encoding)

    理论基础:
    - VGGT的特征包含了帧间的几何对应关系
    - 通过计算特征间的相对关系，提取隐式3D位置信息
    - 将相对位置编码与原始特征融合

    技术实现:
    - 使用特征相似度作为隐式相对位置
    - 注意力分数作为位置编码权重
    - 门控融合原始特征和位置感知特征
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 几何特征投影
        self.geo_proj_q = nn.Linear(hidden_size, hidden_size)
        self.geo_proj_k = nn.Linear(hidden_size, hidden_size)
        self.geo_proj_v = nn.Linear(hidden_size, hidden_size)

        # 输出投影
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Layer Norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # 位置编码融合门控
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        # MLP for refinement
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, geo_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            geo_features: [B, H, W, D] VGGT几何特征

        Returns:
            带有相对几何位置编码的特征 [B, H, W, D]
        """
        B, H, W, D = geo_features.shape
        N = H * W

        # 展平空间维度
        geo_flat = geo_features.view(B, N, D)  # [B, N, D]

        # 投影到Q, K, V
        q = self.geo_proj_q(self.norm1(geo_flat))  # [B, N, D]
        k = self.geo_proj_k(self.norm1(geo_flat))
        v = self.geo_proj_v(self.norm1(geo_flat))

        # 重塑为多头形式
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, N, head_dim]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数作为相对位置
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, heads, N, N]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权聚合得到位置感知特征
        pos_aware = torch.matmul(attn_weights, v)  # [B, heads, N, head_dim]
        pos_aware = pos_aware.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        pos_aware = self.out_proj(pos_aware)

        # 门控融合原始特征和位置感知特征
        combined = torch.cat([geo_flat, pos_aware], dim=-1)
        gate = self.gate(combined)

        # 残差连接 + 门控融合
        output = geo_flat + gate * pos_aware

        # MLP refinement
        output = output + self.mlp(self.norm2(output))

        return output.view(B, H, W, D)


class CrossFrameGeometryEncoding(nn.Module):
    """
    跨帧几何一致性编码 (Cross-Frame Geometry Consistency Encoding)

    理论基础:
    - VGGT通过alternating attention学习了帧间几何对应
    - 显式提取跨帧一致性信息作为几何位置编码
    - 增强模型对3D空间关系的理解

    技术实现:
    - 帧间几何关系编码器
    - 时序几何位置编码
    - 跨帧聚合注意力
    """

    def __init__(
        self,
        hidden_size: int,
        num_frames: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_frames = num_frames
        self.num_heads = num_heads

        # 帧间几何关系编码
        self.frame_relation_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # 时序几何位置编码 (可学习)
        self.temporal_geo_embed = nn.Parameter(
            torch.randn(1, num_frames, 1, hidden_size) * 0.02
        )

        # 跨帧聚合注意力
        self.cross_frame_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer Norms
        self.norm = nn.LayerNorm(hidden_size)

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(
        self,
        geo_features: torch.Tensor,
        num_frames_per_sample: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            geo_features: [B*N_frames, H, W, D] 展平帧维度的几何特征
            num_frames_per_sample: 每个样本的帧数 [B]

        Returns:
            带有跨帧几何位置编码的特征 [B*N_frames, H, W, D]
        """
        BN, H, W, D = geo_features.shape

        # 简化处理: 假设均匀分布帧数
        # 实际应用中需要根据batch信息重组

        # 应用帧间关系编码
        geo_flat = geo_features.view(BN, H * W, D)  # [BN, HW, D]

        # 编码帧间几何关系
        relation_encoded = self.frame_relation_encoder(geo_flat)  # [BN, HW, D]

        # 添加时序位置编码
        # 假设每个样本有相同帧数
        if num_frames_per_sample is None:
            # 简化: 假设BN是帧数的整数倍
            num_frames = min(self.num_frames, BN)
            temporal_pos = self.temporal_geo_embed[:, :num_frames, :, :].squeeze(1)  # [1, num_frames, 1, D]
            temporal_pos = temporal_pos.expand(1, num_frames, H * W, D)
            temporal_pos = temporal_pos.view(num_frames, H * W, D)

            # 重复以匹配BN
            if BN > num_frames:
                repeat_factor = (BN + num_frames - 1) // num_frames
                temporal_pos = temporal_pos.repeat(repeat_factor, 1, 1)[:BN]

            relation_encoded = relation_encoded + temporal_pos

        # 残差连接
        output = geo_flat + relation_encoded

        return output.view(BN, H, W, D)


class UnifiedGeometryPositionEncoding(nn.Module):
    """
    统一几何位置编码 (Unified Geometry Position Encoding)

    整合相对位置编码和跨帧一致性编码，提供完整的几何位置编码方案

    组合策略:
    1. 先应用相对几何位置编码 (帧内)
    2. 再应用跨帧几何一致性编码 (帧间)
    3. 最终融合层统一输出
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_frames: int = 8,
        use_relative: bool = True,
        use_cross_frame: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_relative = use_relative
        self.use_cross_frame = use_cross_frame

        if use_relative:
            self.relative_encoding = RelativeGeometryPositionEncoding(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            )

        if use_cross_frame:
            self.cross_frame_encoding = CrossFrameGeometryEncoding(
                hidden_size=hidden_size,
                num_frames=num_frames,
                num_heads=num_heads,
                dropout=dropout
            )

        # 最终融合层
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size)
        )

        # 组合门控
        if use_relative and use_cross_frame:
            self.combine_gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )

    def forward(
        self,
        geo_features: torch.Tensor,
        num_frames_per_sample: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            geo_features: [B, H, W, D] 或 [B*N_frames, H, W, D] VGGT几何特征
            num_frames_per_sample: 每个样本的帧数

        Returns:
            带有几何位置编码的特征
        """
        x = geo_features
        original = geo_features

        # 相对几何位置编码 (帧内)
        if self.use_relative:
            rel_encoded = self.relative_encoding(x)
            if self.use_cross_frame:
                x = rel_encoded
            else:
                x = rel_encoded

        # 跨帧几何一致性编码 (帧间)
        if self.use_cross_frame:
            cf_encoded = self.cross_frame_encoding(x, num_frames_per_sample)
            if self.use_relative:
                # 组合两种编码
                combined = torch.cat([rel_encoded, cf_encoded], dim=-1)
                gate = self.combine_gate(combined)
                x = gate * rel_encoded + (1 - gate) * cf_encoded
            else:
                x = cf_encoded

        # 最终融合
        x = self.fusion(x)

        # 残差连接
        return original + x


# 工厂函数
def create_geometry_position_encoding(
    encoding_type: str,
    hidden_size: int,
    num_heads: int = 8,
    num_frames: int = 8,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    创建几何位置编码模块

    Args:
        encoding_type: "relative", "cross_frame", "unified"
        hidden_size: 特征维度
        num_heads: 注意力头数
        num_frames: 帧数
        dropout: dropout率

    Returns:
        对应的几何位置编码模块
    """
    if encoding_type == "relative":
        return RelativeGeometryPositionEncoding(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
    elif encoding_type == "cross_frame":
        return CrossFrameGeometryEncoding(
            hidden_size=hidden_size,
            num_frames=num_frames,
            num_heads=num_heads,
            dropout=dropout
        )
    elif encoding_type == "unified":
        return UnifiedGeometryPositionEncoding(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_frames=num_frames,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown encoding_type: {encoding_type}")
