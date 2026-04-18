import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    # Geometry encoder configuration
    use_geometry_encoder: bool = field(default=False)  # Whether to use 3D geometry encoder
    geometry_encoder_type: str = field(default="vggt")  # Type of geometry encoder ("vggt", "pi3")
    geometry_encoder_path: str = field(default="facebook/VGGT-1B/")  # Path to pre-trained geometry encoder model
    reference_frame: str = field(default="first")  # Reference frame for geometry encoding ("first", "last"), only available for vggt
    feature_fusion_method: str = field(default="add")  # Method to fuse geometry and visual features ("add", "concat", "cross_attention", "gate")
    fusion_num_layers: int = field(default=1)  # Number of layers in the cross-attention module when feature_fusion_method is "cross_attention"
    geometry_merger_type: str = field(default="mlp")  # Type of geometry feature merger ("mlp", "avg")
    use_geometry_lastvit_selector: bool = field(default=False)  # Enable LAST-ViT style 2D-guided gating for projected 3D tokens
    geometry_lastvit_top_k: int = field(default=1)  # Top-k patches selected per channel to form the CLS-like token
    use_unique_3d_prefix: bool = field(default=False)  # Whether to build the detached unique_3d prefix branch
    unique_3d_num_queries: int = field(default=0)  # Number of learnable prefix queries extracted from unique_3d
    unique_3d_prefix_num_heads: int = field(default=8)  # Number of heads in the unique_3d cross attention
    unique_3d_prefix_dropout: float = field(default=0.1)  # Dropout in the unique_3d prefix encoder
    unique_3d_hsic_weight: float = field(default=0.0)  # Weight of the RBF-HSIC decorrelation loss
    unique_3d_hsic_sigma_2d: float = field(default=-1.0)  # RBF sigma for 2D features, <=0 enables median heuristic
    unique_3d_hsic_sigma_3d: float = field(default=-1.0)  # RBF sigma for unique_3d features, <=0 enables median heuristic
    unique_3d_hsic_max_samples: int = field(default=256)  # Max token samples per sample for HSIC estimation

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    max_samples: int = field(default=-1)
    shuffle: bool = field(default=True)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
