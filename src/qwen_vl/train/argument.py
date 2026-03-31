import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_mm_vision_lora: bool = field(default=False)  # Use LoRA to tune visual tower when tune_mm_vision=True
    tune_geometry_encoder: bool = field(default=False)  # Tune geometry encoder parameters when use_geometry_encoder=True
    tune_geometry_encoder_lora: bool = field(default=False)  # Use LoRA to tune geometry encoder when tune_geometry_encoder=True
    use_learnable_prefix: bool = field(default=False)  # Enable learnable prefix tokens between visual blocks and question text
    learnable_prefix_len: int = field(default=0)  # Number of learnable prefix tokens to insert at the visual-question boundary
    text_gate_bert_name_or_path: str = field(default="bert-base-uncased")  # Frozen BERT encoder used to extract question summaries for adver/adver_ortho gates
    text_gate_bert_max_length: int = field(default=64)  # Maximum BERT token length for question text encoding
    label_weight_default: float = field(default=1.0)  # Default token weight for supervised assistant tokens when label weighting is enabled
    label_weight_scanrefer_frame: float = field(default=3.0)  # Weight for ScanRefer frame index tokens
    label_weight_scanrefer_bbox: float = field(default=3.0)  # Weight for ScanRefer bbox numeric tokens
    label_weight_scan2cap_category: float = field(default=2.0)  # Weight for Scan2Cap category phrases
    label_weight_scan2cap_attribute: float = field(default=1.5)  # Weight for Scan2Cap attribute phrases
    label_weight_scan2cap_relation: float = field(default=1.5)  # Weight for Scan2Cap relation phrases
    label_weight_scannet_det_label: float = field(default=2.0)  # Weight for ScanNetDet label phrases
    label_weight_scannet_det_bbox: float = field(default=2.5)  # Weight for ScanNetDet bbox numeric tokens
    label_weight_dynamic_iou_alpha: float = field(default=0.0)  # Dynamic IoU weight factor; 0 disables dynamic IoU reweighting
    label_weight_dynamic_iou_eps: float = field(default=1e-6)  # Numerical stabilizer for dynamic IoU weights

    # Geometry encoder configuration
    use_geometry_encoder: bool = field(default=False)  # Whether to use 3D geometry encoder
    geometry_encoder_type: str = field(default="vggt")  # Type of geometry encoder ("vggt", "pi3")
    geometry_encoder_path: str = field(default="facebook/VGGT-1B/")  # Path to pre-trained geometry encoder model
    reference_frame: str = field(default="first")  # Reference frame for geometry encoding ("first", "last"), only available for vggt
    feature_fusion_method: str = field(default="add")  # Method to fuse geometry and visual features ("add", "concat", "cross_attention", "gated", "adver", "adver_ortho", "decompose_add", "decompose_concat", "nrsr_add", "nrsr_concat", "knn_concat")
    fusion_num_layers: int = field(default=1)  # Number of Transformer-style cross-attention blocks when feature_fusion_method is "cross_attention" or "knn_concat"
    geometry_merger_type: str = field(default="mlp")  # Type of geometry feature merger ("mlp", "avg")
    decompose_hidden_size: Optional[int] = field(default=None)  # Bottleneck size for shared/unique decomposition branches; defaults to hidden_size when unset
    nrsr_hidden_size: Optional[int] = field(default=None)  # Hidden size for NRSR VIB MLP
    fusion_recon_mask_ratio: float = field(default=0.3)  # Patch/token mask ratio used by masked reconstruction in adver mode
    adver_compute_align_loss: bool = field(default=False)  # Whether to compute shared alignment loss in adver/adver_ortho; defaults to False to save training time while keeping the code path available
    fusion_align_mode: str = field(default="cosine")  # Alignment loss mode ("cosine", "infonce")
    fusion_ortho_mode: str = field(default="cosine")  # Orthogonality mode ("cosine", "hsic", "mine")
    fusion_lambda_align: float = field(default=1.0)  # Weight for shared alignment loss in decompose fusion methods
    fusion_align_target_ratio: Optional[float] = field(default=None)  # Target late-stage weighted loss_shared / loss_ce ratio; enables dynamic lambda adjustment when set
    fusion_align_lambda_min: float = field(default=1e-8)  # Minimum dynamic lambda for shared alignment loss
    fusion_align_lambda_max: float = field(default=5.0)  # Maximum dynamic lambda for shared alignment loss
    fusion_lambda_ortho: float = field(default=1.0)  # Weight for orthogonality loss
    fusion_ortho_lambda_min: float = field(default=1e-8)  # Minimum dynamic lambda for orthogonality loss
    fusion_ortho_lambda_max: float = field(default=0.2)  # Maximum dynamic lambda for orthogonality loss
    fusion_lambda_recon: float = field(default=1.0)  # Weight for feature reconstruction loss in decompose fusion methods
    fusion_recon_target_ratio: Optional[float] = field(default=None)  # Target late-stage weighted loss_recon / loss_ce ratio; enables dynamic lambda adjustment when set
    fusion_recon_lambda_min: float = field(default=1e-8)  # Minimum dynamic lambda for reconstruction loss
    fusion_recon_lambda_max: float = field(default=5.0)  # Maximum dynamic lambda for reconstruction loss
    fusion_ortho_target_ratio: Optional[float] = field(default=None)  # Target ratio for weighted orthogonality loss vs CE loss
    fusion_lambda_nrsr: float = field(default=1.0)  # Weight for NRSR KL loss
    fusion_lambda_nrsr_dynamic: bool = field(default=True)  # Enable 3-stage dynamic schedule for NRSR lambda
    fusion_lambda_nrsr_stage2_ratio: float = field(default=0.10)  # Forced weighted-KL ratio in stage 2
    fusion_lambda_nrsr_stage3_ratio: float = field(default=0.03)  # Forced weighted-KL ratio in stage 3
    fusion_lambda_warmup: bool = field(default=False)  # Enable ortho lambda warmup for decompose fusion methods
    fusion_lambda_warmup_steps: int = field(default=100)  # Warmup steps for ortho lambda when enabled
    fusion_mine_q_warmup_steps: int = field(default=500)  # q_theta-only warmup updates at each epoch start when ortho mode is "mine"
    fusion_knn_k: int = field(default=9)  # Number of nearest neighbors from other frames for knn_concat; self token is always added.
    fusion_knn_min_valid_ratio: float = field(default=0.25)  # Minimum confidence mass ratio in the center window before a patch/token becomes valid.
    fusion_knn_pos_mlp_hidden_size: Optional[int] = field(default=None)  # Hidden width of the relative-position MLP for knn_concat; defaults to hidden_size.

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    label_weight_masks_dir: Optional[str] = field(default=None)
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
