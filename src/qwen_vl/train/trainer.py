import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import datasets
import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm
from transformers import Trainer
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
)
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
)
from transformers.trainer_utils import seed_worker

from qwen_vl.label_weighting import (
    LABEL_WEIGHT_CODE_SCANREFER_FRAME,
    LABEL_WEIGHT_CODE_SCANREFER_BBOX,
    LABEL_WEIGHT_CODE_SCANNET_DET_BBOX,
    get_label_weight_value_table,
)
from qwen_vl.geometry_tokenization import parse_bbox_inner_values


_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
_SCANREFER_FRAME_PATTERN = re.compile(r'"frame"\s*:\s*(-?\d+)')
_JSON_BBOX_PATTERN = re.compile(r'"bbox_3d"\s*:\s*\[([^\]]*)\]', re.DOTALL)


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """
    assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
    query_states = query_states.squeeze(0)
    key_states = key_states.squeeze(0)
    value_states = value_states.squeeze(0)
    cu_seqlens = attention_mask

    with torch.no_grad():
        max_seqlen = max(
            [
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]
        ).item()

    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
        causal = is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    flash_kwargs = {}

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    attn_output = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout,
        softmax_scale=softmax_scale,
        causal=causal,
        **flash_kwargs,
    )

    attn_output = attn_output.unsqueeze(0)
    query_states = query_states.unsqueeze(0)
    key_states = key_states.unsqueeze(0)
    value_states = value_states.unsqueeze(0)

    return attn_output


def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool,
):
    return attention_mask


def replace_qwen2_vl_attention_class():
    import transformers
    import transformers.modeling_flash_attention_utils

    transformers.models.qwen2_vl.modeling_qwen2_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel._update_causal_mask = (
        _update_causal_mask
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel._update_causal_mask = (
        _update_causal_mask
    )


def print_trainable_parameters_visual(self) -> None:
    """
    Prints the trainable status of all vision components including attention blocks and merger module.
    Outputs the indices of trainable/non-trainable blocks and the merger module status.
    """
    trainable_blocks = []
    non_trainable_blocks = []

    # Check trainable status of vision attention blocks
    for block_idx, block in enumerate(self.blocks):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    # Check trainable status of merger module
    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    # Print results
    print("Vision Module - Attention Blocks:")
    print(
        f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}"
    )
    print(
        f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}"
    )
    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    """
    Prints the trainable status of all LLM components including embeddings, layers, and normalization.
    Outputs the indices of trainable/non-trainable layers and other module statuses.
    """
    # Check embed_tokens
    is_embed_trainable = any(
        param.requires_grad for param in self.embed_tokens.parameters()
    )
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    # Check each decoder layer
    trainable_layers = []
    non_trainable_layers = []

    for layer_idx, layer in enumerate(self.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # Print layer status
    print(
        f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
    )
    print(
        f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
    )


def create_optimizer(self):
    if self.optimizer is None:
        named_parameters = {
            name: param for name, param in self.model.named_parameters() if param.requires_grad
        }
        optimizer_grouped_parameters = _build_optimizer_grouped_parameters(
            self, named_parameters
        )

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


def _build_optimizer_grouped_parameters(
    trainer: Trainer, named_parameters: Dict[str, torch.nn.Parameter]
) -> List[Dict[str, Any]]:
    if not named_parameters:
        return []

    opt_model = trainer.model
    decay_parameters = set(get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS))
    decay_parameters = {name for name in decay_parameters if "bias" not in name}
    projector_lr = getattr(trainer.args, "mm_projector_lr", None)
    vision_tower_lr = getattr(trainer.args, "vision_tower_lr", None)
    fusion_gate_lr = getattr(trainer.args, "fusion_gate_lr", None)
    fusion_feature_3d_lr = getattr(trainer.args, "fusion_feature_3d_lr", None)

    projector_parameters = set()
    if projector_lr is not None and projector_lr != 0:
        projector_parameters = {
            name for name, _ in opt_model.named_parameters() if "merger" in name
        }

    vision_tower_parameters = set()
    if vision_tower_lr is not None and vision_tower_lr != 0:
        vision_tower_parameters = {
            name for name, _ in opt_model.named_parameters() if "visual" in name
        }

    gate_parameters = set()
    if fusion_gate_lr is not None and fusion_gate_lr != 0:
        gate_parameters = {
            name
            for name, _ in opt_model.named_parameters()
            if name.startswith("feature_fusion.adver_gate_")
            or name.startswith("text_gate_projection")
        }

    feature_3d_parameters = set()
    if fusion_feature_3d_lr is not None and fusion_feature_3d_lr != 0:
        feature_3d_prefixes = (
            "feature_fusion.adver_shared_encoder",
            "feature_fusion.adver_unique_encoder",
            "feature_fusion.adver_align_proj_3d",
            "feature_fusion.adver_reconstruct_cross_attention",
            "feature_fusion.adver_reconstruct_memory_norm",
            "feature_fusion.adver_reconstruct_query_norm",
            "feature_fusion.adver_reconstruct_output_norm",
            "feature_fusion.adver_reconstruct_mlp",
            "feature_fusion.adver_reconstruct_2d",
        )
        feature_3d_parameters = {
            name for name, _ in opt_model.named_parameters() if name.startswith(feature_3d_prefixes)
        }

    def _get_group_lr(name: str) -> Optional[float]:
        if name in projector_parameters:
            return float(projector_lr)
        if name in gate_parameters:
            return float(fusion_gate_lr)
        if name in feature_3d_parameters:
            return float(fusion_feature_3d_lr)
        if name in vision_tower_parameters and name not in projector_parameters:
            return float(vision_tower_lr)
        return None

    grouped_parameter_names: Dict[Tuple[float, Optional[float]], List[str]] = {}
    for name in named_parameters:
        weight_decay = trainer.args.weight_decay if name in decay_parameters else 0.0
        group_key = (float(weight_decay), _get_group_lr(name))
        grouped_parameter_names.setdefault(group_key, []).append(name)

    optimizer_grouped_parameters = []
    for (weight_decay, group_lr), group_names in grouped_parameter_names.items():
        if not group_names:
            continue
        group = {
            "params": [named_parameters[name] for name in group_names],
            "weight_decay": weight_decay,
        }
        if group_lr is not None:
            group["lr"] = group_lr
        optimizer_grouped_parameters.append(group)
    return optimizer_grouped_parameters


class VGTrainer(Trainer):
    """Trainer with optional extra q_theta updates for vCLUB in ortho mine mode."""
    NRSR_STAGE1_END = 0.1
    NRSR_STAGE2_END = 0.5
    ADVER_DISABLE_START = 2.0 / 3.0
    ADVER_ORTHO_STAGE1_END = 1.0 / 3.0
    ADVER_ORTHO_STAGE2_END = 2.0 / 3.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mine_cached_features: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._mine_optimizer: Optional[torch.optim.Optimizer] = None
        self._mine_update_steps: int = 5
        self._mine_warmup_done_epochs: Set[int] = set()
        self._aux_monitor_logs: Dict[str, float] = {}
        self._aux_monitor_sums: Dict[str, float] = {}
        self._aux_monitor_counts: Dict[str, int] = {}
        self._latest_dynamic_iou_stats: Dict[str, float] = self._default_dynamic_iou_monitor_stats()
        # EMA for monitoring and dynamic lambda adjustment of decompose auxiliary losses.
        self._loss_ce_ema: Optional[float] = None
        self._aux_loss_emas: Dict[str, float] = {}
        self._ema_decay: float = float(getattr(self.args, "fusion_loss_ema_decay", 0.98))
        self._ema_eps: float = float(getattr(self.args, "fusion_ema_eps", 1e-12))
        self._nrsr_stage2_ratio: float = float(getattr(self.args, "fusion_lambda_nrsr_stage2_ratio", 0.10))
        self._nrsr_stage3_ratio: float = float(getattr(self.args, "fusion_lambda_nrsr_stage3_ratio", 0.03))
        self._nrsr_ratio_eps: float = float(getattr(self.args, "fusion_nrsr_ratio_eps", 1e-12))
        self._dynamic_aux_configs: Dict[str, Dict[str, Union[str, float, None]]] = {
            "loss_shared": {
                "lambda_attr": "fusion_lambda_align",
                "target_ratio": self._maybe_get_float_arg("fusion_align_target_ratio"),
                "lambda_min": float(getattr(self.args, "fusion_align_lambda_min", 1e-8)),
                "lambda_max": float(getattr(self.args, "fusion_align_lambda_max", 5.0)),
            },
            "loss_ortho": {
                "lambda_attr": "fusion_lambda_ortho",
                "target_ratio": self._maybe_get_float_arg("fusion_ortho_target_ratio"),
                "lambda_min": float(getattr(self.args, "fusion_ortho_lambda_min", 1e-8)),
                "lambda_max": float(getattr(self.args, "fusion_ortho_lambda_max", 0.2)),
            },
            "loss_recon": {
                "lambda_attr": "fusion_lambda_recon",
                "target_ratio": self._maybe_get_float_arg("fusion_recon_target_ratio"),
                "lambda_min": float(getattr(self.args, "fusion_recon_lambda_min", 1e-8)),
                "lambda_max": float(getattr(self.args, "fusion_recon_lambda_max", 5.0)),
            },
        }

    @staticmethod
    def _default_dynamic_iou_monitor_stats() -> Dict[str, float]:
        return {
            "dynamic_iou_scanrefer_success_rate": 0.0,
            "dynamic_iou_scannet_det_success_rate": 0.0,
            "dynamic_iou_success_rate": 0.0,
        }

    def _unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        while hasattr(model, "module"):
            model = model.module
        return model

    def _get_feature_fusion_module(self, model: torch.nn.Module):
        unwrapped = self._unwrap_model(model)
        return getattr(unwrapped, "feature_fusion", None)

    def _is_mine_mode(self, feature_fusion) -> bool:
        return (
            feature_fusion is not None
            and hasattr(feature_fusion, "ortho_mode")
            and feature_fusion.ortho_mode == "mine"
        )

    def _get_output_field(self, outputs: Any, name: str):
        if outputs is None:
            return None
        if isinstance(outputs, dict):
            return outputs.get(name)
        return getattr(outputs, name, None)

    def _reduce_output_scalar(self, value: Any, loss: torch.Tensor) -> Optional[float]:
        if value is None:
            return None
        if torch.is_tensor(value):
            reduced = value.to(loss.device, loss.dtype).detach().float()
            if reduced.numel() != 1:
                reduced = reduced.mean()
        else:
            reduced = torch.tensor(float(value), device=loss.device, dtype=torch.float32)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
            reduced = reduced / world_size
        return float(reduced.item())

    def _get_dynamic_iou_params(self, model: torch.nn.Module) -> Tuple[float, float]:
        base_model = self._unwrap_model(model)
        config = getattr(base_model, "config", None)
        alpha = float(getattr(config, "label_weight_dynamic_iou_alpha", 0.0) or 0.0)
        eps = float(getattr(config, "label_weight_dynamic_iou_eps", 1e-6) or 1e-6)
        return alpha, max(eps, 1e-12)

    def _should_skip_invalid_dynamic_iou(self, model: torch.nn.Module) -> bool:
        base_model = self._unwrap_model(model)
        config = getattr(base_model, "config", None)
        return bool(getattr(config, "label_weight_dynamic_iou_skip_invalid", True))

    def _normalize_label_weight_by_weight_sum(self, model: torch.nn.Module) -> bool:
        base_model = self._unwrap_model(model)
        config = getattr(base_model, "config", None)
        return bool(getattr(config, "label_weight_loss_normalize_by_weight_sum", True))

    def _get_tokenizer(self):
        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None:
            tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "decode"):
            return None
        return tokenizer

    @staticmethod
    def _parse_bbox_numbers(text: str) -> Optional[List[float]]:
        parsed_values = parse_bbox_inner_values(text)
        if parsed_values is not None:
            return parsed_values
        numbers = _NUMBER_PATTERN.findall(text)
        if len(numbers) != 9:
            return None
        try:
            return [float(value) for value in numbers]
        except ValueError:
            return None

    def _parse_scanrefer_prediction(self, text: str) -> Tuple[Optional[int], Optional[List[float]]]:
        frame_match = _SCANREFER_FRAME_PATTERN.search(text)
        bbox_match = _JSON_BBOX_PATTERN.search(text)
        frame_idx = None
        if frame_match is not None:
            try:
                frame_idx = int(frame_match.group(1))
            except ValueError:
                frame_idx = None
        bbox = None
        if bbox_match is not None:
            bbox = self._parse_bbox_numbers(bbox_match.group(1))
        return frame_idx, bbox

    def _parse_bbox_matches(self, text: str) -> List[Tuple[List[float], int, int]]:
        matches: List[Tuple[List[float], int, int]] = []
        for match in _JSON_BBOX_PATTERN.finditer(text):
            bbox = self._parse_bbox_numbers(match.group(1))
            if bbox is None:
                continue
            matches.append((bbox, match.start(1), match.end(1)))
        return matches

    @staticmethod
    def _normalize_scannet_det_gt_boxes(boxes: Any) -> List[List[float]]:
        entries: List[List[float]] = []
        if not isinstance(boxes, list):
            return entries
        for item in boxes:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox_3d")
            if not isinstance(bbox, list) or len(bbox) != 9:
                continue
            try:
                entries.append([float(value) for value in bbox])
            except (TypeError, ValueError):
                continue
        return entries

    @staticmethod
    def _clamp_dynamic_weight(weight: float) -> float:
        return float(max(0.0, min(weight, 5.0)))

    def _text_char_span_to_token_span(
        self,
        tokenizer,
        text: str,
        char_start: int,
        char_end: int,
    ) -> Optional[Tuple[int, int]]:
        try:
            prefix_ids = tokenizer(text[:char_start], add_special_tokens=False)["input_ids"]
            span_ids = tokenizer(text[char_start:char_end], add_special_tokens=False)["input_ids"]
        except Exception:
            return None
        if not span_ids:
            return None
        return len(prefix_ids), len(prefix_ids) + len(span_ids)

    @staticmethod
    def _decode_supervised_text(
        tokenizer,
        token_ids: torch.Tensor,
        label_ids: torch.Tensor,
    ) -> str:
        valid_ids = token_ids[label_ids.ne(-100)].tolist()
        if not valid_ids:
            return ""
        return tokenizer.decode(
            valid_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    @staticmethod
    def _compute_box_iou(pred_box: Sequence[float], gt_box: Sequence[float]) -> float:
        try:
            from lmms_eval.tasks.threedod.utils import EulerDepthInstance3DBoxes

            pred_tensor = torch.tensor([pred_box], dtype=torch.float32)
            gt_tensor = torch.tensor([gt_box], dtype=torch.float32)
            iou = EulerDepthInstance3DBoxes.overlaps(
                EulerDepthInstance3DBoxes(pred_tensor, convention="ZXY"),
                EulerDepthInstance3DBoxes(gt_tensor, convention="ZXY"),
            ).item()
            return float(max(0.0, min(1.0, iou)))
        except Exception:
            return 0.0

    def _compute_ordered_box_ious(
        self,
        pred_boxes: List[List[float]],
        gt_boxes: List[List[float]],
    ) -> List[float]:
        if not gt_boxes:
            return []

        ious = [0.0 for _ in gt_boxes]
        common_count = min(len(pred_boxes), len(gt_boxes))
        if common_count <= 0:
            return ious

        try:
            from lmms_eval.tasks.threedod.utils import EulerDepthInstance3DBoxes

            pred_tensor = torch.tensor(pred_boxes[:common_count], dtype=torch.float32)
            gt_tensor = torch.tensor(gt_boxes[:common_count], dtype=torch.float32)
            iou_matrix = EulerDepthInstance3DBoxes.overlaps(
                EulerDepthInstance3DBoxes(pred_tensor, convention="ZXY"),
                EulerDepthInstance3DBoxes(gt_tensor, convention="ZXY"),
            )
            diag = torch.diagonal(iou_matrix, 0).tolist()
            for idx, value in enumerate(diag):
                ious[idx] = float(max(0.0, min(1.0, value)))
        except Exception:
            return ious

        return ious

    def _build_dynamic_iou_token_weights(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        shift_logits: torch.Tensor,
        shift_labels: torch.Tensor,
        shift_label_weight_codes: torch.Tensor,
        token_weights: torch.Tensor,
    ) -> torch.Tensor:
        alpha, eps = self._get_dynamic_iou_params(model)
        if alpha <= 0.0:
            return token_weights

        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return token_weights

        has_scanrefer = bool(
            (shift_label_weight_codes == LABEL_WEIGHT_CODE_SCANREFER_BBOX).any()
            or (shift_label_weight_codes == LABEL_WEIGHT_CODE_SCANREFER_FRAME).any()
        )
        has_scannet_det = bool((shift_label_weight_codes == LABEL_WEIGHT_CODE_SCANNET_DET_BBOX).any())
        if not has_scanrefer and not has_scannet_det:
            return token_weights

        dynamic_weights = token_weights.detach().clone()
        pred_ids = shift_logits.argmax(dim=-1).detach().cpu()
        label_ids = shift_labels.detach().cpu()
        skip_invalid_dynamic_iou = self._should_skip_invalid_dynamic_iou(model)
        scanrefer_targets = inputs.get("scanrefer_targets") or []
        scannet_det_boxes = inputs.get("scannet_det_boxes") or []
        scanrefer_needed_samples = 0
        scanrefer_success_samples = 0
        scannet_det_needed_samples = 0
        scannet_det_success_samples = 0

        for sample_idx in range(shift_labels.size(0)):
            sample_codes = shift_label_weight_codes[sample_idx]
            needs_scanrefer = bool(
                (sample_codes == LABEL_WEIGHT_CODE_SCANREFER_BBOX).any()
                or (sample_codes == LABEL_WEIGHT_CODE_SCANREFER_FRAME).any()
            )
            needs_scannet_det = bool((sample_codes == LABEL_WEIGHT_CODE_SCANNET_DET_BBOX).any())
            if not needs_scanrefer and not needs_scannet_det:
                continue
            if needs_scanrefer:
                scanrefer_needed_samples += 1
            if needs_scannet_det:
                scannet_det_needed_samples += 1

            pred_text = self._decode_supervised_text(tokenizer, pred_ids[sample_idx], label_ids[sample_idx])
            gt_text = self._decode_supervised_text(tokenizer, label_ids[sample_idx], label_ids[sample_idx])
            if needs_scanrefer:
                target = scanrefer_targets[sample_idx] if sample_idx < len(scanrefer_targets) else None
                gt_frame = None
                gt_box = None
                if isinstance(target, dict):
                    frame_value = target.get("frame")
                    bbox_value = target.get("bbox_3d")
                    if isinstance(frame_value, int):
                        gt_frame = frame_value
                    if isinstance(bbox_value, list) and len(bbox_value) == 9:
                        try:
                            gt_box = [float(value) for value in bbox_value]
                        except (TypeError, ValueError):
                            gt_box = None
                pred_frame, pred_box = self._parse_scanrefer_prediction(pred_text)
                scanrefer_iou = 0.0
                valid_scanrefer_target = gt_frame is not None and gt_box is not None
                valid_scanrefer_pred = pred_frame is not None and pred_box is not None
                scanrefer_mask = sample_codes.eq(LABEL_WEIGHT_CODE_SCANREFER_BBOX) | sample_codes.eq(
                    LABEL_WEIGHT_CODE_SCANREFER_FRAME
                )
                can_compute_iou = (
                    valid_scanrefer_target
                    and valid_scanrefer_pred
                    and pred_frame == gt_frame
                )
                if not can_compute_iou:
                    continue
                scanrefer_success_samples += 1
                scanrefer_iou = self._compute_box_iou(pred_box, gt_box)
                scanrefer_weight = self._clamp_dynamic_weight(1.0 - alpha * math.log(scanrefer_iou + eps))
                dynamic_weights[sample_idx] = torch.where(
                    scanrefer_mask,
                    dynamic_weights[sample_idx].new_tensor(scanrefer_weight).detach(),
                    dynamic_weights[sample_idx],
                )

            if needs_scannet_det:
                gt_boxes = scannet_det_boxes[sample_idx] if sample_idx < len(scannet_det_boxes) else None
                gt_box_list = self._normalize_scannet_det_gt_boxes(gt_boxes)
                pred_box_matches = self._parse_bbox_matches(pred_text)
                gt_box_matches = self._parse_bbox_matches(gt_text)
                pred_box_list = [bbox for bbox, _, _ in pred_box_matches]
                gt_text_box_list = [bbox for bbox, _, _ in gt_box_matches]

                if gt_box_list and gt_text_box_list and len(gt_box_list) == len(gt_text_box_list):
                    scannet_det_success_samples += 1
                    box_ious = self._compute_ordered_box_ious(pred_box_list, gt_box_list)
                    valid_positions = torch.nonzero(shift_labels[sample_idx].ne(-100), as_tuple=False).squeeze(-1)
                    for box_idx, (_, char_start, char_end) in enumerate(gt_box_matches):
                        token_span = self._text_char_span_to_token_span(
                            tokenizer,
                            gt_text,
                            char_start,
                            char_end,
                        )
                        if token_span is None:
                            continue
                        token_start, token_end = token_span
                        if token_start >= token_end or token_start >= valid_positions.numel():
                            continue
                        token_end = min(token_end, int(valid_positions.numel()))
                        if token_start >= token_end:
                            continue
                        seq_positions = valid_positions[token_start:token_end]
                        if seq_positions.numel() == 0:
                            continue
                        box_weight = self._clamp_dynamic_weight(
                            1.0 - alpha * math.log(box_ious[box_idx] + eps)
                        )
                        box_mask = torch.zeros_like(sample_codes, dtype=torch.bool)
                        box_mask[seq_positions] = True
                        box_mask = box_mask & sample_codes.eq(LABEL_WEIGHT_CODE_SCANNET_DET_BBOX)
                        dynamic_weights[sample_idx] = torch.where(
                            box_mask,
                            dynamic_weights[sample_idx].new_tensor(box_weight).detach(),
                            dynamic_weights[sample_idx],
                        )

        needed_samples = scanrefer_needed_samples + scannet_det_needed_samples
        success_samples = scanrefer_success_samples + scannet_det_success_samples
        self._latest_dynamic_iou_stats = {
            "dynamic_iou_scanrefer_success_rate": float(
                scanrefer_success_samples / max(scanrefer_needed_samples, 1)
            ),
            "dynamic_iou_scannet_det_success_rate": float(
                scannet_det_success_samples / max(scannet_det_needed_samples, 1)
            ),
            "dynamic_iou_success_rate": float(success_samples / max(needed_samples, 1)),
        }

        return dynamic_weights

    def _recompute_label_weighted_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        outputs: Any,
        fallback_loss: torch.Tensor,
    ) -> torch.Tensor:
        self._latest_dynamic_iou_stats = self._default_dynamic_iou_monitor_stats()
        labels = self._get_output_field(outputs, "aligned_labels")
        if labels is None:
            labels = inputs.get("labels")
        label_weight_codes = self._get_output_field(outputs, "aligned_label_weight_codes")
        if label_weight_codes is None:
            label_weight_codes = inputs.get("label_weight_codes")
        logits = self._get_output_field(outputs, "logits")
        if labels is None or label_weight_codes is None or logits is None:
            return fallback_loss

        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        shift_label_weight_codes = label_weight_codes[..., 1:].contiguous().to(shift_logits.device)
        valid_mask = shift_labels.ne(-100)
        label_weight_mask_factor = self._get_label_weight_mask_factor()

        if label_weight_mask_factor <= 0.0:
            token_weights = torch.where(
                valid_mask,
                torch.ones_like(shift_logits[..., 0]),
                torch.zeros_like(shift_logits[..., 0]),
            )
        else:
            weight_table = get_label_weight_value_table(
                self._unwrap_model(model).config,
                device=shift_logits.device,
                dtype=shift_logits.dtype,
            )
            token_weights = weight_table[shift_label_weight_codes.long()]
            token_weights = torch.where(
                valid_mask,
                token_weights,
                torch.zeros_like(token_weights),
            )

            alpha, _ = self._get_dynamic_iou_params(model)
            has_dynamic_codes = bool(
                (shift_label_weight_codes == LABEL_WEIGHT_CODE_SCANREFER_BBOX).any()
                or (shift_label_weight_codes == LABEL_WEIGHT_CODE_SCANREFER_FRAME).any()
                or (shift_label_weight_codes == LABEL_WEIGHT_CODE_SCANNET_DET_BBOX).any()
            )
            if alpha > 0.0 and has_dynamic_codes:
                token_weights = self._build_dynamic_iou_token_weights(
                    model,
                    inputs,
                    shift_logits,
                    shift_labels,
                    shift_label_weight_codes,
                    token_weights,
                )

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_weights = token_weights.view(-1)
        token_loss = torch.nn.functional.cross_entropy(flat_logits, flat_labels, reduction="none")
        normalize_by_weight_sum = self._normalize_label_weight_by_weight_sum(model)
        weight_sum = flat_weights.sum()
        if float(weight_sum.item()) > 0.0:
            weighted_loss_sum = (token_loss * flat_weights).sum()
            if normalize_by_weight_sum:
                loss_ce = weighted_loss_sum / weight_sum
            else:
                valid_count = flat_labels.ne(-100).sum().to(weighted_loss_sum.dtype)
                if float(valid_count.item()) > 0.0:
                    loss_ce = weighted_loss_sum / valid_count
                else:
                    loss_ce = weighted_loss_sum.new_zeros(())
        elif bool(valid_mask.any()):
            loss_ce = token_loss[flat_labels.ne(-100)].mean()
        else:
            loss_ce = token_loss.new_zeros(())

        total_loss = loss_ce
        base_model = self._unwrap_model(model)
        aux_specs = (
            ("loss_shared", "fusion_lambda_align"),
            ("loss_ortho", "fusion_lambda_ortho"),
            ("loss_recon", "fusion_lambda_recon"),
            ("loss_nrsr_kl", "fusion_lambda_nrsr"),
        )
        for loss_name, lambda_attr in aux_specs:
            aux_loss = self._get_output_field(outputs, loss_name)
            if aux_loss is None:
                continue
            total_loss = total_loss + float(getattr(base_model, lambda_attr, 1.0)) * aux_loss.to(
                total_loss.device,
                total_loss.dtype,
            )
        return total_loss

    def _training_progress(self) -> float:
        max_steps = int(getattr(self.state, "max_steps", 0) or 0)
        if max_steps > 0:
            return float(max(0.0, min(1.0, self.state.global_step / float(max_steps))))

        epoch = getattr(self.state, "epoch", None)
        num_train_epochs = getattr(self.args, "num_train_epochs", None)
        if epoch is not None and num_train_epochs is not None and float(num_train_epochs) > 0:
            return float(max(0.0, min(1.0, float(epoch) / float(num_train_epochs))))
        return 0.0

    def _get_label_weight_mask_factor(self) -> float:
        return 1.0

    def _maybe_get_float_arg(self, name: str) -> Optional[float]:
        value = getattr(self.args, name, None)
        if value is None:
            return None
        return float(value)

    def _get_fusion_method(self) -> str:
        base_model = self._unwrap_model(self.model)
        return str(getattr(getattr(base_model, "config", None), "feature_fusion_method", "add")).lower()

    def _update_ema(self, previous: Optional[float], new_value: float) -> float:
        if previous is None:
            return float(new_value)
        decay = max(min(self._ema_decay, 0.999999), 0.0)
        return decay * float(previous) + (1.0 - decay) * float(new_value)

    def _should_adjust_dynamic_aux_lambda(self) -> bool:
        warmup_enabled = bool(getattr(self.args, "fusion_lambda_warmup", False))
        warmup_steps = int(getattr(self.args, "fusion_lambda_warmup_steps", 0) or 0)
        return (not warmup_enabled) or (self.state.global_step >= warmup_steps)

    def _get_aux_lambda_bases(self, base_model: torch.nn.Module) -> Dict[str, float]:
        lambda_bases = getattr(base_model, "_fusion_aux_lambda_bases", None)
        if lambda_bases is None:
            lambda_bases = {}
            for lambda_attr in ("fusion_lambda_align", "fusion_lambda_ortho", "fusion_lambda_recon"):
                if hasattr(base_model, lambda_attr):
                    lambda_bases[lambda_attr] = float(getattr(base_model, lambda_attr))
            setattr(base_model, "_fusion_aux_lambda_bases", lambda_bases)
        return lambda_bases

    def _set_aux_lambda_base(self, base_model: torch.nn.Module, lambda_attr: str, value: float) -> None:
        lambda_bases = self._get_aux_lambda_bases(base_model)
        lambda_bases[lambda_attr] = float(value)

    def _accumulate_aux_monitor_logs(self, logs: Dict[str, float]) -> None:
        for key, value in logs.items():
            self._aux_monitor_sums[key] = self._aux_monitor_sums.get(key, 0.0) + float(value)
            self._aux_monitor_counts[key] = self._aux_monitor_counts.get(key, 0) + 1

    def _flush_aux_monitor_logs(self) -> Dict[str, float]:
        if not self._aux_monitor_sums:
            return {}
        averaged_logs = {
            key: float(self._aux_monitor_sums[key]) / float(max(self._aux_monitor_counts.get(key, 1), 1))
            for key in self._aux_monitor_sums
        }
        self._aux_monitor_sums = {}
        self._aux_monitor_counts = {}
        return averaged_logs

    def _get_aux_schedule_factor(self, loss_name: str) -> float:
        warmup_enabled = bool(getattr(self.args, "fusion_lambda_warmup", False))
        warmup_steps = max(int(getattr(self.args, "fusion_lambda_warmup_steps", 0) or 0), 1)
        warmup_factor = 1.0
        if warmup_enabled:
            warmup_factor = float(min(max(self.state.global_step, 0) / float(warmup_steps), 1.0))

        fusion_method = self._get_fusion_method()
        if fusion_method == "adver":
            progress = self._training_progress()
            if progress < self.ADVER_DISABLE_START:
                stage_factor = 1.0 if loss_name in {"loss_shared", "loss_recon"} else 0.0
            else:
                stage_factor = 0.0
            if stage_factor <= 0.0:
                return 0.0
            return float(stage_factor) * float(warmup_factor)
        if fusion_method == "adver_ortho":
            progress = self._training_progress()
            if progress < self.ADVER_ORTHO_STAGE1_END:
                stage_factor = 0.0
            elif progress < self.ADVER_ORTHO_STAGE2_END:
                stage_factor = 1.0 if loss_name == "loss_ortho" else 0.0
            else:
                stage_factor = 0.0
            if stage_factor <= 0.0:
                return 0.0
            return float(stage_factor) * float(warmup_factor)

        return warmup_factor

    def _is_nrsr_dynamic_enabled(self, model: torch.nn.Module) -> bool:
        if not bool(getattr(self.args, "fusion_lambda_nrsr_dynamic", False)):
            return False
        base_model = self._unwrap_model(model)
        fusion_method = str(
            getattr(getattr(base_model, "config", None), "feature_fusion_method", "add")
        ).lower()
        return fusion_method in {"nrsr_add", "nrsr_concat"}

    def _nrsr_target_ratio_from_progress(self, progress: float) -> float:
        if progress < self.NRSR_STAGE1_END:
            return 0.0
        if progress < self.NRSR_STAGE2_END:
            return max(0.0, float(self._nrsr_stage2_ratio))
        return max(0.0, float(self._nrsr_stage3_ratio))

    def _apply_nrsr_ratio_constraint(
        self, model: torch.nn.Module, loss: torch.Tensor, outputs: Any
    ) -> torch.Tensor:
        if outputs is None or loss is None or not model.training:
            return loss
        if not self._is_nrsr_dynamic_enabled(model):
            return loss

        loss_shared = self._get_output_field(outputs, "loss_shared")
        if loss_shared is not None:
            loss_shared = loss_shared.to(loss.device, loss.dtype)

        loss_nrsr_kl = self._get_output_field(outputs, "loss_nrsr_kl")
        if loss_nrsr_kl is None:
            return loss
        loss_nrsr_kl = loss_nrsr_kl.to(loss.device, loss.dtype)

        loss_ortho = self._get_output_field(outputs, "loss_ortho")
        if loss_ortho is not None:
            loss_ortho = loss_ortho.to(loss.device, loss.dtype)

        loss_recon = self._get_output_field(outputs, "loss_recon")
        if loss_recon is not None:
            loss_recon = loss_recon.to(loss.device, loss.dtype)

        base_model = self._unwrap_model(model)
        lambda_align = float(getattr(base_model, "fusion_lambda_align", 1.0))
        lambda_ortho = float(getattr(base_model, "fusion_lambda_ortho", 1.0))
        lambda_recon = float(getattr(base_model, "fusion_lambda_recon", 1.0))
        lambda_nrsr_prev = float(getattr(base_model, "fusion_lambda_nrsr", 1.0))

        loss_ce = loss - (lambda_nrsr_prev * loss_nrsr_kl)
        if loss_shared is not None:
            loss_ce = loss_ce - (lambda_align * loss_shared)
        if loss_ortho is not None:
            loss_ce = loss_ce - (lambda_ortho * loss_ortho)
        if loss_recon is not None:
            loss_ce = loss_ce - (lambda_recon * loss_recon)

        # Use globally averaged CE/KL (under DDP) so every rank shares the same lambda.
        scalars = torch.stack([loss_ce.detach().float(), loss_nrsr_kl.detach().float()]).to(loss.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            torch.distributed.all_reduce(scalars, op=torch.distributed.ReduceOp.SUM)
            scalars = scalars / world_size

        loss_ce_scalar = float(scalars[0].item())
        loss_nrsr_scalar = float(scalars[1].item())

        progress = self._training_progress()
        target_ratio = self._nrsr_target_ratio_from_progress(progress)
        if target_ratio <= 0.0 or loss_nrsr_scalar <= 0.0:
            lambda_nrsr_new = 0.0
        else:
            lambda_nrsr_new = target_ratio * max(loss_ce_scalar, 0.0) / (loss_nrsr_scalar + self._nrsr_ratio_eps)
            if not (lambda_nrsr_new >= 0.0):
                lambda_nrsr_new = 0.0

        setattr(base_model, "fusion_lambda_nrsr", float(lambda_nrsr_new))

        loss_new = loss_ce + (float(lambda_nrsr_new) * loss_nrsr_kl)
        if loss_shared is not None:
            loss_new = loss_new + (lambda_align * loss_shared)
        if loss_ortho is not None:
            loss_new = loss_new + (lambda_ortho * loss_ortho)
        if loss_recon is not None:
            loss_new = loss_new + (lambda_recon * loss_recon)
        return loss_new

    def _update_aux_monitor_logs(self, model: torch.nn.Module, loss: torch.Tensor, outputs: Any) -> None:
        if outputs is None or loss is None:
            return

        base_model = self._unwrap_model(model)
        aux_logs: Dict[str, float] = {
            "loss_ce": float(loss.detach().float().item()),
            "label_weight_mask_factor": float(self._get_label_weight_mask_factor()),
        }
        aux_logs.update(self._latest_dynamic_iou_stats)
        aux_terms = []
        aux_specs = (
            ("loss_shared", "fusion_lambda_align"),
            ("loss_ortho", "fusion_lambda_ortho"),
            ("loss_recon", "fusion_lambda_recon"),
            ("loss_nrsr_kl", "fusion_lambda_nrsr"),
        )
        for loss_name, lambda_attr in aux_specs:
            aux_loss = self._get_output_field(outputs, loss_name)
            if aux_loss is None:
                continue
            aux_terms.append(
                (
                    loss_name,
                    aux_loss.to(loss.device, loss.dtype),
                    float(getattr(base_model, lambda_attr, 1.0)),
                    lambda_attr,
                )
            )

        if aux_terms:
            loss_ce = loss
            for _, aux_loss, lambda_value, _ in aux_terms:
                loss_ce = loss_ce - (lambda_value * aux_loss)

            # Convert to float scalars (possibly average across DDP ranks).
            loss_ce_val = loss_ce.detach().float()
            values = [loss_ce_val]
            values.extend(aux_loss.detach().float() for _, aux_loss, _, _ in aux_terms)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                # All-reduce scalars for consistent dynamic lambda across workers.
                reduce_t = torch.stack(values).to(loss.device)
                torch.distributed.all_reduce(reduce_t, op=torch.distributed.ReduceOp.SUM)
                reduce_t = reduce_t / world_size
                values = [reduce_t[i] for i in range(reduce_t.shape[0])]

            loss_ce_val_f = float(values[0].item())
            aux_logs["loss_ce"] = loss_ce_val_f
            aux_values = {}
            for idx, (loss_name, _, _, _) in enumerate(aux_terms, start=1):
                aux_values[loss_name] = float(values[idx].item())

            dynamic_aux_names = [
                loss_name
                for loss_name, config in self._dynamic_aux_configs.items()
                if aux_values.get(loss_name) is not None and config.get("target_ratio") is not None
            ]
            if dynamic_aux_names:
                self._loss_ce_ema = self._update_ema(self._loss_ce_ema, loss_ce_val_f)

            aux_log_specs = (
                ("loss_shared", "fusion_lambda_align"),
                ("loss_ortho", "fusion_lambda_ortho"),
                ("loss_recon", "fusion_lambda_recon"),
            )
            should_adjust_dynamic = self._should_adjust_dynamic_aux_lambda()
            for loss_name, lambda_attr in aux_log_specs:
                loss_value_raw = aux_values.get(loss_name)
                if loss_value_raw is None:
                    continue

                config = self._dynamic_aux_configs.get(loss_name)
                lambda_value = float(getattr(base_model, lambda_attr, 1.0))
                schedule_factor = self._get_aux_schedule_factor(loss_name)
                if config is not None and config.get("target_ratio") is not None:
                    previous_ema = self._aux_loss_emas.get(loss_name)
                    self._aux_loss_emas[loss_name] = self._update_ema(previous_ema, loss_value_raw)
                    scheduled_target_ratio = float(config["target_ratio"]) * float(schedule_factor)
                    if should_adjust_dynamic and self._loss_ce_ema is not None and schedule_factor > 0.0:
                        lambda_target = scheduled_target_ratio * float(self._loss_ce_ema) / (
                            float(self._aux_loss_emas[loss_name]) + self._ema_eps
                        )
                        lambda_target = max(
                            float(config["lambda_min"]),
                            min(float(config["lambda_max"]), float(lambda_target)),
                        )
                        self._set_aux_lambda_base(
                            base_model,
                            str(config["lambda_attr"]),
                            float(lambda_target) / float(schedule_factor),
                        )
                        lambda_value = float(lambda_target)
                    elif schedule_factor <= 0.0:
                        lambda_value = 0.0
                    aux_logs[f"{lambda_attr}_dyn"] = float(lambda_value)
                    aux_logs[f"{loss_name}_ratio_target"] = float(scheduled_target_ratio)

                aux_logs.update(
                    {
                        loss_name: float(loss_value_raw),
                        f"{loss_name}_weighted": float(lambda_value) * float(loss_value_raw),
                    }
                )

            loss_nrsr_val_f = aux_values.get("loss_nrsr_kl")
            if loss_nrsr_val_f is not None:
                lambda_nrsr = float(getattr(base_model, "fusion_lambda_nrsr", 1.0))
                loss_nrsr_weighted = float(lambda_nrsr) * float(loss_nrsr_val_f)
                nrsr_ratio = loss_nrsr_weighted / (loss_ce_val_f + self._ema_eps)
                nrsr_progress = self._training_progress()
                nrsr_ratio_target = self._nrsr_target_ratio_from_progress(nrsr_progress)
                aux_logs.update(
                    {
                        "loss_nrsr_kl": float(loss_nrsr_val_f),
                        "loss_nrsr_kl_weighted": loss_nrsr_weighted,
                        "fusion_lambda_nrsr_dyn": float(lambda_nrsr),
                        "loss_nrsr_ratio": float(nrsr_ratio),
                        "loss_nrsr_ratio_target": float(nrsr_ratio_target),
                        "nrsr_progress": float(nrsr_progress),
                    }
                )

        gate_shared_ratio = self._get_output_field(outputs, "gate_shared_ratio")
        gate_shared_ratio_value = self._reduce_output_scalar(gate_shared_ratio, loss)
        if gate_shared_ratio_value is not None:
            aux_logs["gate_shared_ratio"] = gate_shared_ratio_value

        gate_missing_question_summary = self._get_output_field(outputs, "gate_missing_question_summary")
        gate_missing_question_summary_value = self._reduce_output_scalar(gate_missing_question_summary, loss)
        if gate_missing_question_summary_value is not None:
            aux_logs["gate_missing_question_summary"] = gate_missing_question_summary_value

        fusion_monitor_stats = self._get_output_field(outputs, "fusion_monitor_stats")
        if isinstance(fusion_monitor_stats, dict):
            for key, value in fusion_monitor_stats.items():
                if key in aux_logs:
                    continue
                reduced_value = self._reduce_output_scalar(value, loss)
                if reduced_value is not None:
                    aux_logs[key] = reduced_value

        self._accumulate_aux_monitor_logs(aux_logs)

    def _cache_mine_features(self, outputs: Any) -> None:
        unique_features = self._get_output_field(outputs, "mine_unique_features")
        features_2d = self._get_output_field(outputs, "mine_2d_features")
        if unique_features is None or features_2d is None:
            self._mine_cached_features = None
            return

        self._mine_cached_features = (
            unique_features.detach(),
            features_2d.detach(),
        )

    def _ensure_mine_optimizer(self, feature_fusion) -> None:
        if self._mine_optimizer is not None:
            return
        mine_parameters = list(feature_fusion.get_mine_parameters())
        if not mine_parameters:
            return
        mine_lr = float(getattr(self.args, "mm_projector_lr", None) or self.args.learning_rate)
        self._mine_optimizer = torch.optim.AdamW(mine_parameters, lr=mine_lr, weight_decay=0.0)

    def _get_mine_warmup_steps(self, model: torch.nn.Module) -> int:
        base_model = self._unwrap_model(model)
        config = getattr(base_model, "config", None)
        warmup_steps = int(getattr(config, "fusion_mine_q_warmup_steps", 500) or 0)
        return max(warmup_steps, 0)

    def _run_mine_warmup(self, model: torch.nn.Module, epoch_idx: int, warmup_steps: int) -> None:
        if warmup_steps <= 0:
            return

        feature_fusion = self._get_feature_fusion_module(model)
        if not self._is_mine_mode(feature_fusion):
            return

        supported_methods = {"decompose_add", "decompose_concat"}
        fusion_method = getattr(feature_fusion, "fusion_method", None)
        if fusion_method not in supported_methods:
            if self.is_world_process_zero():
                print(
                    "[MINEWarmup] Skip q_net warmup because "
                    f"fusion_method={fusion_method} does not provide cached mine features."
                )
            return

        self._ensure_mine_optimizer(feature_fusion)
        if self._mine_optimizer is None:
            if self.is_world_process_zero():
                print("[MINEWarmup] Skip q_net warmup because mine optimizer is unavailable.")
            return

        # Build an independent iterator so warmup never consumes the ongoing main-train iterator.
        warmup_dataloader = self.get_train_dataloader()
        warmup_iterator = iter(warmup_dataloader)
        is_rank0 = self.is_world_process_zero()
        if is_rank0:
            print(
                f"[MINEWarmup] Epoch {epoch_idx}: q_net warmup starts "
                f"({warmup_steps} batches / {warmup_steps} updates)."
            )
        progress_bar = tqdm(
            range(warmup_steps),
            desc=f"[MINEWarmup][Epoch {epoch_idx}] q_net",
            disable=not is_rank0,
            leave=False,
        )
        last_q_loss: Optional[float] = None
        feature_fusion.set_mine_network_trainable(False)
        try:
            for _ in progress_bar:
                try:
                    warmup_inputs = next(warmup_iterator)
                except StopIteration:
                    warmup_iterator = iter(warmup_dataloader)
                    warmup_inputs = next(warmup_iterator)

                warmup_inputs = self._prepare_inputs(warmup_inputs)
                with torch.no_grad():
                    with self.compute_loss_context_manager():
                        outputs = model(**warmup_inputs)

                unique_features = self._get_output_field(outputs, "mine_unique_features")
                features_2d = self._get_output_field(outputs, "mine_2d_features")
                if unique_features is None or features_2d is None:
                    continue

                unique_features = unique_features.detach().float()
                features_2d = features_2d.detach().float()

                feature_fusion.set_mine_network_trainable(True)
                self._mine_optimizer.zero_grad(set_to_none=True)
                q_loss = feature_fusion.compute_mine_q_loss_from_cache(
                    unique_features, features_2d
                )
                q_loss.backward()
                self._mine_optimizer.step()
                feature_fusion.set_mine_network_trainable(False)

                last_q_loss = float(q_loss.detach().float().item())
                if is_rank0:
                    progress_bar.set_postfix({"q_loss": f"{last_q_loss:.4f}"})
        finally:
            self._mine_optimizer.zero_grad(set_to_none=True)
            feature_fusion.set_mine_network_trainable(False)
            progress_bar.close()

        if is_rank0:
            if last_q_loss is None:
                print(
                    f"[MINEWarmup] Epoch {epoch_idx}: q_net warmup completed, "
                    "but no valid mine feature pairs were found."
                )
            else:
                print(
                    f"[MINEWarmup] Epoch {epoch_idx}: q_net warmup completed, "
                    f"last_q_loss={last_q_loss:.6f}"
                )

    def _maybe_run_mine_warmup(self, model: torch.nn.Module) -> None:
        feature_fusion = self._get_feature_fusion_module(model)
        if not self._is_mine_mode(feature_fusion):
            return

        warmup_steps = self._get_mine_warmup_steps(model)
        if warmup_steps <= 0:
            return

        epoch_value = 0.0 if self.state.epoch is None else float(self.state.epoch)
        epoch_idx = max(int(epoch_value), 0)
        # Only trigger at epoch boundaries, not in the middle of a resumed epoch.
        if abs(epoch_value - float(epoch_idx)) > 1e-8:
            return
        if epoch_idx in self._mine_warmup_done_epochs:
            return

        self._run_mine_warmup(model, epoch_idx=epoch_idx, warmup_steps=warmup_steps)
        self._mine_warmup_done_epochs.add(epoch_idx)

    def _run_mine_updates(self, model: torch.nn.Module) -> None:
        if self._mine_cached_features is None:
            return

        feature_fusion = self._get_feature_fusion_module(model)
        if not self._is_mine_mode(feature_fusion):
            self._mine_cached_features = None
            return

        feature_fusion.set_mine_network_trainable(True)
        self._ensure_mine_optimizer(feature_fusion)
        if self._mine_optimizer is None:
            feature_fusion.set_mine_network_trainable(False)
            self._mine_cached_features = None
            return

        unique_features, features_2d = self._mine_cached_features
        unique_features = unique_features.float()
        features_2d = features_2d.float()

        for _ in range(self._mine_update_steps):
            self._mine_optimizer.zero_grad(set_to_none=True)
            q_loss = feature_fusion.compute_mine_q_loss_from_cache(
                unique_features, features_2d
            )
            q_loss.backward()
            self._mine_optimizer.step()

        self._mine_optimizer.zero_grad(set_to_none=True)
        feature_fusion.set_mine_network_trainable(False)
        self._mine_cached_features = None

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        feature_fusion = self._get_feature_fusion_module(model)
        should_cache_mine = bool(model.training and self._is_mine_mode(feature_fusion))
        if should_cache_mine:
            feature_fusion.set_mine_network_trainable(False)

        loss_outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            *args,
            **kwargs,
        )
        if isinstance(loss_outputs, tuple):
            loss, outputs = loss_outputs
        else:
            loss, outputs = loss_outputs, None

        if outputs is not None and loss is not None:
            loss = self._recompute_label_weighted_loss(model, inputs, outputs, loss)
        loss = self._apply_nrsr_ratio_constraint(model, loss, outputs)
        self._update_aux_monitor_logs(model, loss, outputs)

        if should_cache_mine:
            self._cache_mine_features(outputs)
        else:
            self._mine_cached_features = None

        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], *args, **kwargs):
        self._maybe_run_mine_warmup(model)
        loss = super().training_step(model, inputs, *args, **kwargs)
        self._run_mine_updates(model)
        return loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        averaged_aux_logs = self._flush_aux_monitor_logs()
        if averaged_aux_logs:
            logs = dict(logs)
            logs.update(averaged_aux_logs)
        super().log(logs, *args, **kwargs)


# Apply monkey patches
Trainer.create_optimizer = create_optimizer

Qwen2VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2VLModel.print_trainable_parameters = print_trainable_parameters
Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2_5_VLModel.print_trainable_parameters = print_trainable_parameters
