import math
import os
from typing import Dict, List, Optional, Sequence

import datasets
import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.utils.data import DataLoader, Sampler
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


class _BBoxResidualLogTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.raw_sum = 0.0
        self.weighted_sum = 0.0
        self.total_sum = 0.0
        self.count = 0.0

    @staticmethod
    def _to_float(value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            value = value.detach().float().mean().item()
        else:
            value = float(value)

        if not math.isfinite(value):
            return None
        return value

    def update(self, raw_loss, weighted_loss, total_loss) -> None:
        raw_value = self._to_float(raw_loss)
        weighted_value = self._to_float(weighted_loss)
        total_value = self._to_float(total_loss)
        if raw_value is None or weighted_value is None or total_value is None:
            return

        self.raw_sum += raw_value
        self.weighted_sum += weighted_value
        self.total_sum += max(total_value, 1e-12)
        self.count += 1.0

    def flush(self, device: torch.device) -> Dict[str, float]:
        if self.count <= 0:
            return {}

        device = torch.device(device)
        sync_device = device
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_backend() == "nccl"
            and sync_device.type != "cuda"
        ):
            sync_device = torch.device("cuda", torch.cuda.current_device())

        stats = torch.tensor(
            [self.raw_sum, self.weighted_sum, self.total_sum, self.count],
            dtype=torch.float64,
            device=sync_device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

        raw_avg = (stats[0] / stats[3].clamp_min(1.0)).item()
        weighted_over_total = (stats[1] / stats[2].clamp_min(1e-12)).item()
        self.reset()
        return {
            "BBOX_RESIDUAL_LOSS_raw": raw_avg,
            "BBOX_RESIDUAL_LOSS_weighted_over_total": weighted_over_total,
        }


class _ConstantLRSchedulerWrapper:
    def __init__(self, scheduler, optimizer, constant_group_indices):
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.constant_group_indices = tuple(int(idx) for idx in constant_group_indices)
        self.constant_lrs = {
            idx: float(self.optimizer.param_groups[idx]["lr"])
            for idx in self.constant_group_indices
        }
        self._restore_constant_lrs()

    def _restore_constant_lrs(self):
        for idx, lr in self.constant_lrs.items():
            if idx < len(self.optimizer.param_groups):
                self.optimizer.param_groups[idx]["lr"] = lr

    def step(self, *args, **kwargs):
        out = self.scheduler.step(*args, **kwargs)
        self._restore_constant_lrs()
        return out

    def state_dict(self):
        return {
            "scheduler": self.scheduler.state_dict(),
            "constant_group_indices": list(self.constant_group_indices),
            "constant_lrs": self.constant_lrs,
        }

    def load_state_dict(self, state_dict):
        if "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])
            self.constant_group_indices = tuple(
                int(idx) for idx in state_dict.get("constant_group_indices", self.constant_group_indices)
            )
            self.constant_lrs = {
                int(idx): float(lr)
                for idx, lr in state_dict.get("constant_lrs", self.constant_lrs).items()
            }
        else:
            self.scheduler.load_state_dict(state_dict)
        self._restore_constant_lrs()

    def get_last_lr(self):
        if hasattr(self.scheduler, "get_last_lr"):
            lrs = list(self.scheduler.get_last_lr())
        else:
            lrs = [group["lr"] for group in self.optimizer.param_groups]
        for idx, lr in self.constant_lrs.items():
            if idx < len(lrs):
                lrs[idx] = lr
        return lrs

    def __getattr__(self, name):
        return getattr(self.scheduler, name)


class VGTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bbox_residual_log_tracker = _BBoxResidualLogTracker()

    @staticmethod
    def _get_output_metric(outputs, key: str):
        if outputs is None:
            return None
        if isinstance(outputs, dict):
            return outputs.get(key)
        return getattr(outputs, key, None)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        try:
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        except TypeError:
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
            )

        if model.training:
            self._bbox_residual_log_tracker.update(
                self._get_output_metric(outputs, "bbox_residual_loss_raw"),
                self._get_output_metric(outputs, "bbox_residual_loss_weighted"),
                loss,
            )

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        logs = dict(logs)
        if "loss" in logs:
            logs.update(self._bbox_residual_log_tracker.flush(self.args.device))

        try:
            return super().log(logs, start_time=start_time)
        except TypeError:
            return super().log(logs)

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        try:
            scheduler = super().create_scheduler(
                num_training_steps=num_training_steps,
                optimizer=optimizer,
            )
        except TypeError:
            scheduler = super().create_scheduler(num_training_steps, optimizer)

        constant_group_indices = getattr(self, "_constant_lr_param_group_indices", [])
        target_optimizer = optimizer if optimizer is not None else self.optimizer
        if (
            scheduler is not None
            and target_optimizer is not None
            and constant_group_indices
            and not isinstance(scheduler, _ConstantLRSchedulerWrapper)
        ):
            scheduler = _ConstantLRSchedulerWrapper(
                scheduler=scheduler,
                optimizer=target_optimizer,
                constant_group_indices=constant_group_indices,
            )
            self.lr_scheduler = scheduler
        return scheduler


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
    opt_model = self.model

    if self.optimizer is None:
        decay_parameters = set(get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS))
        decay_parameters = {name for name in decay_parameters if "bias" not in name}
        named_parameters = [
            (name, parameter)
            for name, parameter in opt_model.named_parameters()
            if parameter.requires_grad
        ]

        use_projector_lr = (
            self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0
        )
        use_vision_lr = (
            self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0
        )
        use_residual_head_lr = (
            getattr(self.args, "bbox_residual_head_lr", None) is not None
            and float(self.args.bbox_residual_head_lr) > 0
        )

        projector_parameters = {
            name for name, _ in named_parameters if "merger" in name
        } if use_projector_lr else set()
        vision_tower_parameters = {
            name for name, _ in named_parameters if "visual" in name
        } if use_vision_lr else set()
        residual_head_parameters = {
            name for name, _ in named_parameters if "bbox_residual_head" in name
        } if use_residual_head_lr else set()

        def resolve_group_tag(name: str) -> str:
            if name in residual_head_parameters:
                return "bbox_residual_head"
            if name in projector_parameters:
                return "projector"
            if name in vision_tower_parameters:
                return "vision"
            return "default"

        optimizer_grouped_parameters = []
        constant_group_indices = []

        group_specs = [
            ("default", None, False),
            ("vision", self.args.vision_tower_lr if use_vision_lr else None, False),
            ("projector", self.args.mm_projector_lr if use_projector_lr else None, False),
            (
                "bbox_residual_head",
                float(self.args.bbox_residual_head_lr) if use_residual_head_lr else None,
                use_residual_head_lr,
            ),
        ]

        for group_tag, lr_override, keep_constant_lr in group_specs:
            if group_tag == "vision" and not use_vision_lr:
                continue
            if group_tag == "projector" and not use_projector_lr:
                continue
            if group_tag == "bbox_residual_head" and not use_residual_head_lr:
                continue

            for use_decay in (True, False):
                group_params = [
                    parameter
                    for name, parameter in named_parameters
                    if resolve_group_tag(name) == group_tag and ((name in decay_parameters) == use_decay)
                ]
                if not group_params:
                    continue

                group = {
                    "params": group_params,
                    "weight_decay": self.args.weight_decay if use_decay else 0.0,
                }
                if lr_override is not None:
                    group["lr"] = lr_override
                optimizer_grouped_parameters.append(group)
                if keep_constant_lr:
                    constant_group_indices.append(len(optimizer_grouped_parameters) - 1)

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        self._constant_lr_param_group_indices = constant_group_indices

    return self.optimizer


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
