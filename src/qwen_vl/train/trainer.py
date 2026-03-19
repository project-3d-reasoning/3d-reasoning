import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
            projector_parameters = [
                name for name, _ in opt_model.named_parameters() if "merger" in name
            ]
            if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                vision_tower_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


class VGTrainer(Trainer):
    """Trainer with optional extra q_theta updates for vCLUB in ortho mine mode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mine_cached_features: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._mine_optimizer: Optional[torch.optim.Optimizer] = None
        self._mine_update_steps: int = 5
        self._aux_monitor_logs: Dict[str, float] = {}

    def _unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        while hasattr(model, "module"):
            model = model.module
        return model

    def _dist_mean(self, value: torch.Tensor) -> torch.Tensor:
        """All-reduce mean for a scalar tensor (no-op if not distributed)."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return value
        v = value.detach().clone()
        torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
        v = v / float(torch.distributed.get_world_size())
        return v

    def _auto_balance_fusion_lambdas(
        self, model: torch.nn.Module, loss: torch.Tensor, outputs: Any
    ) -> None:
        """Dynamically set fusion lambdas so weighted aux losses track CE ratios."""
        if outputs is None or loss is None or (not model.training):
            return

        base_model = self._unwrap_model(model)
        cfg = getattr(base_model, "config", None)
        if cfg is None or not bool(getattr(cfg, "fusion_lambda_auto_balance", False)):
            return

        loss_align = self._get_output_field(outputs, "loss_align")
        loss_ortho = self._get_output_field(outputs, "loss_ortho")
        loss_recon = self._get_output_field(outputs, "loss_recon")
        if loss_align is None and loss_ortho is None and loss_recon is None:
            return

        # Read current lambdas
        lambda_align = float(getattr(base_model, "fusion_lambda_align", 1.0))
        lambda_ortho = float(getattr(base_model, "fusion_lambda_ortho", 1.0))
        lambda_recon = float(getattr(base_model, "fusion_lambda_recon", 1.0))

        eps = float(getattr(cfg, "fusion_lambda_auto_eps", 1e-8))
        min_lambda = float(getattr(cfg, "fusion_lambda_auto_min", 0.0))
        max_lambda = float(getattr(cfg, "fusion_lambda_auto_max", 1000.0))
        ema = float(getattr(cfg, "fusion_lambda_auto_ema", 0.10))
        ema = max(0.0, min(1.0, ema))

        target_align = float(getattr(cfg, "fusion_lambda_target_align", 0.05))
        target_ortho = float(getattr(cfg, "fusion_lambda_target_ortho", 0.05))
        target_recon = float(getattr(cfg, "fusion_lambda_target_recon", 0.10))

        # Compute CE estimate from current total loss and current-weight aux losses.
        weighted_aux = loss.new_zeros(())
        if loss_align is not None:
            weighted_aux = weighted_aux + float(lambda_align) * loss_align.to(loss.device, loss.dtype)
        if loss_ortho is not None:
            weighted_aux = weighted_aux + float(lambda_ortho) * loss_ortho.to(loss.device, loss.dtype)
        if loss_recon is not None:
            weighted_aux = weighted_aux + float(lambda_recon) * loss_recon.to(loss.device, loss.dtype)

        ce_est = torch.clamp(loss - weighted_aux, min=eps)

        # Use distributed-mean scalars to keep all ranks in sync.
        ce_mean = self._dist_mean(ce_est.detach().float())
        align_mean = self._dist_mean(
            (loss_align.detach().float() if loss_align is not None else ce_mean.new_zeros(()))
        )
        ortho_mean = self._dist_mean(
            (loss_ortho.detach().float() if loss_ortho is not None else ce_mean.new_zeros(()))
        )
        recon_mean = self._dist_mean(
            (loss_recon.detach().float() if loss_recon is not None else ce_mean.new_zeros(()))
        )

        def _update(old: float, target_ratio: float, aux_value: torch.Tensor) -> float:
            aux_safe = torch.clamp(aux_value, min=eps)
            new_val = (target_ratio * ce_mean / aux_safe).item()
            blended = (1.0 - ema) * float(old) + ema * float(new_val)
            return float(max(min_lambda, min(max_lambda, blended)))

        with torch.no_grad():
            if loss_align is not None:
                lambda_align = _update(lambda_align, target_align, align_mean)
                setattr(base_model, "fusion_lambda_align", lambda_align)
            if loss_ortho is not None:
                lambda_ortho = _update(lambda_ortho, target_ortho, ortho_mean)
                setattr(base_model, "fusion_lambda_ortho", lambda_ortho)
            if loss_recon is not None:
                lambda_recon = _update(lambda_recon, target_recon, recon_mean)
                setattr(base_model, "fusion_lambda_recon", lambda_recon)

        # For monitoring
        self._aux_monitor_logs.update(
            {
                "fusion_lambda_align": float(lambda_align),
                "fusion_lambda_ortho": float(lambda_ortho),
                "fusion_lambda_recon": float(lambda_recon),
            }
        )

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

    def _update_aux_monitor_logs(self, model: torch.nn.Module, loss: torch.Tensor, outputs: Any) -> None:
        self._aux_monitor_logs = {}
        if outputs is None or loss is None:
            return

        loss_align = self._get_output_field(outputs, "loss_align")
        loss_ortho = self._get_output_field(outputs, "loss_ortho")
        loss_recon = self._get_output_field(outputs, "loss_recon")
        if loss_align is None and loss_ortho is None and loss_recon is None:
            return

        base_model = self._unwrap_model(model)
        lambda_align = float(getattr(base_model, "fusion_lambda_align", 1.0))
        lambda_ortho = float(getattr(base_model, "fusion_lambda_ortho", 1.0))
        lambda_recon = float(getattr(base_model, "fusion_lambda_recon", 1.0))

        weighted_aux = loss.new_zeros(())
        if loss_align is not None:
            weighted_aux = weighted_aux + lambda_align * loss_align.to(loss.device, loss.dtype)
        if loss_ortho is not None:
            weighted_aux = weighted_aux + lambda_ortho * loss_ortho.to(loss.device, loss.dtype)
        if loss_recon is not None:
            weighted_aux = weighted_aux + lambda_recon * loss_recon.to(loss.device, loss.dtype)

        ce_est = loss - weighted_aux
        ce_safe = torch.clamp(ce_est, min=1e-8)
        ratio = weighted_aux / ce_safe

        self._aux_monitor_logs = {
            "loss_ce_est": float(ce_est.detach().float().item()),
            "loss_aux_weighted": float(weighted_aux.detach().float().item()),
            "loss_aux_ratio": float(ratio.detach().float().item()),
            "fusion_lambda_align": float(lambda_align),
            "fusion_lambda_ortho": float(lambda_ortho),
            "fusion_lambda_recon": float(lambda_recon),
        }
        if loss_align is not None:
            self._aux_monitor_logs["loss_align"] = float(loss_align.detach().float().item())
            self._aux_monitor_logs["loss_align_weighted"] = float(
                (lambda_align * loss_align).detach().float().item()
            )
        if loss_ortho is not None:
            self._aux_monitor_logs["loss_ortho"] = float(loss_ortho.detach().float().item())
            self._aux_monitor_logs["loss_ortho_weighted"] = float(
                (lambda_ortho * loss_ortho).detach().float().item()
            )
        if loss_recon is not None:
            self._aux_monitor_logs["loss_recon"] = float(loss_recon.detach().float().item())
            self._aux_monitor_logs["loss_recon_weighted"] = float(
                (lambda_recon * loss_recon).detach().float().item()
            )

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

        self._update_aux_monitor_logs(model, loss, outputs)
        self._auto_balance_fusion_lambdas(model, loss, outputs)

        if should_cache_mine:
            self._cache_mine_features(outputs)
        else:
            self._mine_cached_features = None

        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], *args, **kwargs):
        loss = super().training_step(model, inputs, *args, **kwargs)
        self._run_mine_updates(model)
        return loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if self._aux_monitor_logs:
            logs = dict(logs)
            logs.update(self._aux_monitor_logs)
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
