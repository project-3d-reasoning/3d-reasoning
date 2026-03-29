import os
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

    def _training_progress(self) -> float:
        max_steps = int(getattr(self.state, "max_steps", 0) or 0)
        if max_steps > 0:
            return float(max(0.0, min(1.0, self.state.global_step / float(max_steps))))

        epoch = getattr(self.state, "epoch", None)
        num_train_epochs = getattr(self.args, "num_train_epochs", None)
        if epoch is not None and num_train_epochs is not None and float(num_train_epochs) > 0:
            return float(max(0.0, min(1.0, float(epoch) / float(num_train_epochs))))
        return 0.0

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
                stage_factor = 1.0 if loss_name in {"loss_shared", "loss_recon"} else 0.0
            elif progress < self.ADVER_ORTHO_STAGE2_END:
                stage_factor = 1.0
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

        if not aux_terms:
            loss_ce = loss.detach().float()
            self._accumulate_aux_monitor_logs(
                {
                "loss_ce": float(loss_ce.item()),
                }
            )
            return

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

        aux_logs: Dict[str, float] = {
            "loss_ce": loss_ce_val_f,
        }

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
        if gate_shared_ratio is not None:
            gate_shared_ratio = gate_shared_ratio.to(loss.device, loss.dtype).detach().float()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                torch.distributed.all_reduce(gate_shared_ratio, op=torch.distributed.ReduceOp.SUM)
                gate_shared_ratio = gate_shared_ratio / world_size
            aux_logs["gate_shared_ratio"] = float(gate_shared_ratio.item())

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
