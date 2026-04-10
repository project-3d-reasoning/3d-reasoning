import contextlib
import os
from typing import Any, Dict, List, Optional, Sequence

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
    DistributedType,
    OptimizerNames,
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
)
from transformers.trainer_utils import SaveStrategy, seed_worker

try:
    from transformers.trainer import clear_device_cache
except ImportError:
    from accelerate.utils.memory import clear_device_cache

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward
else:
    smp_forward_backward = None

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


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


class QwenLossLoggingTrainer(Trainer):
    """Trainer that backpropagates the full loss but logs CE and HSIC components separately."""

    FEATURE_STAT_KEYS = (
        "unique_3d_projected_mean",
        "unique_3d_projected_var",
        "feature_2d_mean",
        "feature_2d_var",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tr_loss_hsic_raw = None
        self._tr_loss_hsic_weighted = None
        self._tr_feature_stats = {}

    @staticmethod
    def _get_output_value(outputs: Any, key: str, default: Optional[torch.Tensor] = None):
        if isinstance(outputs, dict):
            return outputs.get(key, default)
        return getattr(outputs, key, default)

    def _ensure_aux_loss_buffers(self, reference: torch.Tensor) -> None:
        if self._tr_loss_hsic_raw is None or self._tr_loss_hsic_weighted is None:
            zero = reference.detach().new_zeros(())
            self._tr_loss_hsic_raw = zero.clone()
            self._tr_loss_hsic_weighted = zero.clone()

    def _ensure_feature_stat_buffers(self, reference: torch.Tensor) -> None:
        if not self._tr_feature_stats:
            zero = reference.detach().new_zeros(())
            self._tr_feature_stats = {
                key: zero.clone() for key in self.FEATURE_STAT_KEYS
            }

    @staticmethod
    def _coerce_scalar_output(
        value: Optional[torch.Tensor | float],
        zero: torch.Tensor,
    ) -> torch.Tensor:
        if value is None:
            return zero
        if not isinstance(value, torch.Tensor):
            return zero.new_tensor(float(value))
        return value

    def _prepare_context_parallel_inputs_compat(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
    ) -> tuple[Any, dict[str, torch.Tensor | Any]]:
        prepare_context_parallel_inputs = getattr(
            super(),
            "_prepare_context_parallel_inputs",
            None,
        )
        if prepare_context_parallel_inputs is None:
            return contextlib.nullcontext, inputs
        return prepare_context_parallel_inputs(model, inputs)

    def _get_gradient_accumulation_steps_compat(self) -> int:
        return getattr(
            self,
            "current_gradient_accumulation_steps",
            self.args.gradient_accumulation_steps,
        )

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        cp_context, inputs = self._prepare_context_parallel_inputs_compat(model, inputs)

        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(
                    model,
                    inputs,
                    return_outputs=True,
                    num_items_in_batch=num_items_in_batch,
                )

            ce_loss = self._get_output_value(outputs, "ce_loss", loss)
            hsic_loss_raw = self._get_output_value(outputs, "hsic_loss_raw")
            hsic_loss_weighted = self._get_output_value(outputs, "hsic_loss_weighted")
            raw_feature_stats = {
                key: self._get_output_value(outputs, key)
                for key in self.FEATURE_STAT_KEYS
            }

            if not isinstance(ce_loss, torch.Tensor):
                ce_loss = loss.detach().new_tensor(float(ce_loss))
            zero = loss.detach().new_zeros(())
            hsic_loss_raw = self._coerce_scalar_output(hsic_loss_raw, zero)
            hsic_loss_weighted = self._coerce_scalar_output(hsic_loss_weighted, zero)
            has_feature_stats = any(value is not None for value in raw_feature_stats.values())
            feature_stats = {}
            if has_feature_stats:
                feature_stats = {
                    key: self._coerce_scalar_output(value, zero)
                    for key, value in raw_feature_stats.items()
                }

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                clear_device_cache()

            kwargs = {}

            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()
                ce_loss = ce_loss.mean()
                hsic_loss_raw = hsic_loss_raw.mean()
                hsic_loss_weighted = hsic_loss_weighted.mean()
                if has_feature_stats:
                    feature_stats = {
                        key: value.mean() for key, value in feature_stats.items()
                    }

            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                grad_accumulation_steps = self._get_gradient_accumulation_steps_compat()
                loss = loss / grad_accumulation_steps
                ce_loss = ce_loss / grad_accumulation_steps
                hsic_loss_raw = hsic_loss_raw / grad_accumulation_steps
                hsic_loss_weighted = hsic_loss_weighted / grad_accumulation_steps
                if has_feature_stats:
                    feature_stats = {
                        key: value / grad_accumulation_steps
                        for key, value in feature_stats.items()
                    }

            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self._ensure_aux_loss_buffers(loss)
            self._tr_loss_hsic_raw += hsic_loss_raw.detach()
            self._tr_loss_hsic_weighted += hsic_loss_weighted.detach()
            if has_feature_stats:
                self._ensure_feature_stat_buffers(loss)
                for key, value in feature_stats.items():
                    self._tr_feature_stats[key] += value.detach()

            self.accelerator.backward(loss, **kwargs)

            return ce_loss.detach()

    def _maybe_log_save_evaluate(
        self,
        tr_loss: torch.Tensor,
        grad_norm: torch.Tensor | float | None,
        model: nn.Module,
        trial: "optuna.Trial | dict[str, Any] | None",
        epoch: float,
        ignore_keys_for_eval: list[str] | None,
        start_time: float,
        learning_rate: float | None = None,
    ) -> None:
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}
            step_delta = self.state.global_step - self._globalstep_last_logged

            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss

            ce_loss_avg = tr_loss_scalar / step_delta
            logs["loss"] = ce_loss_avg
            logs["Loss_ce"] = ce_loss_avg

            if self._tr_loss_hsic_raw is not None and self._tr_loss_hsic_weighted is not None:
                hsic_loss_raw_scalar = self._nested_gather(self._tr_loss_hsic_raw).mean().item()
                hsic_loss_weighted_scalar = self._nested_gather(self._tr_loss_hsic_weighted).mean().item()
                self._tr_loss_hsic_raw -= self._tr_loss_hsic_raw
                self._tr_loss_hsic_weighted -= self._tr_loss_hsic_weighted
                logs["Loss_hsic_raw"] = hsic_loss_raw_scalar / step_delta
                logs["Loss_hsic_weighted"] = hsic_loss_weighted_scalar / step_delta

            if self._tr_feature_stats:
                for key, value in self._tr_feature_stats.items():
                    stat_scalar = self._nested_gather(value).mean().item()
                    logs[key] = stat_scalar / step_delta
                    self._tr_feature_stats[key] -= self._tr_feature_stats[key]

            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


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
