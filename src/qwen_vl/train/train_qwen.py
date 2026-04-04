# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict, Optional
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwen_vl.train.trainer
import qwen_vl.train.sampler
from trainer import replace_qwen2_vl_attention_class, VGTrainer

from transformers import (
    Qwen2VLForConditionalGeneration,
)
from qwen_vl.data.data_qwen import make_supervised_data_module

from qwen_vl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from qwen_vl.text_gate import (
    apply_text_gate_sentence_bert_config,
    resolve_text_gate_sentence_bert_max_length,
    resolve_text_gate_sentence_bert_name_or_path,
)
from qwen_vl.geometry_tokenization import (
    all_geometry_added_tokens,
    geometry_tokens_enabled,
    initialize_geometry_token_embeddings,
    register_geometry_token_gradient_mask,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, AutoConfig, set_seed, enable_full_determinism

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def _maybe_reapply_adver_gate_init(model: torch.nn.Module, loading_info: Dict) -> None:
    if model is None or not loading_info:
        return

    missing_keys = set(loading_info.get("missing_keys") or [])
    gate_output_missing = {
        "feature_fusion.adver_gate_mlp.3.weight",
        "feature_fusion.adver_gate_mlp.3.bias",
    } & missing_keys
    if not gate_output_missing:
        return

    feature_fusion = getattr(model, "feature_fusion", None)
    if feature_fusion is None or not hasattr(feature_fusion, "reset_adver_gate_bias"):
        return

    feature_fusion.reset_adver_gate_bias(-2.0)
    logging.info(
        "Reapplied adver gate output-layer init after loading missing keys: %s",
        sorted(gate_output_missing),
    )


def _infer_lora_target_modules(module: torch.nn.Module):
    """Infer a robust target module list for LoRA from Linear layer names."""
    preferred_linear_names = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "qkv", "proj", "fc1", "fc2",
        "gate_proj", "up_proj", "down_proj",
        "wq", "wk", "wv", "wo",
    }
    preferred_hits = set()
    all_linear_leaf_names = set()
    for name, submodule in module.named_modules():
        if isinstance(submodule, torch.nn.Linear):
            leaf_name = name.split(".")[-1]
            all_linear_leaf_names.add(leaf_name)
            if leaf_name in preferred_linear_names:
                preferred_hits.add(leaf_name)
    target_modules = sorted(preferred_hits if preferred_hits else all_linear_leaf_names)
    if not target_modules:
        raise ValueError("No Linear modules found for LoRA injection.")
    return target_modules


def _wrap_module_with_lora(module: torch.nn.Module, module_name: str):
    """Apply LoRA to a module and return the wrapped module."""
    if hasattr(module, "peft_config"):
        rank0_print(f"[LoRA] {module_name} already wrapped, skip.")
        return module

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError(
            "LoRA tuning is requested but `peft` is not installed. "
            "Please install it with `pip install peft`."
        ) from e

    target_modules = _infer_lora_target_modules(module)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        task_type="FEATURE_EXTRACTION",
    )
    wrapped_module = get_peft_model(module, lora_config)
    rank0_print(f"[LoRA] Applied to {module_name}, target_modules={target_modules}")
    if hasattr(wrapped_module, "print_trainable_parameters"):
        wrapped_module.print_trainable_parameters()
    return wrapped_module


def _get_geometry_backbone_attr(geometry_encoder: torch.nn.Module):
    if hasattr(geometry_encoder, "vggt"):
        return "vggt"
    if hasattr(geometry_encoder, "pi3"):
        return "pi3"
    return None


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        if model_args.tune_mm_vision_lora:
            model.visual = _wrap_module_with_lora(model.visual, "self.visual")
        else:
            for n, p in model.visual.named_parameters():
                p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

    if model_args.use_geometry_encoder:
        if model_args.tune_geometry_encoder:
            if model_args.tune_geometry_encoder_lora:
                backbone_attr = _get_geometry_backbone_attr(model.geometry_encoder)
                if backbone_attr is None:
                    raise ValueError(
                        "Cannot find geometry encoder backbone for LoRA. "
                        "Expected attribute `vggt` or `pi3`."
                    )
                backbone = getattr(model.geometry_encoder, backbone_attr)
                backbone = _wrap_module_with_lora(backbone, f"self.geometry_encoder.{backbone_attr}")
                setattr(model.geometry_encoder, backbone_attr, backbone)
            else:
                for n, p in model.geometry_encoder.named_parameters():
                    p.requires_grad = True
        else:
            for n, p in model.geometry_encoder.named_parameters():
                p.requires_grad = False


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP/DeepSpeed wrappers to get the base model."""
    while hasattr(model, "module"):
        model = model.module
    return model


def _maybe_enable_geometry_tokens(model: torch.nn.Module, tokenizer, data_args) -> int:
    if not geometry_tokens_enabled(data_args):
        return 0

    base_vocab_size = int(getattr(model.config, "geometry_token_base_vocab_size", len(tokenizer)))
    tokenizer_len_before_add = len(tokenizer)
    num_added_tokens = tokenizer.add_tokens(
        all_geometry_added_tokens(),
        special_tokens=False,
    )
    if num_added_tokens > 0:
        base_vocab_size = tokenizer_len_before_add
        model.resize_token_embeddings(len(tokenizer))
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        initialize_geometry_token_embeddings(
            model=model,
            tokenizer=tokenizer,
            base_vocab_size=base_vocab_size,
        )

    model.config.geometry_tokens = True
    model.config.geometry_token_base_vocab_size = int(base_vocab_size)
    model.config.vocab_size = len(tokenizer)
    if hasattr(model, "vocab_size"):
        model.vocab_size = len(tokenizer)

    if getattr(data_args, "geometry_token_freeze_existing_rows", True):
        register_geometry_token_gradient_mask(model, base_vocab_size=base_vocab_size)

    return int(num_added_tokens)


class FusionLambdaWarmupCallback(transformers.TrainerCallback):
    """
    Apply fusion aux scheduling to fusion lambdas.

    Behavior:
    - `adver`: first 2/3 enables align+recon, last 1/3 disables all aux losses.
    - `adver_ortho`: first 1/3 disables aux losses, middle 1/3 enables
      ortho only, last 1/3 disables all aux losses.
    - other supported fusion modes keep the original warmup-only behavior.
    """

    AUX_KEYS = ("fusion_lambda_align", "fusion_lambda_ortho", "fusion_lambda_recon")
    ADVER_DISABLE_START = 2.0 / 3.0
    ADVER_ORTHO_STAGE1_END = 1.0 / 3.0
    ADVER_ORTHO_STAGE2_END = 2.0 / 3.0

    def __init__(self, enabled: bool, warmup_steps: int):
        self.enabled = enabled
        self.warmup_steps = max(int(warmup_steps), 1)
        self.active = False
        self.lambda_bases = {}
        self.fusion_method = None

    def _training_progress(self, args, state) -> float:
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        if max_steps > 0:
            return float(max(0.0, min(1.0, state.global_step / float(max_steps))))

        epoch = getattr(state, "epoch", None)
        num_train_epochs = getattr(args, "num_train_epochs", None)
        if epoch is not None and num_train_epochs is not None and float(num_train_epochs) > 0:
            return float(max(0.0, min(1.0, float(epoch) / float(num_train_epochs))))
        return 0.0

    def _warmup_factor(self, state) -> float:
        if not self.enabled:
            return 1.0
        return float(min(max(state.global_step, 0) / float(self.warmup_steps), 1.0))

    def _adver_stage_factor(self, lambda_attr: str, progress: float) -> float:
        if progress < self.ADVER_DISABLE_START:
            return 1.0 if lambda_attr in {"fusion_lambda_align", "fusion_lambda_recon"} else 0.0
        return 0.0

    def _adver_ortho_stage_factor(self, lambda_attr: str, progress: float) -> float:
        if progress < self.ADVER_ORTHO_STAGE1_END:
            return 0.0
        if progress < self.ADVER_ORTHO_STAGE2_END:
            return 1.0 if lambda_attr == "fusion_lambda_ortho" else 0.0
        return 0.0

    def _schedule_factor(self, lambda_attr: str, progress: float, warmup_factor: float) -> float:
        stage_factor = 1.0
        if self.fusion_method == "adver":
            stage_factor = self._adver_stage_factor(lambda_attr, progress)
        elif self.fusion_method == "adver_ortho":
            stage_factor = self._adver_ortho_stage_factor(lambda_attr, progress)
        if stage_factor <= 0.0:
            return 0.0
        return float(stage_factor) * float(warmup_factor)

    def _apply_schedule(self, model: torch.nn.Module, args, state) -> None:
        progress = self._training_progress(args, state)
        warmup_factor = self._warmup_factor(state)
        lambda_bases = getattr(model, "_fusion_aux_lambda_bases", self.lambda_bases)
        for key, base_value in lambda_bases.items():
            factor = self._schedule_factor(key, progress, warmup_factor)
            setattr(model, key, float(base_value) * float(factor))

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model = _unwrap_model(model)
        fusion_method = getattr(getattr(model, "config", None), "feature_fusion_method", None)
        fusion_method = str(fusion_method).lower() if fusion_method is not None else None
        self.fusion_method = fusion_method
        if fusion_method not in {"adver", "adver_ortho", "decompose_add", "decompose_concat"}:
            if self.enabled:
                rank0_print(
                    f"[FusionLambdaWarmup] Skip: fusion_method={fusion_method} (only adver/adver_ortho/decompose_* supported)."
                )
            return
        self.lambda_bases = {}
        for key in self.AUX_KEYS:
            if hasattr(model, key):
                self.lambda_bases[key] = float(getattr(model, key))
        if not self.lambda_bases:
            rank0_print("[FusionLambdaWarmup] Skip: no fusion lambdas were found on model.")
            return

        setattr(model, "_fusion_aux_lambda_bases", dict(self.lambda_bases))
        self.active = True
        self._apply_schedule(model, args, state)
        if fusion_method == "adver":
            rank0_print(
                "[FusionLambdaWarmup] Enabled adver aux schedule, "
                f"disable_start={self.ADVER_DISABLE_START:.3f}, "
                f"warmup_steps={self.warmup_steps}, base_lambdas={self.lambda_bases}"
            )
        elif fusion_method == "adver_ortho":
            rank0_print(
                "[FusionLambdaWarmup] Enabled adver_ortho 3-stage aux schedule, "
                "stage1=off, stage2=ortho-only, stage3=off, "
                f"stage1_end={self.ADVER_ORTHO_STAGE1_END:.3f}, "
                f"stage2_end={self.ADVER_ORTHO_STAGE2_END:.3f}, "
                f"warmup_steps={self.warmup_steps}, base_lambdas={self.lambda_bases}"
            )
        else:
            rank0_print(
                "[FusionLambdaWarmup] Enabled with warmup-only schedule, "
                f"warmup_steps={self.warmup_steps}, base_lambdas={self.lambda_bases}"
            )

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if not self.active or model is None:
            return
        model = _unwrap_model(model)
        self._apply_schedule(model, args, state)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not self.active or model is None:
            return
        model = _unwrap_model(model)
        lambda_bases = getattr(model, "_fusion_aux_lambda_bases", self.lambda_bases)
        for key, base_value in lambda_bases.items():
            setattr(model, key, float(base_value))


class FusionLambdaNRSRCallback(transformers.TrainerCallback):
    """
    Three-stage schedule for fusion_lambda_nrsr:
    - Stage 1 [0, stage1_end): lambda = 0
    - Stage 2/3 lambda baseline is controlled by trainer-side ratio constraints
      when fusion_lambda_nrsr_dynamic is enabled.
    """

    STAGE1_END = 0.1
    STAGE2_END = 0.5

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self.active = False
        self.target_lambda: float = 0.0
        self.current_lambda: float = 0.0

    def _training_progress(self, state) -> float:
        if state.max_steps is not None and state.max_steps > 0:
            return float(max(0.0, min(1.0, state.global_step / float(state.max_steps))))
        if state.epoch is not None and state.num_train_epochs is not None and state.num_train_epochs > 0:
            return float(max(0.0, min(1.0, float(state.epoch) / float(state.num_train_epochs))))
        return 0.0

    def _scale_from_progress(self, progress: float) -> float:
        if progress < self.STAGE1_END:
            return 0.0
        if progress < self.STAGE2_END:
            ramp = max(self.STAGE2_END - self.STAGE1_END, 1e-12)
            return float((progress - self.STAGE1_END) / ramp)
        return 1.0

    def _apply(self, model: torch.nn.Module, state) -> None:
        progress = self._training_progress(state)
        scale = self._scale_from_progress(progress)
        self.current_lambda = self.target_lambda * scale
        setattr(model, "fusion_lambda_nrsr", self.current_lambda)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model = _unwrap_model(model)
        fusion_method = getattr(getattr(model, "config", None), "feature_fusion_method", None)
        fusion_method = str(fusion_method).lower() if fusion_method is not None else None
        if fusion_method not in {"nrsr_add", "nrsr_concat"}:
            if self.enabled:
                rank0_print(
                    f"[FusionLambdaNRSR] Skip: fusion_method={fusion_method} (only nrsr_* supported)."
                )
            return
        if not self.enabled:
            return
        if not hasattr(model, "fusion_lambda_nrsr"):
            rank0_print("[FusionLambdaNRSR] Skip: fusion_lambda_nrsr is not found on model.")
            return

        self.target_lambda = float(model.fusion_lambda_nrsr)
        self.active = True
        self._apply(model, state)
        rank0_print(
            "[FusionLambdaNRSR] Enabled with "
            f"target={self.target_lambda}, stage1_end={self.STAGE1_END}, "
            f"stage2_end={self.STAGE2_END}, init_lambda={self.current_lambda:.6f}"
        )

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if not self.active or model is None:
            return
        model = _unwrap_model(model)
        self._apply(model, state)

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if not self.active or logs is None:
            return
        # Keep a separate key to avoid overriding trainer-computed dynamic lambda logs.
        logs.setdefault("fusion_lambda_nrsr_sched", float(self.current_lambda))

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not self.active or model is None:
            return
        model = _unwrap_model(model)
        setattr(model, "fusion_lambda_nrsr", self.target_lambda)


class AdverFreezeLlmCallback(transformers.TrainerCallback):
    """
    In adver / adver_ortho with tune_mm_llm, freeze the language decoder and lm_head
    for the first N global steps so fusion / gate branches can stabilize first.

    The LLM parameters are already part of the optimizer because `set_model`
    marks them trainable before the Trainer builds optimizer/scheduler. This
    callback therefore only toggles `requires_grad` during the frozen window.

    On unfreeze, gate param groups have their scheduler base_lr scaled by
    ``gate_lr_decay_on_unfreeze`` so that the gate LR transitions smoothly
    from the pre-training value to one compatible with a jointly-trained LLM.
    """

    def __init__(
        self,
        freeze_steps: int,
        tune_mm_llm: bool,
        fusion_method: Optional[str],
        gate_lr_decay_on_unfreeze: float = 0.1,
        fusion_gate_lr: Optional[float] = None,
    ):
        self.freeze_steps = max(int(freeze_steps), 0)
        self.tune_mm_llm = bool(tune_mm_llm)
        self.fusion_method = str(fusion_method).lower() if fusion_method else ""
        self.gate_lr_decay = float(gate_lr_decay_on_unfreeze)
        self.fusion_gate_lr = float(fusion_gate_lr) if fusion_gate_lr is not None else None
        self.trainer: Optional[transformers.Trainer] = None
        self._active = False
        self._prev_should_freeze: Optional[bool] = None

    def _enabled(self) -> bool:
        if self.freeze_steps <= 0:
            return False
        if not self.tune_mm_llm:
            return False
        return self.fusion_method in {"adver", "adver_ortho"}

    @staticmethod
    def _set_llm_trainable(model: torch.nn.Module, trainable: bool) -> None:
        for _n, p in model.model.named_parameters():
            p.requires_grad = trainable
        model.lm_head.requires_grad = trainable

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None or not self._enabled():
            return
        model = _unwrap_model(model)
        self._active = True
        should_freeze = state.global_step < self.freeze_steps
        self._set_llm_trainable(model, not should_freeze)
        self._prev_should_freeze = should_freeze
        if should_freeze:
            rank0_print(
                f"[AdverFreezeLlm] Freezing LLM (model + lm_head) for the first {self.freeze_steps} global steps."
            )

    def _decay_gate_lr_via_scheduler(self) -> None:
        if self.trainer is None:
            return
        scheduler = getattr(self.trainer, "lr_scheduler", None)
        if scheduler is None or not hasattr(scheduler, "base_lrs"):
            return
        if self.fusion_gate_lr is None or self.fusion_gate_lr <= 0:
            return
        matched = False
        tol = self.fusion_gate_lr * 0.01
        for i, base_lr in enumerate(scheduler.base_lrs):
            if abs(base_lr - self.fusion_gate_lr) < tol:
                old_lr = base_lr
                scheduler.base_lrs[i] = base_lr * self.gate_lr_decay
                matched = True
                rank0_print(
                    f"[AdverFreezeLlm] Decayed gate scheduler base_lr {old_lr:.2e} -> "
                    f"{scheduler.base_lrs[i]:.2e} (x{self.gate_lr_decay})"
                )
        if not matched:
            rank0_print(
                f"[AdverFreezeLlm] No scheduler base_lr matched fusion_gate_lr={self.fusion_gate_lr:.2e}; "
                "gate LR unchanged."
            )

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if not self._active or model is None:
            return
        model = _unwrap_model(model)
        should_freeze = state.global_step < self.freeze_steps
        if self._prev_should_freeze is not None and should_freeze == self._prev_should_freeze:
            return
        self._set_llm_trainable(model, not should_freeze)
        if self._prev_should_freeze and not should_freeze:
            rank0_print(
                f"[AdverFreezeLlm] Unfroze LLM at global_step={state.global_step}."
            )
            self._decay_gate_lr_via_scheduler()
        self._prev_should_freeze = should_freeze


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    # enable_full_determinism(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen2.5" in model_args.model_name_or_path.lower():
        use_custom_model = model_args.use_geometry_encoder or model_args.use_learnable_prefix
        if not use_custom_model:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        else:
            from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            if hasattr(config, "use_geometry_encoder") and config.use_geometry_encoder != model_args.use_geometry_encoder:
                raise ValueError(
                    "The use_geometry_encoder in config and model_args are not consistent. "
                    "Please check the model config."
                )

            for k in [
                "use_geometry_encoder", 
                "geometry_encoder_type", 
                "reference_frame",
                "feature_fusion_method", 
                "fusion_num_layers",
                "geometry_merger_type",
                "decompose_hidden_size",
                "nrsr_hidden_size",
                "fusion_recon_mask_ratio",
                "fusion_gate_temperature",
                "fusion_gate_question_dim",
                "adver_compute_align_loss",
                "fusion_align_mode",
                "fusion_ortho_mode",
                "fusion_lambda_align",
                "fusion_lambda_ortho",
                "fusion_lambda_recon",
                "fusion_ortho_target_ratio",
                "fusion_lambda_nrsr",
                "fusion_lambda_nrsr_dynamic",
                "fusion_lambda_nrsr_stage2_ratio",
                "fusion_lambda_nrsr_stage3_ratio",
                "fusion_lambda_warmup",
                "fusion_lambda_warmup_steps",
                "fusion_mine_q_warmup_steps",
                "fusion_knn_k",
                "fusion_knn_min_valid_ratio",
                "fusion_knn_pos_mlp_hidden_size",
                "tune_mm_vision",
                "tune_mm_vision_lora",
                "tune_geometry_encoder",
                "tune_geometry_encoder_lora",
                "use_learnable_prefix",
                "learnable_prefix_len",
                "label_weight_default",
                "label_weight_scanrefer_frame",
                "label_weight_scanrefer_bbox",
                "label_weight_scan2cap_category",
                "label_weight_scan2cap_attribute",
                "label_weight_scan2cap_relation",
                "label_weight_scannet_det_label",
                "label_weight_scannet_det_bbox",
                "label_weight_dynamic_iou_alpha",
                "label_weight_dynamic_iou_eps",
                "label_weight_dynamic_iou_skip_invalid",
                "label_weight_loss_normalize_by_weight_sum",
            ]:
                setattr(config, k, getattr(model_args, k))
            apply_text_gate_sentence_bert_config(config, cache_dir=training_args.cache_dir)

            if model_args.use_geometry_encoder:
                assert model_args.geometry_encoder_path is not None, \
                    "geometry_encoder_path must be set in the config when use_geometry_encoder is True."
            model, loading_info = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                geometry_encoder_path=(model_args.geometry_encoder_path if model_args.use_geometry_encoder else None),
                output_loading_info=True,
            )
            _maybe_reapply_adver_gate_init(model, loading_info)

        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    text_gate_sentence_bert_name_or_path = resolve_text_gate_sentence_bert_name_or_path(model_args)
    text_gate_sentence_bert_max_length = resolve_text_gate_sentence_bert_max_length(model_args)
    sentence_bert_tokenizer = None
    if (
        model_args.use_geometry_encoder
        and str(model_args.feature_fusion_method).lower() in {"adver", "adver_ortho"}
        and text_gate_sentence_bert_name_or_path
    ):
        sentence_bert_tokenizer = transformers.AutoTokenizer.from_pretrained(
            text_gate_sentence_bert_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=text_gate_sentence_bert_max_length,
            padding_side="right",
            use_fast=True,
        )
    for attr in (
        "fusion_align_target_ratio",
        "fusion_align_lambda_min",
        "fusion_align_lambda_max",
        "fusion_ortho_target_ratio",
        "fusion_ortho_lambda_min",
        "fusion_ortho_lambda_max",
        "fusion_recon_target_ratio",
        "fusion_recon_lambda_min",
        "fusion_recon_lambda_max",
    ):
        value = getattr(model_args, attr, None)
        if value is not None:
            setattr(training_args, attr, float(value))
    set_model(model_args, model)
    geometry_tokens_added = _maybe_enable_geometry_tokens(model, tokenizer, data_args)
    if geometry_tokens_added > 0:
        logging.info(
            "Added %d geometry tokens. New tokenizer size=%d",
            geometry_tokens_added,
            len(tokenizer),
        )

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    print(model.config)
    if model_args.use_geometry_encoder:
        setattr(data_args, "use_geometry_encoder", model_args.use_geometry_encoder)
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        sentence_bert_tokenizer=sentence_bert_tokenizer,
        sentence_bert_max_length=text_gate_sentence_bert_max_length,
    )
    fusion_lambda_warmup_callback = FusionLambdaWarmupCallback(
        enabled=model_args.fusion_lambda_warmup,
        warmup_steps=model_args.fusion_lambda_warmup_steps,
    )
    fusion_lambda_nrsr_callback = FusionLambdaNRSRCallback(
        enabled=model_args.fusion_lambda_nrsr_dynamic,
    )
    adver_freeze_llm_callback = AdverFreezeLlmCallback(
        freeze_steps=model_args.adver_freeze_llm_steps,
        tune_mm_llm=model_args.tune_mm_llm,
        fusion_method=model_args.feature_fusion_method,
        gate_lr_decay_on_unfreeze=getattr(model_args, "gate_lr_decay_on_unfreeze", 0.1),
        fusion_gate_lr=getattr(training_args, "fusion_gate_lr", None),
    )
    trainer = VGTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[
            fusion_lambda_warmup_callback,
            fusion_lambda_nrsr_callback,
            adver_freeze_llm_callback,
        ],
        **data_module,
    )
    adver_freeze_llm_callback.trainer = trainer
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    tokenizer.save_pretrained(training_args.output_dir)
    data_args.image_processor.save_pretrained(training_args.output_dir)

    source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    shutil.copy2(source_path, template_path)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    try:
        train(attn_implementation="flash_attention_2")
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
