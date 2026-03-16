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
from typing import Dict
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
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, AutoConfig, set_seed, enable_full_determinism

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


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


class FusionLambdaWarmupCallback(transformers.TrainerCallback):
    """
    Linearly warm up fusion lambdas from 0 to target values.

    Only applies when:
    - `fusion_lambda_warmup` is enabled
    - fusion method is `decompose_add` or `decompose_concat`
    """

    def __init__(self, enabled: bool, warmup_steps: int):
        self.enabled = enabled
        self.warmup_steps = max(int(warmup_steps), 1)
        self.active = False
        self.target_lambdas = {}

    def _apply_factor(self, model: torch.nn.Module, factor: float) -> None:
        factor = float(max(0.0, min(1.0, factor)))
        for key, target in self.target_lambdas.items():
            setattr(model, key, target * factor)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model = _unwrap_model(model)
        fusion_method = getattr(getattr(model, "config", None), "feature_fusion_method", None)
        if fusion_method not in {"decompose_add", "decompose_concat"}:
            if self.enabled:
                rank0_print(
                    f"[FusionLambdaWarmup] Skip: fusion_method={fusion_method} (only decompose_* supported)."
                )
            return
        if not self.enabled:
            return
        if not all(hasattr(model, key) for key in ("fusion_lambda_align", "fusion_lambda_ortho", "fusion_lambda_recon")):
            rank0_print("[FusionLambdaWarmup] Skip: fusion lambda attributes are not found on model.")
            return

        self.target_lambdas = {
            "fusion_lambda_align": float(model.fusion_lambda_align),
            "fusion_lambda_ortho": float(model.fusion_lambda_ortho),
            "fusion_lambda_recon": float(model.fusion_lambda_recon),
        }
        self.active = True
        init_factor = min(max(state.global_step, 0) / float(self.warmup_steps), 1.0)
        self._apply_factor(model, init_factor)
        rank0_print(
            "[FusionLambdaWarmup] Enabled for "
            f"{self.warmup_steps} steps, target={self.target_lambdas}, init_factor={init_factor:.4f}"
        )

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if not self.active or model is None:
            return
        model = _unwrap_model(model)
        factor = min(max(state.global_step, 0) / float(self.warmup_steps), 1.0)
        self._apply_factor(model, factor)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not self.active or model is None:
            return
        model = _unwrap_model(model)
        self._apply_factor(model, 1.0)


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
        if not model_args.use_geometry_encoder:
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
                "fusion_align_mode",
                "fusion_ortho_mode",
                "fusion_lambda_align",
                "fusion_lambda_ortho",
                "fusion_lambda_recon",
                "fusion_lambda_warmup",
                "fusion_lambda_warmup_steps",
                "tune_mm_vision",
                "tune_mm_vision_lora",
                "tune_geometry_encoder",
                "tune_geometry_encoder_lora",
            ]:
                setattr(config, k, getattr(model_args, k))

            assert model_args.geometry_encoder_path is not None, \
                "geometry_encoder_path must be set in the config when use_geometry_encoder is True."
            model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                geometry_encoder_path=model_args.geometry_encoder_path
            )

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
    set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    print(model.config)
    if model_args.use_geometry_encoder:
        setattr(data_args, "use_geometry_encoder", model_args.use_geometry_encoder)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    fusion_lambda_warmup_callback = FusionLambdaWarmupCallback(
        enabled=model_args.fusion_lambda_warmup,
        warmup_steps=model_args.fusion_lambda_warmup_steps,
    )
    trainer = VGTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[fusion_lambda_warmup_callback],
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
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
