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
from qwen_vl.train.bbox_probe import BBoxFormatProbeCallback
from qwen_vl.train.bbox_residual_schedule import BBoxResidualWeightWarmupCallback
from trainer import VGTrainer, replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
)
from qwen_vl.data.data_qwen import make_supervised_data_module
from qwen_vl.bbox_special_tokens import (
    add_bbox_tokens,
    resize_model_embeddings_for_bbox_tokens,
)

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
        if getattr(model, "bbox_residual_head", None) is not None:
            for p in model.bbox_residual_head.parameters():
                p.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False
        if getattr(model, "bbox_residual_head", None) is not None:
            for p in model.bbox_residual_head.parameters():
                p.requires_grad = False

    if model_args.use_geometry_encoder:
        # vggt is frozen
        for n, p in model.geometry_encoder.named_parameters():
            p.requires_grad = False

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
            "use_bbox_special_tokens",
            "bbox_coordinate_label_smoothing",
            "bbox_coordinate_smoothing_neighbor_radius",
            "use_bbox_residual_head",
            "bbox_residual_loss_weight",
            "bbox_residual_loss_weight_warmup_ratio",
            "bbox_residual_loss_ratio_target",
            "bbox_residual_loss_ratio_start",
            "bbox_residual_loss_weight_max",
            "bbox_position_residual_head_range",
            "bbox_size_residual_head_range",
            "bbox_angle_residual_head_range",
        ]:
            setattr(config, k, getattr(model_args, k))

        model_load_kwargs = dict(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        if model_args.use_geometry_encoder:
            assert model_args.geometry_encoder_path is not None, \
                "geometry_encoder_path must be set in the config when use_geometry_encoder is True."
            model_load_kwargs["geometry_encoder_path"] = model_args.geometry_encoder_path

        model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
            **model_load_kwargs
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
        use_fast=model_args.tokenizer_use_fast,
    )
    if model_args.use_bbox_special_tokens:
        num_new_tokens = add_bbox_tokens(tokenizer)
        resize_model_embeddings_for_bbox_tokens(
            model,
            tokenizer,
            num_new_tokens,
            coordinate_label_smoothing=model_args.bbox_coordinate_label_smoothing,
            coordinate_smoothing_neighbor_radius=model_args.bbox_coordinate_smoothing_neighbor_radius,
        )
    set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    print(model.config)
    if model_args.use_geometry_encoder:
        setattr(data_args, "use_geometry_encoder", model_args.use_geometry_encoder)
    setattr(data_args, "use_bbox_special_tokens", model_args.use_bbox_special_tokens)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = VGTrainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )
    if model_args.use_bbox_residual_head and (
        model_args.bbox_residual_loss_weight > 0
        or 0.0 < model_args.bbox_residual_loss_ratio_target < 1.0
    ):
        trainer.add_callback(
            BBoxResidualWeightWarmupCallback(
                target_weight=model_args.bbox_residual_loss_weight,
                warmup_ratio=model_args.bbox_residual_loss_weight_warmup_ratio,
                target_loss_ratio=model_args.bbox_residual_loss_ratio_target,
                target_loss_ratio_start=model_args.bbox_residual_loss_ratio_start,
                max_dynamic_weight=model_args.bbox_residual_loss_weight_max,
            )
        )
    if training_args.bbox_probe_interval > 0:
        trainer.add_callback(
            BBoxFormatProbeCallback(
                tokenizer=tokenizer,
                image_processor=data_args.image_processor,
                output_dir=training_args.output_dir,
                probe_interval=training_args.bbox_probe_interval,
                probe_num_samples=training_args.bbox_probe_num_samples,
                probe_max_new_tokens=training_args.bbox_probe_max_new_tokens,
                use_bbox_special_tokens=model_args.use_bbox_special_tokens,
                use_geometry_encoder=model_args.use_geometry_encoder,
            )
        )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    shutil.copy2(source_path, template_path)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
