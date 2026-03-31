from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT
from qwen_vl.data.utils import load_and_preprocess_images

@register_model("vgllm")
class VGLLM(lmms):

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        max_length: Optional[int] = None,
        add_frame_index: bool=False,
        use_geometry_encoder: Optional[bool] = None,
        geometry_encoder_type: Optional[str] = None,
        reference_frame: Optional[str] = None,
        tune_mm_vision: Optional[bool] = None,
        tune_mm_vision_lora: Optional[bool] = None,
        tune_geometry_encoder: Optional[bool] = None,
        tune_geometry_encoder_lora: Optional[bool] = None,
        use_learnable_prefix: Optional[bool] = None,
        learnable_prefix_len: Optional[int] = None,
        text_gate_bert_name_or_path: Optional[str] = None,
        feature_fusion_method: Optional[str] = None,
        fusion_num_layers: Optional[int] = None,
        geometry_merger_type: Optional[str] = None,
        decompose_hidden_size: Optional[int] = None,
        nrsr_hidden_size: Optional[int] = None,
        fusion_align_mode: Optional[str] = None,
        fusion_ortho_mode: Optional[str] = None,
        fusion_lambda_align: Optional[float] = None,
        fusion_lambda_ortho: Optional[float] = None,
        fusion_lambda_recon: Optional[float] = None,
        fusion_ortho_target_ratio: Optional[float] = None,
        fusion_lambda_nrsr: Optional[float] = None,
        fusion_lambda_nrsr_dynamic: Optional[bool] = None,
        fusion_lambda_nrsr_stage2_ratio: Optional[float] = None,
        fusion_lambda_nrsr_stage3_ratio: Optional[float] = None,
        fusion_lambda_warmup: Optional[bool] = None,
        fusion_lambda_warmup_steps: Optional[int] = None,
        fusion_mine_q_warmup_steps: Optional[int] = None,
        fusion_knn_k: Optional[int] = None,
        fusion_knn_min_valid_ratio: Optional[float] = None,
        fusion_knn_pos_mlp_hidden_size: Optional[int] = None,
        legacy_image_roundtrip: Optional[Union[bool, str]] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Keep strict kwargs check for safety.
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.add_frame_index = add_frame_index
        self.legacy_image_roundtrip = str(legacy_image_roundtrip).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
            "none",
        }
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        config = AutoConfig.from_pretrained(pretrained)
        if use_geometry_encoder is not None:
            setattr(config, "use_geometry_encoder", use_geometry_encoder)
        if geometry_encoder_type is not None:
            setattr(config, "geometry_encoder_type", geometry_encoder_type)
        if reference_frame is not None:
            setattr(config, "reference_frame", reference_frame)
        if feature_fusion_method is not None:
            setattr(config, "feature_fusion_method", feature_fusion_method)
        if fusion_num_layers is not None:
            setattr(config, "fusion_num_layers", fusion_num_layers)
        if geometry_merger_type is not None:
            setattr(config, "geometry_merger_type", geometry_merger_type)
        if decompose_hidden_size is not None:
            setattr(config, "decompose_hidden_size", decompose_hidden_size)
        if nrsr_hidden_size is not None:
            setattr(config, "nrsr_hidden_size", nrsr_hidden_size)
        if fusion_align_mode is not None:
            setattr(config, "fusion_align_mode", fusion_align_mode)
        if fusion_ortho_mode is not None:
            setattr(config, "fusion_ortho_mode", fusion_ortho_mode)
        if fusion_lambda_align is not None:
            setattr(config, "fusion_lambda_align", fusion_lambda_align)
        if fusion_lambda_ortho is not None:
            setattr(config, "fusion_lambda_ortho", fusion_lambda_ortho)
        if fusion_lambda_recon is not None:
            setattr(config, "fusion_lambda_recon", fusion_lambda_recon)
        if fusion_ortho_target_ratio is not None:
            setattr(config, "fusion_ortho_target_ratio", fusion_ortho_target_ratio)
        if fusion_lambda_nrsr is not None:
            setattr(config, "fusion_lambda_nrsr", fusion_lambda_nrsr)
        if fusion_lambda_nrsr_dynamic is not None:
            setattr(config, "fusion_lambda_nrsr_dynamic", fusion_lambda_nrsr_dynamic)
        if fusion_lambda_nrsr_stage2_ratio is not None:
            setattr(config, "fusion_lambda_nrsr_stage2_ratio", fusion_lambda_nrsr_stage2_ratio)
        if fusion_lambda_nrsr_stage3_ratio is not None:
            setattr(config, "fusion_lambda_nrsr_stage3_ratio", fusion_lambda_nrsr_stage3_ratio)
        if fusion_lambda_warmup is not None:
            setattr(config, "fusion_lambda_warmup", fusion_lambda_warmup)
        if fusion_lambda_warmup_steps is not None:
            setattr(config, "fusion_lambda_warmup_steps", fusion_lambda_warmup_steps)
        if fusion_mine_q_warmup_steps is not None:
            setattr(config, "fusion_mine_q_warmup_steps", fusion_mine_q_warmup_steps)
        if fusion_knn_k is not None:
            setattr(config, "fusion_knn_k", fusion_knn_k)
        if fusion_knn_min_valid_ratio is not None:
            setattr(config, "fusion_knn_min_valid_ratio", fusion_knn_min_valid_ratio)
        if fusion_knn_pos_mlp_hidden_size is not None:
            setattr(config, "fusion_knn_pos_mlp_hidden_size", fusion_knn_pos_mlp_hidden_size)
        if tune_mm_vision is not None:
            setattr(config, "tune_mm_vision", tune_mm_vision)
        if tune_mm_vision_lora is not None:
            setattr(config, "tune_mm_vision_lora", tune_mm_vision_lora)
        if tune_geometry_encoder is not None:
            setattr(config, "tune_geometry_encoder", tune_geometry_encoder)
        if tune_geometry_encoder_lora is not None:
            setattr(config, "tune_geometry_encoder_lora", tune_geometry_encoder_lora)
        if use_learnable_prefix is not None:
            setattr(config, "use_learnable_prefix", use_learnable_prefix)
        if learnable_prefix_len is not None:
            setattr(config, "learnable_prefix_len", learnable_prefix_len)
        if text_gate_bert_name_or_path is not None:
            setattr(config, "text_gate_bert_name_or_path", text_gate_bert_name_or_path)

        if (
            getattr(config, "use_geometry_encoder", False)
            or getattr(config, "use_vggt_feature", False)
            or getattr(config, "use_learnable_prefix", False)
            or getattr(config, "learnable_prefix_len", 0) > 0
        ):
            load_class = Qwen2_5_VLForConditionalGenerationWithVGGT
            eval_logger.info("Using Qwen2_5_VLForConditionalGenerationWithVGGT")
        else:
            load_class = Qwen2_5_VLForConditionalGeneration
            eval_logger.info("Using Qwen2_5_VLForConditionalGeneration")
        if use_flash_attention_2:
            self._model = load_class.from_pretrained(
                pretrained,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = load_class.from_pretrained(pretrained, config=config, torch_dtype="auto", device_map=self.device_map).eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels, padding_side="left")
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")
        self._text_gate_bert_tokenizer = None
        if (
            getattr(config, "use_geometry_encoder", False)
            and str(getattr(config, "feature_fusion_method", "add")).lower() in {"adver", "adver_ortho"}
            and getattr(config, "text_gate_bert_name_or_path", None)
        ):
            self._text_gate_bert_tokenizer = AutoTokenizer.from_pretrained(
                config.text_gate_bert_name_or_path,
                padding_side="right",
                use_fast=True,
            )

        if max_length is not None:
            eval_logger.warning(f"Setting max_length to {max_length}")
            setattr(self.processor.tokenizer, "model_max_length", max_length)
            setattr(self._tokenizer, "model_max_length", max_length)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
            self._model = self.model.to("cuda").to(torch.bfloat16)

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _maybe_roundtrip_image(self, image: Image.Image) -> Image.Image:
        if not self.legacy_image_roundtrip:
            return image

        source_cache_key = None
        if hasattr(image, "info"):
            source_cache_key = image.info.get("vgllm_cache_key")

        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
        buffer.seek(0)
        with Image.open(buffer) as reopened:
            roundtripped_image = reopened.convert("RGB")
        if source_cache_key:
            roundtripped_image.info["vgllm_cache_key"] = source_cache_key
        roundtripped_image.info["vgllm_cache_variant"] = "jpeg_roundtrip"
        return roundtripped_image

    def _prepare_image_tensor(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image_tensor = load_and_preprocess_images([image])[0]
        geometry_tensor = image_tensor

        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size
        _, height, width = image_tensor.shape
        if (width // patch_size) % merge_size > 0:
            width = width - (width // patch_size) % merge_size * patch_size
        if (height // patch_size) % merge_size > 0:
            height = height - (height // patch_size) % merge_size * patch_size
        processor_tensor = image_tensor[:, :height, :width]
        return processor_tensor, geometry_tensor

    def _prepare_message_inputs(self, context: str, visual) -> Tuple[List[Dict[str, object]], List[torch.Tensor], List[torch.Tensor]]:
        message = [{"role": "system", "content": "You are a helpful assistant."}]
        sample_image_inputs = []
        sample_geometry_inputs = []

        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
            vr = decord.VideoReader(visual)
            image_num = len(vr)
            if image_num < self.max_num_frames:
                frame_indices = np.arange(image_num)
            else:
                frame_indices = np.linspace(0, image_num - 1, self.max_num_frames).astype(int)
            visual_content = []
            for frame_idx in frame_indices:
                image = Image.fromarray(vr[frame_idx].asnumpy()).convert("RGB")
                visual_content.append({"type": "image", "image": image})
                processor_tensor, geometry_tensor = self._prepare_image_tensor(image)
                sample_image_inputs.append(processor_tensor)
                sample_geometry_inputs.append(geometry_tensor)
            message.append({"role": "user", "content": visual_content + [{"type": "text", "text": context}]})
            return message, sample_image_inputs, sample_geometry_inputs

        if isinstance(visual, Image.Image):
            visual = self._maybe_roundtrip_image(visual)
            processor_tensor, geometry_tensor = self._prepare_image_tensor(visual)
            sample_image_inputs.append(processor_tensor)
            sample_geometry_inputs.append(geometry_tensor)
            message.append({"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]})
            return message, sample_image_inputs, sample_geometry_inputs

        if isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
            image_content = []
            for image_count, image in enumerate(visual):
                image = self._maybe_roundtrip_image(image)
                if self.add_frame_index:
                    image_content.append({"type": "text", "text": "Frame-{}: ".format(image_count)})
                image_content.append({"type": "image", "image": image})
                processor_tensor, geometry_tensor = self._prepare_image_tensor(image)
                sample_image_inputs.append(processor_tensor)
                sample_geometry_inputs.append(geometry_tensor)
            message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
            return message, sample_image_inputs, sample_geometry_inputs

        message.append({"role": "user", "content": [{"type": "text", "text": context}]})
        return message, sample_image_inputs, sample_geometry_inputs

    def _prepare_generation_chunk(self, chunk) -> Dict[str, object]:
        contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
        task_name = task[0]
        split_name = split[0]
        visuals = [doc_to_visual[0](self.task_dict[task_name][split_name][ids]) for ids in doc_id]
        visuals = self.flatten(visuals)

        messages = []
        image_inputs = []
        geometry_encoder_inputs = []
        for visual_idx, context in enumerate(contexts):
            visual = visuals[visual_idx] if visual_idx < len(visuals) else None
            message, sample_image_inputs, sample_geometry_inputs = self._prepare_message_inputs(context, visual)
            messages.append(message)
            image_inputs.extend(sample_image_inputs)
            if sample_geometry_inputs:
                geometry_encoder_inputs.append(torch.stack(sample_geometry_inputs))

        return {
            "contexts": list(contexts),
            "messages": messages,
            "image_inputs": image_inputs,
            "geometry_encoder_inputs": geometry_encoder_inputs,
            "gen_kwargs": dict(all_gen_kwargs[0]),
        }

    def _prefetch_generation_chunks(self, chunks: Iterable) -> Iterator[Dict[str, object]]:
        chunk_iter = iter(chunks)
        try:
            current_chunk = next(chunk_iter)
        except StopIteration:
            return

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._prepare_generation_chunk, current_chunk)
            for next_chunk in chunk_iter:
                prepared_chunk = future.result()
                future = executor.submit(self._prepare_generation_chunk, next_chunk)
                yield prepared_chunk
            yield future.result()

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for prepared_chunk in self._prefetch_generation_chunks(chunks):
            contexts = prepared_chunk["contexts"]
            messages = prepared_chunk["messages"]
            image_inputs = prepared_chunk["image_inputs"]
            geometry_encoder_inputs = prepared_chunk["geometry_encoder_inputs"]
            gen_kwargs = prepared_chunk["gen_kwargs"]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=text,
                images=image_inputs if image_inputs else None,
                videos=None,
                padding=True,
                return_tensors="pt",
                do_rescale=False
            )
            if self._text_gate_bert_tokenizer is not None:
                bert_batch = self._text_gate_bert_tokenizer(
                    list(contexts),
                    padding=True,
                    truncation=True,
                    max_length=getattr(self.model.config, "text_gate_bert_max_length", 64),
                    return_tensors="pt",
                )
                inputs["bert_question_input_ids"] = bert_batch["input_ids"]
                inputs["bert_question_attention_mask"] = bert_batch["attention_mask"]
            device = "cuda" if self.device_map == "auto" else self.device
            if getattr(self.model.config, "use_geometry_encoder", False) or getattr(self.model.config, "use_vggt_feature", False):
                inputs["geometry_encoder_inputs"] = [feat.to(device) for feat in geometry_encoder_inputs]
            inputs = inputs.to(device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
