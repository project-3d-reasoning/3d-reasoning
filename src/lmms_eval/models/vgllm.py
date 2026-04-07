import os
from typing import List, Optional, Tuple, Union

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
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT
from qwen_vl.data.utils import load_and_preprocess_images, load_first_frame_coord_inputs

try:
    # from qwen_vl_utils import process_vision_info
    from qwen_vl_utils import extract_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


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
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.add_frame_index = add_frame_index
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

        if getattr(config, "use_geometry_encoder", False) or getattr(config, "use_vggt_feature", False):
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

        if max_length is not None:
            eval_logger.warning(f"Setting max_length to {max_length}")
            setattr(self.processor.tokenizer, "model_max_length", max_length)
            setattr(self._tokenizer, "model_max_length", max_length)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self._logged_coord_pe_eval = False

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

    def _resolve_doc_image_paths(self, task_name: str, doc) -> Optional[List[str]]:
        if "images" not in doc or not isinstance(doc["images"], (list, tuple)):
            return None

        if task_name in {"scanrefer", "scanrefer_first_frame"}:
            from lmms_eval.tasks.scanrefer import utils as task_utils
        elif task_name == "scan2cap":
            from lmms_eval.tasks.scan2cap import utils as task_utils
        elif task_name in {"scannet_4frames", "scannet_6frames"}:
            from lmms_eval.tasks.threedod import utils as task_utils
        else:
            return None

        return [os.path.join(task_utils.media_dir, image_file) for image_file in doc["images"]]

    def _load_eval_coord_pe(self, task_name: str, doc):
        image_paths = self._resolve_doc_image_paths(task_name, doc)
        if image_paths is None:
            return None, None
        return load_first_frame_coord_inputs(image_paths)

    def _build_empty_coord_pe(self, processed_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = processed_images.shape
        points = torch.zeros(
            processed_images.shape[0],
            height,
            width,
            3,
            dtype=processed_images.dtype,
        )
        masks = torch.zeros(
            processed_images.shape[0],
            height,
            width,
            dtype=torch.bool,
        )
        return points, masks

    def _prepare_coord_pe_batch(self, processed_images: torch.Tensor, coord_points, coord_masks) -> Tuple[torch.Tensor, torch.Tensor]:
        if coord_points is None or coord_masks is None:
            return self._build_empty_coord_pe(processed_images)

        if len(coord_points) != processed_images.shape[0] or len(coord_masks) != processed_images.shape[0]:
            eval_logger.warning("coord_pe image count does not match visual count during evaluation; falling back to empty coord_pe.")
            return self._build_empty_coord_pe(processed_images)

        target_h = processed_images.shape[-2]
        target_w = processed_images.shape[-1]
        for points, masks in zip(coord_points, coord_masks):
            if points.shape[:2] != (target_h, target_w) or masks.shape[:2] != (target_h, target_w):
                eval_logger.warning("coord_pe spatial shape does not match processed images during evaluation; falling back to empty coord_pe.")
                return self._build_empty_coord_pe(processed_images)

        return torch.stack(coord_points), torch.stack(coord_masks)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        device = "cuda" if self.device_map == "auto" else self.device
        use_geometry_encoder = getattr(self.model.config, "use_geometry_encoder", False) or getattr(self.model.config, "use_vggt_feature", False)
        use_coord_pe = getattr(self.model.config, "use_coord_pe", False)
        patch_multiple = self.processor.image_processor.patch_size * self.processor.image_processor.merge_size

        def _ensure_rgb(image: Image.Image) -> Image.Image:
            return image if image.mode == "RGB" else image.convert("RGB")

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
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            docs = [self.task_dict[task][split][ids] for ids in doc_id]
            visuals = [doc_to_visual[0](doc) for doc in docs]
            coord_pe_batches = [self._load_eval_coord_pe(task, doc) for doc in docs] if use_coord_pe else [(None, None)] * len(docs)
            visuals = self.flatten(visuals)

            gen_kwargs = dict(all_gen_kwargs[0])

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            messages = []
            for i, context in enumerate(contexts):

                message = [{"role": "system", "content": "You are a helpful assistant."}]

                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        vr = decord.VideoReader(visual)
                        image_num = len(vr)
                        # sample max_num_frames frame indices from the video
                        if image_num < self.max_num_frames:
                            frame_indices = np.arange(image_num)
                        else:
                            frame_indices = np.linspace(0, image_num - 1, self.max_num_frames).astype(int)
                        # read the frames
                        frames = [vr[i].asnumpy() for i in frame_indices]
                        visual_content = []
                        for frame in frames:
                            image = Image.fromarray(frame).convert("RGB")
                            visual_content.append({"type": "image", "image": image})
                        message.append({"role": "user", "content": visual_content + [{"type": "text", "text": context}]})

                    elif isinstance(visual, Image.Image):  # Single image
                        message.append({"role": "user", "content": [{"type": "image", "image": _ensure_rgb(visual)}, {"type": "text", "text": context}]})
                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                        image_content = []
                        image_count = 0
                        for v in visual:
                            if self.add_frame_index:
                                image_content.append({"type": "text", "text": "Frame-{}: ".format(image_count)})    
                            image_content.append({"type": "image", "image": _ensure_rgb(v)})
                            image_count += 1
                        message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                    else:
                        message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                else:
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})

                messages.append(message)

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # image_inputs, video_inputs = process_vision_info(messages)

            geometry_encoder_inputs = []
            coord_pe_points = []
            coord_pe_masks = []
            image_inputs = []
            for sample_idx, message in enumerate(messages):
                vision_info = extract_vision_info(message)
                sample_images = []
                for ele in vision_info:
                    if "image" in ele:
                        image = ele["image"]
                        if isinstance(image, Image.Image):
                            sample_images.append(image)
                        elif isinstance(image, str) and "base64," in image:
                            raise NotImplementedError("Base64-encoded images are no longer supported in VGLLM generate_until")
                        elif isinstance(image, str) and os.path.exists(image):
                            with Image.open(image) as pil_image:
                                sample_images.append(_ensure_rgb(pil_image))
                        else:
                            raise NotImplementedError("Unsupported image type")
                    elif "video" in ele:
                        raise NotImplementedError("Video inputs are not supported with geometry encoder preprocessing")
                    else:
                        raise NotImplementedError("Unsupported vision info type")

                if not sample_images:
                    continue

                processed_images = load_and_preprocess_images(sample_images)
                geometry_encoder_inputs.append(processed_images)
                if use_coord_pe:
                    sample_coord_points, sample_coord_masks = coord_pe_batches[sample_idx]
                    sample_coord_points, sample_coord_masks = self._prepare_coord_pe_batch(
                        processed_images,
                        sample_coord_points,
                        sample_coord_masks,
                    )
                    coord_pe_points.append(sample_coord_points)
                    coord_pe_masks.append(sample_coord_masks)
                for image in processed_images:
                    _, height, width = image.shape
                    if width % patch_multiple > 0:
                        width -= width % patch_multiple
                    if height % patch_multiple > 0:
                        height -= height % patch_multiple
                    image_inputs.append(image[:, :height, :width])
            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
                do_rescale=False
            )
            if use_geometry_encoder:
                inputs["geometry_encoder_inputs"] = [feat.to(device) for feat in geometry_encoder_inputs]
                if use_coord_pe:
                    inputs["coord_pe_points"] = [feat.to(device) for feat in coord_pe_points]
                    inputs["coord_pe_masks"] = [feat.to(device) for feat in coord_pe_masks]
                    if coord_pe_points and not self._logged_coord_pe_eval:
                        first_points = coord_pe_points[0]
                        first_masks = coord_pe_masks[0]
                        eval_logger.info(
                            "Passing coord_pe into model during evaluation: "
                            f"batch={len(coord_pe_points)}, "
                            f"first_points_shape={tuple(first_points.shape)}, "
                            f"first_masks_shape={tuple(first_masks.shape)}, "
                            f"first_valid_pixels={int(first_masks.sum().item())}"
                        )
                        self._logged_coord_pe_eval = True
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

            with torch.inference_mode():
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
