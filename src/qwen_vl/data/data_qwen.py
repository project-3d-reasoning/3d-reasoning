import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
try:
    from decord import VideoReader
except ImportError:
    VideoReader = None
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2
from .utils import prepare_image_inputs
from qwen_vl.label_weighting import (
    LabelWeightMaskStore,
    build_full_label_weight_codes,
    get_label_weight_mask_path,
)
from qwen_vl.geometry_tokenization import (
    build_geometry_assistant_codes,
    geometry_tokens_enabled,
    transform_conversations_for_geometry_tokens,
)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def _require_decord() -> None:
    if VideoReader is None:
        raise ImportError(
            "decord is required only when loading video samples, but it is not installed. "
            "Install `decord` or make sure the current training batch contains image-only data."
        )


def read_jsonl(path, max_samples: int=-1):
    with open(path, "r") as f:
        # return [json.loads(line) for line in f]
        ret = []
        for line in f:
            ret.append(json.loads(line))
            if max_samples !=-1 and len(ret) >= max_samples:
                break
    return ret


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List = [],
    visual_type: str = "image",
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>"
                            * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def extract_question_text_from_conversations(conversations) -> str:
    roles = {"human": "user", "gpt": "assistant"}
    question_segments = []
    for conv in conversations:
        role = conv.get("role", conv.get("from"))
        content = conv.get("content", conv.get("value", ""))
        role = roles.get(role, role)
        if role != "user":
            continue

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    text_parts.append(item)
            content = " ".join(text_parts)
        else:
            content = str(content)

        content = (
            content.replace(DEFAULT_IMAGE_TOKEN, " ")
            .replace(DEFAULT_VIDEO_TOKEN, " ")
            .replace("<image>", " ")
            .replace("<video>", " ")
        )
        content = " ".join(content.split())
        if content:
            question_segments.append(content)

    return " ".join(question_segments)


def extract_assistant_text_from_conversations(conversations) -> str:
    roles = {"human": "user", "gpt": "assistant"}
    for conv in reversed(conversations):
        role = conv.get("role", conv.get("from"))
        role = roles.get(role, role)
        if role != "assistant":
            continue
        return str(conv.get("content", conv.get("value", "")))
    return ""


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"], max_samples=data_args.max_samples)
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            indexed_annotations = list(enumerate(annotations))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                indexed_annotations = random.sample(
                    indexed_annotations, int(len(indexed_annotations) * sampling_rate)
                )
                print(f"sampling {len(indexed_annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for source_ann_idx, ann in indexed_annotations:
                ann["data_path"] = data["data_path"]
                ann["tag"] = data["tag"]
                ann["_source_dataset_name"] = data["dataset_name"]
                ann["_source_sample_idx"] = source_ann_idx
            list_data_dict += [ann for _, ann in indexed_annotations]

        print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.label_weight_mask_enabled = bool(getattr(data_args, "label_weight_masks_dir", None))
        self.label_weight_mask_stores = {}
        if self.label_weight_mask_enabled:
            masks_dir = getattr(data_args, "label_weight_masks_dir", None)
            for data in dataset_list:
                try:
                    mask_path = get_label_weight_mask_path(data["annotation_path"], masks_dir)
                    if os.path.exists(mask_path):
                        self.label_weight_mask_stores[data["dataset_name"]] = LabelWeightMaskStore(mask_path)
                    else:
                        rank0_print(
                            f"[LabelWeightMask] Missing sidecar for dataset={data['dataset_name']}: {mask_path}"
                        )
                except Exception as exc:
                    rank0_print(
                        f"[LabelWeightMask] Failed to load sidecar for dataset={data['dataset_name']}: {exc}"
                    )
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            if "image" in sample:
                image_num = len(sample["image"])
            elif "images" in sample:
                image_num = len(sample["images"])
            elif "video" in sample:
                image_num = getattr(self.data_args, "video_max_frames", 8)
            else:
                image_num = 0
            length_list.append(image_num * 252 + cur_len)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            if "image" in sample:
                image_num = len(sample["image"])
            elif "images" in sample:
                image_num = len(sample["images"])
            elif "video" in sample:
                image_num = getattr(self.data_args, "video_max_frames", 8)
            else:
                image_num = 0
            cur_len += image_num*252
            tag = sample.get("tag", "2d")
            cur_len = -cur_len if tag == "2d" else cur_len
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw
    
    def draw_visual_marks(self, images, spar_info):

        if spar_info is None:
            return
        info = json.loads(spar_info)
        task_type = info["type"]
        from .draw_marker import DRAW_FUNCTIONS
        draw_fn = DRAW_FUNCTIONS[task_type]
        if len(images) == 1:
            draw_fn(images[0], info)
        else:
            draw_fn(images, info)
        # for j, img in enumerate(images):
        #     # write to local
        #     img.save(f"images/img_{j}.jpg", format="JPEG")

    def process_video(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        _require_decord()
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts
    

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e
    
    def read_video_images(self, source):
        # read video images from the source
        assert isinstance(source["video"], str), "video should be a string"
        video_file = os.path.join(source["data_path"], source["video"])
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
            raise FileNotFoundError
        
        def get_frame_indices(total_frames, fps=1):
            video_length = total_frames / fps
            interval = getattr(self.data_args, "base_interval", 2)
            num_frames_to_sample = round(video_length / interval)
            video_min_frames = getattr(self.data_args, "video_min_frames", 4)
            video_max_frames = getattr(self.data_args, "video_max_frames", 8)
            target_frames = min(
                max(num_frames_to_sample, video_min_frames), video_max_frames
            )
            frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
            frame_idx = np.unique(frame_idx)
            return frame_idx        

        # check whether video_file is a directory
        if os.path.isdir(video_file):
            frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
            frame_files.sort()
            frame_idx = get_frame_indices(len(frame_files), 1)
            images = [frame_files[i] for i in frame_idx]
            images = [Image.open(frame).convert("RGB") for frame in images]
        elif any([video_file.endswith(ext) for ext in [".mp4", ".avi", ".mov"]]):
            _require_decord()
            vr = VideoReader(video_file, num_threads=4)
            total_frames = len(vr)
            avg_fps = vr.get_avg_fps()
            frame_idx = get_frame_indices(total_frames, avg_fps)
            video = vr.get_batch(frame_idx).asnumpy()
            
            images = [Image.fromarray(frame).convert("RGB") for frame in video]
        return images

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        original_source_item = self.list_data_dict[i]
        source_item = copy.deepcopy(original_source_item)
        original_conversations = original_source_item["conversations"]
        if isinstance(i, int):
            sources = [source_item]
        else:
            sources = source_item
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        video = None
        
        if "video" in sources[0]:
            sources[0]["images"] = self.read_video_images(sources[0])
            num_image = len(sources[0]["images"])
            sources[0]["conversations"][0]["value"] = sources[0]["conversations"][0]["value"].replace(
                DEFAULT_VIDEO_TOKEN, "".join([DEFAULT_IMAGE_TOKEN] * num_image)
            )
            del sources[0]["video"]
        
        # # replace <image>\n with <image>
        sources[0]["conversations"][0]["value"] = sources[0]["conversations"][0]["value"].replace(
            f"{DEFAULT_IMAGE_TOKEN}\n", DEFAULT_IMAGE_TOKEN
        )

        source_dataset_name = sources[0].get("_source_dataset_name")
        if geometry_tokens_enabled(self.data_args):
            sources[0]["conversations"] = transform_conversations_for_geometry_tokens(
                sources[0]["conversations"],
                dataset_name=source_dataset_name,
            )
        transformed_conversations = sources[0]["conversations"]

        has_image_inputs = False

        # rename images tag
        if "images" in sources[0]:
            sources[0]["image"] = sources[0]["images"]

        # notice that we use images as the tag
        if "image" in sources[0]:
            has_image_inputs = True
            image_folder = original_source_item["data_path"]
            image_file = sources[0]["image"]
            if isinstance(image_file, List):

                if isinstance(image_file[0], str):
                    image_file = [
                        os.path.join(image_folder, file) for file in image_file
                    ]
                    image_file = [Image.open(img).convert("RGB") for img in image_file]
                elif isinstance(image_file[0], Image.Image):
                    pass
                else:
                    raise NotImplementedError
                # draw visual markers
                self.draw_visual_marks(image_file, sources[0].get("spar_info", None))

                image, grid_thw, geometry_encoder_inputs = [], [], []
                for file in image_file:
                    ret = prepare_image_inputs(file, self.data_args.image_processor)
                    image.append(ret["pixel_values"])
                    geometry_encoder_inputs.append(ret["geometry_encoder_inputs"])
                    grid_thw.append(ret["image_grid_thw"])
            else:
                raise NotImplementedError

            grid_thw_merged = copy.deepcopy(grid_thw)
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="image"
            )
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                torch.stack(grid_thw, dim=0),
            )
        elif "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [
                        os.path.join(video_folder, file) for file in video_file
                    ]
                    results = [self.process_video(file) for file in video_file]
                    video, grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                video = [video]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="video"
            )
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                video_grid_thw=torch.stack(grid_thw, dim=0),
                second_per_grid_ts=second_per_grid_ts,
            )
        else:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                position_ids=position_ids,
                question_text=extract_question_text_from_conversations(original_conversations),
            )

        if has_image_inputs:
            data_dict["pixel_values"] = image
            data_dict["image_grid_thw"] = grid_thw
            if getattr(self.data_args, "use_geometry_encoder", False):
                data_dict["geometry_encoder_inputs"] = geometry_encoder_inputs
        # video exist in the data
        elif "video" in original_source_item:
            data_dict["pixel_values_videos"] = video
            data_dict["video_grid_thw"] = grid_thw
        
        data_dict["tag"] = original_source_item.get("tag", "2d")
        if self.label_weight_mask_enabled:
            source_sample_idx = int(original_source_item.get("_source_sample_idx", -1))
            assistant_codes = None
            use_runtime_geometry_codes = bool(
                geometry_tokens_enabled(self.data_args)
                and source_dataset_name in {"scanrefer", "scannet_det"}
            )
            if (
                not use_runtime_geometry_codes
                and source_dataset_name in self.label_weight_mask_stores
                and source_sample_idx >= 0
            ):
                assistant_codes = self.label_weight_mask_stores[source_dataset_name].get_codes(source_sample_idx)
            elif use_runtime_geometry_codes:
                assistant_text = extract_assistant_text_from_conversations(transformed_conversations)
                assistant_codes = build_geometry_assistant_codes(
                    self.tokenizer,
                    assistant_text=assistant_text,
                    dataset_name=source_dataset_name,
                )
            data_dict["label_weight_codes"] = build_full_label_weight_codes(
                data_dict["labels"],
                assistant_codes=assistant_codes,
                ignore_index=IGNORE_INDEX,
            )
        if source_dataset_name == "scanrefer" and "target" in original_source_item:
            data_dict["scanrefer_target"] = copy.deepcopy(original_source_item["target"])
        if source_dataset_name == "scannet_det" and "boxes" in original_source_item:
            data_dict["scannet_det_boxes"] = copy.deepcopy(original_source_item["boxes"])
        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    sentence_bert_tokenizer: Optional[transformers.PreTrainedTokenizer] = None
    sentence_bert_max_length: int = 64
    bert_tokenizer: Optional[transformers.PreTrainedTokenizer] = None
    bert_max_length: Optional[int] = None

    def _get_sentence_bert_tokenizer(self) -> Optional[transformers.PreTrainedTokenizer]:
        if self.sentence_bert_tokenizer is not None:
            return self.sentence_bert_tokenizer
        return self.bert_tokenizer

    def _get_sentence_bert_max_length(self) -> int:
        if self.bert_max_length is not None:
            return int(self.bert_max_length)
        return int(self.sentence_bert_max_length)

    def _maybe_add_sentence_bert_question_inputs(
        self,
        batch: Dict[str, torch.Tensor],
        instances: Sequence[Dict],
    ) -> Dict[str, torch.Tensor]:
        sentence_bert_tokenizer = self._get_sentence_bert_tokenizer()
        if sentence_bert_tokenizer is None:
            return batch

        question_texts = [instance.get("question_text", "") for instance in instances]
        sentence_bert_batch = sentence_bert_tokenizer(
            question_texts,
            padding=True,
            truncation=True,
            max_length=self._get_sentence_bert_max_length(),
            return_tensors="pt",
        )
        batch["sentence_bert_question_input_ids"] = sentence_bert_batch["input_ids"]
        batch["sentence_bert_question_attention_mask"] = sentence_bert_batch["attention_mask"]
        return batch

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if any("label_weight_codes" in instance for instance in instances):
            label_weight_codes = [
                instance.get(
                    "label_weight_codes",
                    torch.zeros_like(instance["labels"], dtype=torch.uint8),
                )
                for instance in instances
            ]
            label_weight_codes = torch.nn.utils.rnn.pad_sequence(
                label_weight_codes,
                batch_first=True,
                padding_value=0,
            )
            label_weight_codes = label_weight_codes[:, : self.tokenizer.model_max_length]
            batch["label_weight_codes"] = label_weight_codes
        if any("scanrefer_target" in instance for instance in instances):
            batch["scanrefer_targets"] = [
                instance.get("scanrefer_target")
                for instance in instances
            ]
        if any("scannet_det_boxes" in instance for instance in instances):
            batch["scannet_det_boxes"] = [
                instance.get("scannet_det_boxes")
                for instance in instances
            ]
        images = list(
            itertools.chain(
                *(
                    instance["pixel_values"]
                    for instance in instances
                    if "pixel_values" in instance
                )
            )
        )
        videos = list(
            itertools.chain(
                *(
                    instance["pixel_values_videos"]
                    for instance in instances
                    if "pixel_values_videos" in instance
                )
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(
                    *(
                        instance["video_grid_thw"]
                        for instance in instances
                        if "video_grid_thw" in instance
                    )
                )
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
                
        # assume all data in a batch has geometry_encoder_inputs
        if "geometry_encoder_inputs" in instances[0]:
            geometry_encoder_inputs = [torch.stack(instance["geometry_encoder_inputs"]) for instance in instances]
            batch["geometry_encoder_inputs"] = geometry_encoder_inputs
            assert len(set([instance["tag"] for instance in instances])) == 1, "all data in a batch should have the same tag"
            batch["tag"] = instances[0]["tag"]
        return self._maybe_add_sentence_bert_question_inputs(batch, instances)


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )

        seq_lens = torch.tensor(
            [0] + [len(seq) for seq in input_ids], dtype=torch.int32
        )
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids.unsqueeze(0),
            labels=labels.unsqueeze(0),
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        if any("label_weight_codes" in instance for instance in instances):
            label_weight_codes = [
                instance.get(
                    "label_weight_codes",
                    torch.zeros_like(instance["labels"], dtype=torch.uint8),
                )
                for instance in instances
            ]
            batch["label_weight_codes"] = torch.cat(label_weight_codes, dim=0).unsqueeze(0)
        if any("scanrefer_target" in instance for instance in instances):
            batch["scanrefer_targets"] = [
                instance.get("scanrefer_target")
                for instance in instances
            ]
        if any("scannet_det_boxes" in instance for instance in instances):
            batch["scannet_det_boxes"] = [
                instance.get("scannet_det_boxes")
                for instance in instances
            ]
        images = list(
            itertools.chain(
                *(
                    instance["pixel_values"]
                    for instance in instances
                    if "pixel_values" in instance
                )
            )
        )
        videos = list(
            itertools.chain(
                *(
                    instance["pixel_values_videos"]
                    for instance in instances
                    if "pixel_values_videos" in instance
                )
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(
                    *(
                        instance["video_grid_thw"]
                        for instance in instances
                        if "video_grid_thw" in instance
                    )
                )
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

                
        # assume all data in a batch has geometry_encoder_inputs
        if "geometry_encoder_inputs" in instances[0]:
            raise NotImplementedError("FlattenedDataCollatorForSupervisedDataset does not support geometry_encoder_inputs")

        return self._maybe_add_sentence_bert_question_inputs(batch, instances)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    sentence_bert_tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
    sentence_bert_max_length: int = 64,
    bert_tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
    bert_max_length: Optional[int] = None,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if sentence_bert_tokenizer is None:
        sentence_bert_tokenizer = bert_tokenizer
    if bert_max_length is not None:
        sentence_bert_max_length = int(bert_max_length)
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(
            tokenizer=tokenizer,
            sentence_bert_tokenizer=sentence_bert_tokenizer,
            sentence_bert_max_length=sentence_bert_max_length,
        )
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        sentence_bert_tokenizer=sentence_bert_tokenizer,
        sentence_bert_max_length=sentence_bert_max_length,
    )
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
