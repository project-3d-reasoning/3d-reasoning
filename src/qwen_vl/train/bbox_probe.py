import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from PIL import Image
from transformers import TrainerCallback

from qwen_vl.bbox_special_tokens import (
    decode_bbox_values,
    parse_json_like_with_special_tokens,
    restore_scanrefer_bbox_prompt,
    restore_threedod_bbox_prompt,
)
from qwen_vl.data.utils import prepare_generation_images


SYSTEM_MESSAGE = "You are a helpful assistant."
CHAT_TEMPLATE = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


@dataclass
class ProbeSample:
    dataset: str
    doc_id: int
    doc: Dict


class BBoxFormatProbeCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        image_processor,
        output_dir: str,
        probe_interval: int,
        probe_num_samples: int,
        probe_max_new_tokens: int,
        use_bbox_special_tokens: bool,
        use_geometry_encoder: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.output_dir = Path(output_dir)
        self.probe_interval = probe_interval
        self.probe_num_samples = probe_num_samples
        self.probe_max_new_tokens = probe_max_new_tokens
        self.use_bbox_special_tokens = use_bbox_special_tokens
        self.use_geometry_encoder = use_geometry_encoder
        self._last_probe_step = -1
        self._probe_samples = None
        self._project_root = Path(__file__).resolve().parents[3]
        self._media_dir = self._project_root / "data" / "media"

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.probe_interval <= 0:
            return control
        if state.global_step <= 0 or state.global_step % self.probe_interval != 0:
            return control
        if state.global_step == self._last_probe_step:
            return control

        self._last_probe_step = state.global_step
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if state.is_world_process_zero:
            self._run_probe(model=model, global_step=state.global_step)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return control

    def _run_probe(self, model, global_step: int) -> None:
        if model is None:
            return

        if self._probe_samples is None:
            self._probe_samples = self._load_probe_samples()
        if not self._probe_samples:
            return

        was_training = model.training
        model.eval()
        try:
            summary = {
                "global_step": global_step,
                "scanrefer": {"ok": 0, "total": 0},
                "scannet_det": {"ok": 0, "total": 0},
                "samples": [],
            }
            for sample in self._probe_samples:
                result = self._probe_one_sample(model=model, sample=sample)
                dataset_summary = summary[sample.dataset]
                dataset_summary["total"] += 1
                dataset_summary["ok"] += int(result["format_ok"])
                summary["samples"].append(result)

            output_dir = self.output_dir / "bbox_probe"
            output_dir.mkdir(parents=True, exist_ok=True)
            step_file = output_dir / f"step_{global_step:07d}.json"
            summary_file = output_dir / "summary.jsonl"
            with open(step_file, "w") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            with open(summary_file, "a") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")

            print(
                "[bbox_probe] step {} | scanrefer {}/{} | scannet_det {}/{}".format(
                    global_step,
                    summary["scanrefer"]["ok"],
                    summary["scanrefer"]["total"],
                    summary["scannet_det"]["ok"],
                    summary["scannet_det"]["total"],
                )
            )
        finally:
            if was_training:
                model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load_probe_samples(self) -> List[ProbeSample]:
        candidates = [
            (
                "scanrefer",
                self._project_root / "data" / "evaluation" / "scanrefer" / "scanrefer_val_32frames.json",
            ),
            (
                "scannet_det",
                self._project_root / "data" / "evaluation" / "threedod" / "scannet" / "scannet_det_val_4frames.json",
            ),
        ]

        samples = []
        for dataset_name, path in candidates:
            if not path.exists():
                print(f"[bbox_probe] skip missing probe file: {path}")
                continue
            with open(path) as f:
                data = json.load(f)
            for doc_id, doc in enumerate(data[: self.probe_num_samples]):
                samples.append(ProbeSample(dataset=dataset_name, doc_id=doc_id, doc=doc))
        return samples

    def _probe_one_sample(self, model, sample: ProbeSample) -> Dict:
        prompt, image_paths = self._build_prompt_and_images(sample)
        try:
            generated_text = self._generate_text(
                model=model, prompt=prompt, image_paths=image_paths
            )
            format_ok, detail = self._validate_output(
                sample=sample, generated_text=generated_text
            )
        except Exception as exc:
            generated_text = ""
            format_ok = False
            detail = {
                "error": f"probe_failed: {type(exc).__name__}: {exc}",
            }
        return {
            "dataset": sample.dataset,
            "doc_id": sample.doc_id,
            "format_ok": format_ok,
            "detail": detail,
            "generated_text": generated_text,
        }

    def _build_prompt_and_images(self, sample: ProbeSample) -> Tuple[str, List[str]]:
        doc = copy.deepcopy(sample.doc)
        prompt = doc["conversations"][0]["value"]
        if self.use_bbox_special_tokens:
            if sample.dataset == "scanrefer":
                prompt = restore_scanrefer_bbox_prompt(prompt)
            elif sample.dataset == "scannet_det":
                prompt = restore_threedod_bbox_prompt(prompt)

        image_paths = [str(self._media_dir / image_path) for image_path in doc["images"]]
        return prompt, image_paths

    def _generate_text(self, model, prompt: str, image_paths: List[str]) -> str:
        pil_images = [Image.open(path).convert("RGB") for path in image_paths]
        image_token = "<image>"
        model_image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        user_content = prompt.replace(image_token, model_image_token)

        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.chat_template = CHAT_TEMPLATE
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        trimmed_images, geometry_encoder_inputs = prepare_generation_images(
            pil_images, self.image_processor
        )
        inputs = self._prepare_generation_inputs(text=text, images=trimmed_images)
        input_length = inputs["input_ids"].shape[1]
        max_length = self._get_model_max_length()
        if max_length is not None and input_length > max_length:
            raise ValueError(
                f"input_too_long: {input_length} > model_max_length {max_length}"
            )

        device = next(model.parameters()).device
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(device)
        if self.use_geometry_encoder:
            inputs["geometry_encoder_inputs"] = [geometry_encoder_inputs.to(device)]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.probe_max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
        return self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _get_model_max_length(self) -> int | None:
        max_length = getattr(self.tokenizer, "model_max_length", None)
        if not isinstance(max_length, int):
            return None
        if max_length <= 0 or max_length >= 1_000_000:
            return None
        return max_length

    def _prepare_generation_inputs(self, text: str, images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        image_inputs = self.image_processor(images=images, videos=None, return_tensors="pt", do_rescale=False)
        image_grid_thw = image_inputs["image_grid_thw"]
        image_token = getattr(self.tokenizer, "image_token", "<|image_pad|>")
        merge_length = self.image_processor.merge_size**2
        index = 0
        while image_token in text:
            text = text.replace(
                image_token,
                "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length),
                1,
            )
            index += 1
        text = text.replace("<|placeholder|>", image_token)
        text_inputs = self.tokenizer([text], padding=True, return_tensors="pt")
        return {**text_inputs, **image_inputs}

    def _validate_output(self, sample: ProbeSample, generated_text: str) -> Tuple[bool, Dict]:
        try:
            parsed = parse_json_like_with_special_tokens(generated_text)
        except Exception as exc:
            return False, {"error": f"parse_failed: {exc}"}

        if sample.dataset == "scanrefer":
            if not isinstance(parsed, dict):
                return False, {"error": "expected_dict"}
            frame = parsed.get("frame")
            bbox = parsed.get("bbox_3d")
            if not isinstance(frame, int):
                return False, {"error": "invalid_frame_type"}
            if frame < 0 or frame >= len(sample.doc["images"]):
                return False, {"error": "frame_out_of_range", "frame": frame}
            if not isinstance(bbox, list) or len(bbox) != 9:
                return False, {"error": "invalid_bbox_length"}
            decoded_bbox = decode_bbox_values(bbox)
            return True, {"frame": frame, "decoded_bbox": decoded_bbox}

        if sample.dataset == "scannet_det":
            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list) or len(parsed) == 0:
                return False, {"error": "expected_non_empty_list"}

            decoded = []
            for item in parsed:
                if not isinstance(item, dict):
                    return False, {"error": "list_item_not_dict"}
                if not isinstance(item.get("label"), str) or not item["label"]:
                    return False, {"error": "invalid_label"}
                bbox = item.get("bbox_3d")
                if not isinstance(bbox, list) or len(bbox) != 9:
                    return False, {"error": "invalid_bbox_length"}
                decoded.append({"label": item["label"], "decoded_bbox": decode_bbox_values(bbox)})
            return True, {"num_boxes": len(decoded), "preview": decoded[:3]}

        return False, {"error": f"unsupported_dataset:{sample.dataset}"}
