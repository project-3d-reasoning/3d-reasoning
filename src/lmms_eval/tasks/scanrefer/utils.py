import re
import os
import torch
from pathlib import Path
import yaml
import pickle
import numpy as np
from PIL import Image
from functools import lru_cache
from loguru import logger as eval_logger
from scipy.spatial.transform import Rotation as R
from lmms_eval.tasks.threedod.utils import EulerDepthInstance3DBoxes

with open(Path(__file__).parent / "scanrefer.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
media_dir = yaml.safe_load("".join(safe_data))["metadata"]["media_dir"]
SCANREFER_IMAGE_CACHE_SIZE = max(1, int(os.getenv("SCANREFER_IMAGE_CACHE_SIZE", "256")))
_SCANREFER_FRAME_EXTRINSIC_CACHE = {}
# embodiedscan_path = yaml.safe_load("".join(safe_data))["metadata"]["embodiedscan_path"]
# with open(embodiedscan_path, "rb") as f:
#     data = pickle.load(f)["data_list"]
#     id2scene = {sample["sample_id"]: sample for sample in data}


@lru_cache(maxsize=SCANREFER_IMAGE_CACHE_SIZE)
def _load_scanrefer_image(image_file):
    image_path = os.path.join(media_dir, image_file)
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        rgb_image.load()
    setattr(rgb_image, "_vgllm_source_path", image_path)
    return rgb_image

def scanrefer_doc_to_visual(doc):
    image_files = doc["images"]
    images = [_load_scanrefer_image(image_file) for image_file in image_files]
    return [images]    


def scanrefer_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = doc["prompt"]
    return prompt


def scanrefer_bbox_to_9dof(bbox, convention, extrinsic=None):
    center = bbox[0: 3]
    sizes = bbox[3:6]
    rot = R.from_euler(convention, np.array(bbox[6:9]))
    if extrinsic is not None:
        center = (extrinsic @ np.array([*center, 1]).reshape(4, 1)).reshape(4)[:3].tolist()
        mat = extrinsic[:3, :3] @ rot.as_matrix()
        rot = R.from_matrix(mat)
    euler = list(rot.as_euler(convention))

    return center + sizes + euler


def _get_scanrefer_frame_extrinsic(doc, frame_idx):
    if not doc["images"]:
        return np.array(doc["axis_align_matrix"]) @ np.array(doc["cam2global"][frame_idx])

    scene_id = doc["images"][0].split("/")[-2]
    cache_key = (scene_id, frame_idx)
    if cache_key not in _SCANREFER_FRAME_EXTRINSIC_CACHE:
        _SCANREFER_FRAME_EXTRINSIC_CACHE[cache_key] = (
            np.array(doc["axis_align_matrix"]) @ np.array(doc["cam2global"][frame_idx])
        )
    return _SCANREFER_FRAME_EXTRINSIC_CACHE[cache_key]


def scanrefer_process_results(doc, results):
    lines = results[0].strip('\n').strip("```").strip("json").strip("\n").split("\n")
    gt_bbox = doc["gt_bbox"]
    pred_dict = None
    for line in lines:
        if "bbox_3d" in line:
            try:
                pred_dict = eval(line.strip())
            except Exception as e:
                eval_logger.error(f"Error parsing bbox_3d: {line.strip()}")
            break
    
    iou = 0
    pred_bbox = None
    if pred_dict is not None:
        try:
            assert "frame" in pred_dict and isinstance(pred_dict["frame"], int) and pred_dict["frame"] >= 0 and pred_dict["frame"] < len(doc["cam2global"]), \
                "Invalid frame index"
            assert "bbox_3d" in pred_dict and isinstance(pred_dict["bbox_3d"], list) and len(pred_dict["bbox_3d"]) == 9, \
                "Invalid bbox_3d format"
            
            frame_idx = pred_dict["frame"]
            extrinsic = _get_scanrefer_frame_extrinsic(doc, frame_idx)
            pred_bbox = scanrefer_bbox_to_9dof(pred_dict["bbox_3d"], convention="ZXY", extrinsic=extrinsic)
            iou = EulerDepthInstance3DBoxes.overlaps(
                EulerDepthInstance3DBoxes(torch.tensor([pred_bbox]), convention="ZXY"),
                EulerDepthInstance3DBoxes(torch.tensor([gt_bbox]), convention="ZXY")
            ).item()
        except Exception as e:
            eval_logger.error(f"Error parsing pred_dict: {pred_dict} with error: {e}")

    ret = {
        'iou': iou,
        'pred_bbox': pred_bbox,
        'gt_bbox': gt_bbox,
        "images": doc["images"]
    }
    return {"scanrefer_score": ret}


def scanrefer_aggregate_results(results):
    ious = np.asarray([result.get("iou", 0.0) for result in results], dtype=np.float32)
    output = {}
    output["iou25"] = float((ious >= 0.25).mean() * 100) if len(ious) > 0 else 0.0
    output["iou50"] = float((ious >= 0.50).mean() * 100) if len(ious) > 0 else 0.0

    eval_logger.info(f"Scanrefer results: {output}")
    return output["iou25"]
