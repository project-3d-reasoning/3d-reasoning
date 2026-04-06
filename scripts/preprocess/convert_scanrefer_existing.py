#!/usr/bin/env python3

import argparse
import copy
import json
import os
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


CONVENTION = "ZXY"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert existing ScanRefer train/val JSONs between target-frame and first-frame bbox coordinates."
    )
    parser.add_argument(
        "--train-input",
        type=str,
        default="data/train/scanrefer_train_32frames.json",
        help="Existing ScanRefer train JSON.",
    )
    parser.add_argument(
        "--val-input",
        type=str,
        default="data/evaluation/scanrefer/scanrefer_val_32frames.json",
        help="Existing ScanRefer val JSON.",
    )
    parser.add_argument(
        "--train-output",
        type=str,
        default="data/train/scanrefer_train_32frames_first_frame.json",
        help="Converted ScanRefer train JSON.",
    )
    parser.add_argument(
        "--val-output",
        type=str,
        default="data/evaluation/scanrefer_first_frame/scanrefer_val_32frames.json",
        help="Converted ScanRefer val JSON.",
    )
    parser.add_argument(
        "--media-dir",
        type=str,
        default="data/media",
        help="Root directory that contains ScanNet posed_images.",
    )
    parser.add_argument(
        "--bbox-coordinate-frame",
        type=str,
        default="first_frame",
        choices=["target_frame", "first_frame"],
        help="Target coordinate frame for bbox_3d in the converted data.",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=2,
        help="Number of decimals used for converted train bbox_3d labels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def get_scene_id(item):
    return item["images"][0].split("/")[2]


def get_source_bbox_coordinate_frame(item):
    return item.get("bbox_coordinate_frame", "target_frame")


def get_coord_frame_index(item, bbox_coordinate_frame):
    if bbox_coordinate_frame == "target_frame":
        if item.get("target") is None:
            raise ValueError("target_frame requires item['target']")
        return int(item["target"]["frame"])
    if bbox_coordinate_frame == "first_frame":
        return 0
    raise ValueError(f"Unsupported bbox coordinate frame: {bbox_coordinate_frame}")


def get_pose_path(media_dir, image_path):
    stem, _ = os.path.splitext(image_path)
    return os.path.join(media_dir, f"{stem}.txt")


def load_pose(media_dir, image_path, pose_cache):
    pose_path = get_pose_path(media_dir, image_path)
    if pose_path not in pose_cache:
        pose_cache[pose_path] = np.loadtxt(pose_path)
    return pose_cache[pose_path]


def orthonormalize_rotation(rot):
    u, _, vt = np.linalg.svd(rot)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return rot


def mean_rotation(rotations):
    accumulator = np.sum(rotations, axis=0)
    return orthonormalize_rotation(accumulator)


def infer_scene_axis_align_matrix(scene_items, media_dir, pose_cache):
    if "axis_align_matrix" in scene_items[0]:
        return np.array(scene_items[0]["axis_align_matrix"], dtype=float)

    rot_estimates = []
    translation_terms = []
    for item in scene_items:
        source_frame = get_source_bbox_coordinate_frame(item)
        coord_frame_idx = get_coord_frame_index(item, source_frame)
        pose = load_pose(media_dir, item["images"][coord_frame_idx], pose_cache)

        gt_bbox = np.array(item["gt_bbox"], dtype=float)
        src_bbox = np.array(item["target"]["bbox_3d"], dtype=float)

        center_world = gt_bbox[:3]
        rot_world = R.from_euler(CONVENTION, gt_bbox[6:9]).as_matrix()
        rot_src = R.from_euler(CONVENTION, src_bbox[6:9]).as_matrix()

        rot_est = rot_world @ (pose[:3, :3] @ rot_src).T
        rot_estimates.append(orthonormalize_rotation(rot_est))
        translation_terms.append(
            (
                center_world,
                pose[:3, :3] @ src_bbox[:3] + pose[:3, 3],
            )
        )

    axis_align_matrix = np.eye(4)
    axis_align_matrix[:3, :3] = mean_rotation(rot_estimates)
    axis_align_matrix[:3, 3] = np.mean(
        [center_world - axis_align_matrix[:3, :3] @ center_global for center_world, center_global in translation_terms],
        axis=0,
    )
    return axis_align_matrix


def bbox_world_to_cam(gt_bbox, extrinsic, round_decimals=None):
    gt_bbox = np.array(gt_bbox, dtype=float)
    center_world = gt_bbox[:3]
    size = gt_bbox[3:6].tolist()
    rot_world = R.from_euler(CONVENTION, gt_bbox[6:9]).as_matrix()

    world2cam = np.linalg.inv(extrinsic)
    center_cam = (world2cam @ np.array([*center_world, 1.0]))[:3]
    rot_cam = world2cam[:3, :3] @ rot_world
    euler_cam = R.from_matrix(rot_cam).as_euler(CONVENTION)

    bbox = center_cam.tolist() + size + euler_cam.tolist()
    if round_decimals is not None:
        bbox = [round(float(x), round_decimals) for x in bbox]
    return bbox


def build_prompt(prompt, bbox_coordinate_frame):
    prefix = prompt.split('Output a JSON dictionary')[0]
    if bbox_coordinate_frame == "target_frame":
        suffix = 'Output a JSON dictionary with the frame index in "frame" and its 3D bounding box in "bbox_3d" in the frame\'s coordinates.\n'
    elif bbox_coordinate_frame == "first_frame":
        suffix = 'Output a JSON dictionary with the frame index in "frame" and its 3D bounding box in "bbox_3d" in the first frame\'s coordinates.\n'
    else:
        raise ValueError(f"Unsupported bbox coordinate frame: {bbox_coordinate_frame}")
    return prefix + suffix


def build_human_message(num_images, prompt):
    image_tokens = "".join([f"Frame-{i}: <image>" for i in range(num_images)])
    return f"{image_tokens}\n{prompt}"


def build_train_answer(target):
    answer = {
        "frame": int(target["frame"]),
        "bbox_3d": target["bbox_3d"],
    }
    return f"```json\n\t{json.dumps(answer, ensure_ascii=False)}\n```"


def infer_axis_align_matrices(train_data, media_dir, pose_cache):
    scene_to_items = defaultdict(list)
    for item in train_data:
        scene_to_items[get_scene_id(item)].append(item)

    scene_to_axis_align = {}
    for scene_id, scene_items in tqdm(scene_to_items.items(), desc="Inferring scene axis-align matrices"):
        scene_to_axis_align[scene_id] = infer_scene_axis_align_matrix(scene_items, media_dir, pose_cache)
    return scene_to_axis_align


def convert_train_data(train_data, scene_to_axis_align, media_dir, pose_cache, bbox_coordinate_frame, round_decimals):
    converted = []
    for item in tqdm(train_data, desc="Converting train data"):
        new_item = copy.deepcopy(item)
        scene_id = get_scene_id(item)
        axis_align_matrix = scene_to_axis_align[scene_id]

        coord_frame_idx = get_coord_frame_index(item, bbox_coordinate_frame)
        pose = load_pose(media_dir, item["images"][coord_frame_idx], pose_cache)
        extrinsic = axis_align_matrix @ pose
        converted_bbox = bbox_world_to_cam(
            item["gt_bbox"],
            extrinsic=extrinsic,
            round_decimals=round_decimals,
        )

        new_item["target"] = {
            "frame": int(item["target"]["frame"]),
            "bbox_3d": converted_bbox,
        }
        new_item["bbox_coordinate_frame"] = bbox_coordinate_frame

        new_prompt = build_prompt(item["prompt"], bbox_coordinate_frame)
        new_item["prompt"] = new_prompt
        new_item["conversations"][0]["value"] = build_human_message(len(item["images"]), new_prompt)
        new_item["conversations"][1]["value"] = build_train_answer(new_item["target"])
        converted.append(new_item)
    return converted


def convert_val_data(val_data, bbox_coordinate_frame):
    converted = []
    for item in tqdm(val_data, desc="Converting val data"):
        new_item = copy.deepcopy(item)
        new_prompt = build_prompt(item["prompt"], bbox_coordinate_frame)
        new_item["prompt"] = new_prompt
        new_item["conversations"][0]["value"] = build_human_message(len(item["images"]), new_prompt)
        new_item["bbox_coordinate_frame"] = bbox_coordinate_frame
        converted.append(new_item)
    return converted


def validate_output_paths(args):
    for path in [args.train_output, args.val_output]:
        if os.path.exists(path) and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")


def main():
    args = parse_args()
    validate_output_paths(args)

    train_data = load_json(args.train_input)
    val_data = load_json(args.val_input)

    pose_cache = {}
    scene_to_axis_align = infer_axis_align_matrices(train_data, args.media_dir, pose_cache)

    converted_train = convert_train_data(
        train_data=train_data,
        scene_to_axis_align=scene_to_axis_align,
        media_dir=args.media_dir,
        pose_cache=pose_cache,
        bbox_coordinate_frame=args.bbox_coordinate_frame,
        round_decimals=args.round_decimals,
    )
    converted_val = convert_val_data(
        val_data=val_data,
        bbox_coordinate_frame=args.bbox_coordinate_frame,
    )

    save_json(converted_train, args.train_output)
    save_json(converted_val, args.val_output)

    print(f"Saved train split to {args.train_output}")
    print(f"Saved val split to {args.val_output}")
    print(f"bbox_coordinate_frame={args.bbox_coordinate_frame}")


if __name__ == "__main__":
    main()
