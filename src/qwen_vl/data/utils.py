# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np
import copy
from pathlib import Path

def load_and_preprocess_images(image_path_list, mode="crop", target_size=518):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()

    # First process all images and collect their shapes
    for image_path in image_path_list:

        # Open image
        if isinstance(image_path, str):
            img = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            img = image_path
        else:
            raise NotImplementedError(f"Unsupported image type: {type(image_path)}")

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def get_preprocess_params(width, height, target_size=518, patch_multiple=14):
    new_width = target_size
    new_height = int(round((height * (new_width / width)) / patch_multiple) * patch_multiple)
    crop_top = 0
    crop_bottom = new_height
    if new_height > target_size:
        crop_top = (new_height - target_size) // 2
        crop_bottom = crop_top + target_size
    return {
        "new_width": new_width,
        "new_height": new_height,
        "crop_top": crop_top,
        "crop_bottom": crop_bottom,
    }


def preprocess_depth_and_intrinsics(depth, intrinsic, target_size=518):
    height, width = depth.shape
    params = get_preprocess_params(width, height, target_size)

    depth_img = Image.fromarray(depth)
    depth_img = depth_img.resize((params["new_width"], params["new_height"]), Image.Resampling.NEAREST)
    depth_resized = np.asarray(depth_img, dtype=np.float32)
    if params["new_height"] > target_size:
        depth_resized = depth_resized[params["crop_top"] : params["crop_bottom"], :]

    scale_x = params["new_width"] / width
    scale_y = params["new_height"] / height
    intrinsic = intrinsic.copy()
    intrinsic[0, 0] *= scale_x
    intrinsic[1, 1] *= scale_y
    intrinsic[0, 2] *= scale_x
    intrinsic[1, 2] *= scale_y
    if params["new_height"] > target_size:
        intrinsic[1, 2] -= params["crop_top"]

    return depth_resized, intrinsic


def unproject_depth_to_cam(depth_m, intrinsic):
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    points = np.stack([x, y, z], axis=-1)
    valid = depth_m > 1e-6
    return points.astype(np.float32), valid


def transform_points(points, transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return (points @ rotation.T + translation).astype(np.float32)


def load_first_frame_coord_inputs(image_path_list, target_size=518):
    coord_points = []
    coord_masks = []
    first_pose = None
    first_pose_inv = None

    for image_path in image_path_list:
        rgb_path = Path(image_path)
        depth_path = rgb_path.with_suffix(".png")
        pose_path = rgb_path.with_suffix(".txt")
        depth_intrinsic_path = rgb_path.parent / "depth_intrinsic.txt"

        if not (depth_path.exists() and pose_path.exists() and depth_intrinsic_path.exists()):
            return None, None

        depth_raw = np.asarray(Image.open(depth_path), dtype=np.float32)
        depth_m = depth_raw / 1000.0
        depth_intrinsic = np.loadtxt(depth_intrinsic_path, dtype=np.float32)[:3, :3]
        depth_m, depth_intrinsic = preprocess_depth_and_intrinsics(depth_m, depth_intrinsic, target_size=target_size)
        cam_points, valid = unproject_depth_to_cam(depth_m, depth_intrinsic)

        pose = np.loadtxt(pose_path, dtype=np.float32)
        if first_pose is None:
            first_pose = pose
            first_pose_inv = np.linalg.inv(first_pose).astype(np.float32)

        cam_to_first = (first_pose_inv @ pose).astype(np.float32)
        first_frame_points = transform_points(cam_points, cam_to_first)
        coord_points.append(torch.from_numpy(first_frame_points))
        coord_masks.append(torch.from_numpy(valid))

    return coord_points, coord_masks


def prepare_image_inputs(image, image_processor):

    images = load_and_preprocess_images([image])
    geometry_encoder_inputs = copy.deepcopy(images[0])
    merge_size: int = getattr(image_processor, "merge_size")
    patch_size: int = getattr(image_processor, "patch_size")
    _, height, width = images[0].shape

    if width % (patch_size * merge_size) > 0:
        width = width - (width % (patch_size * merge_size))
    if height % (patch_size * merge_size) > 0:
        height = height - (height % (patch_size * merge_size))

    images = images[:,:, :height, :width]
    visual_processed = image_processor(images, return_tensors="pt", do_rescale=False)
    image_tensor = visual_processed["pixel_values"]
    grid_thw = visual_processed["image_grid_thw"]

    return {
        "pixel_values": image_tensor,
        "image_grid_thw": grid_thw[0],
        "geometry_encoder_inputs": geometry_encoder_inputs
    }
