# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from collections import OrderedDict
from threading import Lock

import torch
from PIL import Image
from torchvision import transforms as TF


def _get_preprocess_cache_size() -> int:
    try:
        return max(int(os.environ.get("VGLLM_PREPROCESS_CACHE_SIZE", "64")), 0)
    except ValueError:
        return 64


_PREPROCESS_CACHE_MAX_ENTRIES = _get_preprocess_cache_size()
_PREPROCESS_CACHE = OrderedDict()
_PREPROCESS_CACHE_LOCK = Lock()


def _build_preprocess_cache_key(image_source, mode: str, target_size: int):
    if _PREPROCESS_CACHE_MAX_ENTRIES <= 0:
        return None

    source_path = None
    if isinstance(image_source, str):
        source_path = image_source
    elif isinstance(image_source, Image.Image):
        source_path = image_source.info.get("vgllm_cache_key") if hasattr(image_source, "info") else None
        if not source_path:
            source_path = getattr(image_source, "filename", None)

    if not source_path:
        return None

    source_path = os.path.abspath(source_path)
    try:
        stat = os.stat(source_path)
        source_version = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        source_version = None

    return (source_path, source_version, mode, target_size)


def _get_cached_preprocessed_image(image_source, mode: str, target_size: int):
    cache_key = _build_preprocess_cache_key(image_source, mode, target_size)
    if cache_key is None:
        return None

    with _PREPROCESS_CACHE_LOCK:
        cached = _PREPROCESS_CACHE.get(cache_key)
        if cached is not None:
            _PREPROCESS_CACHE.move_to_end(cache_key)
        return cached


def _store_cached_preprocessed_image(image_source, mode: str, target_size: int, image_tensor: torch.Tensor):
    cache_key = _build_preprocess_cache_key(image_source, mode, target_size)
    if cache_key is None:
        return

    with _PREPROCESS_CACHE_LOCK:
        _PREPROCESS_CACHE[cache_key] = image_tensor
        _PREPROCESS_CACHE.move_to_end(cache_key)
        while len(_PREPROCESS_CACHE) > _PREPROCESS_CACHE_MAX_ENTRIES:
            _PREPROCESS_CACHE.popitem(last=False)


def _load_single_preprocessed_image(image_source, mode="crop", target_size=518, to_tensor=None):
    cached_image = _get_cached_preprocessed_image(image_source, mode, target_size)
    if cached_image is not None:
        return cached_image

    if to_tensor is None:
        to_tensor = TF.ToTensor()

    if isinstance(image_source, str):
        with Image.open(image_source) as opened_image:
            img = opened_image.copy()
    elif isinstance(image_source, Image.Image):
        img = image_source
    else:
        raise NotImplementedError(f"Unsupported image type: {type(image_source)}")

    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    img = img.convert("RGB")
    width, height = img.size

    if mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
    else:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14

    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    img = to_tensor(img)

    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        img = img[:, start_y : start_y + target_size, :]

    if mode == "pad":
        h_padding = target_size - img.shape[1]
        w_padding = target_size - img.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            img = torch.nn.functional.pad(
                img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )

    _store_cached_preprocessed_image(image_source, mode, target_size, img)
    return img


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
        img = _load_single_preprocessed_image(
            image_path,
            mode=mode,
            target_size=target_size,
            to_tensor=to_tensor,
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
