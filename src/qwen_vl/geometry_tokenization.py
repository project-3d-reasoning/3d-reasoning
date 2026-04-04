"""Utilities for quantized geometry tokens used by 3D box outputs."""

from __future__ import annotations

import ast
import copy
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AddedToken

from qwen_vl.label_weighting import (
    LABEL_WEIGHT_CODE_SCANNET_DET_BBOX,
    LABEL_WEIGHT_CODE_SCANNET_DET_LABEL,
    LABEL_WEIGHT_CODE_SCANREFER_BBOX,
    LABEL_WEIGHT_CODE_SCANREFER_FRAME,
)


POSITION_MIN = -12.0
POSITION_MAX = 12.0
POSITION_STEP = 0.03
SIZE_MIN = 0.0
SIZE_MAX = 12.0
SIZE_STEP = 0.03
ANGLE_MIN = -math.pi
ANGLE_MAX = math.pi
ANGLE_BINS = 256
TILT_MIN = -math.pi / 2.0
TILT_MAX = math.pi / 2.0
TILT_BINS = 128

POSITION_COUNT = int(round((POSITION_MAX - POSITION_MIN) / POSITION_STEP)) + 1
SIZE_COUNT = int(round((SIZE_MAX - SIZE_MIN) / SIZE_STEP)) + 1
POSITION_WIDTH = len(str(POSITION_COUNT - 1))
SIZE_WIDTH = len(str(SIZE_COUNT - 1))
ANGLE_WIDTH = len(str(ANGLE_BINS - 1))
TILT_WIDTH = len(str(TILT_BINS - 1))

NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
SCANREFER_FRAME_PATTERN = re.compile(r'"frame"\s*:\s*(-?\d+)')
JSON_LIST_LABEL_PATTERN = re.compile(r'"label"\s*:\s*"([^"]+)"')
BBOX_FIELD_PATTERN = re.compile(r'("bbox_3d"\s*:\s*)\[([^\]]*)\]', re.DOTALL)
GEOMETRY_TOKEN_PATTERN = re.compile(r"\|(pos|size|angle|tilt)_(\d+)\|")

SCANREFER_PROMPT_NEEDLE = (
    'Output a JSON dictionary with the frame index in "frame" and its 3D bounding box in '
    '"bbox_3d" in the frame\'s coordinates.'
)
SCANREFER_PROMPT_REPLACEMENT = (
    'Output a JSON dictionary with the frame index in "frame" and its quantized 3D bounding '
    'box token list in "bbox_3d" in the frame\'s coordinates.\n'
    'Use 9 quoted geometry tokens in order [x_center, y_center, z_center, x_size, y_size, '
    'z_size, yaw, roll, pitch]. Use |pos_xxx| for positions, |size_xxx| for sizes, |angle_xxx| '
    'for yaw and pitch, and |tilt_xxx| for roll. Do not output decimal numbers inside bbox_3d.'
)
THREEDOD_PROMPT_NEEDLE = (
    'The 3D bounding box format should be [x_center, y_center, z_center, x_size, y_size, '
    'z_size, yaw, roll, pitch].'
)
THREEDOD_PROMPT_REPLACEMENT = (
    'Represent each bbox_3d as 9 quoted geometry tokens in that order. Use |pos_xxx| for '
    'positions, |size_xxx| for sizes, |angle_xxx| for yaw and pitch, and |tilt_xxx| for roll. '
    'Do not output decimal numbers inside bbox_3d.'
)


def geometry_tokens_enabled(obj: Any) -> bool:
    return bool(getattr(obj, "geometry_tokens", False))


def _clip(value: float, minimum: float, maximum: float) -> float:
    return float(max(minimum, min(maximum, float(value))))


def _zero_safe(value: float, ndigits: int = 6) -> float:
    value = round(float(value), ndigits)
    if abs(value) < 10 ** (-ndigits):
        return 0.0
    return value


def _quantize_linear(value: float, minimum: float, maximum: float, count: int) -> int:
    value = _clip(value, minimum, maximum)
    if count <= 1:
        return 0
    step = (maximum - minimum) / float(count - 1)
    idx = int(round((value - minimum) / step))
    return max(0, min(count - 1, idx))


def _dequantize_linear(index: int, minimum: float, maximum: float, count: int) -> float:
    if count <= 1:
        return float(minimum)
    index = max(0, min(count - 1, int(index)))
    step = (maximum - minimum) / float(count - 1)
    return float(minimum + index * step)


def _format_geometry_token(kind: str, index: int) -> str:
    if kind == "pos":
        return f"|pos_{int(index):0{POSITION_WIDTH}d}|"
    if kind == "size":
        return f"|size_{int(index):0{SIZE_WIDTH}d}|"
    if kind == "angle":
        return f"|angle_{int(index):0{ANGLE_WIDTH}d}|"
    if kind == "tilt":
        return f"|tilt_{int(index):0{TILT_WIDTH}d}|"
    raise ValueError(f"Unsupported geometry token kind: {kind}")


def _token_kind_for_index(coord_index: int) -> str:
    if 0 <= coord_index <= 2:
        return "pos"
    if 3 <= coord_index <= 5:
        return "size"
    if coord_index == 7:
        return "tilt"
    if coord_index in (6, 8):
        return "angle"
    raise ValueError(f"Unsupported bbox coordinate index: {coord_index}")


def geometry_token_to_value(token: str) -> float:
    match = GEOMETRY_TOKEN_PATTERN.fullmatch(str(token).strip())
    if match is None:
        raise ValueError(f"Invalid geometry token: {token}")
    kind, index_text = match.groups()
    index = int(index_text)
    if kind == "pos":
        return _dequantize_linear(index, POSITION_MIN, POSITION_MAX, POSITION_COUNT)
    if kind == "size":
        return _dequantize_linear(index, SIZE_MIN, SIZE_MAX, SIZE_COUNT)
    if kind == "angle":
        return _dequantize_linear(index, ANGLE_MIN, ANGLE_MAX, ANGLE_BINS)
    if kind == "tilt":
        return _dequantize_linear(index, TILT_MIN, TILT_MAX, TILT_BINS)
    raise ValueError(f"Unsupported geometry token kind: {kind}")


def value_to_geometry_token(value: float, coord_index: int) -> str:
    kind = _token_kind_for_index(coord_index)
    if kind == "pos":
        index = _quantize_linear(value, POSITION_MIN, POSITION_MAX, POSITION_COUNT)
    elif kind == "size":
        index = _quantize_linear(value, SIZE_MIN, SIZE_MAX, SIZE_COUNT)
    elif kind == "angle":
        index = _quantize_linear(value, ANGLE_MIN, ANGLE_MAX, ANGLE_BINS)
    elif kind == "tilt":
        index = _quantize_linear(value, TILT_MIN, TILT_MAX, TILT_BINS)
    else:
        raise ValueError(f"Unsupported geometry token kind: {kind}")
    return _format_geometry_token(kind, index)


def all_geometry_token_strings() -> List[str]:
    tokens: List[str] = []
    tokens.extend(_format_geometry_token("pos", idx) for idx in range(POSITION_COUNT))
    tokens.extend(_format_geometry_token("size", idx) for idx in range(SIZE_COUNT))
    tokens.extend(_format_geometry_token("angle", idx) for idx in range(ANGLE_BINS))
    tokens.extend(_format_geometry_token("tilt", idx) for idx in range(TILT_BINS))
    return tokens


def all_geometry_added_tokens() -> List[AddedToken]:
    return [
        AddedToken(token, normalized=False, special=False)
        for token in all_geometry_token_strings()
    ]


def encode_bbox_values(values: Sequence[Any]) -> List[str]:
    if len(values) != 9:
        raise ValueError(f"bbox_3d must have 9 values, got {len(values)}")
    encoded: List[str] = []
    for idx, value in enumerate(values):
        if isinstance(value, str) and GEOMETRY_TOKEN_PATTERN.fullmatch(value.strip()):
            encoded.append(value.strip())
            continue
        encoded.append(value_to_geometry_token(float(value), idx))
    return encoded


def decode_bbox_values(values: Sequence[Any], ndigits: int = 6) -> List[float]:
    if len(values) != 9:
        raise ValueError(f"bbox_3d must have 9 values, got {len(values)}")
    decoded: List[float] = []
    for value in values:
        if isinstance(value, str):
            stripped = value.strip()
            if GEOMETRY_TOKEN_PATTERN.fullmatch(stripped):
                decoded.append(_zero_safe(geometry_token_to_value(stripped), ndigits=ndigits))
                continue
            decoded.append(_zero_safe(float(stripped), ndigits=ndigits))
            continue
        decoded.append(_zero_safe(float(value), ndigits=ndigits))
    return decoded


def _parse_bbox_inner_text(inner_text: str) -> Optional[List[Any]]:
    try:
        parsed = ast.literal_eval(f"[{inner_text}]")
    except Exception:
        return None
    if not isinstance(parsed, list):
        return None
    return parsed


def encode_bbox_3d_text(text: str) -> str:
    if not isinstance(text, str) or '"bbox_3d"' not in text:
        return text

    def repl(match: re.Match) -> str:
        prefix = match.group(1)
        inner_text = match.group(2)
        parsed = _parse_bbox_inner_text(inner_text)
        if parsed is None or len(parsed) != 9:
            return match.group(0)
        try:
            encoded = encode_bbox_values(parsed)
        except Exception:
            return match.group(0)
        return f"{prefix}{json.dumps(encoded, ensure_ascii=True)}"

    return BBOX_FIELD_PATTERN.sub(repl, text)


def decode_bbox_3d_text(text: str, ndigits: int = 6) -> str:
    if not isinstance(text, str) or '"bbox_3d"' not in text:
        return text

    def repl(match: re.Match) -> str:
        prefix = match.group(1)
        inner_text = match.group(2)
        parsed = _parse_bbox_inner_text(inner_text)
        if parsed is None or len(parsed) != 9:
            return match.group(0)
        if not any(isinstance(value, str) and GEOMETRY_TOKEN_PATTERN.fullmatch(value.strip()) for value in parsed):
            return match.group(0)
        try:
            decoded = decode_bbox_values(parsed, ndigits=ndigits)
        except Exception:
            return match.group(0)
        return f"{prefix}{json.dumps(decoded, ensure_ascii=True)}"

    return BBOX_FIELD_PATTERN.sub(repl, text)


def parse_bbox_inner_values(inner_text: str, ndigits: int = 6) -> Optional[List[float]]:
    parsed = _parse_bbox_inner_text(inner_text)
    if parsed is None or len(parsed) != 9:
        return None
    try:
        return decode_bbox_values(parsed, ndigits=ndigits)
    except Exception:
        return None


def rewrite_prompt_text(text: str, dataset_name: str) -> str:
    if not isinstance(text, str):
        return text
    if "|pos_" in text or "|size_" in text or "|angle_" in text or "|tilt_" in text:
        return text
    if dataset_name == "scanrefer":
        return text.replace(SCANREFER_PROMPT_NEEDLE, SCANREFER_PROMPT_REPLACEMENT)
    if dataset_name in {"scannet_det", "threedod"}:
        return text.replace(THREEDOD_PROMPT_NEEDLE, THREEDOD_PROMPT_REPLACEMENT)
    return text


def transform_conversations_for_geometry_tokens(
    conversations: Sequence[Dict[str, Any]],
    dataset_name: str,
) -> List[Dict[str, Any]]:
    transformed = copy.deepcopy(list(conversations))
    if dataset_name not in {"scanrefer", "scannet_det"}:
        return transformed

    for message in transformed:
        role = message.get("from", message.get("role", ""))
        content_key = "value" if "value" in message else "content"
        text = str(message.get(content_key, ""))
        if role in {"human", "user"}:
            text = rewrite_prompt_text(text, dataset_name)
        elif role in {"gpt", "assistant"}:
            text = encode_bbox_3d_text(text)
        message[content_key] = text
    return transformed


def _geometry_seed_text(token_text: str) -> str:
    value = geometry_token_to_value(token_text)
    return f"{value:.2f}"


def initialize_geometry_token_embeddings(
    model,
    tokenizer,
    base_vocab_size: int,
) -> None:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        return

    input_weight = input_embeddings.weight.data
    output_weight = output_embeddings.weight.data if output_embeddings is not None else None
    input_snapshot = input_weight[:base_vocab_size].clone()
    output_snapshot = (
        output_weight[:base_vocab_size].clone()
        if output_weight is not None and output_weight.shape[0] >= base_vocab_size
        else None
    )
    fallback_ids = [
        token_id
        for token_id in tokenizer.encode("0.00", add_special_tokens=False)
        if token_id < base_vocab_size
    ]
    if not fallback_ids:
        fallback_ids = [0]

    for token_text in all_geometry_token_strings():
        token_id = tokenizer.convert_tokens_to_ids(token_text)
        if token_id is None or int(token_id) < base_vocab_size:
            continue
        seed_text = _geometry_seed_text(token_text)
        seed_ids = [
            token_id
            for token_id in tokenizer.encode(seed_text, add_special_tokens=False)
            if token_id < base_vocab_size
        ]
        if not seed_ids:
            seed_ids = fallback_ids

        seed_vector = input_snapshot[seed_ids].mean(dim=0)
        input_weight[token_id].copy_(seed_vector)

        if output_weight is not None:
            if output_snapshot is not None:
                output_vector = output_snapshot[seed_ids].mean(dim=0)
            else:
                output_vector = seed_vector
            output_weight[token_id].copy_(output_vector)


def register_geometry_token_gradient_mask(model, base_vocab_size: int) -> None:
    if base_vocab_size <= 0:
        return

    def hook_fn(grad: torch.Tensor) -> torch.Tensor:
        if grad is None or grad.ndim != 2 or grad.shape[0] <= base_vocab_size:
            return grad
        masked = grad.clone()
        masked[:base_vocab_size].zero_()
        return masked

    seen_params = set()
    for embedding in (model.get_input_embeddings(), model.get_output_embeddings()):
        if embedding is None or not hasattr(embedding, "weight"):
            continue
        parameter = embedding.weight
        if id(parameter) in seen_params:
            continue
        seen_params.add(id(parameter))
        parameter.requires_grad = True
        parameter.register_hook(hook_fn)


@dataclass
class SpanMatch:
    start_char: int
    end_char: int
    code: int


class TextTokenMapper:
    """Map character spans to token spans for assistant content."""

    def __init__(self, tokenizer, text: str):
        self.tokenizer = tokenizer
        self.text = text
        self._offsets: Optional[List[Tuple[int, int]]] = None
        self._prefix_cache: Dict[int, int] = {0: 0}
        try:
            encoding = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = encoding.get("offset_mapping")
            if offsets is not None:
                self._offsets = [(int(start), int(end)) for start, end in offsets]
        except Exception:
            self._offsets = None

    def token_count(self) -> int:
        if self._offsets is not None:
            return len(self._offsets)
        return self._prefix_token_count(len(self.text))

    def token_span(self, start_char: int, end_char: int) -> Optional[Tuple[int, int]]:
        if start_char >= end_char:
            return None
        if self._offsets is not None:
            token_start = None
            token_end = None
            for idx, (tok_start, tok_end) in enumerate(self._offsets):
                if tok_end <= start_char:
                    continue
                if tok_start >= end_char:
                    break
                if token_start is None:
                    token_start = idx
                token_end = idx + 1
            if token_start is None or token_end is None:
                return None
            return token_start, token_end

        token_start = self._prefix_token_count(start_char)
        token_end = self._prefix_token_count(end_char)
        if token_end <= token_start:
            return None
        return token_start, token_end

    def _prefix_token_count(self, char_idx: int) -> int:
        char_idx = max(0, min(int(char_idx), len(self.text)))
        cached = self._prefix_cache.get(char_idx)
        if cached is not None:
            return cached
        count = len(self.tokenizer.encode(self.text[:char_idx], add_special_tokens=False))
        self._prefix_cache[char_idx] = count
        return count


def _build_assistant_codes(tokenizer, assistant_text: str, spans: Sequence[SpanMatch]) -> torch.Tensor:
    mapper = TextTokenMapper(tokenizer, assistant_text)
    codes = torch.zeros(mapper.token_count(), dtype=torch.uint8)
    if not spans:
        return codes

    occupied = torch.zeros_like(codes, dtype=torch.bool)
    sorted_spans = sorted(
        spans,
        key=lambda item: ((item.end_char - item.start_char), -item.start_char),
        reverse=True,
    )
    for span in sorted_spans:
        token_span = mapper.token_span(span.start_char, span.end_char)
        if token_span is None:
            continue
        token_start, token_end = token_span
        if token_end <= token_start:
            continue
        if occupied[token_start:token_end].any():
            continue
        codes[token_start:token_end] = int(span.code)
        occupied[token_start:token_end] = True
    return codes


def _collect_numeric_spans(inner_text: str, inner_start: int, code: int) -> List[SpanMatch]:
    spans: List[SpanMatch] = []
    for match in NUMBER_PATTERN.finditer(inner_text):
        spans.append(
            SpanMatch(
                start_char=inner_start + match.start(),
                end_char=inner_start + match.end(),
                code=code,
            )
        )
    return spans


def _collect_bbox_value_spans(inner_text: str, inner_start: int, code: int) -> List[SpanMatch]:
    spans: List[SpanMatch] = []
    geometry_matches = list(GEOMETRY_TOKEN_PATTERN.finditer(inner_text))
    if geometry_matches:
        for match in geometry_matches:
            spans.append(
                SpanMatch(
                    start_char=inner_start + match.start(),
                    end_char=inner_start + match.end(),
                    code=code,
                )
            )
        return spans
    return _collect_numeric_spans(inner_text, inner_start, code)


def build_geometry_assistant_codes(tokenizer, assistant_text: str, dataset_name: str) -> torch.Tensor:
    spans: List[SpanMatch] = []

    if dataset_name == "scanrefer":
        for match in SCANREFER_FRAME_PATTERN.finditer(assistant_text):
            spans.append(
                SpanMatch(
                    start_char=match.start(1),
                    end_char=match.end(1),
                    code=LABEL_WEIGHT_CODE_SCANREFER_FRAME,
                )
            )
        for match in BBOX_FIELD_PATTERN.finditer(assistant_text):
            spans.extend(
                _collect_bbox_value_spans(
                    inner_text=match.group(2),
                    inner_start=match.start(2),
                    code=LABEL_WEIGHT_CODE_SCANREFER_BBOX,
                )
            )
        return _build_assistant_codes(tokenizer, assistant_text, spans)

    if dataset_name == "scannet_det":
        for match in JSON_LIST_LABEL_PATTERN.finditer(assistant_text):
            spans.append(
                SpanMatch(
                    start_char=match.start(1),
                    end_char=match.end(1),
                    code=LABEL_WEIGHT_CODE_SCANNET_DET_LABEL,
                )
            )
        for match in BBOX_FIELD_PATTERN.finditer(assistant_text):
            spans.extend(
                _collect_bbox_value_spans(
                    inner_text=match.group(2),
                    inner_start=match.start(2),
                    code=LABEL_WEIGHT_CODE_SCANNET_DET_BBOX,
                )
            )
        return _build_assistant_codes(tokenizer, assistant_text, spans)

    return torch.zeros(0, dtype=torch.uint8)
