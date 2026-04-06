import ast
import copy
import json
import math
import re
from dataclasses import dataclass
from typing import Any, List, Sequence


BBOX_QUANTIZATION_STEP = 0.03
SCAN2CAP_POSITION_FINE_BINS = 4

POSITION_PROMPT_TOKEN = "|position|"
SIZE_PROMPT_TOKEN = "|size|"
ANGLE_PROMPT_TOKEN = "|angle|"
POSITION_COARSE_PROMPT_FAMILY = "position_coarse"
POSITION_FINE_PROMPT_FAMILY = "position_fine"
PROMPT_PLACEHOLDER_TOKENS = [
    POSITION_PROMPT_TOKEN,
    SIZE_PROMPT_TOKEN,
    ANGLE_PROMPT_TOKEN,
]


@dataclass(frozen=True)
class QuantizationSpec:
    family: str
    min_value: float
    max_value: float
    step: float = BBOX_QUANTIZATION_STEP

    @property
    def num_bins(self) -> int:
        return int(round((self.max_value - self.min_value) / self.step)) + 1

    @property
    def max_index(self) -> int:
        return self.num_bins - 1


POSITION_SPEC = QuantizationSpec("position", -15.0, 15.0)
SIZE_SPEC = QuantizationSpec("size", 0.0, 12.0)
ANGLE_YAW_SPEC = QuantizationSpec("angle", -math.pi, math.pi)
ANGLE_ROLL_SPEC = QuantizationSpec("angle", -math.pi / 2, math.pi / 2)
ANGLE_PITCH_SPEC = QuantizationSpec("angle", -math.pi, math.pi)
SCAN2CAP_POSITION_COARSE_NUM_BINS = math.ceil(
    POSITION_SPEC.num_bins / SCAN2CAP_POSITION_FINE_BINS
)

BBOX_QUANTIZATION_SPECS = [
    POSITION_SPEC,
    POSITION_SPEC,
    POSITION_SPEC,
    SIZE_SPEC,
    SIZE_SPEC,
    SIZE_SPEC,
    ANGLE_YAW_SPEC,
    ANGLE_ROLL_SPEC,
    ANGLE_PITCH_SPEC,
]

BBOX_PLACEHOLDER_TOKENS = [
    POSITION_PROMPT_TOKEN,
    POSITION_PROMPT_TOKEN,
    POSITION_PROMPT_TOKEN,
    SIZE_PROMPT_TOKEN,
    SIZE_PROMPT_TOKEN,
    SIZE_PROMPT_TOKEN,
    ANGLE_PROMPT_TOKEN,
    ANGLE_PROMPT_TOKEN,
    ANGLE_PROMPT_TOKEN,
]

_FAMILY_TO_MAX_BINS = {}
for spec in BBOX_QUANTIZATION_SPECS:
    _FAMILY_TO_MAX_BINS[spec.family] = max(
        _FAMILY_TO_MAX_BINS.get(spec.family, 0),
        spec.num_bins,
    )

SPECIAL_TOKEN_PATTERN = re.compile(r"\|(position|size|angle)(?:_(\d+))?\|")
_FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_NUMERIC_TRIPLE_PATTERN = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
_POSITION_TOKEN_TRIPLE_PATTERN = re.compile(
    r'\[\s*"(\|position_\d+\|)"\s*,\s*"(\|position_\d+\|)"\s*,\s*"(\|position_\d+\|)"\s*\]'
)
_TOKEN_LITERAL_PATTERN = re.compile(
    r'(?<!["\'])\|(position|size|angle)(?:_\d+)?\|(?!["\'])'
)


def get_bbox_special_tokens(include_prompt_placeholders: bool = True) -> List[str]:
    tokens = []
    tokens.extend(get_bbox_prompt_tokens(include_prompt_placeholders=include_prompt_placeholders))
    tokens.extend(get_bbox_coordinate_tokens())
    return tokens


def get_bbox_coordinate_tokens() -> List[str]:
    tokens = []
    for family, num_bins in _FAMILY_TO_MAX_BINS.items():
        tokens.extend([f"|{family}_{idx}|" for idx in range(num_bins)])
    return tokens


def get_bbox_prompt_tokens(include_prompt_placeholders: bool = True) -> List[str]:
    tokens = []
    if include_prompt_placeholders:
        tokens.extend(PROMPT_PLACEHOLDER_TOKENS)
    tokens.extend(
        [f"|{POSITION_COARSE_PROMPT_FAMILY}_{idx}|" for idx in range(SCAN2CAP_POSITION_COARSE_NUM_BINS)]
    )
    tokens.extend(
        [f"|{POSITION_FINE_PROMPT_FAMILY}_{idx}|" for idx in range(SCAN2CAP_POSITION_FINE_BINS)]
    )
    return tokens


def parse_bbox_coordinate_token(token: str) -> tuple[str, int]:
    token_match = SPECIAL_TOKEN_PATTERN.fullmatch(token.strip())
    if token_match is None or token_match.group(2) is None:
        raise ValueError(f"Not a bbox coordinate token: {token!r}")
    family, token_index = token_match.groups()
    return family, int(token_index)


def get_bbox_coordinate_token_metadata(tokenizer) -> List[dict]:
    metadata = []
    for token, token_id in zip(
        get_bbox_coordinate_tokens(),
        tokenizer.convert_tokens_to_ids(get_bbox_coordinate_tokens()),
    ):
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if token_id is None or token_id < 0 or token_id == unk_token_id:
            continue
        family, bin_index = parse_bbox_coordinate_token(token)
        metadata.append(
            {
                "token": token,
                "token_id": int(token_id),
                "family": family,
                "bin_index": bin_index,
            }
        )
    return metadata


def build_bbox_coordinate_config(tokenizer, neighbor_radius: int = 2) -> dict:
    metadata = get_bbox_coordinate_token_metadata(tokenizer)
    coordinate_token_ids = [item["token_id"] for item in metadata]
    token_family_map = {str(item["token_id"]): item["family"] for item in metadata}
    token_bin_index_map = {str(item["token_id"]): item["bin_index"] for item in metadata}

    family_to_entries = {}
    for item in metadata:
        family_to_entries.setdefault(item["family"], {})[item["bin_index"]] = item["token_id"]

    neighbor_map = {}
    for item in metadata:
        family_entries = family_to_entries[item["family"]]
        neighbors = []
        for offset in range(1, max(int(neighbor_radius), 0) + 1):
            left_token_id = family_entries.get(item["bin_index"] - offset)
            right_token_id = family_entries.get(item["bin_index"] + offset)
            if left_token_id is not None:
                neighbors.append(left_token_id)
            if right_token_id is not None:
                neighbors.append(right_token_id)
        neighbor_map[str(item["token_id"])] = neighbors

    return {
        "coordinate_token_ids": coordinate_token_ids,
        "coordinate_neighbor_map": neighbor_map,
        "coordinate_family_map": token_family_map,
        "coordinate_bin_index_map": token_bin_index_map,
    }


def get_bbox_coordinate_token_ids(tokenizer) -> List[int]:
    token_ids = tokenizer.convert_tokens_to_ids(get_bbox_coordinate_tokens())
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    return [
        int(token_id)
        for token_id in token_ids
        if token_id is not None and token_id >= 0 and token_id != unk_token_id
    ]


def add_bbox_tokens(tokenizer) -> int:
    # These are regular added tokens instead of tokenizer special tokens so they
    # remain visible during decoding with `skip_special_tokens=True`.
    new_tokens = list(dict.fromkeys(get_bbox_coordinate_tokens() + get_bbox_prompt_tokens(False)))
    return tokenizer.add_tokens(new_tokens)


def resize_model_embeddings_for_bbox_tokens(
    model,
    tokenizer,
    num_new_tokens: int,
    coordinate_label_smoothing: float | None = None,
    coordinate_smoothing_neighbor_radius: int | None = None,
) -> None:
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg

        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            output_weight = output_embeddings.weight.data
            output_embeddings_avg = output_weight[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_weight[-num_new_tokens:] = output_embeddings_avg

    config_values = build_bbox_coordinate_config(
        tokenizer,
        neighbor_radius=(
            coordinate_smoothing_neighbor_radius
            if coordinate_smoothing_neighbor_radius is not None
            else int(getattr(model.config, "bbox_coordinate_smoothing_neighbor_radius", 2))
        ),
    )
    model.config.vocab_size = len(tokenizer)
    setattr(model.config, "use_bbox_special_tokens", True)
    setattr(model.config, "bbox_coordinate_token_ids", config_values["coordinate_token_ids"])
    setattr(model.config, "bbox_coordinate_neighbor_token_ids", config_values["coordinate_neighbor_map"])
    setattr(model.config, "bbox_coordinate_family_map", config_values["coordinate_family_map"])
    setattr(model.config, "bbox_coordinate_bin_index_map", config_values["coordinate_bin_index_map"])
    if coordinate_label_smoothing is not None:
        setattr(model.config, "bbox_coordinate_label_smoothing", float(coordinate_label_smoothing))
    elif not hasattr(model.config, "bbox_coordinate_label_smoothing"):
        setattr(model.config, "bbox_coordinate_label_smoothing", 0.0)
    if coordinate_smoothing_neighbor_radius is not None:
        setattr(
            model.config,
            "bbox_coordinate_smoothing_neighbor_radius",
            int(coordinate_smoothing_neighbor_radius),
        )
    elif not hasattr(model.config, "bbox_coordinate_smoothing_neighbor_radius"):
        setattr(model.config, "bbox_coordinate_smoothing_neighbor_radius", 2)


add_bbox_special_tokens = add_bbox_tokens
resize_model_embeddings_for_special_tokens = resize_model_embeddings_for_bbox_tokens


def quantize_value(value: float, spec: QuantizationSpec) -> int:
    clamped_value = min(max(float(value), spec.min_value), spec.max_value)
    index = int(round((clamped_value - spec.min_value) / spec.step))
    return min(max(index, 0), spec.max_index)


def dequantize_index(index: int, spec: QuantizationSpec) -> float:
    clamped_index = min(max(int(index), 0), spec.max_index)
    value = spec.min_value + clamped_index * spec.step
    return min(max(value, spec.min_value), spec.max_value)


def quantize_value_to_token(value: float, spec: QuantizationSpec) -> str:
    return f"|{spec.family}_{quantize_value(value, spec)}|"


def quantize_value_with_residual(value: float, spec: QuantizationSpec) -> tuple[str, float]:
    token_index = quantize_value(value, spec)
    token = f"|{spec.family}_{token_index}|"
    token_center = dequantize_index(token_index, spec)
    residual = (min(max(float(value), spec.min_value), spec.max_value) - token_center) / spec.step
    residual = min(max(residual, -0.5), 0.5)
    return token, float(residual)


def quantize_position_triplet(position: Sequence[float]) -> List[str]:
    if len(position) != 3:
        raise ValueError(f"Expected 3 position values, got {len(position)}")
    return [quantize_value_to_token(value, POSITION_SPEC) for value in position]


def quantize_position_value_to_coarse_fine_tokens(value: float) -> List[str]:
    global_index = quantize_value(value, POSITION_SPEC)
    coarse_index = global_index // SCAN2CAP_POSITION_FINE_BINS
    fine_index = global_index % SCAN2CAP_POSITION_FINE_BINS
    return [
        f"|{POSITION_COARSE_PROMPT_FAMILY}_{coarse_index}|",
        f"|{POSITION_FINE_PROMPT_FAMILY}_{fine_index}|",
    ]


def quantize_position_triplet_to_coarse_fine_tokens(position: Sequence[float]) -> List[str]:
    if len(position) != 3:
        raise ValueError(f"Expected 3 position values, got {len(position)}")
    tokens = []
    for value in position:
        tokens.extend(quantize_position_value_to_coarse_fine_tokens(value))
    return tokens


def format_position_triplet_for_prompt(position: Sequence[float]) -> str:
    return json.dumps(quantize_position_triplet_to_coarse_fine_tokens(position), ensure_ascii=False)


def quantize_bbox_values(bbox_3d: Sequence[float]) -> List[str]:
    if len(bbox_3d) != 9:
        raise ValueError(f"Expected 9 bbox values, got {len(bbox_3d)}")
    return [
        quantize_value_to_token(value, spec)
        for value, spec in zip(bbox_3d, BBOX_QUANTIZATION_SPECS)
    ]


def quantize_bbox_values_with_residuals(bbox_3d: Sequence[float]) -> tuple[List[str], List[float]]:
    if len(bbox_3d) != 9:
        raise ValueError(f"Expected 9 bbox values, got {len(bbox_3d)}")
    tokens = []
    residuals = []
    for value, spec in zip(bbox_3d, BBOX_QUANTIZATION_SPECS):
        token, residual = quantize_value_with_residual(value, spec)
        tokens.append(token)
        residuals.append(residual)
    return tokens, residuals


def decode_special_token_value(value: Any, spec: QuantizationSpec) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        stripped_value = value.strip()
        token_match = SPECIAL_TOKEN_PATTERN.fullmatch(stripped_value)
        if token_match is not None and token_match.group(2) is not None:
            family, token_index = token_match.groups()
            if family != spec.family:
                raise ValueError(f"Token family mismatch: {family} vs {spec.family}")
            return dequantize_index(int(token_index), spec)
        try:
            return float(stripped_value)
        except ValueError as exc:
            raise ValueError(f"Unable to decode value {value!r}") from exc

    raise TypeError(f"Unsupported bbox value type: {type(value)}")


def decode_bbox_values(bbox_3d: Sequence[Any]) -> List[float]:
    if len(bbox_3d) != 9:
        raise ValueError(f"Expected 9 bbox values, got {len(bbox_3d)}")
    return [
        decode_special_token_value(value, spec)
        for value, spec in zip(bbox_3d, BBOX_QUANTIZATION_SPECS)
    ]


def get_bbox_token_instruction() -> str:
    placeholder_json = json.dumps(BBOX_PLACEHOLDER_TOKENS)
    return (
        'Represent "bbox_3d" as 9 quantized special tokens in the format '
        f'{placeholder_json}. Use one special token per coordinate, such as "|position_512|".'
    )


SCANREFER_BBOX_LEGACY_LINE = (
    'Output a JSON dictionary with the frame index in "frame" and its 3D bounding box '
    'in "bbox_3d" in the frame\'s coordinates.'
)
THREEDOD_BBOX_LEGACY_LINE = (
    "The 3D bounding box format should be [x_center, y_center, z_center, "
    "x_size, y_size, z_size, yaw, roll, pitch]."
)
THREEDOD_BBOX_TOKEN_INSTRUCTION = (
    'The 3D bounding box format should be '
    f'{json.dumps(BBOX_PLACEHOLDER_TOKENS)}. '
    'Use quantized special tokens such as "|size_120|".'
)


def inject_scanrefer_bbox_token_instruction(text: str) -> str:
    instruction = get_bbox_token_instruction()
    if instruction in text:
        return text

    if SCANREFER_BBOX_LEGACY_LINE in text:
        return text.replace(SCANREFER_BBOX_LEGACY_LINE, f"{SCANREFER_BBOX_LEGACY_LINE}\n{instruction}")
    return text.rstrip() + "\n" + instruction


def inject_threedod_bbox_token_instruction(text: str) -> str:
    if THREEDOD_BBOX_TOKEN_INSTRUCTION in text:
        return text
    if THREEDOD_BBOX_LEGACY_LINE in text:
        return text.replace(THREEDOD_BBOX_LEGACY_LINE, THREEDOD_BBOX_TOKEN_INSTRUCTION)
    return text.rstrip() + "\n" + THREEDOD_BBOX_TOKEN_INSTRUCTION


def restore_scanrefer_bbox_prompt(text: str) -> str:
    instruction = get_bbox_token_instruction()
    restored = text.replace(f"\n{instruction}", "").replace(instruction, "")
    return restored.rstrip()


def restore_threedod_bbox_prompt(text: str) -> str:
    if THREEDOD_BBOX_TOKEN_INSTRUCTION in text:
        return text.replace(THREEDOD_BBOX_TOKEN_INSTRUCTION, THREEDOD_BBOX_LEGACY_LINE)
    return text


def convert_legacy_position_token_triplet_to_coarse_fine(text: str) -> str:
    match = _POSITION_TOKEN_TRIPLE_PATTERN.search(text)
    if match is None:
        return text

    coarse_fine_tokens = []
    for token in match.groups():
        _, global_index = parse_bbox_coordinate_token(token)
        coarse_index = global_index // SCAN2CAP_POSITION_FINE_BINS
        fine_index = global_index % SCAN2CAP_POSITION_FINE_BINS
        coarse_fine_tokens.extend(
            [
                f"|{POSITION_COARSE_PROMPT_FAMILY}_{coarse_index}|",
                f"|{POSITION_FINE_PROMPT_FAMILY}_{fine_index}|",
            ]
        )

    replacement = json.dumps(coarse_fine_tokens, ensure_ascii=False)
    return text[: match.start()] + replacement + text[match.end() :]


def rewrite_scan2cap_prompt_with_position_tokens(text: str) -> str:
    if f"|{POSITION_COARSE_PROMPT_FAMILY}_" in text:
        return text

    if "|position_" in text:
        return convert_legacy_position_token_triplet_to_coarse_fine(text)

    match = _NUMERIC_TRIPLE_PATTERN.search(text)
    if match is None:
        return text

    values = [float(match.group(i)) for i in range(1, 4)]
    replacement = format_position_triplet_for_prompt(values)
    return text[: match.start()] + replacement + text[match.end() :]


def strip_code_fence(text: str) -> str:
    match = _FENCED_JSON_PATTERN.search(text)
    if match is not None:
        return match.group(1).strip()
    return text.strip()


def normalize_special_token_literals(text: str) -> str:
    return _TOKEN_LITERAL_PATTERN.sub(lambda match: f'"{match.group(0)}"', text)


def parse_json_like_with_special_tokens(text: str) -> Any:
    base_content = strip_code_fence(text)
    candidate_texts = [base_content]

    for open_char, close_char in [("{", "}"), ("[", "]")]:
        start_idx = base_content.find(open_char)
        end_idx = base_content.rfind(close_char)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate_texts.append(base_content[start_idx : end_idx + 1])

    last_error = None
    for candidate in candidate_texts:
        try:
            return ast.literal_eval(normalize_special_token_literals(candidate))
        except Exception as exc:
            last_error = exc

    raise last_error


def quantize_bbox_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        new_payload = {}
        for key, value in payload.items():
            if key == "bbox_3d" and isinstance(value, (list, tuple)) and len(value) == 9:
                new_payload[key] = quantize_bbox_values(value)
            else:
                new_payload[key] = quantize_bbox_payload(value)
        return new_payload

    if isinstance(payload, list):
        return [quantize_bbox_payload(item) for item in payload]

    if isinstance(payload, tuple):
        return [quantize_bbox_payload(item) for item in payload]

    return payload


def quantize_bbox_payload_with_residuals(payload: Any) -> tuple[Any, List[float]]:
    residual_targets = []

    def _quantize(value: Any) -> Any:
        if isinstance(value, dict):
            new_payload = {}
            for key, nested_value in value.items():
                if key == "bbox_3d" and isinstance(nested_value, (list, tuple)) and len(nested_value) == 9:
                    quantized_values, residuals = quantize_bbox_values_with_residuals(nested_value)
                    residual_targets.extend(residuals)
                    new_payload[key] = quantized_values
                else:
                    new_payload[key] = _quantize(nested_value)
            return new_payload

        if isinstance(value, list):
            return [_quantize(item) for item in value]

        if isinstance(value, tuple):
            return [_quantize(item) for item in value]

        return value

    return _quantize(copy.deepcopy(payload)), residual_targets


def format_bbox_payload(payload: Any, fenced: bool = True) -> str:
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if fenced:
        return f"```json\n{json_text}\n```"
    return json_text


def format_quantized_bbox_payload(payload: Any, fenced: bool = True) -> str:
    quantized_payload = quantize_bbox_payload(copy.deepcopy(payload))
    return format_bbox_payload(quantized_payload, fenced=fenced)


def rewrite_bbox_response_with_special_tokens(text: str) -> str:
    rewritten_text, _ = rewrite_bbox_response_with_auxiliary_targets(text)
    return rewritten_text


def rewrite_bbox_response_with_auxiliary_targets(text: str) -> tuple[str, List[float]]:
    if "|position_" in text or "|size_" in text or "|angle_" in text:
        return text, []

    try:
        payload = parse_json_like_with_special_tokens(text)
    except Exception:
        return text, []

    quantized_payload, residual_targets = quantize_bbox_payload_with_residuals(payload)
    return format_bbox_payload(
        quantized_payload,
        fenced=text.strip().startswith("```"),
    ), residual_targets


def refine_bbox_values_with_residuals(
    bbox_3d: Sequence[Any],
    residual_values: Sequence[float],
    decimal_places: int = 4,
) -> tuple[List[Any], int]:
    if len(bbox_3d) != 9:
        raise ValueError(f"Expected 9 bbox values, got {len(bbox_3d)}")

    refined_values = []
    residual_index = 0
    for value, spec in zip(bbox_3d, BBOX_QUANTIZATION_SPECS):
        if isinstance(value, str):
            token_match = SPECIAL_TOKEN_PATTERN.fullmatch(value.strip())
        else:
            token_match = None
        if token_match is None or token_match.group(2) is None:
            refined_values.append(value)
            continue
        if residual_index >= len(residual_values):
            refined_values.append(value)
            continue
        residual = residual_values[residual_index]
        residual_index += 1
        bin_center = decode_special_token_value(value, spec)
        refined_value = bin_center + float(residual) * spec.step
        refined_value = min(max(refined_value, spec.min_value), spec.max_value)
        refined_values.append(round(refined_value, decimal_places))
    return refined_values, residual_index


def refine_bbox_payload_with_residuals(
    payload: Any,
    residual_values: Sequence[float],
    decimal_places: int = 4,
) -> tuple[Any, int]:
    residual_index = 0

    def _refine(value: Any) -> Any:
        nonlocal residual_index
        if isinstance(value, dict):
            new_payload = {}
            for key, nested_value in value.items():
                if key == "bbox_3d" and isinstance(nested_value, (list, tuple)) and len(nested_value) == 9:
                    refined_bbox, used_count = refine_bbox_values_with_residuals(
                        nested_value,
                        residual_values[residual_index:],
                        decimal_places=decimal_places,
                    )
                    new_payload[key] = refined_bbox
                    residual_index += used_count
                else:
                    new_payload[key] = _refine(nested_value)
            return new_payload

        if isinstance(value, list):
            return [_refine(item) for item in value]

        if isinstance(value, tuple):
            return [_refine(item) for item in value]

        return value

    return _refine(copy.deepcopy(payload)), residual_index


def refine_bbox_response_with_residuals(
    text: str,
    residual_values: Sequence[float],
    decimal_places: int = 4,
) -> str:
    try:
        payload = parse_json_like_with_special_tokens(text)
    except Exception:
        return text

    refined_payload, used_residuals = refine_bbox_payload_with_residuals(
        payload,
        residual_values,
        decimal_places=decimal_places,
    )
    if used_residuals == 0:
        return text
    return format_bbox_payload(refined_payload, fenced=text.strip().startswith("```"))
