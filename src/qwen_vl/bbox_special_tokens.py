import ast
import copy
import json
import math
import re
from dataclasses import dataclass
from typing import Any, List, Sequence


BBOX_QUANTIZATION_STEP = 0.03

POSITION_PROMPT_TOKEN = "|position|"
SIZE_PROMPT_TOKEN = "|size|"
ANGLE_PROMPT_TOKEN = "|angle|"
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
_TOKEN_LITERAL_PATTERN = re.compile(
    r'(?<!["\'])\|(position|size|angle)(?:_\d+)?\|(?!["\'])'
)


def get_bbox_special_tokens(include_prompt_placeholders: bool = True) -> List[str]:
    tokens = []
    if include_prompt_placeholders:
        tokens.extend(PROMPT_PLACEHOLDER_TOKENS)
    for family, num_bins in _FAMILY_TO_MAX_BINS.items():
        tokens.extend([f"|{family}_{idx}|" for idx in range(num_bins)])
    return tokens


def get_bbox_coordinate_tokens() -> List[str]:
    return get_bbox_special_tokens(include_prompt_placeholders=False)


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
    new_tokens = list(dict.fromkeys(get_bbox_coordinate_tokens()))
    return tokenizer.add_tokens(new_tokens)


def resize_model_embeddings_for_bbox_tokens(model, tokenizer, num_new_tokens: int) -> None:
    if num_new_tokens <= 0:
        return

    model.resize_token_embeddings(len(tokenizer))

    input_embeddings = model.get_input_embeddings().weight.data
    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    input_embeddings[-num_new_tokens:] = input_embeddings_avg

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        output_weight = output_embeddings.weight.data
        output_embeddings_avg = output_weight[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_weight[-num_new_tokens:] = output_embeddings_avg

    model.config.vocab_size = len(tokenizer)
    setattr(model.config, "use_bbox_special_tokens", True)
    setattr(model.config, "bbox_coordinate_token_ids", get_bbox_coordinate_token_ids(tokenizer))
    setattr(model.config, "bbox_coordinate_label_smoothing", 0.1)


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


def quantize_position_triplet(position: Sequence[float]) -> List[str]:
    if len(position) != 3:
        raise ValueError(f"Expected 3 position values, got {len(position)}")
    return [quantize_value_to_token(value, POSITION_SPEC) for value in position]


def format_position_triplet_for_prompt(position: Sequence[float]) -> str:
    return json.dumps(quantize_position_triplet(position), ensure_ascii=False)


def quantize_bbox_values(bbox_3d: Sequence[float]) -> List[str]:
    if len(bbox_3d) != 9:
        raise ValueError(f"Expected 9 bbox values, got {len(bbox_3d)}")
    return [
        quantize_value_to_token(value, spec)
        for value, spec in zip(bbox_3d, BBOX_QUANTIZATION_SPECS)
    ]


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


def rewrite_scan2cap_prompt_with_position_tokens(text: str) -> str:
    if "|position_" in text:
        return text

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


def format_quantized_bbox_payload(payload: Any, fenced: bool = True) -> str:
    quantized_payload = quantize_bbox_payload(copy.deepcopy(payload))
    json_text = json.dumps(quantized_payload, ensure_ascii=False, indent=2)
    if fenced:
        return f"```json\n{json_text}\n```"
    return json_text


def rewrite_bbox_response_with_special_tokens(text: str) -> str:
    if "|position_" in text or "|size_" in text or "|angle_" in text:
        return text

    try:
        payload = parse_json_like_with_special_tokens(text)
    except Exception:
        return text

    return format_quantized_bbox_payload(payload, fenced=text.strip().startswith("```"))
