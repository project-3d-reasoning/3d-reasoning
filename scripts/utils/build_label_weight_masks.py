#!/usr/bin/env python3
"""Build token-aligned label-weight masks for 3D training datasets."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_vl.label_weighting import (  # noqa: E402
    LABEL_WEIGHT_CODE_SCANNET_DET_BBOX,
    LABEL_WEIGHT_CODE_SCANNET_DET_LABEL,
    LABEL_WEIGHT_CODE_SCAN2CAP_ATTRIBUTE,
    LABEL_WEIGHT_CODE_SCAN2CAP_CATEGORY,
    LABEL_WEIGHT_CODE_SCAN2CAP_RELATION,
    LABEL_WEIGHT_CODE_SCANREFER_BBOX,
    LABEL_WEIGHT_CODE_SCANREFER_FRAME,
    LABEL_WEIGHT_CODE_TO_NAME,
    LABEL_WEIGHT_MASK_SUFFIX,
    SCAN2CAP_LEXICON_FILENAME,
)


NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
SCANREFER_FRAME_PATTERN = re.compile(r'"frame"\s*:\s*(-?\d+)')
JSON_LIST_LABEL_PATTERN = re.compile(r'"label"\s*:\s*"([^"]+)"')
JSON_BBOX_PATTERN = re.compile(r'"bbox_3d"\s*:\s*\[([^\]]*)\]', re.DOTALL)

SCAN2CAP_RELATION_PATTERNS = [
    "next to",
    "beside",
    "near",
    "on top of",
    "on the top of",
    "on the left side of",
    "on the right side of",
    "on the left of",
    "on the right of",
    "left side of",
    "right side of",
    "left of",
    "right of",
    "in front of",
    "behind",
    "across from",
    "in the corner",
    "in the center",
    "in the center of",
    "in the middle",
    "in the middle of",
    "on the side of",
    "at the side of",
    "on the bottom part",
    "at the bottom of",
    "at the top of",
    "under",
    "below",
    "above",
    "between",
    "against",
    "inside",
    "outside",
    "from the door",
    "from the entrance",
    "from the inside",
    "from the outside",
]

# A lightweight adjective lexicon for Scan2Cap object descriptions.
SCAN2CAP_ATTRIBUTE_WORDS = {
    "white",
    "black",
    "brown",
    "beige",
    "grey",
    "gray",
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "pink",
    "purple",
    "silver",
    "gold",
    "wooden",
    "wood",
    "plastic",
    "metal",
    "metallic",
    "glass",
    "fabric",
    "leather",
    "round",
    "rectangular",
    "square",
    "small",
    "large",
    "big",
    "tiny",
    "long",
    "short",
    "tall",
    "low",
    "high",
    "double",
    "single",
    "greenish",
    "yellowish",
    "designed",
}

SCAN2CAP_CATEGORY_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "on",
    "in",
    "at",
    "from",
    "to",
    "left",
    "right",
    "side",
    "top",
    "bottom",
    "corner",
    "center",
    "middle",
    "inside",
    "outside",
    "front",
    "behind",
    "between",
    "near",
    "above",
    "below",
    "against",
    "next",
    "beside",
    "across",
    "direction",
}

SCAN2CAP_PREFIX_PATTERN = re.compile(
    r"^(?:there is|there are|it is|this is|these are|that is|those are|a|an|the)\s+",
    re.IGNORECASE,
)


@dataclass
class SpanMatch:
    start_char: int
    end_char: int
    code: int


@dataclass
class CompiledPhraseMatcher:
    code: int
    patterns: List[re.Pattern]


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
                self._offsets = [(int(s), int(e)) for s, e in offsets]
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
        prefix = self.text[:char_idx]
        count = len(self.tokenizer.encode(prefix, add_special_tokens=False))
        self._prefix_cache[char_idx] = count
        return count


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def strip_scan2cap_prefixes(text: str) -> str:
    text = normalize_text(text)
    while True:
        stripped = SCAN2CAP_PREFIX_PATTERN.sub("", text).strip()
        if stripped == text:
            return stripped
        text = stripped


def is_valid_scan2cap_category_phrase(phrase: Optional[str]) -> bool:
    if not phrase:
        return False
    tokens = [token for token in re.findall(r"[a-z]+(?:-[a-z]+)?", phrase.lower()) if token]
    if not tokens:
        return False
    if all(token in SCAN2CAP_ATTRIBUTE_WORDS for token in tokens):
        return False
    content_tokens = [token for token in tokens if token not in SCAN2CAP_CATEGORY_STOPWORDS]
    if not content_tokens:
        return False
    if all(token in SCAN2CAP_ATTRIBUTE_WORDS for token in content_tokens):
        return False
    normalized_phrase = " ".join(tokens)
    if normalized_phrase in SCAN2CAP_RELATION_PATTERNS:
        return False
    return True


def get_assistant_text(sample: Dict) -> str:
    conversations = sample.get("conversations", [])
    for message in reversed(conversations):
        role = message.get("from", message.get("role", ""))
        if role in {"gpt", "assistant"}:
            return str(message.get("value", message.get("content", "")))
    raise ValueError("Sample does not contain an assistant response.")


def build_assistant_codes(tokenizer, assistant_text: str, spans: Sequence[SpanMatch]) -> torch.Tensor:
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


def collect_number_spans(inner_text: str, inner_start: int, code: int) -> List[SpanMatch]:
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


def build_scanrefer_spans(assistant_text: str) -> List[SpanMatch]:
    spans: List[SpanMatch] = []
    for match in SCANREFER_FRAME_PATTERN.finditer(assistant_text):
        spans.append(
            SpanMatch(
                start_char=match.start(1),
                end_char=match.end(1),
                code=LABEL_WEIGHT_CODE_SCANREFER_FRAME,
            )
        )
    for match in JSON_BBOX_PATTERN.finditer(assistant_text):
        spans.extend(
            collect_number_spans(
                inner_text=match.group(1),
                inner_start=match.start(1),
                code=LABEL_WEIGHT_CODE_SCANREFER_BBOX,
            )
        )
    return spans


def build_scannet_det_spans(assistant_text: str) -> List[SpanMatch]:
    spans: List[SpanMatch] = []
    for match in JSON_LIST_LABEL_PATTERN.finditer(assistant_text):
        spans.append(
            SpanMatch(
                start_char=match.start(1),
                end_char=match.end(1),
                code=LABEL_WEIGHT_CODE_SCANNET_DET_LABEL,
            )
        )
    for match in JSON_BBOX_PATTERN.finditer(assistant_text):
        spans.extend(
            collect_number_spans(
                inner_text=match.group(1),
                inner_start=match.start(1),
                code=LABEL_WEIGHT_CODE_SCANNET_DET_BBOX,
            )
        )
    return spans


def extract_scan2cap_primary_phrase(text: str) -> Tuple[List[str], Optional[str]]:
    text = normalize_text(text).lower()
    if not text:
        return [], None
    text = strip_scan2cap_prefixes(text).lower()
    clause = re.split(r"[.;,]", text, maxsplit=1)[0].strip()
    if not clause:
        return [], None

    for relation in sorted(SCAN2CAP_RELATION_PATTERNS, key=len, reverse=True):
        idx = clause.find(relation)
        if idx > 0:
            clause = clause[:idx].strip()
            break

    tokens = [token for token in re.findall(r"[a-z]+(?:-[a-z]+)?", clause) if token]
    if not tokens:
        return [], None

    attributes: List[str] = []
    while len(tokens) > 1 and tokens[0] in SCAN2CAP_ATTRIBUTE_WORDS:
        attributes.append(tokens.pop(0))

    category = " ".join(tokens).strip()
    if not category:
        return attributes, None
    return attributes, category


def build_scan2cap_lexicon(
    samples: Sequence[Dict],
    progress_desc: str = "Scan2Cap lexicon",
) -> Dict[str, List[Dict[str, int]]]:
    category_counts: Counter[str] = Counter()
    attribute_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()

    for sample in tqdm(samples, desc=progress_desc, unit="sample", dynamic_ncols=True):
        assistant_text = get_assistant_text(sample)
        normalized = normalize_text(assistant_text).lower()

        attributes, category = extract_scan2cap_primary_phrase(normalized)
        if is_valid_scan2cap_category_phrase(category):
            category_counts[category] += 1
        for attribute in attributes:
            attribute_counts[attribute] += 1
        for relation in SCAN2CAP_RELATION_PATTERNS:
            if relation in normalized:
                relation_counts[relation] += normalized.count(relation)

    def counter_to_list(counter: Counter[str]) -> List[Dict[str, int]]:
        return [
            {"phrase": phrase, "count": int(count)}
            for phrase, count in counter.most_common()
        ]

    return {
        "categories": counter_to_list(category_counts),
        "attributes": counter_to_list(attribute_counts),
        "relations": counter_to_list(relation_counts),
    }


def compile_phrase_matcher(
    phrases: Iterable[str],
    code: int,
    chunk_size: int = 2048,
) -> CompiledPhraseMatcher:
    normalized_phrases = sorted(
        {
            normalize_text(phrase)
            for phrase in phrases
            if normalize_text(phrase)
        },
        key=lambda phrase: (-len(phrase), phrase),
    )
    patterns: List[re.Pattern] = []
    for start_idx in range(0, len(normalized_phrases), chunk_size):
        chunk = normalized_phrases[start_idx : start_idx + chunk_size]
        if not chunk:
            continue
        chunk_pattern = r"(?<!\w)(?:%s)(?!\w)" % "|".join(re.escape(phrase) for phrase in chunk)
        patterns.append(re.compile(chunk_pattern, re.IGNORECASE))
    return CompiledPhraseMatcher(code=code, patterns=patterns)


def iter_phrase_spans(text: str, matcher: CompiledPhraseMatcher) -> List[SpanMatch]:
    spans: List[SpanMatch] = []
    for pattern in matcher.patterns:
        for match in pattern.finditer(text):
            spans.append(
                SpanMatch(
                    start_char=match.start(),
                    end_char=match.end(),
                    code=matcher.code,
                )
            )
    return spans


def build_scan2cap_matchers(lexicon: Dict[str, List[Dict[str, int]]]) -> List[CompiledPhraseMatcher]:
    category_phrases = [
        item["phrase"]
        for item in lexicon.get("categories", [])
        if is_valid_scan2cap_category_phrase(item.get("phrase"))
    ]
    attribute_phrases = [item["phrase"] for item in lexicon.get("attributes", []) if item.get("phrase")]
    relation_phrases = [item["phrase"] for item in lexicon.get("relations", []) if item.get("phrase")]

    return [
        compile_phrase_matcher(category_phrases, LABEL_WEIGHT_CODE_SCAN2CAP_CATEGORY),
        compile_phrase_matcher(attribute_phrases, LABEL_WEIGHT_CODE_SCAN2CAP_ATTRIBUTE),
        compile_phrase_matcher(relation_phrases, LABEL_WEIGHT_CODE_SCAN2CAP_RELATION),
    ]


def build_scan2cap_spans(
    assistant_text: str,
    lexicon: Optional[Dict[str, List[Dict[str, int]]]] = None,
    matchers: Optional[Sequence[CompiledPhraseMatcher]] = None,
) -> List[SpanMatch]:
    if matchers is None:
        matchers = build_scan2cap_matchers(lexicon or {})

    spans: List[SpanMatch] = []
    for matcher in matchers:
        spans.extend(iter_phrase_spans(assistant_text, matcher))
    return spans


def build_dataset_payload(code_tensors: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
    offsets = torch.zeros(len(code_tensors) + 1, dtype=torch.long)
    if code_tensors:
        lengths = torch.tensor([int(t.numel()) for t in code_tensors], dtype=torch.long)
        offsets[1:] = torch.cumsum(lengths, dim=0)
        flat_codes = torch.cat([tensor.to(torch.uint8) for tensor in code_tensors], dim=0)
    else:
        flat_codes = torch.empty((0,), dtype=torch.uint8)
    return {
        "flat_codes": flat_codes,
        "offsets": offsets,
        "code_to_name": dict(LABEL_WEIGHT_CODE_TO_NAME),
    }


def save_payload(
    output_path: Path,
    annotation_path: str,
    code_tensors: Sequence[torch.Tensor],
    extra_metadata: Optional[Dict[str, object]] = None,
) -> None:
    payload = build_dataset_payload(code_tensors)
    payload.update(
        {
            "version": 1,
            "annotation_path": annotation_path,
            "mask_type": "assistant_token_codes",
        }
    )
    if extra_metadata:
        payload.update(extra_metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_codes_for_samples(
    samples: Sequence[Dict],
    tokenizer,
    span_builder,
    progress_desc: str,
) -> List[torch.Tensor]:
    code_tensors: List[torch.Tensor] = []
    for sample in tqdm(samples, desc=progress_desc, unit="sample", dynamic_ncols=True):
        assistant_text = get_assistant_text(sample)
        spans = span_builder(assistant_text)
        code_tensors.append(build_assistant_codes(tokenizer, assistant_text, spans))
    return code_tensors


def build_output_path(output_dir: Path, dataset_path: str) -> Path:
    base_name = os.path.splitext(os.path.basename(dataset_path))[0]
    return output_dir / f"{base_name}{LABEL_WEIGHT_MASK_SUFFIX}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--scanrefer_path", type=str, default="")
    parser.add_argument("--scan2cap_path", type=str, default="")
    parser.add_argument("--scannet_det_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scan2cap_lexicon_path", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Deprecated alias. Fast tokenizer is used by default.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="Force the slow tokenizer implementation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_fast = True
    if args.use_slow_tokenizer:
        use_fast = False
    elif args.use_fast_tokenizer:
        use_fast = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=use_fast,
    )

    scan2cap_lexicon = None
    scan2cap_matchers: Optional[List[CompiledPhraseMatcher]] = None
    if args.scan2cap_path:
        scan2cap_lexicon_path = (
            Path(args.scan2cap_lexicon_path)
            if args.scan2cap_lexicon_path
            else output_dir / SCAN2CAP_LEXICON_FILENAME
        )
        if args.overwrite or not scan2cap_lexicon_path.exists():
            scan2cap_samples = load_json(args.scan2cap_path)
            scan2cap_lexicon = build_scan2cap_lexicon(
                scan2cap_samples,
                progress_desc="Scan2Cap lexicon",
            )
            save_json(scan2cap_lexicon_path, scan2cap_lexicon)
            print(f"Saved Scan2Cap lexicon to {scan2cap_lexicon_path}")
        else:
            with open(scan2cap_lexicon_path, "r", encoding="utf-8") as handle:
                scan2cap_lexicon = json.load(handle)
            print(f"Loaded existing Scan2Cap lexicon from {scan2cap_lexicon_path}")
        scan2cap_matchers = build_scan2cap_matchers(scan2cap_lexicon or {})

    if args.scanrefer_path:
        output_path = build_output_path(output_dir, args.scanrefer_path)
        if args.overwrite or not output_path.exists():
            samples = load_json(args.scanrefer_path)
            code_tensors = build_codes_for_samples(
                samples,
                tokenizer,
                build_scanrefer_spans,
                progress_desc="ScanRefer masks",
            )
            save_payload(
                output_path,
                args.scanrefer_path,
                code_tensors,
                extra_metadata={"dataset_name": "scanrefer"},
            )
            print(f"Saved ScanRefer label-weight masks to {output_path}")
        else:
            print(f"Skip existing ScanRefer mask file: {output_path}")

    if args.scan2cap_path:
        output_path = build_output_path(output_dir, args.scan2cap_path)
        if args.overwrite or not output_path.exists():
            samples = load_json(args.scan2cap_path)
            code_tensors = build_codes_for_samples(
                samples,
                tokenizer,
                lambda text: build_scan2cap_spans(text, matchers=scan2cap_matchers or []),
                progress_desc="Scan2Cap masks",
            )
            save_payload(
                output_path,
                args.scan2cap_path,
                code_tensors,
                extra_metadata={
                    "dataset_name": "scan2cap",
                    "scan2cap_lexicon_path": str(
                        Path(args.scan2cap_lexicon_path)
                        if args.scan2cap_lexicon_path
                        else output_dir / SCAN2CAP_LEXICON_FILENAME
                    ),
                },
            )
            print(f"Saved Scan2Cap label-weight masks to {output_path}")
        else:
            print(f"Skip existing Scan2Cap mask file: {output_path}")

    if args.scannet_det_path:
        output_path = build_output_path(output_dir, args.scannet_det_path)
        if args.overwrite or not output_path.exists():
            samples = load_json(args.scannet_det_path)
            code_tensors = build_codes_for_samples(
                samples,
                tokenizer,
                build_scannet_det_spans,
                progress_desc="ScanNetDet masks",
            )
            save_payload(
                output_path,
                args.scannet_det_path,
                code_tensors,
                extra_metadata={"dataset_name": "scannet_det"},
            )
            print(f"Saved ScanNetDet label-weight masks to {output_path}")
        else:
            print(f"Skip existing ScanNetDet mask file: {output_path}")


if __name__ == "__main__":
    main()
