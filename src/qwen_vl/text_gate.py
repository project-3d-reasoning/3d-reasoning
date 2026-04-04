from typing import Any, Optional


DEFAULT_TEXT_GATE_SENTENCE_BERT_NAME_OR_PATH = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TEXT_GATE_SENTENCE_BERT_MAX_LENGTH = 64


def _read_optional_name_or_path(source: Any, attr_name: str) -> Optional[str]:
    if not hasattr(source, attr_name):
        return None

    value = getattr(source, attr_name)
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return ""
    return str(value)


def resolve_text_gate_sentence_bert_name_or_path(source: Any) -> Optional[str]:
    for attr_name in (
        "text_gate_sentence_bert_name_or_path",
        "text_gate_bert_name_or_path",
    ):
        value = _read_optional_name_or_path(source, attr_name)
        if value is None:
            continue
        if value == "":
            return None
        return value
    return DEFAULT_TEXT_GATE_SENTENCE_BERT_NAME_OR_PATH


def resolve_text_gate_sentence_bert_max_length(source: Any) -> int:
    for attr_name in (
        "text_gate_sentence_bert_max_length",
        "text_gate_bert_max_length",
    ):
        if not hasattr(source, attr_name):
            continue
        value = getattr(source, attr_name)
        if value is None:
            continue
        return int(value)
    return DEFAULT_TEXT_GATE_SENTENCE_BERT_MAX_LENGTH


def resolve_text_gate_sentence_bert_cache_dir(source: Any) -> Optional[str]:
    for attr_name in (
        "text_gate_sentence_bert_cache_dir",
        "text_gate_bert_cache_dir",
    ):
        if not hasattr(source, attr_name):
            continue
        value = getattr(source, attr_name)
        if value is not None:
            return value
    return None


def apply_text_gate_sentence_bert_config(
    config: Any,
    *,
    name_or_path: Optional[str] = None,
    max_length: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> None:
    if isinstance(name_or_path, str):
        name_or_path = name_or_path.strip() or None
    resolved_name_or_path = (
        name_or_path
        if name_or_path is not None
        else resolve_text_gate_sentence_bert_name_or_path(config)
    )
    resolved_max_length = (
        int(max_length)
        if max_length is not None
        else resolve_text_gate_sentence_bert_max_length(config)
    )
    resolved_cache_dir = (
        cache_dir
        if cache_dir is not None
        else resolve_text_gate_sentence_bert_cache_dir(config)
    )

    setattr(config, "text_gate_sentence_bert_name_or_path", resolved_name_or_path)
    setattr(config, "text_gate_sentence_bert_max_length", resolved_max_length)
    setattr(config, "text_gate_bert_name_or_path", resolved_name_or_path)
    setattr(config, "text_gate_bert_max_length", resolved_max_length)

    if resolved_cache_dir is not None:
        setattr(config, "text_gate_sentence_bert_cache_dir", resolved_cache_dir)
        setattr(config, "text_gate_bert_cache_dir", resolved_cache_dir)
