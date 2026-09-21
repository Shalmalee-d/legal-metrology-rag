"""Small, environment-based settings shared by the RAG service."""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def get_data_root() -> Path:
    """Root for all runtime data. Defaults to ``data`` for production."""
    return Path(os.getenv("RAG_DATA_ROOT", "data"))


def get_registry_dir() -> Path:
    """Directory holding registry + update-status files."""
    override = os.getenv("RAG_REGISTRY_DIR")
    if override:
        return Path(override)
    return get_data_root() / "registry"


def get_registry_file() -> Path:
    """Configured document-registry path. Defaults to production registry."""
    override = os.getenv("RAG_REGISTRY_FILE")
    if override:
        return Path(override)
    return get_registry_dir() / "document_registry.json"


def get_update_status_file() -> Path:
    """Configured update-status path."""
    override = os.getenv("RAG_UPDATE_STATUS_FILE")
    if override:
        return Path(override)
    return get_registry_dir() / "update_status.json"


def get_documents_dir() -> Path:
    """Directory holding downloaded PDFs."""
    override = os.getenv("RAG_DOCUMENTS_DIR")
    if override:
        return Path(override)
    return get_data_root() / "documents"


def get_metadata_file() -> Path:
    """Configured fetcher metadata path."""
    override = os.getenv("RAG_METADATA_FILE")
    if override:
        return Path(override)
    return get_documents_dir() / "metadata.json"


def get_rules_dir() -> Path:
    """Directory holding the rule repository."""
    override = os.getenv("RAG_RULES_DIR")
    if override:
        return Path(override)
    return get_data_root() / "rules"


def get_rules_file() -> Path:
    """Configured compliance-rules path."""
    override = os.getenv("RAG_RULES_FILE")
    if override:
        return Path(override)
    return get_rules_dir() / "compliance_rules.json"


def get_extracted_dir() -> Path:
    """Directory holding extracted page JSON."""
    override = os.getenv("RAG_EXTRACTED_DIR")
    if override:
        return Path(override)
    return get_data_root() / "extracted"


def get_tesseract_command() -> str | None:
    """Return an optional executable override; PATH remains the default."""
    return os.getenv("TESSERACT_CMD") or None


def get_ocr_languages() -> str:
    """Tesseract language list, e.g. ``eng`` or ``eng+hin``."""
    return os.getenv("TESSERACT_LANG", "eng+hin")


def _positive_float_setting(name: str, default: str) -> float:
    value = os.getenv(name, default)
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number.") from error
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


def get_ocr_render_scale() -> float:
    """PDF render scale used for OCR (three is a quality-oriented default)."""
    return _positive_float_setting("OCR_RENDER_SCALE", "3")


def get_ocr_timeout_seconds() -> float:
    """Maximum time Tesseract may spend on a single page."""
    return _positive_float_setting("OCR_TIMEOUT_SECONDS", "120")


def get_ocr_min_alphanumeric_ratio() -> float:
    """Minimum visible OCR characters that must be letters or digits."""
    return _positive_float_setting("OCR_MIN_ALPHANUMERIC_RATIO", "0.35")


def get_ocr_max_replacement_ratio() -> float:
    """Maximum replacement-character ratio tolerated in OCR text."""
    return _positive_float_setting("OCR_MAX_REPLACEMENT_RATIO", "0.02")


def get_ocr_max_malformed_devanagari_ratio() -> float:
    """Maximum malformed Devanagari mark ratio tolerated in OCR text."""
    return _positive_float_setting("OCR_MAX_MALFORMED_DEVANAGARI_RATIO", "0.08")


def get_ocr_min_devanagari_ratio() -> float:
    """Minimum Devanagari ratio when an OCR request is Hindi-only."""
    return _positive_float_setting("OCR_MIN_DEVANAGARI_RATIO", "0.01")


def get_check_interval_days() -> float:
    """Weekly regulatory update interval; configurable for deployments/tests."""
    return _positive_float_setting("RAG_CHECK_INTERVAL_DAYS", "7")


def get_rag_chunk_size() -> int:
    """Maximum regulatory-text characters supplied to one model request."""
    return int(_positive_float_setting("RAG_CHUNK_SIZE", "2400"))


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "qwen3:4b")


def get_ollama_timeout_seconds() -> float:
    return _positive_float_setting("OLLAMA_TIMEOUT_SECONDS", "180")


def get_ollama_connect_timeout_seconds() -> float:
    """Fail quickly when the local Ollama service is unavailable."""
    return _positive_float_setting("OLLAMA_CONNECT_TIMEOUT_SECONDS", "10")


def get_ollama_max_output_tokens() -> int:
    """Bound one compact structured extraction response from the local model."""
    return int(_positive_float_setting("OLLAMA_MAX_OUTPUT_TOKENS", "256"))
