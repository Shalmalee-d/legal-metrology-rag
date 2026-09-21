import json
import logging
import shutil
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import pymupdf
import pytesseract
from PIL import Image
from pytesseract import TesseractNotFoundError

from app.config import (
    get_ocr_languages,
    get_ocr_max_malformed_devanagari_ratio,
    get_ocr_max_replacement_ratio,
    get_ocr_min_alphanumeric_ratio,
    get_ocr_min_devanagari_ratio,
    get_ocr_render_scale,
    get_ocr_timeout_seconds,
    get_tesseract_command,
)


EXTRACTED_DIR = Path("data/extracted")
_DEFAULT_EXTRACTED_DIR = Path("data/extracted")


def _resolve_extracted_dir() -> Path:
    if EXTRACTED_DIR != _DEFAULT_EXTRACTED_DIR:
        return EXTRACTED_DIR
    from app.config import get_extracted_dir

    return get_extracted_dir()

MIN_DIRECT_TEXT_CHARACTERS = 20

logger = logging.getLogger(__name__)

OCR_GOOD = "GOOD"
OCR_DEGRADED = "DEGRADED"
OCR_UNAVAILABLE = "OCR_UNAVAILABLE"
LANGUAGE_AVAILABLE = "AVAILABLE"
LANGUAGE_UNAVAILABLE = "LANGUAGE_UNAVAILABLE"
NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class OcrLanguageResolution:
    languages: str
    status: str
    requested: str
    hindi_requested: bool


def configure_tesseract() -> None:
    """Use an explicit override when supplied, otherwise rely on PATH."""
    command = get_tesseract_command()
    if command:
        pytesseract.pytesseract.tesseract_cmd = command
    elif not shutil.which("tesseract"):
        raise RuntimeError("Tesseract was not found. Install it and add it to PATH, or set TESSERACT_CMD.")


def available_ocr_languages() -> set[str]:
    configure_tesseract()
    try:
        return set(pytesseract.get_languages(config=""))
    except TesseractNotFoundError as error:
        raise RuntimeError("Tesseract was not found. Set TESSERACT_CMD or update PATH.") from error


def _resolve_ocr_language_details(requested: str | None = None) -> OcrLanguageResolution:
    """Resolve installed OCR languages while retaining Hindi availability state."""
    requested = requested or get_ocr_languages()
    available = available_ocr_languages()
    usable = [part for part in requested.split("+") if part and part in available]
    hindi_requested = "hin" in requested.split("+")
    hindi_available = "hin" in available
    if hindi_requested and not hindi_available:
        warnings.warn(
            "Hindi OCR requested but hin.traineddata is not installed; OCR pages will be marked LANGUAGE_UNAVAILABLE.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not usable:
        raise RuntimeError(f"None of the requested OCR languages ({requested}) are installed. Available: {', '.join(sorted(available)) or 'none'}.")
    return OcrLanguageResolution(
        languages="+".join(usable),
        status=LANGUAGE_AVAILABLE if not hindi_requested or hindi_available else LANGUAGE_UNAVAILABLE,
        requested=requested,
        hindi_requested=hindi_requested,
    )


def resolve_ocr_languages(requested: str | None = None) -> str:
    """Backward-compatible language resolver for callers needing only a string."""
    return _resolve_ocr_language_details(requested).languages


def _is_devanagari(character: str) -> bool:
    return "\u0900" <= character <= "\u097f"


def _malformed_devanagari_marks(text: str) -> int:
    """Count combining marks that cannot belong to a preceding Devanagari base."""
    malformed = 0
    previous_was_devanagari_base = False
    for character in text:
        if not _is_devanagari(character):
            previous_was_devanagari_base = False
            continue
        if unicodedata.category(character).startswith("M"):
            if not previous_was_devanagari_base:
                malformed += 1
            continue
        previous_was_devanagari_base = True
    return malformed


def assess_ocr_quality(text: str, *, hindi_expected: bool = False) -> str:
    """Classify obvious transcription failures without attempting legal-text repair."""
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return OCR_DEGRADED

    replacement_ratio = text.count("\ufffd") / len(visible)
    if replacement_ratio > get_ocr_max_replacement_ratio():
        return OCR_DEGRADED

    # This is the characteristic UTF-8-as-Windows-1252 sequence. It is
    # flagged for review, never converted or guessed back into legal text.
    if "à¤" in text or "à¥" in text:
        return OCR_DEGRADED

    alphanumeric_ratio = sum(character.isalnum() for character in visible) / len(visible)
    if alphanumeric_ratio < get_ocr_min_alphanumeric_ratio():
        return OCR_DEGRADED

    devanagari_count = sum(_is_devanagari(character) for character in visible)
    if hindi_expected and devanagari_count / len(visible) < get_ocr_min_devanagari_ratio():
        return OCR_DEGRADED
    if devanagari_count:
        malformed_ratio = _malformed_devanagari_marks(text) / devanagari_count
        if malformed_ratio > get_ocr_max_malformed_devanagari_ratio():
            return OCR_DEGRADED

    return OCR_GOOD


def _ocr_page(page: pymupdf.Page, languages: str | None = None) -> str:
    scale = get_ocr_render_scale()
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    try:
        return pytesseract.image_to_string(
            image,
            lang=languages or resolve_ocr_languages(),
            timeout=get_ocr_timeout_seconds(),
        ).strip()
    except RuntimeError as error:
        raise RuntimeError(
            f"OCR timed out or failed for a PDF page after "
            f"{get_ocr_timeout_seconds()} seconds."
        ) from error


def extract_pdf_text(
    pdf_path: str,
    ocr_languages: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Extract each page with provenance and OCR quality metadata."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    try:
        document = pymupdf.open(pdf_path)
    except pymupdf.FileDataError as error:
        raise ValueError(f"Could not read PDF: {pdf_path}") from error

    pages = []
    language_resolution: OcrLanguageResolution | None = None
    language_error: str | None = None
    try:
        total_pages = len(document)
        for page_number, page in enumerate(document, start=1):
            if progress_callback:
                progress_callback(page_number, total_pages)
            direct_text = page.get_text("text").strip()
            if len(direct_text) >= MIN_DIRECT_TEXT_CHARACTERS:
                pages.append({
                    "page": page_number,
                    "text": direct_text,
                    "method": "direct",
                    "extraction_method": "direct_text",
                    "ocr_languages": None,
                    "ocr_status": OCR_GOOD,
                    "ocr_language_status": NOT_APPLICABLE,
                })
            else:
                # Querying installed Tesseract languages launches a subprocess.
                # Resolve it once per document rather than once per scanned page.
                if language_resolution is None and language_error is None:
                    try:
                        language_resolution = _resolve_ocr_language_details(ocr_languages)
                    except RuntimeError as error:
                        language_error = str(error)
                        logger.warning("OCR_UNAVAILABLE page=%s error=%s", page_number, error)

                if language_error:
                    pages.append({
                        "page": page_number,
                        "text": "",
                        "method": "ocr",
                        "extraction_method": "tesseract_ocr",
                        "ocr_languages": None,
                        "ocr_status": OCR_UNAVAILABLE,
                        "ocr_language_status": OCR_UNAVAILABLE,
                        "ocr_error": language_error,
                    })
                    continue

                assert language_resolution is not None
                try:
                    text = _ocr_page(page, language_resolution.languages)
                except RuntimeError as error:
                    logger.warning("OCR_UNAVAILABLE page=%s error=%s", page_number, error)
                    pages.append({
                        "page": page_number,
                        "text": "",
                        "method": "ocr",
                        "extraction_method": "tesseract_ocr",
                        "ocr_languages": language_resolution.languages,
                        "ocr_status": OCR_UNAVAILABLE,
                        "ocr_language_status": language_resolution.status,
                        "ocr_error": str(error),
                    })
                    continue

                # A Hindi-only request has an explicit script expectation. A
                # mixed eng+hin request remains valid for English-only pages.
                hindi_expected = language_resolution.requested.split("+") == ["hin"]
                quality = assess_ocr_quality(text, hindi_expected=hindi_expected)
                if quality == OCR_DEGRADED:
                    logger.warning(
                        "OCR_DEGRADED page=%s languages=%s language_status=%s",
                        page_number,
                        language_resolution.languages,
                        language_resolution.status,
                    )
                elif language_resolution.status == LANGUAGE_UNAVAILABLE:
                    logger.warning(
                        "OCR_LANGUAGE_UNAVAILABLE page=%s requested=%s using=%s",
                        page_number,
                        language_resolution.requested,
                        language_resolution.languages,
                    )
                pages.append({
                    "page": page_number,
                    "text": text,
                    "method": "ocr",
                    "extraction_method": "tesseract_ocr",
                    "ocr_languages": language_resolution.languages,
                    "ocr_status": quality,
                    "ocr_language_status": language_resolution.status,
                })
    finally:
        document.close()

    return pages


def save_extracted_text(pdf_path: str, pages: list[dict]) -> Path:
    extracted_dir = _resolve_extracted_dir()
    extracted_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = Path(pdf_path).stem
    output_path = extracted_dir / f"{pdf_name}.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            pages,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path
