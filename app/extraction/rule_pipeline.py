import hashlib
import inspect
import logging
import re

from app.extraction.rule_extractor import extract_rules_from_chunk
from app.extraction.pdf_extractor import extract_pdf_text, save_extracted_text
from app.extraction.chunker import chunk_pages
from app.validation.rule_validator import validate_rule
from app.repository.rule_repository import add_rule, update_rules


logger = logging.getLogger(__name__)


class OcrQualityError(ValueError):
    """Raised when no trustworthy extracted page remains for rule generation."""


# Explicit document processing outcomes. Every "zero rules" document is
# classified so a genuine no-rule document is never confused with a failure.
SUCCESS_WITH_RULES = "SUCCESS_WITH_RULES"
NO_RULES_FOUND = "NO_RULES_FOUND"
VALIDATION_REJECTED_ALL = "VALIDATION_REJECTED_ALL"
EXTRACTION_ERROR = "EXTRACTION_ERROR"
OCR_ERROR = "OCR_ERROR"

_MAX_OUTCOME_REASONS = 5
_MAX_OUTCOME_REASON_CHARS = 200


def _new_outcome() -> dict:
    return {
        "status": None,
        "candidate_count": 0,
        "chunks_processed": 0,
        "chunks_with_candidates": 0,
        "rules_generated": 0,
        "rules_validated": 0,
        "rules_rejected": 0,
        "model_errors": [],
        "validation_rejection_reasons": [],
    }


def _record_rejection(outcome: dict, reason: str) -> None:
    if len(outcome["validation_rejection_reasons"]) < _MAX_OUTCOME_REASONS:
        outcome["validation_rejection_reasons"].append(reason[:_MAX_OUTCOME_REASON_CHARS])


def _extractor_accepts_diagnostics(extractor) -> bool:
    """Whether an extractor callable accepts an optional ``diagnostics`` kwarg."""
    try:
        parameters = inspect.signature(extractor).parameters.values()
    except (TypeError, ValueError):
        return False
    for parameter in parameters:
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "diagnostics":
            return True
    return False

def process_chunk(
    chunk: dict,
    source_document: str,
) -> list:
    """
    Extract compliance rules from a chunk and return
    only rules that pass validation.
    """

    extracted_rules = extract_rules_from_chunk(
        chunk,
        source_document,
    )

    valid_rules = []

    for rule in extracted_rules:
        errors = validate_rule(rule)

        if not errors:
            valid_rules.append(rule)

    return valid_rules


def process_chunks(
    chunks: list[dict],
    source_document: str,
    extractor=extract_rules_from_chunk,
    progress_callback=None,
    outcome: dict | None = None,
) -> list:
    """
    Process multiple chunks and merge duplicate rules.

    Rules with the same rule_id are combined while preserving
    all source pages where the rule was found.

    When ``outcome`` is provided it is filled with per-chunk diagnostics
    (candidate counts, model errors, generated/validated/rejected tallies).
    Extractor exceptions propagate after being recorded, preserving the
    existing fail-the-document behavior.
    """

    unique_rules = {}
    use_diagnostics = outcome is not None and _extractor_accepts_diagnostics(extractor)
    if outcome is not None:
        outcome["chunks_processed"] = len(chunks)

    for chunk_index, chunk in enumerate(chunks, start=1):
        if progress_callback:
            progress_callback("ollama_rule_generation", chunk, chunk_index, len(chunks))
        chunk_diagnostics: dict = {}
        try:
            if use_diagnostics:
                extracted_rules = extractor(chunk, source_document, diagnostics=chunk_diagnostics)
            else:
                extracted_rules = extractor(chunk, source_document)
        except Exception as error:
            if outcome is not None:
                outcome["model_errors"].append({
                    "chunk_id": chunk.get("chunk_id", f"chunk_{chunk_index:04d}"),
                    "error_type": type(error).__name__,
                    "kind": getattr(error, "kind", None),
                    "message": str(error)[:_MAX_OUTCOME_REASON_CHARS],
                })
            raise
        if progress_callback:
            progress_callback("validation", chunk, chunk_index, len(chunks))
        if outcome is not None:
            if chunk_diagnostics.get("skipped_no_candidates"):
                pass
            elif "proposed" in chunk_diagnostics:
                outcome["rules_generated"] += chunk_diagnostics.get("proposed", 0) or 0
                outcome["rules_rejected"] += chunk_diagnostics.get("validation_rejected", 0) or 0
                for reason in chunk_diagnostics.get("rejection_reasons", []) or []:
                    _record_rejection(outcome, reason)
            else:
                outcome["rules_generated"] += len(extracted_rules)
        rules = []
        for rule in extracted_rules:
            errors = validate_rule(rule)
            if not errors:
                rules.append(rule)
            elif outcome is not None:
                outcome["rules_rejected"] += 1
                _record_rejection(outcome, f"{getattr(rule, 'rule_id', '?')}: {errors[0]}")
        if outcome is not None:
            outcome["rules_validated"] += len(rules)

        for rule in rules:
            identity = _source_identity(
                source_document,
                rule.parameter,
                rule.condition,
            )
            if identity not in unique_rules:
                unique_rules[identity] = rule
            else:
                existing_rule = unique_rules[identity]

                merged_pages = sorted(
                    set(existing_rule.source_pages + rule.source_pages)
                )

                evidence = existing_rule.evidence_text
                if rule.evidence_text not in evidence:
                    evidence = f"{evidence}\n\n{rule.evidence_text}"
                unique_rules[identity] = existing_rule.model_copy(
                    update={"source_pages": merged_pages, "evidence_text": evidence}
                )

    return list(unique_rules.values())

def save_processed_rules(
    rules: list,
) -> None:
    """
    Save validated rules into the rule repository.
    """

    for rule in rules:
        add_rule(rule)


def process_pdf_document(
    pdf_path: str,
    source_document: str,
    version: str | None = None,
    extractor=extract_rules_from_chunk,
    persist: bool = True,
    require_rules: bool = True,
    progress_callback=None,
    source_identity_document: str | None = None,
    outcome: dict | None = None,
) -> list:
    """Run one downloaded regulatory document through the local RAG pipeline.

    A document hash is a practical version value for fetched official material.
    The repository keeps any previous version and marks it superseded.

    When ``outcome`` is provided it is filled with a structured result
    (``status`` of SUCCESS_WITH_RULES, NO_RULES_FOUND,
    VALIDATION_REJECTED_ALL, EXTRACTION_ERROR, or OCR_ERROR plus counters).
    Raising behavior is unchanged: empty results with ``require_rules`` still
    raise ``ValueError`` and unusable input still raises ``OcrQualityError``.
    """
    if outcome is not None:
        outcome.update(_new_outcome())
    if progress_callback:
        progress_callback("pdf_extraction", None, None, None)
    def report_pdf_page(page_number: int, total_pages: int) -> None:
        if progress_callback:
            progress_callback(
                "pdf_extraction",
                {"chunk_id": f"page_{page_number:04d}"},
                page_number,
                total_pages,
            )

    pages = extract_pdf_text(pdf_path, progress_callback=report_pdf_page)
    save_extracted_text(pdf_path, pages)
    usable_pages = [
        page for page in pages
        if page.get("ocr_status", "GOOD") == "GOOD" and page.get("text", "").strip()
    ]
    skipped_pages = [
        page for page in pages
        if page not in usable_pages
    ]
    if skipped_pages:
        logger.warning(
            "OCR_QUALITY_PAGES_SKIPPED document=%s pages=%s statuses=%s",
            source_document,
            [page["page"] for page in skipped_pages],
            [page.get("ocr_status", "GOOD") for page in skipped_pages],
        )
    if not usable_pages:
        if outcome is not None:
            outcome["status"] = OCR_ERROR
        raise OcrQualityError(
            "No trustworthy pages were available after OCR quality checks."
        )

    chunks = chunk_pages(usable_pages)
    if outcome is not None:
        from app.extraction.llm_rule_generator import extract_normative_candidates

        for chunk in chunks:
            chunk_candidates = extract_normative_candidates(chunk.get("text", ""))
            outcome["candidate_count"] += len(chunk_candidates)
            if chunk_candidates:
                outcome["chunks_with_candidates"] += 1
    try:
        rules = process_chunks(chunks, source_document, extractor=extractor,
                               progress_callback=progress_callback, outcome=outcome)
    except Exception:
        if outcome is not None and outcome.get("status") is None:
            outcome["status"] = EXTRACTION_ERROR
        raise
    if not rules and require_rules:
        if outcome is not None:
            if outcome.get("model_errors"):
                outcome["status"] = EXTRACTION_ERROR
            elif outcome.get("rules_generated", 0) > 0:
                outcome["status"] = VALIDATION_REJECTED_ALL
            else:
                outcome["status"] = NO_RULES_FOUND
        raise ValueError("No validated compliance rules were extracted; active rules were not changed.")
    if outcome is not None:
        outcome["status"] = SUCCESS_WITH_RULES
    versioned_rules = [
        rule.model_copy(
            update={
                "version": version,
                # This is deterministic and source-aware, unlike an LLM's
                # descriptive rule_id. It lets a revised version of the same
                # official document replace its prior interpretation without
                # guessing that a separate amendment supersedes another rule.
                "source_identity": _source_identity(
                    source_identity_document or rule.source_document,
                    rule.parameter,
                    rule.condition,
                ),
                "source_document_id": source_identity_document,
            }
        )
        for rule in rules
    ]
    if persist:
        if version is None:
            save_processed_rules(versioned_rules)
        else:
            update_rules(versioned_rules)
    return versioned_rules


def _source_identity(source_document: str, parameter: str, condition: str) -> str:
    normalized = "|".join(
        re.sub(r"\s+", " ", value).strip().casefold()
        for value in (source_document, parameter, condition)
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
