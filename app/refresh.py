"""Shared two-document production refresh for Legal Metrology RAG.

Production source set is EXACTLY two official documents (base 2011 Rules and
the Readymade Garments/Hosiery advisory). Both the frontend/manual
"Check for Updates" action and the weekly scheduler call
:func:`check_for_updates` — there is intentionally only one implementation.

Per document: discover, fingerprint-compare, and only on change download,
extract/OCR, chunk, candidate detection, Qwen, validation, then replace ONLY
that document's rules while preserving the other document's rules.
Unchanged fingerprints never trigger extraction, OCR, chunking, or Qwen.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from threading import Lock

from app.sources.source_registry import SOURCE_URL, TargetNotFoundError, select_targets
from app.extraction.llm_rule_generator import generate_rules_from_chunk
from app.extraction.rule_pipeline import NO_RULES_FOUND, process_pdf_document
from app.fetcher.document_fetcher import (
    check_document_change,
    discover_documents,
    download_document,
    mark_document_processed,
)
from app.repository.rule_repository import get_active_rules, update_rules

logger = logging.getLogger(__name__)

_refresh_lock = Lock()
_last_result: dict = {
    "status": "idle",
    "documents_checked": 0,
    "documents_changed": 0,
    "unchanged_documents": 0,
    "rules_updated": False,
    "active_rule_count": 0,
    "errors": [],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_update_running() -> bool:
    return _refresh_lock.locked()


def get_refresh_status() -> dict:
    """Latest refresh summary plus live rule count (never triggers work)."""
    status = dict(_last_result)
    status.update(
        {
            "update_in_progress": is_update_running(),
            "active_rule_count": len(get_active_rules()),
        }
    )
    return status


def _remember(result: dict) -> dict:
    global _last_result
    _last_result = {key: result.get(key) for key in (
        "status", "documents_checked", "documents_changed",
        "unchanged_documents", "rules_updated", "errors")}
    _last_result.setdefault("status", "idle")
    return result


def check_for_updates() -> dict:
    """Fingerprint-gated refresh of the two production documents.

    Returns ``status`` of ``up_to_date``, ``updated``, ``failed``, or
    ``in_progress``, with ``documents_checked``, ``documents_changed``,
    ``unchanged_documents``, ``rules_updated`` (bool), ``active_rule_count``,
    and ``errors``.
    """
    checked_at = _now()
    if not _refresh_lock.acquire(blocking=False):
        return {"status": "in_progress", "checked_at": checked_at,
                "message": "A regulatory update is already in progress."}
    try:
        discovered = discover_documents(SOURCE_URL)
        try:
            targets = select_targets(discovered)
        except TargetNotFoundError as error:
            logger.error("Refresh target missing: %s", error)
            return _remember({
                "status": "failed", "checked_at": checked_at,
                "documents_checked": 2, "documents_changed": 0,
                "unchanged_documents": 0, "rules_updated": False,
                "active_rule_count": len(get_active_rules()),
                "errors": [str(error)],
                "message": str(error),
            })

        documents_changed = 0
        unchanged_documents = 0
        staged: list = []
        processed: list[tuple[str, str]] = []
        errors: list[str] = []
        document_results: list[dict] = []

        for target in targets:
            title = target["title"]
            try:
                detected = check_document_change(target)
                if detected == "UNCHANGED":
                    unchanged_documents += 1
                    logger.info("UNCHANGED: %s", title)
                    document_results.append({"title": title, "url": target["url"],
                                             "outcome": "UNCHANGED"})
                    continue
                record = download_document(target)
                if record.get("status") == "UNCHANGED":
                    unchanged_documents += 1
                    logger.info("UNCHANGED_AFTER_DOWNLOAD: %s", title)
                    document_results.append({"title": title, "url": target["url"],
                                             "outcome": "UNCHANGED"})
                    continue
                local_path = record.get("local_path")
                if not local_path:
                    raise ValueError(f"No local PDF path available for {title}.")
                documents_changed += 1
                outcome: dict = {}
                try:
                    rules = process_pdf_document(
                        local_path, record["title"], record["sha256"],
                        extractor=generate_rules_from_chunk,
                        persist=False, require_rules=True,
                        source_identity_document=target["url"],
                        outcome=outcome,
                    )
                except ValueError as empty_error:
                    if outcome.get("status") == NO_RULES_FOUND:
                        logger.info("NO_RULES: %s", title)
                        processed.append((record.get("url", target["url"]), record["sha256"]))
                        document_results.append({"title": title, "url": target["url"],
                                                 "outcome": NO_RULES_FOUND,
                                                 "candidates": outcome.get("candidate_count", 0)})
                        continue
                    raise
                staged.extend(rules)
                processed.append((record.get("url", target["url"]), record["sha256"]))
                document_results.append({"title": title, "url": target["url"],
                                         "outcome": "SUCCESS_WITH_RULES",
                                         "rules": len(rules)})
                logger.info("DOCUMENT_UPDATED: %s rules=%s", title, len(rules))
            except Exception as error:  # noqa: BLE001 - per-document isolation
                logger.exception("REFRESH_FAILED: %s", title)
                errors.append(f"{title}: {error}")
                document_results.append({"title": title, "url": target.get("url"),
                                         "outcome": "FAILED", "error": str(error)[:200]})

        if errors:
            return _remember({
                "status": "failed", "checked_at": checked_at,
                "documents_checked": 2, "documents_changed": documents_changed,
                "unchanged_documents": unchanged_documents, "rules_updated": False,
                "active_rule_count": len(get_active_rules()),
                "errors": errors, "document_results": document_results,
                "message": "; ".join(errors),
            })
        rules_updated = False
        if staged:
            repository_result = update_rules(staged) or {}
            rules_updated = True
            logger.info("REPOSITORY_UPDATED rules=%s superseded=%s",
                        repository_result.get("rules_added"), repository_result.get("rules_superseded"))
        for url, sha256 in processed:
            mark_document_processed(url, sha256)
        if documents_changed:
            return _remember({
                "status": "updated", "checked_at": checked_at,
                "documents_checked": 2, "documents_changed": documents_changed,
                "unchanged_documents": unchanged_documents, "rules_updated": rules_updated,
                "active_rule_count": len(get_active_rules()),
                "errors": [], "document_results": document_results,
                "message": "Regulatory rules were updated." if rules_updated else
                           "Documents changed but yielded no new rules.",
            })
        return _remember({
            "status": "up_to_date", "checked_at": checked_at,
            "documents_checked": 2, "documents_changed": 0,
            "unchanged_documents": unchanged_documents, "rules_updated": False,
            "active_rule_count": len(get_active_rules()),
            "errors": [], "document_results": document_results,
            "message": "Rules are up to date.",
        })
    except Exception as error:  # noqa: BLE001 - catastrophic guard
        logger.exception("REFRESH_FAILED")
        return _remember({
            "status": "failed", "checked_at": checked_at,
            "documents_checked": 0, "documents_changed": 0,
            "unchanged_documents": 0, "rules_updated": False,
            "active_rule_count": len(get_active_rules()),
            "errors": [str(error)],
            "message": str(error),
        })
    finally:
        _refresh_lock.release()
