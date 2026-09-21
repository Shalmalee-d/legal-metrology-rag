"""Shared regulatory refresh orchestration for scheduler and manual API calls."""

import logging
from datetime import datetime, timezone
from threading import Lock

from app.extraction.llm_rule_generator import generate_rules_from_chunk
from app.extraction.rule_pipeline import (
    NO_RULES_FOUND,
    process_pdf_document,
)
from app.fetcher.document_fetcher import (
    check_document_change,
    discover_documents,
    download_document,
    get_verified_cached_document,
    mark_document_processed,
)
from app.repository.rule_repository import get_active_rules, load_rules, save_rules, update_rules
from app.registry.document_registry import load_update_status, mark_missing, save_update_status
from app.sources.source_registry import OFFICIAL_SOURCES


logger = logging.getLogger(__name__)

_update_lock = Lock()
_progress_lock = Lock()

_last_successful_update: str | None = None
_last_successful_update_loaded = False


def _get_last_successful_update() -> str | None:
    """Lazily load durable status so imports never touch production files."""
    global _last_successful_update, _last_successful_update_loaded
    if not _last_successful_update_loaded:
        try:
            _last_successful_update = load_update_status().get("last_successful_update")
        except Exception:
            _last_successful_update = None
        _last_successful_update_loaded = True
    return _last_successful_update


def reset_update_state_for_tests() -> None:
    """Reset in-memory progress between tests; never touches production files."""
    global _last_successful_update, _last_successful_update_loaded
    with _progress_lock:
        _update_progress.update(
            {
                "update_in_progress": False,
                "current_stage": "idle",
                "current_document": None,
                "current_chunk": None,
                "documents_checked": 0,
                "rebuild_documents_attempted": 0,
                "documents_total": 0,
                "documents_changed": 0,
                "documents_discovered": 0,
                "unchanged_documents": 0,
                "new_documents": 0,
                "changed_documents": 0,
                "rules_updated": 0,
                "rules_added": 0,
                "rules_superseded": 0,
                "rules_rejected": 0,
                "successful_documents": 0,
                "failed_documents": 0,
                "documents_no_rules": 0,
                "status": "idle",
                "failed_document_details": [],
                "last_error": None,
            }
        )
    _last_successful_update = None
    _last_successful_update_loaded = False
    # Drain a stuck lock left by a failed test without blocking production.
    try:
        while _update_lock.locked():
            _update_lock.release()
    except RuntimeError:
        pass

_update_progress = {
    "update_in_progress": False,
    "current_stage": "idle",
    "current_document": None,
    "current_chunk": None,
    "documents_checked": 0,
    "rebuild_documents_attempted": 0,
    "documents_total": 0,
    "documents_changed": 0,
    "documents_discovered": 0,
    "unchanged_documents": 0,
    "new_documents": 0,
    "changed_documents": 0,
    "rules_updated": 0,
    "rules_added": 0,
    "rules_superseded": 0,
    "rules_rejected": 0,
    "successful_documents": 0,
    "failed_documents": 0,
    "documents_no_rules": 0,
    "status": "idle",
    "failed_document_details": [],
    "last_error": None,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_update_running() -> bool:
    return _update_lock.locked()


def _set_progress(**updates) -> None:
    with _progress_lock:
        _update_progress.update(updates)


def get_update_status() -> dict:
    with _progress_lock:
        status = _update_progress.copy()

    status.update(
        {
            "update_in_progress": is_update_running(),
            "last_successful_update": _get_last_successful_update(),
            "active_rule_count": len(get_active_rules()),
        }
    )

    return status


def _persist_completed_status(result: dict) -> dict:
    """Keep the latest terminal summary available across backend restarts."""
    if result.get("status") not in {"updated", "up_to_date", "completed_with_errors", "failed"}:
        return result
    summary = {
        key: result.get(key) for key in (
            "status", "checked_at", "documents_discovered", "documents_checked",
            "new_documents", "changed_documents", "unchanged_documents",
            "successful_documents", "failed_documents", "documents_no_rules",
            "rules_added", "rules_updated",
            "rules_superseded", "rules_rejected", "active_rule_count",
            "failed_document_details", "message",
        )
    }
    summary["last_successful_update"] = _get_last_successful_update()
    save_update_status(summary)
    return result


def _doc_result(document: dict, record: dict | None, outcome: dict,
                status: str, error: str | None = None) -> dict:
    """Compact per-document diagnostics; never embeds raw model output."""
    result = {
        "title": document.get("title"),
        "url": document.get("url"),
        "outcome": status,
        "candidate_count": outcome.get("candidate_count", 0),
        "chunks_processed": outcome.get("chunks_processed", 0),
        "rules_generated": outcome.get("rules_generated", 0),
        "rules_validated": outcome.get("rules_validated", 0),
        "rules_rejected": outcome.get("rules_rejected", 0),
        "model_errors": [
            {key: item.get(key) for key in ("chunk_id", "error_type", "kind")}
            for item in outcome.get("model_errors", [])
        ],
        "rejection_reasons": list(outcome.get("validation_rejection_reasons", [])),
    }
    if error is not None:
        result["error"] = error[:200]
    return result


def _pipeline_progress(document: str):
    def report(
        stage: str,
        chunk: dict | None,
        index: int | None,
        total: int | None,
    ) -> None:
        current_chunk = (
            None
            if chunk is None
            else f"{chunk['chunk_id']} ({index}/{total})"
        )

        _set_progress(
            current_stage=stage,
            current_document=document,
            current_chunk=current_chunk,
        )

        logger.info(
            "RAG_PROGRESS stage=%s document=%s chunk=%s",
            stage,
            document,
            current_chunk,
        )

    return report


def _summary_fields(*, documents_discovered: int = 0, unchanged_documents: int = 0,
                    new_documents: int = 0, changed_documents: int = 0,
                    rules_added: int = 0, rules_updated: int = 0,
                    rules_superseded: int = 0, rules_rejected: int = 0) -> dict:
    """Stable additive status fields for API clients during the transition."""
    return {
        "documents_discovered": documents_discovered,
        "unchanged_documents": unchanged_documents,
        "new_documents": new_documents,
        "changed_documents": changed_documents,
        "rules_added": rules_added,
        "rules_updated": rules_updated,
        "rules_superseded": rules_superseded,
        "rules_rejected": rules_rejected,
    }


def run_rag_update(allow_bootstrap: bool = True, rebuild: bool = False) -> dict:
    """Check official documents and safely activate only complete valid updates.

    ``rebuild=True`` is an explicit controlled operation: it reprocesses the
    discovered scoped set and replaces the snapshot only when that full run
    succeeds. Routine API/scheduler calls remain incremental.
    """

    global _last_successful_update, _last_successful_update_loaded

    checked_at = _now()

    if not _update_lock.acquire(blocking=False):
        progress = get_update_status()

        return {
            "status": "in_progress",
            "checked_at": checked_at,
            "documents_checked": progress["documents_checked"],
            "documents_total": progress["documents_total"],
            "documents_changed": progress["documents_changed"],
            "rules_updated": progress["rules_updated"],
            "successful_documents": progress["successful_documents"],
            "failed_documents": progress["failed_documents"],
            "message": "A regulatory update is already in progress.",
        }

    try:
        _set_progress(
            update_in_progress=True,
            current_stage="source_check",
            current_document=None,
            current_chunk=None,
            documents_checked=0,
            rebuild_documents_attempted=0,
            documents_total=0,
            documents_changed=0,
            documents_discovered=0,
            unchanged_documents=0,
            new_documents=0,
            changed_documents=0,
            rules_updated=0,
            rules_added=0,
            rules_superseded=0,
            rules_rejected=0,
            successful_documents=0,
            failed_documents=0,
            documents_no_rules=0,
            status="in_progress",
            failed_document_details=[],
            last_error=None,
        )

        logger.info("CHECK_STARTED")

        documents_checked = 0
        documents_changed = 0
        unchanged_documents = 0
        new_documents = 0
        changed_documents = 0
        rules_updated = 0
        rules_added = 0
        rules_superseded = 0
        successful_documents = 0
        documents_no_rules = 0
        rebuild_documents_attempted = 0

        failures: list[str] = []
        staged_rules = []
        processed_documents = []
        document_results: list[dict] = []
        documents = []

        # If the repository has no active rules, this is the initial
        # regulatory knowledge-base bootstrap.
        # Bootstrap depends on the repository snapshot itself, not on today's
        # applicability. A populated repository containing only future rules
        # must not trigger a historical corpus rebuild.
        bootstrap_required = not load_rules()

        logger.info(
            "RAG_BOOTSTRAP required=%s active_rule_count=%s",
            bootstrap_required,
            len(get_active_rules()),
        )

        # ---------------------------------------------------------
        # 1. Discover official regulatory documents
        # ---------------------------------------------------------

        source_documents: dict[str, list[dict]] = {}
        for source in OFFICIAL_SOURCES:
            logger.info(
                "RAG_PROGRESS stage=source_check source=%s",
                source["url"],
            )

            discovered = discover_documents(source["url"])
            for document in discovered:
                document.setdefault("source", source["url"])
                document.setdefault("category", source.get("document_category", "legal_metrology_packaged_commodities"))
            source_documents[source["url"]] = discovered
            documents.extend(discovered)

        # Disappearance is an observable source state, not a legal conclusion;
        # rules/history remain in the repository.
        for source_url, discovered in source_documents.items():
            mark_missing(source_url, {item["url"] for item in discovered})

        _set_progress(
            documents_total=len(documents), documents_discovered=len(documents)
        )

        # ---------------------------------------------------------
        # 2. Check each document
        # ---------------------------------------------------------

        for document in documents:
            documents_checked += 1

            _set_progress(
                current_stage="change_detection",
                current_document=document["title"],
                current_chunk=None,
                documents_checked=documents_checked,
            )

            logger.info(
                "RAG_PROGRESS stage=change_detection document=%s",
                document["title"],
            )

            try:
                doc_outcome: dict = {}
                detected = check_document_change(document)

                if detected == "UNCHANGED":
                    unchanged_documents += 1
                    _set_progress(unchanged_documents=unchanged_documents)
                elif detected == "NEW":
                    new_documents += 1
                    _set_progress(new_documents=new_documents)
                else:
                    changed_documents += 1
                    _set_progress(changed_documents=changed_documents)

                # The startup check is intentionally non-bootstrap: it may
                # inspect all sources but must never restart a costly initial
                # corpus build after every Uvicorn restart.
                if bootstrap_required and not allow_bootstrap:
                    logger.info("BOOTSTRAP_DEFERRED: %s", document["title"])
                    continue

                # Normal weekly/manual refresh:
                # unchanged documents require no expensive processing.
                if detected == "UNCHANGED" and not bootstrap_required and not rebuild:
                    logger.info(
                        "UNCHANGED: %s",
                        document["title"],
                    )
                    continue

                if detected == "UNCHANGED" and bootstrap_required:
                    logger.info(
                        "BOOTSTRAP_DOCUMENT: %s",
                        document["title"],
                    )
                elif detected == "NEW":
                    logger.info(
                        "NEW_DOCUMENT: %s",
                        document["title"],
                    )
                else:
                    logger.info(
                        "DOCUMENT_CHANGED: %s",
                        document["title"],
                    )

                # -------------------------------------------------
                # 3. Download / reuse local document
                # -------------------------------------------------

                _set_progress(
                    current_stage="download",
                    current_document=document["title"],
                    current_chunk=None,
                )

                logger.info(
                    "RAG_PROGRESS stage=download document=%s",
                    document["title"],
                )

                if rebuild:
                    rebuild_documents_attempted += 1
                    _set_progress(
                        rebuild_documents_attempted=rebuild_documents_attempted
                    )

                try:
                    record = download_document(document)
                except Exception:
                    # A bootstrap may use only a previously downloaded,
                    # hash-verified official PDF when the source host is
                    # temporarily unavailable.  Normal refreshes still fail
                    # rather than treating cached content as current.
                    record = (
                        get_verified_cached_document(document)
                        if bootstrap_required
                        else None
                    )
                    if record is None:
                        raise
                    logger.warning(
                        "BOOTSTRAP_USING_VERIFIED_CACHE: %s",
                        document["title"],
                    )

                # During bootstrap, download_document() can correctly
                # report UNCHANGED because the PDF already exists.
                # We still need to process that existing PDF.
                if (
                    record["status"] == "UNCHANGED"
                    and not bootstrap_required
                    and not rebuild
                ):
                    # Defensive: UNCHANGED from download is only safe to skip
                    # when the matching fingerprint was already processed.
                    # A fingerprint downloaded but never activated (e.g. prior
                    # Ollama/validation failure) must still enter OCR/Qwen.
                    sha = record.get("sha256")
                    processed = record.get("processed_sha256")
                    if sha is not None and processed is not None:
                        if sha == processed:
                            logger.info(
                                "UNCHANGED_AFTER_DOWNLOAD: %s",
                                document["title"],
                            )
                            continue
                        # Unprocessed content: fall through to processing.
                    else:
                        # Missing hash info: consult registry before skipping.
                        try:
                            from app.registry.document_registry import (
                                get_record as _get_registry_record,
                            )

                            reg = _get_registry_record(document["url"])
                            if reg is not None:
                                fp = reg.get("fingerprint")
                                pp = reg.get("processed_fingerprint")
                                if fp is not None and pp is not None and fp == pp:
                                    logger.info(
                                        "UNCHANGED_AFTER_DOWNLOAD: %s",
                                        document["title"],
                                    )
                                    continue
                            # No proof of prior processing: must process.
                        except Exception:
                            # Registry check must never mask as up-to-date.
                            pass

                local_path = record.get("local_path")

                if not local_path:
                    raise ValueError(
                        "No local PDF path available for "
                        f"{document['title']}"
                    )

                # -------------------------------------------------
                # 4. PDF extraction -> Qwen -> validation
                # -------------------------------------------------

                logger.info(
                    "RAG_UPDATE_STARTED: %s",
                    document["title"],
                )

                # This document has a new/changed content version that
                # requires processing. Track that independently from whether
                # extraction and validation eventually succeed.
                documents_changed += 1
                _set_progress(documents_changed=documents_changed, changed_documents=changed_documents)

                doc_outcome: dict = {}
                try:
                    rules = process_pdf_document(
                        local_path,
                        record["title"],
                        record["sha256"],
                        extractor=generate_rules_from_chunk,
                        persist=False,
                        require_rules=True,
                        progress_callback=_pipeline_progress(
                            document["title"]
                        ),
                        # URL is the durable source identity.  A government page
                        # can correct a display title without creating a second
                        # active interpretation of the same PDF.
                        source_identity_document=document["url"],
                        outcome=doc_outcome,
                    )
                except ValueError as empty_error:
                    # The pipeline found no surviving rules. A document whose
                    # corpus genuinely holds no actionable rule is resolved
                    # (marked processed, stays retryable via future content
                    # changes); anything else remains a retryable failure.
                    if doc_outcome.get("status") == NO_RULES_FOUND:
                        documents_changed += 1
                        documents_no_rules += 1
                        _set_progress(documents_changed=documents_changed,
                                      documents_no_rules=documents_no_rules)
                        processed_documents.append(
                            (record.get("url", document["url"]), record["sha256"])
                        )
                        document_results.append(
                            _doc_result(document, record, doc_outcome, NO_RULES_FOUND))
                        logger.info("RAG_NO_RULES: %s", document["title"])
                        continue
                    raise

                if not rules:
                    raise ValueError("No validated compliance rules were extracted; document remains retryable.")

                rules_updated += len(rules)
                successful_documents += 1

                _set_progress(
                    documents_changed=documents_changed,
                    rules_updated=rules_updated,
                    successful_documents=successful_documents,
                )

                staged_rules.extend(rules)

                processed_documents.append(
                    (
                        record.get(
                            "url",
                            document["url"],
                        ),
                        record["sha256"],
                    )
                )
                document_results.append(
                    _doc_result(document, record, doc_outcome, "SUCCESS_WITH_RULES"))

                logger.info(
                    "RAG_UPDATE_COMPLETED: %s rules=%s",
                    document["title"],
                    len(rules),
                )

            except Exception as error:
                logger.exception(
                    "RAG_UPDATE_FAILED: %s",
                    document["title"],
                )

                failures.append(
                    f"{document['title']}: {error}"
                )
                outcome_status = doc_outcome.get("status") or "EXTRACTION_ERROR"
                document_results.append(
                    _doc_result(document, None, doc_outcome, outcome_status,
                                error=str(error)))

                _set_progress(
                    last_error=str(error),
                    failed_documents=len(failures),
                    failed_document_details=failures.copy(),
                )

                # Continue checking the remaining documents.
                # Successfully processed documents are activated below; this
                # failed version remains unprocessed and will be retried.
                continue

        # ---------------------------------------------------------
        # 5. Persist successful documents without losing unrelated rules
        # ---------------------------------------------------------

        rebuild_ready = (
            rebuild
            and not failures
            and len(processed_documents) == len(documents)
            and bool(staged_rules)
        )
        if rebuild and not rebuild_ready:
            # A controlled rebuild must never replace an established snapshot
            # with a partial/empty corpus. An all-empty extraction is also
            # treated as unsafe because it would erase the usable knowledge
            # base without proving that the source corpus contains no rules.
            staged_rules = []

        if staged_rules:
            _set_progress(
                current_stage="repository_activation",
                current_document=None,
                current_chunk=None,
            )

            logger.info(
                "RAG_PROGRESS stage=repository_activation rules=%s",
                len(staged_rules),
            )

            # update_rules replaces matching source identities and supersedes
            # older versions, while retaining rules from documents that were
            # unchanged or failed in this run.
            if rebuild:
                save_rules(staged_rules)
                rules_added = len(staged_rules)
            else:
                repository_result = update_rules(staged_rules) or {}
                rules_added = repository_result.get("rules_added", len(staged_rules))
                rules_superseded = repository_result.get("rules_superseded", 0)
            _set_progress(
                rules_added=rules_added,
                rules_superseded=rules_superseded,
            )

        # A document which completed valid extraction (including a document
        # with no standalone machine-evaluable rule) is safe to mark only
        # after its generated rules have been durably written.
        for url, sha256 in (processed_documents if not rebuild or rebuild_ready else []):
            mark_document_processed(
                url,
                sha256,
            )

        if processed_documents:
            _last_successful_update = checked_at
            _last_successful_update_loaded = True

        if rebuild and not rebuild_ready:
            message = (
                "; ".join(failures)
                if failures
                else "Controlled rebuild produced no validated compliance rules; existing snapshot was preserved."
            )
            _set_progress(
                current_stage="failed",
                status="failed",
                last_error=message,
            )
            return {
                "status": "failed",
                **_summary_fields(
                    documents_discovered=len(documents),
                    unchanged_documents=unchanged_documents,
                    new_documents=new_documents,
                    changed_documents=changed_documents,
                ),
                "checked_at": checked_at,
                "documents_checked": documents_checked,
                "documents_changed": documents_changed,
                "rebuild_documents_attempted": rebuild_documents_attempted,
                "rules_updated": 0,
                "successful_documents": successful_documents,
                "failed_documents": len(failures),
                "documents_no_rules": documents_no_rules,
                "document_results": document_results,
                "failed_document_details": failures,
                "active_rule_count": len(get_active_rules()),
                "rules_preserved": True,
                "message": message,
            }

        if failures and successful_documents:
            error_message = "; ".join(failures)
            _set_progress(
                current_stage="completed_with_errors",
                status="completed_with_errors",
                last_error=error_message,
            )
            logger.warning(
                "RAG_UPDATE_COMPLETED_WITH_ERRORS successful_documents=%s failures=%s",
                successful_documents,
                len(failures),
            )

            return {
                "status": "completed_with_errors",
                **_summary_fields(documents_discovered=len(documents), unchanged_documents=unchanged_documents,
                                  new_documents=new_documents, changed_documents=changed_documents,
                                  rules_added=rules_updated, rules_updated=rules_updated),
                "checked_at": checked_at,
                "documents_checked": documents_checked,
                "documents_changed": documents_changed,
                "rebuild_documents_attempted": rebuild_documents_attempted,
                "rules_generated": rules_updated,
                "rules_activated": rules_updated,
                "rules_updated": rules_updated,
                "successful_documents": successful_documents,
                "failed_documents": len(failures),
                "documents_no_rules": documents_no_rules,
                "document_results": document_results,
                "failed_document_details": failures,
                "active_rule_count": len(get_active_rules()),
                "rules_preserved": True,
                "message": (
                    "Regulatory knowledge base updated with errors: "
                    f"{error_message}"
                ),
            }

        if failures:
            error_message = "; ".join(failures)
            _set_progress(
                current_stage="failed",
                status="failed",
                last_error=error_message,
            )
            logger.error("RAG_UPDATE_BATCH_FAILED failures=%s", len(failures))
            return {
                "status": "failed",
                **_summary_fields(documents_discovered=len(documents), unchanged_documents=unchanged_documents,
                                  new_documents=new_documents, changed_documents=changed_documents),
                "checked_at": checked_at,
                "documents_checked": documents_checked,
                "documents_changed": documents_changed,
                "rebuild_documents_attempted": rebuild_documents_attempted,
                "rules_updated": rules_updated,
                "successful_documents": 0,
                "failed_documents": len(failures),
                "documents_no_rules": documents_no_rules,
                "document_results": document_results,
                "failed_document_details": failures,
                "active_rule_count": len(get_active_rules()),
                "rules_preserved": True,
                "message": error_message,
            }

        if documents_changed:
            _set_progress(current_stage="completed", status="updated")

            return {
                "status": "updated",
                **_summary_fields(documents_discovered=len(documents), unchanged_documents=unchanged_documents,
                                  new_documents=new_documents, changed_documents=changed_documents,
                                  rules_added=rules_updated, rules_updated=rules_updated),
                "checked_at": checked_at,
                "documents_checked": documents_checked,
                "documents_changed": documents_changed,
                "rebuild_documents_attempted": rebuild_documents_attempted,
                "rules_generated": rules_updated,
                "rules_activated": rules_updated,
                "rules_updated": rules_updated,
                "successful_documents": successful_documents,
                "failed_documents": 0,
                "documents_no_rules": documents_no_rules,
                "document_results": document_results,
                "active_rule_count": len(get_active_rules()),
                "message": "Regulatory knowledge base updated successfully.",
            }

        # ---------------------------------------------------------
        # 7. Nothing changed
        # ---------------------------------------------------------

        _set_progress(
            current_stage="completed",
            status="up_to_date",
        )

        return {
            "status": "up_to_date",
            **_summary_fields(documents_discovered=len(documents), unchanged_documents=unchanged_documents,
                              new_documents=new_documents, changed_documents=changed_documents),
            "checked_at": checked_at,
            "documents_checked": documents_checked,
            "documents_changed": 0,
            "rebuild_documents_attempted": rebuild_documents_attempted,
            "rules_updated": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "documents_no_rules": 0,
            "document_results": [],
            "active_rule_count": len(get_active_rules()),
            "message": (
                "Regulatory rules are already up to date."
            ),
        }

    except Exception as error:
        logger.exception("RAG_UPDATE_FAILED")

        _set_progress(
            current_stage="failed",
            status="failed",
            last_error=str(error),
        )

        return {
            "status": "failed",
            **_summary_fields(),
            "checked_at": checked_at,
            "documents_checked": 0,
            "documents_changed": 0,
            "rebuild_documents_attempted": 0,
            "rules_updated": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "active_rule_count": len(get_active_rules()),
            "rules_preserved": True,
            "message": str(error),
        }

    finally:
        _update_lock.release()

        _set_progress(
            update_in_progress=False,
            current_document=None,
            current_chunk=None,
        )
        try:
            save_update_status(get_update_status())
        except Exception:
            # Observability persistence must never mask the update result or
            # replace a valid rules snapshot with a reporting failure.
            logger.exception("Could not persist RAG update status")
