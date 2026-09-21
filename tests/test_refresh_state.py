"""Regression for suspicious refresh: 39 discovered, 32 unchanged, 7 changed.

Covers:
- changed documents actually enter processing (OCR/Qwen path)
- unchanged documents never invoke download/processing
- successful processing increments counters, failed increments failed
- status is never up_to_date while changed remain unprocessed
- test registry isolated from production
"""

from pathlib import Path

from app.extraction.rule_schema import ComplianceRule


PROD_REGISTRY = Path("data/registry/document_registry.json")


def _rule(doc: str) -> ComplianceRule:
    return ComplianceRule(
        rule_id=f"LM_{doc}",
        parameter="mrp",
        condition="must_exist",
        applies_to={"product_type": "packaged_commodity"},
        requirement="The package must declare the retail sale price.",
        source_document=doc,
        source_pages=[1],
        evidence_text="Every package shall declare the retail sale price.",
    )


def _snapshot_prod():
    if not PROD_REGISTRY.exists():
        return None
    return PROD_REGISTRY.read_bytes()


def _make_docs(n=39):
    return [{"title": f"Doc {i}", "url": f"https://example.test/{i}.pdf"} for i in range(1, n + 1)]


def test_39_discovered_32_unchanged_7_changed_all_processed(monkeypatch):
    import app.rag_update_service as service

    before = _snapshot_prod()
    docs = _make_docs(39)
    changed_urls = {f"https://example.test/{i}.pdf" for i in range(33, 40)}  # 7 changed

    def fake_check(doc):
        return "CHANGED" if doc["url"] in changed_urls else "UNCHANGED"

    download_calls = []
    process_calls = []

    def fake_download(doc):
        assert doc["url"] in changed_urls, "unchanged must never reach download"
        download_calls.append(doc["url"])
        return {
            "title": doc["title"],
            "url": doc["url"],
            "local_path": f"/tmp/{doc['title']}.pdf",
            "sha256": f"hash-{doc['title']}",
            "processed_sha256": None,
            "status": "CHANGED",
        }

    def fake_process(path, *args, **kwargs):
        process_calls.append(path)
        return [_rule(path)]

    monkeypatch.setattr(service, "OFFICIAL_SOURCES", [{"url": "https://example.test"}])
    monkeypatch.setattr(service, "discover_documents", lambda url: docs)
    monkeypatch.setattr(service, "mark_missing", lambda *a: [])
    monkeypatch.setattr(service, "load_rules", lambda: [_rule("existing")])
    monkeypatch.setattr(service, "get_active_rules", lambda: [_rule("existing")])
    monkeypatch.setattr(service, "check_document_change", fake_check)
    monkeypatch.setattr(service, "download_document", fake_download)
    monkeypatch.setattr(service, "process_pdf_document", fake_process)
    monkeypatch.setattr(service, "update_rules", lambda rules: {"rules_added": len(rules), "rules_superseded": 0})
    monkeypatch.setattr(service, "mark_document_processed", lambda *a: None)

    result = service.run_rag_update()

    assert result["documents_discovered"] == 39
    assert result["unchanged_documents"] == 32
    assert result["changed_documents"] == 7
    assert result["successful_documents"] == 7
    assert result["failed_documents"] == 0
    assert result["status"] == "updated"
    assert result["status"] != "up_to_date"
    assert sorted(download_calls) == sorted(changed_urls)
    assert len(process_calls) == 7
    assert _snapshot_prod() == before


def test_39_mixed_success_failure_is_not_up_to_date(monkeypatch):
    import app.rag_update_service as service

    before = _snapshot_prod()
    docs = _make_docs(39)
    changed_urls = {f"https://example.test/{i}.pdf" for i in range(33, 40)}

    def fake_check(doc):
        return "CHANGED" if doc["url"] in changed_urls else "UNCHANGED"

    def fake_download(doc):
        return {
            "title": doc["title"],
            "url": doc["url"],
            "local_path": f"/tmp/{doc['title']}.pdf",
            "sha256": f"hash-{doc['title']}",
            "processed_sha256": None,
            "status": "CHANGED",
        }

    def fake_process(path, *args, **kwargs):
        if "Doc 33" in path or "Doc 34" in path:
            raise RuntimeError("Simulated Qwen failure")
        return [_rule(path)]

    monkeypatch.setattr(service, "OFFICIAL_SOURCES", [{"url": "https://example.test"}])
    monkeypatch.setattr(service, "discover_documents", lambda url: docs)
    monkeypatch.setattr(service, "mark_missing", lambda *a: [])
    monkeypatch.setattr(service, "load_rules", lambda: [_rule("existing")])
    monkeypatch.setattr(service, "get_active_rules", lambda: [_rule("existing")])
    monkeypatch.setattr(service, "check_document_change", fake_check)
    monkeypatch.setattr(service, "download_document", fake_download)
    monkeypatch.setattr(service, "process_pdf_document", fake_process)
    monkeypatch.setattr(service, "update_rules", lambda rules: {"rules_added": len(rules), "rules_superseded": 0})
    monkeypatch.setattr(service, "mark_document_processed", lambda *a: None)

    result = service.run_rag_update()

    assert result["changed_documents"] == 7
    assert result["successful_documents"] == 5
    assert result["failed_documents"] == 2
    assert result["status"] == "completed_with_errors"
    assert result["status"] != "up_to_date"
    assert _snapshot_prod() == before


def test_unprocessed_download_unchanged_still_enters_processing(monkeypatch):
    """Download hash-match but never processed must NOT be skipped."""
    import app.rag_update_service as service

    doc = {"title": "Unprocessed", "url": "https://example.test/unprocessed.pdf"}
    monkeypatch.setattr(service, "OFFICIAL_SOURCES", [{"url": "https://example.test"}])
    monkeypatch.setattr(service, "discover_documents", lambda url: [doc])
    monkeypatch.setattr(service, "mark_missing", lambda *a: [])
    monkeypatch.setattr(service, "load_rules", lambda: [_rule("existing")])
    monkeypatch.setattr(service, "get_active_rules", lambda: [_rule("existing")])
    monkeypatch.setattr(service, "check_document_change", lambda d: "CHANGED")
    # Simulate fetcher returning UNCHANGED (hash matches) but processed is None.
    monkeypatch.setattr(
        service,
        "download_document",
        lambda d: {
            "title": d["title"],
            "url": d["url"],
            "local_path": "/tmp/unprocessed.pdf",
            "sha256": "same-hash",
            "processed_sha256": None,
            "status": "UNCHANGED",
        },
    )
    processed = []
    monkeypatch.setattr(service, "process_pdf_document", lambda *a, **k: processed.append(True) or [_rule("unprocessed")])
    monkeypatch.setattr(service, "update_rules", lambda rules: {"rules_added": 1, "rules_superseded": 0})
    monkeypatch.setattr(service, "mark_document_processed", lambda *a: None)

    result = service.run_rag_update()

    assert processed == [True], "unprocessed UNCHANGED_AFTER_DOWNLOAD must still process"
    assert result["successful_documents"] == 1
    assert result["status"] == "updated"


def test_already_processed_download_unchanged_is_safely_skipped(monkeypatch):
    """Download hash-match and already processed may skip without false work."""
    import app.rag_update_service as service

    doc = {"title": "Done", "url": "https://example.test/done.pdf"}
    monkeypatch.setattr(service, "OFFICIAL_SOURCES", [{"url": "https://example.test"}])
    monkeypatch.setattr(service, "discover_documents", lambda url: [doc])
    monkeypatch.setattr(service, "mark_missing", lambda *a: [])
    monkeypatch.setattr(service, "load_rules", lambda: [_rule("existing")])
    monkeypatch.setattr(service, "get_active_rules", lambda: [_rule("existing")])
    monkeypatch.setattr(service, "check_document_change", lambda d: "CHANGED")
    monkeypatch.setattr(
        service,
        "download_document",
        lambda d: {
            "title": d["title"],
            "url": d["url"],
            "local_path": "/tmp/done.pdf",
            "sha256": "same-hash",
            "processed_sha256": "same-hash",
            "status": "UNCHANGED",
        },
    )
    monkeypatch.setattr(
        service, "process_pdf_document", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not process"))
    )
    monkeypatch.setattr(service, "update_rules", lambda rules: (_ for _ in ()).throw(AssertionError("no rules")))

    result = service.run_rag_update()

    # No processing happened, but detection said CHANGED and download proved
    # already-processed: service correctly reports up_to_date (no outstanding work).
    assert result["successful_documents"] == 0
    assert result["failed_documents"] == 0
    assert result["status"] == "up_to_date"
