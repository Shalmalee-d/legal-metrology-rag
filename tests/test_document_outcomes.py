"""Document processing outcome classification and partial-success tests."""

import pytest

from app.extraction.rule_pipeline import (
    EXTRACTION_ERROR,
    NO_RULES_FOUND,
    OCR_ERROR,
    SUCCESS_WITH_RULES,
    VALIDATION_REJECTED_ALL,
    process_pdf_document,
)
from app.extraction.rule_schema import ComplianceRule


def _pages(texts):
    return [{"page": number, "text": text, "method": "direct", "ocr_status": "GOOD"}
            for number, text in enumerate(texts, start=1)]


def _configure_pages(monkeypatch, texts):
    import app.extraction.rule_pipeline as pipeline

    monkeypatch.setattr(pipeline, "extract_pdf_text", lambda *args, **kwargs: _pages(texts))
    monkeypatch.setattr(pipeline, "save_extracted_text", lambda *args: None)


def _rule():
    return ComplianceRule(rule_id="LM_001", parameter="mrp", condition="must_exist",
                          requirement="The package must declare the retail sale price.",
                          source_document="Rules", source_pages=[1],
                          evidence_text="Every package shall declare the retail sale price.")


def test_success_with_rules_outcome(monkeypatch):
    _configure_pages(monkeypatch, ["Every package shall declare the retail sale price."])
    outcome = {}
    rules = process_pdf_document("x.pdf", "Rules", persist=False, require_rules=True, outcome=outcome)
    assert len(rules) == 1
    assert outcome["status"] == SUCCESS_WITH_RULES
    assert outcome["chunks_processed"] >= 1
    assert outcome["rules_validated"] == 1


def test_no_rules_found_when_no_normative_candidates(monkeypatch):
    _configure_pages(monkeypatch, ["Government of India\nMinistry of Consumer Affairs\nPhone 011-23389489"])
    outcome = {}
    with pytest.raises(ValueError, match="No validated compliance rules"):
        process_pdf_document("x.pdf", "Notice", persist=False, require_rules=True, outcome=outcome)
    assert outcome["status"] == NO_RULES_FOUND
    assert outcome["candidate_count"] == 0
    assert outcome["rules_generated"] == 0


def test_validation_rejected_all_outcome(monkeypatch):
    _configure_pages(monkeypatch, ["Every package shall bear the official rule name on the package."])
    bad = ComplianceRule(rule_id="bad", parameter="rule heading", condition="must_exist",
                         source_document="Rules", source_pages=[1],
                         evidence_text="These Rules shall be called Rules.")
    outcome = {}
    with pytest.raises(ValueError, match="No validated compliance rules"):
        process_pdf_document("x.pdf", "Rules", persist=False, require_rules=True,
                             extractor=lambda *args, **kwargs: [bad], outcome=outcome)
    assert outcome["status"] == VALIDATION_REJECTED_ALL
    assert outcome["rules_generated"] == 1
    assert outcome["rules_rejected"] == 1
    assert outcome["validation_rejection_reasons"]


def test_extraction_error_outcome_on_model_failure(monkeypatch):
    _configure_pages(monkeypatch, ["Every package shall declare the retail sale price."])

    def _boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    outcome = {}
    with pytest.raises(RuntimeError, match="model exploded"):
        process_pdf_document("x.pdf", "Rules", persist=False, require_rules=True,
                             extractor=_boom, outcome=outcome)
    assert outcome["status"] == EXTRACTION_ERROR
    assert outcome["model_errors"]
    assert outcome["model_errors"][0]["error_type"] == "RuntimeError"


def test_ocr_error_outcome(monkeypatch):
    import app.extraction.rule_pipeline as pipeline

    monkeypatch.setattr(pipeline, "extract_pdf_text", lambda *args, **kwargs: [
        {"page": 1, "text": "\ufffd\ufffd\ufffd", "method": "ocr", "ocr_status": "DEGRADED"}])
    monkeypatch.setattr(pipeline, "save_extracted_text", lambda *args: None)
    outcome = {}
    with pytest.raises(pipeline.OcrQualityError):
        process_pdf_document("x.pdf", "Rules", persist=False, require_rules=False, outcome=outcome)
    assert outcome["status"] == OCR_ERROR


def _service_rule():
    from app.extraction.rule_schema import ComplianceRule as _CR
    return _CR(rule_id="LM_001", parameter="mrp", condition="must_exist",
               requirement="The package must declare MRP.", source_document="Rules",
               source_pages=[1], evidence_text="MRP shall be declared.")


def _service_config(monkeypatch, documents):
    import app.rag_update_service as service

    monkeypatch.setattr(service, "OFFICIAL_SOURCES", [{"url": "https://example.test"}])
    monkeypatch.setattr(service, "discover_documents", lambda url: documents)
    monkeypatch.setattr(service, "mark_missing", lambda *args: [])
    monkeypatch.setattr(service, "load_rules", lambda: [_service_rule()])
    monkeypatch.setattr(service, "get_active_rules", lambda: [_service_rule()])
    monkeypatch.setattr(service, "check_document_change", lambda item: "NEW")
    return service


def test_no_rules_document_is_processed_not_failed(monkeypatch):
    service = _service_config(monkeypatch, [{"title": "Empty", "url": "https://example.test/empty.pdf"}])
    monkeypatch.setattr(service, "download_document", lambda item: {
        "title": item["title"], "url": item["url"], "local_path": "x.pdf",
        "sha256": "hash-empty", "status": "NEW"})
    marked = []
    monkeypatch.setattr(service, "mark_document_processed", lambda url, sha: marked.append((url, sha)))

    def fake_process(path, *args, **kwargs):
        outcome = kwargs.get("outcome")
        if outcome is not None:
            outcome.update(status=NO_RULES_FOUND, candidate_count=0, chunks_processed=1,
                           chunks_with_candidates=0, rules_generated=0, rules_validated=0,
                           rules_rejected=0, model_errors=[], validation_rejection_reasons=[])
        raise ValueError("No validated compliance rules were extracted; active rules were not changed.")

    monkeypatch.setattr(service, "process_pdf_document", fake_process)
    monkeypatch.setattr(service, "update_rules", lambda rules: (_ for _ in ()).throw(AssertionError("no rules")))

    result = service.run_rag_update()
    assert result["failed_documents"] == 0
    assert result["documents_no_rules"] == 1
    assert result["status"] == "updated"
    assert marked == [("https://example.test/empty.pdf", "hash-empty")]
    assert result["document_results"][0]["outcome"] == NO_RULES_FOUND


def test_validation_rejected_document_stays_failed_with_reasons(monkeypatch):
    service = _service_config(monkeypatch, [{"title": "Bad", "url": "https://example.test/bad.pdf"}])
    monkeypatch.setattr(service, "download_document", lambda item: {
        "title": item["title"], "url": item["url"], "local_path": "x.pdf",
        "sha256": "hash-bad", "status": "NEW"})
    marked = []
    monkeypatch.setattr(service, "mark_document_processed", lambda url, sha: marked.append((url, sha)))

    def fake_process(path, *args, **kwargs):
        outcome = kwargs.get("outcome")
        if outcome is not None:
            outcome.update(status=VALIDATION_REJECTED_ALL, candidate_count=2, chunks_processed=1,
                           chunks_with_candidates=1, rules_generated=1, rules_validated=0,
                           rules_rejected=1, model_errors=[],
                           validation_rejection_reasons=["rule_1: requirement must describe an actionable package or product check."])
        raise ValueError("No validated compliance rules were extracted; active rules were not changed.")

    monkeypatch.setattr(service, "process_pdf_document", fake_process)

    result = service.run_rag_update()
    assert result["failed_documents"] == 1
    assert result["status"] == "failed"
    assert marked == []
    assert result["document_results"][0]["outcome"] == VALIDATION_REJECTED_ALL
    assert result["document_results"][0]["rejection_reasons"]


def test_partial_mix_of_rules_empty_and_failed(monkeypatch):
    docs = [{"title": name, "url": f"https://example.test/{name}.pdf"} for name in ("good", "empty", "bad")]
    service = _service_config(monkeypatch, docs)
    monkeypatch.setattr(service, "download_document", lambda item: {
        "title": item["title"], "url": item["url"], "local_path": item["title"],
        "sha256": item["title"] + "-hash", "status": "NEW"})
    monkeypatch.setattr(service, "mark_document_processed", lambda *args: None)

    def fake_process(path, *args, **kwargs):
        outcome = kwargs.get("outcome")
        if path == "good":
            return [_service_rule()]
        if path == "empty":
            if outcome is not None:
                outcome.update(status=NO_RULES_FOUND, candidate_count=0, chunks_processed=1,
                               chunks_with_candidates=0, rules_generated=0, rules_validated=0,
                               rules_rejected=0, model_errors=[], validation_rejection_reasons=[])
            raise ValueError("No validated compliance rules were extracted.")
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "process_pdf_document", fake_process)
    activated = []
    monkeypatch.setattr(service, "update_rules",
                        lambda rules: activated.extend(rules) or {"rules_added": 1, "rules_superseded": 0})

    result = service.run_rag_update()
    assert result["status"] == "completed_with_errors"
    assert result["successful_documents"] == 1
    assert result["documents_no_rules"] == 1
    assert result["failed_documents"] == 1
    assert len(activated) == 1
    assert {entry["outcome"] for entry in result["document_results"]} == {
        "SUCCESS_WITH_RULES", NO_RULES_FOUND, "EXTRACTION_ERROR"}


def test_rebuild_with_only_empty_documents_preserves_repository(tmp_path, monkeypatch):
    import app.repository.rule_repository as repository

    docs = [{"title": "Empty", "url": "https://example.test/empty.pdf"}]
    service = _service_config(monkeypatch, docs)
    monkeypatch.setattr(repository, "RULES_DIR", tmp_path)
    monkeypatch.setattr(repository, "RULES_FILE", tmp_path / "compliance_rules.json")
    repository.save_rules([_service_rule()])
    monkeypatch.setattr(service, "download_document", lambda item: {
        "title": item["title"], "url": item["url"], "local_path": item["title"],
        "sha256": "hash", "status": "NEW"})

    def fake_process(path, *args, **kwargs):
        outcome = kwargs.get("outcome")
        if outcome is not None:
            outcome.update(status=NO_RULES_FOUND, candidate_count=0, chunks_processed=1,
                           chunks_with_candidates=0, rules_generated=0, rules_validated=0,
                           rules_rejected=0, model_errors=[], validation_rejection_reasons=[])
        raise ValueError("No validated compliance rules were extracted.")

    monkeypatch.setattr(service, "process_pdf_document", fake_process)

    result = service.run_rag_update(rebuild=True)
    assert result["status"] == "failed"
    assert repository.load_rules()[0].rule_id == "LM_001"
