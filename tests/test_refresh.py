"""Two-document fingerprint-gated refresh: unchanged skips Qwen, changed reprocesses."""

import pytest

from app.extraction.rule_schema import ComplianceRule


BASE = {"title": "Download The Legal Metrology (Packaged Commodities) Rules, 2011",
        "url": "https://example.test/base.pdf", "source": "https://example.test",
        "category": "legal_metrology_packaged_commodities"}
GARMENTS = {"title": "Download Advisory for Readymade Garments/ Hosiery products",
            "url": "https://example.test/garments.pdf", "source": "https://example.test",
            "category": "legal_metrology_packaged_commodities"}


def _rule(source="https://example.test/base.pdf", rule_id="LM_001"):
    return ComplianceRule(rule_id=rule_id, parameter="mrp", condition="must_exist",
                          requirement="The package must declare the retail sale price.",
                          source_document="Rules", source_pages=[1],
                          evidence_text="Every package shall declare the retail sale price.",
                          source_document_id=source, source_identity=f"{rule_id}-v1",
                          version="v1")


def _configure(monkeypatch):
    import app.refresh as refresh

    monkeypatch.setattr(refresh, "discover_documents", lambda url: [BASE, GARMENTS])
    return refresh


def test_unchanged_documents_skip_qwen_entirely(monkeypatch):
    refresh = _configure(monkeypatch)
    monkeypatch.setattr(refresh, "check_document_change", lambda document: "UNCHANGED")
    monkeypatch.setattr(refresh, "download_document",
                        lambda document: (_ for _ in ()).throw(AssertionError("no download")))
    monkeypatch.setattr(refresh, "process_pdf_document",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no Qwen")))
    monkeypatch.setattr(refresh, "get_active_rules", lambda: [])

    result = refresh.check_for_updates()

    assert result["status"] == "up_to_date"
    assert result["documents_checked"] == 2
    assert result["documents_changed"] == 0
    assert result["rules_updated"] is False
    assert result["unchanged_documents"] == 2


def test_changed_document_reprocessing_preserves_other_rules(tmp_path, monkeypatch):
    import app.repository.rule_repository as repository
    import app.refresh as refresh

    monkeypatch.setattr(repository, "RULES_DIR", tmp_path)
    monkeypatch.setattr(repository, "RULES_FILE", tmp_path / "compliance_rules.json")
    unrelated = _rule(source="https://example.test/other.pdf", rule_id="LM_OTHER")
    repository.save_rules([unrelated])
    _configure(monkeypatch)
    monkeypatch.setattr(refresh, "check_document_change",
                        lambda document: "CHANGED" if document["url"] == BASE["url"] else "UNCHANGED")
    monkeypatch.setattr(refresh, "download_document", lambda document: {
        "title": document["title"], "url": document["url"], "local_path": "x.pdf",
        "sha256": "newhash", "processed_sha256": None, "status": "CHANGED"})
    monkeypatch.setattr(refresh, "process_pdf_document",
                        lambda *args, **kwargs: [_rule(source=BASE["url"])])
    marked = []
    monkeypatch.setattr(refresh, "mark_document_processed", lambda url, sha: marked.append((url, sha)))

    result = refresh.check_for_updates()

    assert result["status"] == "updated"
    assert result["documents_changed"] == 1
    assert result["rules_updated"] is True
    assert (BASE["url"], "newhash") in marked
    kept = {rule.rule_id: rule for rule in repository.load_rules()}
    assert kept["LM_001"].status == "active"
    assert kept["LM_OTHER"].status == "active"
    assert kept["LM_OTHER"].source_document_id == "https://example.test/other.pdf"


def test_genuinely_empty_document_marks_processed_without_failure(monkeypatch):
    refresh = _configure(monkeypatch)
    monkeypatch.setattr(refresh, "check_document_change", lambda document: "CHANGED")
    monkeypatch.setattr(refresh, "download_document", lambda document: {
        "title": document["title"], "url": document["url"], "local_path": "x.pdf",
        "sha256": "hash", "processed_sha256": None, "status": "CHANGED"})
    monkeypatch.setattr(refresh, "get_active_rules", lambda: [])

    from app.extraction.rule_pipeline import NO_RULES_FOUND

    def fake_process(*args, **kwargs):
        outcome = kwargs.get("outcome")
        if outcome is not None:
            outcome.update(status=NO_RULES_FOUND, candidate_count=0, chunks_processed=1,
                           chunks_with_candidates=0, rules_generated=0, rules_validated=0,
                           rules_rejected=0, model_errors=[], validation_rejection_reasons=[])
        raise ValueError("No validated compliance rules were extracted.")

    monkeypatch.setattr(refresh, "process_pdf_document", fake_process)
    marked = []
    monkeypatch.setattr(refresh, "mark_document_processed", lambda url, sha: marked.append(url))
    monkeypatch.setattr(refresh, "update_rules",
                        lambda rules: (_ for _ in ()).throw(AssertionError("nothing to activate")))

    result = refresh.check_for_updates()

    assert result["status"] == "updated"
    assert result["rules_updated"] is False
    assert result["errors"] == []
    assert sorted(marked) == sorted([BASE["url"], GARMENTS["url"]])


def test_missing_target_fails_without_substitution(monkeypatch):
    refresh = _configure(monkeypatch)
    monkeypatch.setattr(refresh, "discover_documents", lambda url: [BASE])
    monkeypatch.setattr(refresh, "get_active_rules", lambda: [])

    result = refresh.check_for_updates()

    assert result["status"] == "failed"
    assert result["errors"]
    assert result["documents_changed"] == 0


def test_atomic_update_failure_marks_nothing(monkeypatch):
    refresh = _configure(monkeypatch)
    monkeypatch.setattr(refresh, "check_document_change", lambda document: "CHANGED")
    monkeypatch.setattr(refresh, "download_document", lambda document: {
        "title": document["title"], "url": document["url"], "local_path": "x.pdf",
        "sha256": "hash", "processed_sha256": None, "status": "CHANGED"})
    monkeypatch.setattr(refresh, "process_pdf_document", lambda *args, **kwargs: [_rule()])
    marked = []
    monkeypatch.setattr(refresh, "mark_document_processed", lambda url, sha: marked.append(url))
    monkeypatch.setattr(refresh, "update_rules",
                        lambda rules: (_ for _ in ()).throw(RuntimeError("disk full")))

    result = refresh.check_for_updates()

    assert result["status"] == "failed"
    assert marked == []
    assert result["errors"]


def test_manual_and_weekly_share_one_function(monkeypatch):
    import app.refresh as refresh
    import app.scheduler.scheduler as scheduler

    calls = []
    monkeypatch.setattr(refresh, "check_for_updates", lambda: calls.append(True) or {"status": "ok"})
    assert scheduler.check_for_updates() == {"status": "ok"}
    assert calls == [True]
