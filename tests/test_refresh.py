"""Two-document fingerprint-gated refresh with incremental catalog persistence."""

import json

import pytest

from app.extraction.rule_schema import ComplianceRule


BASE = {"title": "Download The Legal Metrology (Packaged Commodities) Rules, 2011",
        "url": "https://example.test/base.pdf", "source": "https://example.test",
        "category": "legal_metrology_packaged_commodities"}
GARMENTS = {"title": "Download Advisory for Readymade Garments/ Hosiery products",
            "url": "https://example.test/garments.pdf", "source": "https://example.test",
            "category": "legal_metrology_packaged_commodities"}


def _extraction_rule():
    return ComplianceRule(rule_id="LM_001", parameter="mrp", condition="must_exist",
                          requirement="The package must declare the retail sale price.",
                          source_document="Rules", source_pages=[1],
                          evidence_text="Every package shall declare the retail sale price.")


def _with_overlay(rule, category="common"):
    rule._product_overlay = {
        "category": category,
        "title": "Retail sale price declaration",
        "applies_when": "Always applies.",
        "check_type": "must_exist",
    }
    return rule


def _product_fixture(rule_id="LM_X", category="common"):
    return {
        "rule_id": rule_id, "category": category, "title": "T",
        "requirement": "Every package must declare the test item.",
        "applies_when": "Always.", "check_type": "must_exist", "check_parameters": {},
        "evidence_required": ["test_item"],
        "source": [{"document": "Rules", "rule_or_section": "Rule 6", "page": 1}],
        "source_text": "Every package shall declare the test item.",
        "effective_from": None, "status": "active",
    }


def _seed_catalog(tmp_path, common=None, categories=None):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    (rules_dir / "compliance_rules.json").write_text(
        json.dumps({"rules": common if common is not None else []}), encoding="utf-8")
    for name, rules in (categories or {}).items():
        (rules_dir / f"{name}.json").write_text(
            json.dumps({"rules": rules}), encoding="utf-8")


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
    monkeypatch.setattr(refresh, "load_all_active_rules", lambda: [])

    result = refresh.check_for_updates()

    assert result["status"] == "up_to_date"
    assert result["documents_checked"] == 2
    assert result["documents_changed"] == 0
    assert result["rules_updated"] is False
    assert result["unchanged_documents"] == 2


def test_changed_document_appends_and_preserves_other_rules(tmp_path, monkeypatch):
    import app.refresh as refresh

    _seed_catalog(tmp_path, common=[_product_fixture("LM_OLD")],
                  categories={"food": [_product_fixture("LM_F", category="food")]})
    _configure(monkeypatch)
    monkeypatch.setattr(refresh, "check_document_change",
                        lambda document: "CHANGED" if document["url"] == BASE["url"] else "UNCHANGED")
    monkeypatch.setattr(refresh, "download_document", lambda document: {
        "title": document["title"], "url": document["url"], "local_path": "x.pdf",
        "sha256": "newhash", "processed_sha256": None, "status": "CHANGED"})
    monkeypatch.setattr(refresh, "process_pdf_document",
                        lambda *args, **kwargs: [_with_overlay(_extraction_rule())])
    marked = []
    monkeypatch.setattr(refresh, "mark_document_processed", lambda url, sha: marked.append((url, sha)))

    result = refresh.check_for_updates()

    assert result["status"] == "updated"
    assert result["documents_changed"] == 1
    assert result["rules_updated"] is True
    assert (BASE["url"], "newhash") in marked
    from app.repository.rule_repository import load_applicable_rules, load_category_rules

    ids = [rule.rule_id for rule in load_applicable_rules(None)]
    assert "LM_OLD" in ids
    assert len([rule for rule in load_category_rules("food") if rule.rule_id == "LM_F"]) == 1


def test_genuinely_empty_document_marks_processed_without_failure(monkeypatch):
    refresh = _configure(monkeypatch)
    monkeypatch.setattr(refresh, "check_document_change", lambda document: "CHANGED")
    monkeypatch.setattr(refresh, "download_document", lambda document: {
        "title": document["title"], "url": document["url"], "local_path": "x.pdf",
        "sha256": "hash", "processed_sha256": None, "status": "CHANGED"})
    monkeypatch.setattr(refresh, "load_all_active_rules", lambda: [])

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

    result = refresh.check_for_updates()

    assert result["status"] == "updated"
    assert result["rules_updated"] is False
    assert result["errors"] == []
    assert sorted(marked) == sorted([BASE["url"], GARMENTS["url"]])


def test_missing_target_fails_without_substitution(monkeypatch):
    refresh = _configure(monkeypatch)
    monkeypatch.setattr(refresh, "discover_documents", lambda url: [BASE])
    monkeypatch.setattr(refresh, "load_all_active_rules", lambda: [])

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
    monkeypatch.setattr(refresh, "process_pdf_document",
                        lambda *args, **kwargs: [_with_overlay(_extraction_rule())])
    marked = []
    monkeypatch.setattr(refresh, "mark_document_processed", lambda url, sha: marked.append(url))
    monkeypatch.setattr(refresh, "add_or_update_rule",
                        lambda rule: (_ for _ in ()).throw(RuntimeError("disk full")))

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
