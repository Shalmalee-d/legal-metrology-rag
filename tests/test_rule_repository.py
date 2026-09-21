from pathlib import Path

from app.extraction.rule_schema import ComplianceRule
from app.repository.rule_repository import (
    add_rule,
    load_rules,
    update_rule,
    get_active_rules,
)
import pytest


def test_rule_repository(tmp_path, monkeypatch):
    import app.repository.rule_repository as repository

    rules_file = tmp_path / "compliance_rules.json"

    monkeypatch.setattr(repository, "RULES_FILE", rules_file)
    monkeypatch.setattr(repository, "RULES_DIR", tmp_path)

    rule = ComplianceRule(
        rule_id="LM_001",
        parameter="mrp",
        condition="must_exist",
        requirement="MRP must be declared",
        source_document="Packaged Commodities Rules, 2011",
        source_pages=[25],
        evidence_text="MRP shall be declared.",
    )

    add_rule(rule)

    rules = load_rules()

    assert len(rules) == 1
    assert rules[0].rule_id == "LM_001"
    assert rules[0].parameter == "mrp"


def test_update_existing_rule(tmp_path, monkeypatch):
    import app.repository.rule_repository as repository

    rules_file = tmp_path / "compliance_rules.json"

    monkeypatch.setattr(repository, "RULES_FILE", rules_file)
    monkeypatch.setattr(repository, "RULES_DIR", tmp_path)

    rule_v1 = ComplianceRule(
        rule_id="LM_001",
        parameter="mrp",
        condition="must_exist",
        requirement="MRP must be declared",
        source_document="Packaged Commodities Rules, 2011",
        source_pages=[25],
        evidence_text="Every package shall declare the MRP.",
        version="2011",
    )

    rule_v2 = ComplianceRule(
        rule_id="LM_001",
        parameter="mrp",
        condition="must_match",
        requirement="MRP declaration must follow updated format",
        expected_value="updated format",
        source_document="Packaged Commodities Amendment Rules",
        source_pages=[10],
        evidence_text="Every package shall declare the updated MRP format.",
        version="amended",
    )

    add_rule(rule_v1)
    update_rule(rule_v2)

    rules = load_rules()

    assert len(rules) == 2

    old_rule = next(
        rule for rule in rules if rule.version == "2011"
    )

    new_rule = next(
        rule for rule in rules if rule.version == "amended"
    )

    assert old_rule.status == "superseded"
    assert new_rule.status == "active"
    assert new_rule.condition == "must_match"
    assert new_rule.version == "amended"

    active_rules = get_active_rules()

    assert len(active_rules) == 1
    assert active_rules[0].version == "amended"
    assert active_rules[0].status == "active"


def test_add_rule_merges_exact_version_evidence(tmp_path, monkeypatch):
    import app.repository.rule_repository as repository

    monkeypatch.setattr(repository, "RULES_FILE", tmp_path / "compliance_rules.json")
    monkeypatch.setattr(repository, "RULES_DIR", tmp_path)

    first = ComplianceRule(rule_id="LM_001", parameter="mrp", condition="must_exist",
        requirement="The package must declare MRP.", source_document="Rules", source_pages=[1], evidence_text="MRP shall be declared.", version="2011")
    second = first.model_copy(update={"source_pages": [2], "evidence_text": "Second evidence."})

    add_rule(first)
    add_rule(second)

    rules = load_rules()
    assert len(rules) == 1
    assert rules[0].source_pages == [1, 2]
    assert "Second evidence." in rules[0].evidence_text


def test_active_rules_exclude_future_and_expired_versions(tmp_path, monkeypatch):
    import app.repository.rule_repository as repository

    monkeypatch.setattr(repository, "RULES_FILE", tmp_path / "compliance_rules.json")
    monkeypatch.setattr(repository, "RULES_DIR", tmp_path)
    future = ComplianceRule(rule_id="LM_FUTURE", parameter="mrp", condition="must_exist",
        requirement="The package must declare MRP.", source_document="Rules", source_pages=[1], evidence_text="MRP shall be declared.", effective_from="2999-01-01")
    expired = ComplianceRule(rule_id="LM_EXPIRED", parameter="name", condition="must_exist",
        requirement="The package must declare manufacturer name.", source_document="Rules", source_pages=[1], evidence_text="Every package shall declare manufacturer name.", effective_until="2000-01-01")
    add_rule(future)
    add_rule(expired)

    assert repository.get_active_rules() == []


def test_invalid_rule_cannot_be_persisted_or_returned_active(tmp_path, monkeypatch):
    import app.repository.rule_repository as repository
    monkeypatch.setattr(repository, "RULES_FILE", tmp_path / "compliance_rules.json")
    monkeypatch.setattr(repository, "RULES_DIR", tmp_path)
    invalid = ComplianceRule(rule_id="bad", parameter="rule heading", condition="must_exist",
        source_document="Rules", source_pages=[1], evidence_text="These Rules shall be called Rules.")
    with pytest.raises(ValueError, match="invalid compliance rules"):
        repository.save_rules([invalid])
    # Legacy malformed JSON must not become an active API rule merely because
    # it satisfies the permissive transport schema.
    repository.RULES_FILE.write_text("[" + invalid.model_dump_json() + "]", encoding="utf-8")
    assert repository.get_active_rules() == []


def test_update_retires_rules_removed_from_changed_source(tmp_path, monkeypatch):
    import app.repository.rule_repository as repository
    monkeypatch.setattr(repository, "RULES_FILE", tmp_path / "compliance_rules.json")
    monkeypatch.setattr(repository, "RULES_DIR", tmp_path)
    base = ComplianceRule(rule_id="kept", parameter="mrp", condition="must_exist",
        requirement="The package must declare MRP.", source_document="Rules", source_pages=[1],
        evidence_text="MRP shall be declared.", source_document_id="https://example/rules", source_identity="keep", version="v1")
    removed = base.model_copy(update={"rule_id": "removed", "parameter": "manufacturer", "requirement": "The package must declare manufacturer.", "evidence_text": "Every package shall declare manufacturer.", "source_identity": "removed"})
    unrelated = base.model_copy(update={"rule_id": "other", "source_document_id": "https://example/other", "source_identity": "other"})
    repository.save_rules([base, removed, unrelated])
    replacement = base.model_copy(update={"version": "v2"})
    result = repository.update_rules([replacement])
    rules = {rule.rule_id: rule for rule in repository.load_rules()}
    assert rules["removed"].status == "superseded"
    assert rules["other"].status == "active"
    assert result["rules_superseded"] == 1

if __name__ == "__main__":
    test_rule_repository(Path("."))
