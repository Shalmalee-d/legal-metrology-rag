"""Repository boundary: invalid quality can never be activated."""

import pytest

from app.extraction.rule_schema import ComplianceRule
from app.validation.rule_validator import validate_rule


def _valid(rule_id="LM_OK", source_id="https://example/doc1", identity="ok-1", version="h1"):
    return ComplianceRule(
        rule_id=rule_id,
        parameter="mrp",
        condition="must_exist",
        applies_to={"product_type": "packaged_commodity"},
        requirement="The package must declare the retail sale price.",
        source_document="Rules",
        source_pages=[1],
        evidence_text="Every package shall declare the retail sale price.",
        source_document_id=source_id,
        source_identity=identity,
        version=version,
    )


def test_invalid_rules_cannot_be_saved_or_activated():
    import app.repository.rule_repository as repository

    # Null requirement (legacy shape) is invalid.
    legacy = _valid().model_copy(update={"requirement": None})
    assert validate_rule(legacy)
    with pytest.raises(ValueError, match="invalid compliance rules"):
        repository.save_rules([legacy])
    with pytest.raises(ValueError, match="invalid compliance rules"):
        repository.update_rules([legacy])

    # Metadata-only candidate is invalid.
    meta = _valid(rule_id="LM_META").model_copy(
        update={
            "parameter": "rule_name",
            "condition": "must_equal",
            "requirement": "The rules must have this name.",
            "expected_value": "Rules, 2025",
            "evidence_text": "These rules may be called the Packaged Commodities Rules, 2025.",
        }
    )
    assert validate_rule(meta)
    with pytest.raises(ValueError, match="invalid"):
        repository.save_rules([meta])


def test_valid_rules_persisted_and_unrelated_preserved():
    import app.repository.rule_repository as repository

    unrelated = _valid(rule_id="LM_UN", source_id="https://example/other", identity="other-1", version="h0")
    repository.save_rules([unrelated])

    changed = _valid(rule_id="LM_NEW", source_id="https://example/doc1", identity="new-1", version="h1")
    result = repository.update_rules([changed])

    loaded = {r.rule_id: r for r in repository.load_rules()}
    assert loaded["LM_UN"].status == "active"
    assert loaded["LM_NEW"].status == "active"
    assert result["rules_added"] >= 1


def test_source_specific_update_does_not_modify_unrelated():
    import app.repository.rule_repository as repository

    base = _valid(rule_id="LM_A", source_id="https://example/A", identity="a-1", version="v1")
    other = _valid(rule_id="LM_B", source_id="https://example/B", identity="b-1", version="v1")
    repository.save_rules([base, other])

    replacement = base.model_copy(update={"version": "v2"})
    # Keep requirement/evidence identical so only version changes.
    repository.update_rules([replacement])

    after = {r.rule_id: r for r in repository.load_rules()}
    assert after["LM_B"].version == "v1"
    assert after["LM_B"].status == "active"
    assert after["LM_A"].version == "v2"
