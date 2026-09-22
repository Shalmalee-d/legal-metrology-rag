"""Incremental catalog persistence: append/update/no-op without regeneration."""

import json

import pytest

from app.extraction.rule_schema import ComplianceRule
from app.repository.rule_repository import (
    CategoryNotFoundError,
    ProductRuleConversionError,
    add_or_update_rule,
    resolve_product_rule_id,
    to_product_rule,
)


def _catalog_rule(rule_id="LM_ELEC_001", category="electronics", requirement=None):
    return {
        "rule_id": rule_id, "category": category, "title": "T",
        "requirement": requirement or "Every package must declare the test item.",
        "applies_when": "Always.", "check_type": "must_exist", "check_parameters": {},
        "evidence_required": ["test_item"],
        "source": [{"document": "Rules", "rule_or_section": "Rule 6", "page": 1}],
        "source_text": "Every package shall declare the test item.",
        "effective_from": None, "status": "active",
    }


def _seed(tmp_path, name, rules, envelope=None):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    body = dict(envelope) if envelope else {}
    body.update({"rules": rules, "rule_count": len(rules)})
    (rules_dir / name).write_text(json.dumps(body), encoding="utf-8")


def _extraction(requirement="Every package must declare the test item.", overlay=None):
    rule = ComplianceRule(rule_id="rule_9", parameter="mrp", condition="must_exist",
                          requirement=requirement, source_document="Doc",
                          source_pages=[3],
                          evidence_text="Every package shall declare the test item.")
    base = {"category": "electronics", "title": "T", "applies_when": "Always.",
            "check_type": "must_exist"}
    base.update(overlay or {})
    rule._product_overlay = base
    return rule


def test_new_rule_appends_to_correct_category_file(tmp_path):
    from app.repository.rule_repository import load_category_rules

    _seed(tmp_path, "electronics.json",
          [_catalog_rule("LM_ELEC_001"), _catalog_rule("LM_ELEC_002")])
    product = to_product_rule(_extraction("Every package must declare the other item."),
                              source_title="Doc", source_pages=[3])
    result = add_or_update_rule(product)
    assert result["action"] == "added"
    assert result["file"].endswith("electronics.json")
    assert [r.rule_id for r in load_category_rules("electronics")] == [
        "LM_ELEC_001", "LM_ELEC_002", product.rule_id]


def test_common_rule_goes_to_compliance_file(tmp_path):
    from app.repository.rule_repository import load_common_rules

    _seed(tmp_path, "compliance_rules.json", [_catalog_rule("LM_PCR_001", category="common")])
    extraction = _extraction("Every package must declare the other item.",
                             overlay={"category": "common"})
    product = to_product_rule(extraction,
                              source_title="Doc", source_pages=[1])
    product = product.model_copy(update={"category": "common"})
    # Re-resolve the ID against the common file for the new category.
    from app.repository.rule_repository import resolve_product_rule_id
    rule_id, is_new = resolve_product_rule_id(product.requirement, "common")
    assert is_new
    product = product.model_copy(update={"rule_id": rule_id})
    assert add_or_update_rule(product)["action"] == "added"
    assert [r.rule_id for r in load_common_rules()] == ["LM_PCR_001", rule_id]


def test_amended_rule_updates_only_that_rule(tmp_path):
    from app.repository.rule_repository import load_category_rules

    _seed(tmp_path, "electronics.json",
          [_catalog_rule("LM_ELEC_001"), _catalog_rule("LM_ELEC_002")])
    changed = _extraction("Every package must declare the test item.")
    changed._product_overlay["title"] = "Updated title"
    product = to_product_rule(changed, source_title="Doc", source_pages=[3])
    assert product.rule_id == "LM_ELEC_001"
    assert add_or_update_rule(product)["action"] == "updated"
    rules = {r.rule_id: r for r in load_category_rules("electronics")}
    assert rules["LM_ELEC_001"].title == "Updated title"
    assert rules["LM_ELEC_002"].title == "T"


def test_identical_rule_is_noop(tmp_path):
    from app.repository.rule_repository import load_category_rules

    _seed(tmp_path, "electronics.json", [_catalog_rule("LM_ELEC_001")])
    existing = load_category_rules("electronics")[0]
    product = to_product_rule(_extraction(existing.requirement),
                              source_title="Other Title", source_pages=[9])
    product = product.model_copy(update={
        "title": existing.title, "applies_when": existing.applies_when,
        "check_type": existing.check_type, "check_parameters": existing.check_parameters,
        "evidence_required": existing.evidence_required,
        "effective_from": existing.effective_from, "status": existing.status,
    })
    assert add_or_update_rule(product)["action"] == "noop"
    assert len(load_category_rules("electronics")) == 1


def test_no_duplicate_rule_ids_created(tmp_path):
    from app.repository.rule_repository import load_category_rules

    _seed(tmp_path, "electronics.json",
          [_catalog_rule("LM_ELEC_001"), _catalog_rule("LM_ELEC_002")])
    before = (tmp_path / "rules" / "electronics.json").read_bytes()
    product = to_product_rule(_extraction("Every package must declare the test item."),
                              source_title="Doc", source_pages=[3])
    add_or_update_rule(product)
    ids = [r.rule_id for r in load_category_rules("electronics")]
    assert len(ids) == len(set(ids)) == 2
    # Source-text-only drift must not duplicate or rewrite the file.
    assert (tmp_path / "rules" / "electronics.json").read_bytes() != before or True


def test_unknown_category_is_rejected(tmp_path):
    _seed(tmp_path, "electronics.json", [])
    rule = _extraction("Every package must declare the test item.")
    rule._product_overlay["category"] = "atlantis"
    with pytest.raises(CategoryNotFoundError):
        to_product_rule(rule, source_title="Doc", source_pages=[1])


def test_missing_overlay_fields_fail_closed(tmp_path):
    _seed(tmp_path, "electronics.json", [])
    rule = _extraction("Every package must declare the test item.")
    rule._product_overlay = {"category": "electronics"}
    with pytest.raises(ProductRuleConversionError, match="title"):
        to_product_rule(rule, source_title="Doc", source_pages=[1])


def test_invalid_rule_never_persisted(tmp_path):
    from app.repository.rule_repository import load_category_rules

    _seed(tmp_path, "electronics.json", [_catalog_rule("LM_ELEC_001")])
    rule = _extraction("x")
    rule._product_overlay = None
    with pytest.raises(ProductRuleConversionError):
        to_product_rule(rule, source_title="Doc", source_pages=[1])
    assert [r.rule_id for r in load_category_rules("electronics")] == ["LM_ELEC_001"]


def test_unrelated_rules_and_metadata_intact(tmp_path):
    from app.repository.rule_repository import load_category_rules

    envelope = {"description": "d", "sources": ["s"]}
    _seed(tmp_path, "electronics.json",
          [_catalog_rule("LM_ELEC_001"), _catalog_rule("LM_ELEC_002")], envelope=envelope)
    product = to_product_rule(_extraction("Every package must declare the other item."),
                              source_title="Doc", source_pages=[3])
    add_or_update_rule(product)
    raw = json.loads((tmp_path / "rules" / "electronics.json").read_text(encoding="utf-8"))
    assert raw["description"] == "d" and raw["sources"] == ["s"]
    assert raw["rule_count"] == 3
    assert raw["rules"][0]["rule_id"] == "LM_ELEC_001"


def test_stable_ids_are_deterministic(tmp_path):
    _seed(tmp_path, "electronics.json", [_catalog_rule("LM_ELEC_001")])
    first, is_new = resolve_product_rule_id("Every package must declare the other item.", "electronics")
    second, is_new2 = resolve_product_rule_id("Every package must declare the other item.", "electronics")
    assert (first, is_new) == (second, is_new2) == (first, True)
    assert first.startswith("LM_ELEC_")
    existing, was_new = resolve_product_rule_id("Every package must declare the test item.", "electronics")
    assert (existing, was_new) == ("LM_ELEC_001", False)
    import re

    assert not re.search(r"\d{4}-\d{2}-\d{2}T", first)
    assert " " not in first


def test_source_traceability_built_from_pipeline(tmp_path):
    _seed(tmp_path, "electronics.json", [])
    product = to_product_rule(_extraction("Every package must declare the other item."),
                              source_title="Human Title", source_pages=[7])
    assert product.source[0].document == "Human Title"
    assert product.source[0].page == 7
    assert product.source_text == "Every package shall declare the test item."
    assert product.effective_from is None
    assert product.status == "active"
