"""Curated product-rule catalog: common/category/applicable loading."""

import json

import pytest

from app.repository.rule_repository import (
    CategoryNotFoundError,
    list_categories,
    load_applicable_rules,
    load_category_rules,
    load_common_rules,
)


def _product_rule(rule_id="LM_T_001", **updates):
    rule = {
        "rule_id": rule_id,
        "category": "common",
        "title": "Test rule",
        "requirement": "Every package must declare the test item.",
        "applies_when": "Always applies.",
        "check_type": "must_exist",
        "check_parameters": {},
        "evidence_required": ["test_item"],
        "source": [{"document": "Rules, 2011", "rule_or_section": "Rule 6", "page": 1}],
        "source_text": "Every package shall declare the test item.",
        "effective_from": None,
        "status": "active",
    }
    rule.update(updates)
    return rule


def _rules_dir(tmp_path):
    path = tmp_path / "rules"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_catalog(tmp_path, name, rules):
    path = _rules_dir(tmp_path) / name
    path.write_text(json.dumps({"rules": rules}, ensure_ascii=False), encoding="utf-8")
    return path


def _write_common(tmp_path, rules):
    return _write_catalog(tmp_path, "compliance_rules.json", rules)


def test_load_common_rules(tmp_path):
    _write_common(tmp_path, [_product_rule("LM_A"), _product_rule("LM_B")])
    rules = load_common_rules()
    assert [rule.rule_id for rule in rules] == ["LM_A", "LM_B"]
    assert all(rule.category == "common" for rule in rules)


def test_applicable_excludes_scheduled_future_rules(tmp_path):
    _write_common(tmp_path, [
        _product_rule("LM_NOW"),
        _product_rule("LM_FUTURE", effective_from="2999-01-01", status="scheduled"),
    ])
    assert [rule.rule_id for rule in load_applicable_rules(None)] == ["LM_NOW"]


def test_category_and_applicable_combination(tmp_path):
    _write_common(tmp_path, [_product_rule("LM_A"), _product_rule("LM_B")])
    _write_catalog(tmp_path, "food.json", [_product_rule("LM_F", category="food")])
    _write_catalog(tmp_path, "baby_and_childcare.json", [])
    assert [rule.rule_id for rule in load_category_rules("food")] == ["LM_F"]
    assert load_category_rules("baby_and_childcare") == []
    assert [rule.rule_id for rule in load_applicable_rules("food")] == ["LM_A", "LM_B", "LM_F"]
    assert [rule.rule_id for rule in load_applicable_rules("baby_and_childcare")] == ["LM_A", "LM_B"]


def test_unknown_category_raises_not_found(tmp_path):
    _write_common(tmp_path, [_product_rule()])
    with pytest.raises(CategoryNotFoundError):
        load_category_rules("no_such_category")
    with pytest.raises(CategoryNotFoundError):
        load_applicable_rules("no_such_category")


def test_malformed_category_file_is_isolated(tmp_path):
    _write_common(tmp_path, [_product_rule()])
    (_rules_dir(tmp_path) / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_category_rules("broken")
    # Other files keep loading.
    assert len(load_common_rules()) == 1
    assert list_categories() == ["broken"]


def test_missing_required_field_rejected(tmp_path):
    bad = _product_rule()
    del bad["requirement"]
    _write_catalog(tmp_path, "bad.json", [bad])
    with pytest.raises(ValueError, match="requirement"):
        load_category_rules("bad")


def test_missing_common_file_raises(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_common_rules()


def test_list_categories_ignores_reserved_names(tmp_path):
    _write_common(tmp_path, [_product_rule()])
    for name in ("README.json", "rag_generated.json", "backup.bak.1.json", ".hidden.json"):
        (_rules_dir(tmp_path) / name).write_text("{}", encoding="utf-8")
    _write_catalog(tmp_path, "food.json", [])
    assert list_categories() == ["food"]


def test_returned_data_is_json_serializable(tmp_path):
    _write_common(tmp_path, [_product_rule()])
    _write_catalog(tmp_path, "food.json", [_product_rule("LM_F", category="food")])
    payload = [rule.model_dump() for rule in load_applicable_rules("food")]
    assert json.loads(json.dumps(payload))
