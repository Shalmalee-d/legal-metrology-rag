import pytest


def test_manual_refresh_queues_central_update_function(monkeypatch):
    import app.api.routes as routes
    started = []
    class Thread:
        def __init__(self, target, **kwargs): self.target = target
        def start(self): started.append(True)
        def is_alive(self): return bool(started)
    monkeypatch.setattr(routes, "Thread", Thread)
    monkeypatch.setattr(routes, "is_update_running", lambda: False)
    monkeypatch.setattr(routes, "get_refresh_status", lambda: {"current_stage": "idle"})
    monkeypatch.setattr(routes, "check_for_updates", lambda: {"status": "updated"})
    routes._refresh_thread = None
    result = routes.refresh_regulatory_knowledge()
    assert result["accepted"] is True
    assert result["status"] == "running"
    assert started == [True]
    assert routes._refresh_thread.target() == {"status": "updated"}
    routes._refresh_thread = None
    result = routes.check_regulatory_updates()
    assert result["accepted"] is True
    assert started == [True, True]


def test_active_rule_endpoint_returns_serialized_active_rules(tmp_path):
    import json

    import app.api.routes as routes

    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    common = [{
        "rule_id": "LM_A", "category": "common", "title": "T",
        "requirement": "Every package must declare the test item.",
        "applies_when": "Always.", "check_type": "must_exist", "check_parameters": {},
        "evidence_required": ["test_item"],
        "source": [{"document": "Rules", "rule_or_section": "Rule 6", "page": 1}],
        "source_text": "Every package shall declare the test item.",
        "effective_from": None, "status": "active",
    }]
    (rules_dir / "compliance_rules.json").write_text(
        json.dumps({"rules": common}), encoding="utf-8")

    from app.extraction.product_rule import ProductRule

    assert routes.active_rules() == [ProductRule(**common[0]).model_dump()]


def test_category_endpoints(tmp_path):
    import json

    from fastapi import HTTPException

    import app.api.routes as routes

    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    (rules_dir / "compliance_rules.json").write_text(
        json.dumps({"rules": []}), encoding="utf-8")
    food = [{
        "rule_id": "LM_F", "category": "food", "title": "T",
        "requirement": "Every package must declare the test item.",
        "applies_when": "Always.", "check_type": "must_exist", "check_parameters": {},
        "evidence_required": ["test_item"],
        "source": [{"document": "Rules", "rule_or_section": "Rule 6", "page": 1}],
        "source_text": "Every package shall declare the test item.",
        "effective_from": None, "status": "active",
    }]
    (rules_dir / "food.json").write_text(
        json.dumps({"rules": food}), encoding="utf-8")

    assert routes.rule_categories() == ["food"]
    from app.extraction.product_rule import ProductRule

    expected = [ProductRule(**food[0]).model_dump()]
    assert routes.category_rules("food") == expected
    assert routes.applicable_rules("food") == expected
    with pytest.raises(HTTPException) as missing:
        routes.category_rules("nope")
    assert missing.value.status_code == 404
    with pytest.raises(HTTPException) as missing_applicable:
        routes.applicable_rules("nope")
    assert missing_applicable.value.status_code == 404
    (rules_dir / "broken.json").write_text("{bad json", encoding="utf-8")
    with pytest.raises(HTTPException) as malformed:
        routes.category_rules("broken")
    assert malformed.value.status_code == 500
