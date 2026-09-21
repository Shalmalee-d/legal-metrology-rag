def test_manual_refresh_queues_central_update_function(monkeypatch):
    import app.api.routes as routes
    started = []
    class Thread:
        def __init__(self, target, **kwargs): self.target = target
        def start(self): started.append(True)
        def is_alive(self): return bool(started)
    monkeypatch.setattr(routes, "Thread", Thread)
    monkeypatch.setattr(routes, "is_update_running", lambda: False)
    monkeypatch.setattr(routes, "get_update_status", lambda: {"current_stage": "idle"})
    routes._refresh_thread = None
    result = routes.refresh_regulatory_knowledge()
    assert result["accepted"] is True
    assert result["status"] == "running"
    assert started == [True]


def test_active_rule_endpoint_returns_serialized_active_rules(monkeypatch):
    import app.api.routes as routes
    from app.extraction.rule_schema import ComplianceRule

    rule = ComplianceRule(rule_id="LM_001", parameter="mrp", condition="must_exist",
        source_document="Rules", source_pages=[1], evidence_text="MRP shall be declared.")
    monkeypatch.setattr(routes, "get_active_rules", lambda: [rule])

    assert routes.active_rules() == [rule.model_dump()]
