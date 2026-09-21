"""Bounded Qwen retry, error classification, and call-gating regression tests."""

import httpx
import pytest

from app.extraction.llm_rule_generator import (
    OllamaModelError,
    OllamaRuleGenerator,
    OllamaUnavailableError,
    generate_rules_from_chunk,
)


def _chunk(text="The retail sale price shall be declared."):
    return {"chunk_id": "one", "source_pages": [44, 45], "text": text}


def _valid_body(evidence="The retail sale price shall be declared."):
    import json as _json  # local import to keep module import light
    return {"response": _json.dumps({"rules": [{
        "rule_id": "LM_MRP_001", "parameter": "mrp", "condition": "must_exist",
        "requirement": "The package must declare the retail sale price.",
        "evidence_text": evidence}]})}


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def _tracker(monkeypatch, responses):
    import app.extraction.llm_rule_generator as generator

    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs.get("json", {}))
        outcome = responses[min(len(calls) - 1, len(responses) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)

    monkeypatch.setattr(generator.httpx, "post", fake_post)
    return calls


def test_valid_first_response_uses_single_call(monkeypatch):
    import json

    calls = _tracker(monkeypatch, [{"response": json.dumps({"rules": [{
        "rule_id": "LM_MRP_001", "parameter": "mrp", "condition": "must_exist",
        "requirement": "The package must declare the retail sale price.",
        "evidence_text": "The retail sale price shall be declared."}]})}])
    rules = generate_rules_from_chunk(_chunk(), "Rules")
    assert len(rules) == 1
    assert len(calls) == 1


def test_malformed_then_valid_succeeds_after_one_retry(monkeypatch):
    import json

    calls = _tracker(monkeypatch, [
        {"response": '{"rules": [{"rule_id": "LM_MRP_001", "parameter": "mrp",'},
        {"response": json.dumps({"rules": [{
            "rule_id": "LM_MRP_001", "parameter": "mrp", "condition": "must_exist",
            "requirement": "The package must declare the retail sale price.",
            "evidence_text": "The retail sale price shall be declared."}]})},
    ])
    rules = generate_rules_from_chunk(_chunk(), "Rules")
    assert len(rules) == 1
    assert len(calls) == 2
    # Same deterministic prompt/schema on retry.
    assert calls[0]["prompt"] == calls[1]["prompt"]
    assert calls[0]["format"] == calls[1]["format"]


def test_malformed_twice_is_model_json_error(monkeypatch):
    calls = _tracker(monkeypatch, [
        {"response": '{"rules": [{"broken"'},
        {"response": '{"rules": [{"broken"'},
    ])
    with pytest.raises(OllamaModelError) as raised:
        generate_rules_from_chunk(_chunk(), "Rules")
    assert raised.value.kind == OllamaUnavailableError.MODEL_JSON_ERROR
    assert len(calls) == 2


def test_schema_invalid_twice_is_model_schema_error(monkeypatch):
    import json

    calls = _tracker(monkeypatch, [
        {"response": json.dumps({"rules": "not-a-list"})},
        {"response": json.dumps({"rules": {"not": "a-list"}})},
    ])
    with pytest.raises(OllamaModelError) as raised:
        generate_rules_from_chunk(_chunk(), "Rules")
    assert raised.value.kind == OllamaUnavailableError.MODEL_SCHEMA_ERROR
    assert len(calls) == 2


def test_model_error_remains_ollama_unavailable_compatible(monkeypatch):
    _tracker(monkeypatch, [{"response": "not json"}, {"response": "not json"}])
    with pytest.raises(OllamaUnavailableError):
        generate_rules_from_chunk(_chunk(), "Rules")


def test_validator_rejection_does_not_retry(monkeypatch):
    import json

    chunk = {"chunk_id": "m", "source_pages": [1],
             "text": "Every package shall bear the official rule name on the package."}
    calls = _tracker(monkeypatch, [{"response": json.dumps({"rules": [{
        "rule_id": "RULE_NAME", "parameter": "rule_name", "condition": "must_equal",
        "requirement": "The rules must have this name.",
        "expected_value": "Legal Metrology Rules, 2025",
        "evidence_text": chunk["text"]}]})}])
    assert generate_rules_from_chunk(chunk, "Rules") == []
    assert len(calls) == 1


def test_empty_rules_does_not_retry(monkeypatch):
    import json

    calls = _tracker(monkeypatch, [{"response": json.dumps({"rules": []})}])
    assert generate_rules_from_chunk(_chunk(), "Rules") == []
    assert len(calls) == 1


def test_no_normative_candidates_skips_model_call(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    called = []
    monkeypatch.setattr(generator.httpx, "post",
                        lambda *a, **k: called.append(True) or _Response({"response": "{}"}))
    chunk = {"chunk_id": "m", "source_pages": [1],
             "text": "Government of India\nMinistry of Consumer Affairs\nPhone 011-23389489"}
    assert generate_rules_from_chunk(chunk, "Notice") == []
    assert called == []


def test_hindi_normative_text_still_calls_model(monkeypatch):
    import json

    hindi = "पैकेज पर अधिकतम खुदरा मूल्य अंकित होना चाहिए"
    calls = _tracker(monkeypatch, [{"response": json.dumps({"rules": []})}])
    assert generate_rules_from_chunk(
        {"chunk_id": "h", "source_pages": [1], "text": hindi}, "Hindi Rules") == []
    assert len(calls) == 1


def test_timeout_is_not_retried_and_keeps_policy(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    calls = []
    monkeypatch.setattr(
        generator.httpx, "post",
        lambda *a, **k: (calls.append(True), (_ for _ in ()).throw(httpx.ReadTimeout("slow")))[1])
    with pytest.raises(OllamaUnavailableError, match="read timeout") as raised:
        generate_rules_from_chunk(_chunk(), "Rules")
    assert raised.value.kind == OllamaUnavailableError.MODEL_TIMEOUT
    assert len(calls) == 1


def test_transport_error_is_not_retried(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise httpx.ConnectError("down")

    monkeypatch.setattr(generator.httpx, "post", fail)
    with pytest.raises(OllamaUnavailableError) as raised:
        generate_rules_from_chunk(_chunk(), "Rules")
    assert raised.value.kind == OllamaUnavailableError.MODEL_TRANSPORT_ERROR
    assert len(calls) == 1
