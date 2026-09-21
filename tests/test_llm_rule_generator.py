import json

import pytest

from app.extraction.llm_rule_generator import (
    OllamaRuleGenerator,
    OllamaUnavailableError,
    generate_rules_from_chunk,
)


def _chunk():
    return {"chunk_id": "one", "source_pages": [44, 45], "text": "The retail sale price shall be declared."}


class _Response:
    def __init__(self, payload): self.payload = payload
    def raise_for_status(self): pass
    def json(self): return self.payload


def test_valid_ollama_rule_inherits_source_metadata(monkeypatch):
    import app.extraction.llm_rule_generator as generator
    body = {"rules": [{"rule_id": "LM_MRP_001", "parameter": "mrp", "condition": "must_exist", "requirement": "The package must declare the retail sale price.", "evidence_text": "The retail sale price shall be declared."}]}
    monkeypatch.setattr(generator.httpx, "post", lambda *args, **kwargs: _Response({"response": json.dumps(body)}))

    rules = generate_rules_from_chunk(_chunk(), "Rules")

    assert rules[0].source_document == "Rules"
    assert rules[0].source_pages == [44, 45]


def test_public_generator_class_uses_local_http_configuration(monkeypatch):
    import app.extraction.llm_rule_generator as generator
    body = {"rules": [{"rule_id": "LM_MRP_001", "parameter": "mrp", "condition": "must_exist", "requirement": "The package must declare the retail sale price.", "evidence_text": "The retail sale price shall be declared."}]}
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Response({"response": json.dumps(body)})

    monkeypatch.setattr(generator.httpx, "post", fake_post)
    rules = OllamaRuleGenerator("http://localhost:11434", "qwen3:4b", 5).generate_rules(_chunk(), "Rules")

    assert rules[0].rule_id == "LM_MRP_001"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["json"]["model"] == "qwen3:4b"
    assert captured["json"]["think"] is False
    assert captured["json"]["options"]["num_predict"] > 0
    assert captured["timeout"].read == 5


def test_read_timeout_is_preserved_as_typed_local_ollama_failure(monkeypatch):
    import httpx
    import app.extraction.llm_rule_generator as generator

    monkeypatch.setattr(generator.httpx, "post", lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ReadTimeout("slow model")))

    with pytest.raises(OllamaUnavailableError, match="read timeout") as raised:
        generate_rules_from_chunk(_chunk(), "Rules")

    assert raised.value.error_type == "ReadTimeout"
    assert raised.value.timeout_type == "read"


def test_malformed_ollama_json_is_rejected(monkeypatch):
    import app.extraction.llm_rule_generator as generator
    monkeypatch.setattr(generator.httpx, "post", lambda *args, **kwargs: _Response({"response": "not json"}))

    with pytest.raises(OllamaUnavailableError):
        generate_rules_from_chunk(_chunk(), "Rules")


def test_unsupported_evidence_is_rejected(monkeypatch):
    import app.extraction.llm_rule_generator as generator
    body = {"rules": [{"rule_id": "LM_001", "parameter": "mrp", "condition": "must_exist", "evidence_text": "Invented requirement."}]}
    monkeypatch.setattr(generator.httpx, "post", lambda *args, **kwargs: _Response({"response": json.dumps(body)}))

    assert generate_rules_from_chunk(_chunk(), "Rules") == []


def test_formatting_equivalent_evidence_is_grounded_to_exact_source(monkeypatch):
    import app.extraction.llm_rule_generator as generator
    chunk = {
        "chunk_id": "amendment",
        "source_pages": [2],
        "text": "the declaration\nof the quantity under\nthese rules shall not contain any word",
    }
    body = {"rules": [{
        "rule_id": "LM_AMENDMENT_001",
        "parameter": "quantity declaration",
        "condition": "must_not_exist",
        "requirement": "The package must not contain the prohibited wording in its quantity declaration.",
        "evidence_text": "the declaration of the quantity under these rules shall not contain any word",
    }]}
    monkeypatch.setattr(generator.httpx, "post", lambda *args, **kwargs: _Response({"response": json.dumps(body)}))

    rules = generate_rules_from_chunk(chunk, "Third Amendment Rules, 2011")

    assert rules[0].evidence_text == chunk["text"]


def test_paraphrased_evidence_remains_rejected(monkeypatch):
    import app.extraction.llm_rule_generator as generator
    chunk = {
        "chunk_id": "amendment",
        "source_pages": [2],
        "text": "the declaration\nof the quantity under\nthese rules shall not contain any word",
    }
    body = {"rules": [{
        "rule_id": "LM_AMENDMENT_001",
        "parameter": "quantity declaration",
        "condition": "must_not_exist",
        "evidence_text": "quantity labels must never use misleading wording",
    }]}
    monkeypatch.setattr(generator.httpx, "post", lambda *args, **kwargs: _Response({"response": json.dumps(body)}))

    assert generate_rules_from_chunk(chunk, "Third Amendment Rules, 2011") == []


def test_non_product_metadata_chunk_returns_no_rules(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    chunk = {
        "chunk_id": "metadata",
        "source_pages": [1],
        "text": "These rules may be called the Legal Metrology Rules, 2025.",
    }
    body = [{
        "rule_id": "RULE_NAME",
        "parameter": "rule_name",
        "condition": "must_equal",
        "requirement": "The rules must have this name.",
        "expected_value": "Legal Metrology Rules, 2025",
        "evidence_text": chunk["text"],
    }]
    monkeypatch.setattr(generator.httpx, "post", lambda *args, **kwargs: _Response({"response": json.dumps(body)}))

    assert generate_rules_from_chunk(chunk, "Rules") == []
