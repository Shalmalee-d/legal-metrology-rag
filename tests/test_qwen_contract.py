"""Qwen3 4B response-contract regression: compact deterministic JSON only."""

import json
import pytest

from app.extraction.llm_rule_generator import (
    OLLAMA_RESPONSE_SCHEMA,
    OllamaRuleGenerator,
    OllamaUnavailableError,
    _build_request,
    _prompt,
    generate_rules_from_chunk,
)


def _chunk():
    return {
        "text": "The retail sale price of the package shall be declared on the package.",
        "page_number": 1,
        "source_pages": [1],
    }


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_malformed_truncated_json_is_rejected_without_side_effects(monkeypatch, tmp_path):
    import app.extraction.llm_rule_generator as generator
    import app.repository.rule_repository as repository

    # Exact failure mode: reachable Ollama, truncated mid-object (~char 832).
    truncated = '{"rules": [{"rule_id": "LM_MRP_001", "parameter": "mrp", "condition": "must_exist", "requirement": "The package must declare'
    monkeypatch.setattr(generator.httpx, "post", lambda *a, **k: _Response({"response": truncated}))

    # Isolate repository (autouse fixture already does, but assert explicitly).
    prod_file = repository._resolve_rules_file()
    before = prod_file.read_bytes() if prod_file.exists() else None

    chunk = _chunk()
    with pytest.raises(OllamaUnavailableError, match="invalid data"):
        generate_rules_from_chunk(chunk, "test_regulation.pdf")

    after = prod_file.read_bytes() if prod_file.exists() else None
    assert before == after, "malformed Ollama output must not modify compliance_rules.json"


def test_truncated_json_does_not_mark_document_processed(monkeypatch):
    import app.rag_update_service as service
    from app.extraction.llm_rule_generator import OllamaUnavailableError as OUE

    doc = {"title": "T", "url": "https://example.test/t.pdf"}
    monkeypatch.setattr(service, "OFFICIAL_SOURCES", [{"url": "https://example.test"}])
    monkeypatch.setattr(service, "discover_documents", lambda url: [doc])
    monkeypatch.setattr(service, "mark_missing", lambda *a: [])
    from app.extraction.rule_schema import ComplianceRule
    valid = ComplianceRule(
        rule_id="LM_X", parameter="mrp", condition="must_exist",
        requirement="The package must declare the retail sale price.",
        source_document="R", source_pages=[1],
        evidence_text="Every package shall declare the retail sale price.",
    )
    monkeypatch.setattr(service, "load_rules", lambda: [valid])
    monkeypatch.setattr(service, "get_active_rules", lambda: [valid])
    monkeypatch.setattr(service, "check_document_change", lambda d: "NEW")
    monkeypatch.setattr(
        service, "download_document",
        lambda d: {"title": d["title"], "url": d["url"], "local_path": "/tmp/x.pdf", "sha256": "h1", "status": "NEW"},
    )

    def _raise(*a, **k):
        raise OUE("Local Ollama rule generation returned invalid data.", error_type="JSONDecodeError")

    monkeypatch.setattr(service, "process_pdf_document", _raise)
    monkeypatch.setattr(service, "update_rules", lambda rules: (_ for _ in ()).throw(AssertionError("must not activate")))
    marked = []
    monkeypatch.setattr(service, "mark_document_processed", lambda *a: marked.append(a))

    result = service.run_rag_update()
    assert result["failed_documents"] == 1
    assert result["successful_documents"] == 0
    assert marked == []
    assert result["status"] == "failed"


def test_mocked_structured_success_parses_with_evidence_and_validation(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    chunk = _chunk()
    body = {
        "rules": [
            {
                "rule_id": "LM_MRP_001",
                "parameter": "mrp",
                "condition": "must_exist",
                "requirement": "The package must declare the retail sale price.",
                "evidence_text": "The retail sale price of the package shall be declared on the package.",
            }
        ]
    }
    monkeypatch.setattr(generator.httpx, "post", lambda *a, **k: _Response({"response": json.dumps(body)}))

    rules = generate_rules_from_chunk(chunk, "test_regulation.pdf")
    assert len(rules) == 1
    assert rules[0].parameter == "mrp"
    assert rules[0].requirement.startswith("The package must declare")
    # Evidence preserved as exact source span (formatting-equivalent, not paraphrase).
    assert chunk["text"].startswith(rules[0].evidence_text.rstrip("."))
    assert rules[0].evidence_text in chunk["text"] or chunk["text"].startswith(rules[0].evidence_text)
    assert rules[0].source_pages == [1]
    # Deterministic validation ran (would be [] for this actionable rule).
    from app.validation.rule_validator import validate_rule
    assert validate_rule(rules[0]) == []


def test_prompt_requires_compact_json_only_contract():
    text = _prompt(_chunk())
    low = text.lower()
    for required in [
        "json only",
        "no markdown",
        "no explanations",
        "exact",
        "evidence_text",
        "[]",
        "concise",
        "single-line",
        "at most one",
        "fewer precise",
    ]:
        assert required in low, required


def test_request_payload_is_bounded_deterministic(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs.get("json", {}))
        captured["_url"] = url
        return _Response({"response": json.dumps({"rules": []})})

    monkeypatch.setattr(generator.httpx, "post", fake_post)
    OllamaRuleGenerator("http://localhost:11434", "qwen3:4b", 5).generate_rules(_chunk(), "test_regulation.pdf")

    assert captured["_url"] == "http://localhost:11434/api/generate"
    assert captured["model"] == "qwen3:4b"
    assert captured["stream"] is False
    assert captured["think"] is False
    # Structured output via JSON Schema, not weak "json" string.
    assert isinstance(captured["format"], dict)
    assert captured["format"]["type"] == "object"
    assert captured["format"]["properties"]["rules"]["maxItems"] == 1
    assert captured["format"] == OLLAMA_RESPONSE_SCHEMA
    num = captured["options"]["num_predict"]
    assert 0 < num <= 256, f"num_predict must stay bounded small, got {num}"
    assert captured["options"].get("temperature") == 0


def test_requirement_spacing_repairs_onthe_using_source_vocabulary():
    from app.extraction.llm_rule_generator import _normalize_requirement_text

    source = "The retail sale price of the package shall be declared on the package."
    assert _normalize_requirement_text("shall be declared onthe package", source) == "shall be declared on the package"
    # Correct spacing is preserved, not collapsed.
    assert "on the" in _normalize_requirement_text("shall be declared on the package", source)
    assert "onthe" not in _normalize_requirement_text("shall be declared onthe package", source).lower()


def test_qwen_onthe_requirement_is_repaired_before_validation(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    chunk = _chunk()
    body = {
        "rules": [
            {
                "rule_id": "LM_MRP_001",
                "parameter": "mrp",
                "condition": "must_exist",
                "requirement": "The retail sale price shall be declared onthe package.",
                "evidence_text": "The retail sale price of the package shall be declared on the package.",
            }
        ]
    }
    monkeypatch.setattr(generator.httpx, "post", lambda *a, **k: _Response({"response": json.dumps(body)}))

    rules = generate_rules_from_chunk(chunk, "test_regulation.pdf")
    assert len(rules) == 1
    assert "onthe" not in rules[0].requirement.lower()
    assert "on the" in rules[0].requirement.lower()
    # Exact source evidence is never modified to hide the issue.
    assert rules[0].evidence_text in chunk["text"] or chunk["text"].startswith(rules[0].evidence_text)


def test_empty_rules_output_is_valid(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    monkeypatch.setattr(
        generator.httpx, "post", lambda *a, **k: _Response({"response": json.dumps({"rules": []})})
    )
    assert generate_rules_from_chunk(_chunk(), "test_regulation.pdf") == []


def test_metadata_only_candidate_is_rejected_by_validation(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    chunk = {"text": "These rules may be called the Legal Metrology Rules, 2025.", "source_pages": [1]}
    body = {
        "rules": [
            {
                "rule_id": "RULE_NAME",
                "parameter": "rule_name",
                "condition": "must_equal",
                "requirement": "The rules must have this name.",
                "expected_value": "Legal Metrology Rules, 2025",
                "evidence_text": chunk["text"],
            }
        ]
    }
    monkeypatch.setattr(generator.httpx, "post", lambda *a, **k: _Response({"response": json.dumps(body)}))
    assert generate_rules_from_chunk(chunk, "Rules") == []


def test_real_sized_chunk_uses_compact_single_rule_contract(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    real_text = (
        "Every package shall bear thereon or on a label securely affixed thereto a declaration "
        "of the retail sale price in Indian currency. The declaration shall appear in a conspicuous "
        "place and in easily readable letters. " * 12
    )[:2350]
    assert 2000 < len(real_text) <= 2400
    chunk = {"text": real_text, "source_pages": [4]}
    evidence = real_text[:180].rsplit(" ", 1)[0]
    body = {
        "rules": [
            {
                "rule_id": "LM_MRP_001",
                "parameter": "mrp",
                "condition": "must_exist",
                "requirement": "The package must declare the retail sale price.",
                "evidence_text": evidence,
            }
        ]
    }
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs.get("json", {}))
        return _Response({"response": json.dumps(body)})

    monkeypatch.setattr(generator.httpx, "post", fake_post)
    rules = generate_rules_from_chunk(chunk, "Real Rules, 2011")
    assert len(rules) == 1
    # Single-rule contract keeps the mocked compact response well inside budget.
    assert len(json.dumps(body)) < 1200
    assert captured["format"]["properties"]["rules"]["maxItems"] == 1
    assert captured["options"]["num_predict"] <= 256


def _agri_chunk():
    text = (
        "Every package containing agricultural farm produce shall not exceed "
        "a net quantity of 50 kg under the Rules."
    )
    return {"text": text, "source_pages": [1]}


def test_comparison_rule_with_expected_value_and_unit_is_accepted(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    chunk = _agri_chunk()
    body = {
        "rules": [
            {
                "rule_id": "LM_QTY_050",
                "parameter": "net_quantity",
                "condition": "must_be_less_than",
                "requirement": "Packages containing agricultural farm produce must not exceed a net quantity of 50 kg.",
                "evidence_text": chunk["text"],
                "expected_value": 50,
                "expected_unit": "kg",
            }
        ]
    }
    monkeypatch.setattr(generator.httpx, "post", lambda *a, **k: _Response({"response": json.dumps(body)}))

    rules = generate_rules_from_chunk(chunk, "Agri Advisory")
    assert len(rules) == 1
    assert rules[0].expected_value == 50
    assert rules[0].expected_unit == "kg"
    assert rules[0].requirement != "50 kg"
    from app.validation.rule_validator import validate_rule
    assert validate_rule(rules[0]) == []


def test_bare_numeric_requirement_with_noun_phrase_evidence_is_avoided(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    chunk = _agri_chunk()
    body = {
        "rules": [
            {
                "rule_id": "rule_1",
                "parameter": "net_quantity",
                "condition": "must_be_less_than",
                "requirement": "50 kg",
                "evidence_text": "packages of agriculture farm produce upto 50 kg",
            }
        ]
    }
    monkeypatch.setattr(generator.httpx, "post", lambda *a, **k: _Response({"response": json.dumps(body)}))

    # Bare value + fragment evidence must not survive the pipeline.
    assert generate_rules_from_chunk(chunk, "Agri Advisory") == []


def test_prompt_requires_comparison_values_and_normative_evidence():
    low = _prompt(_chunk()).lower()
    for required in [
        "expected_value",
        "bare value",
        "one complete actionable sentence",
        "shall",
        "allowed",
        "covered under the rules",
        "never use a bare noun phrase",
        "same regulatory effect",
    ]:
        assert required in low, required


def test_prompt_comparison_fields_unconditional_and_subject_tiebreak():
    low = _prompt(_chunk()).lower()
    for required in [
        "you must provide both expected_value and expected_unit",
        "never omit them",
        "do not choose a comparison condition",
        "do not combine a 50 kg requirement with 25 kg evidence",
        "same threshold",
        "do not automatically use the first numeric clause",
        "actual regulatory effect for the target product",
    ]:
        assert required in low, required
    assert "only when needed" not in low
    assert "only when explicitly supported" not in low


def test_prompt_requires_same_effect_evidence_selection():
    low = _prompt(_chunk()).lower()
    for required in [
        "complete contiguous source sentence",
        "compare parameter",
        "condition",
        "expected_value",
        "expected_unit",
        "must support the exact threshold",
        "merely because it contains",
        "operative sentence",
        "different regulatory effect",
        "must not be a noun phrase",
    ]:
        assert required in low, required


def test_mismatched_threshold_evidence_is_rejected(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    chunk = {
        "text": (
            "The provisions shall not apply to packages of more than 25 kilogram. "
            "Packages of agricultural farm produce up to 50 kg are covered under the Rules "
            "and must declare the net quantity."
        ),
        "source_pages": [1],
    }
    body = {
        "rules": [
            {
                "rule_id": "rule_1",
                "parameter": "net_quantity",
                "condition": "must_be_less_than",
                "requirement": "Packages of agricultural farm produce must have a net quantity less than 50 kilograms",
                "evidence_text": "packages of more than 25 kilogram",
                "expected_value": "50",
                "expected_unit": "kilograms",
            }
        ]
    }
    monkeypatch.setattr(generator.httpx, "post", lambda *a, **k: _Response({"response": json.dumps(body)}))
    # 50 kg rule grounded in 25 kg exclusion evidence must not survive.
    assert generate_rules_from_chunk(chunk, "Diag") == []


def test_matching_normative_evidence_with_comparison_is_accepted(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    chunk = _agri_chunk()
    evidence = chunk["text"]
    body = {
        "rules": [
            {
                "rule_id": "LM_QTY_050",
                "parameter": "net_quantity",
                "condition": "must_be_less_than",
                "requirement": "Packages containing agricultural farm produce must not exceed a net quantity of 50 kg.",
                "evidence_text": evidence,
                "expected_value": 50,
                "expected_unit": "kg",
            }
        ]
    }
    monkeypatch.setattr(generator.httpx, "post", lambda *a, **k: _Response({"response": json.dumps(body)}))
    rules = generate_rules_from_chunk(chunk, "Agri Advisory")
    assert len(rules) == 1
    assert rules[0].expected_value == 50
    assert "50" in rules[0].evidence_text


def _smoke_chunk_1696():
    from pathlib import Path as _Path

    f = _Path("data/extracted/Download_Advisory_On_Packages_of_agriculture_farm_produce_upto_50kg_under_the_Legal_Metrology__Packaged_Commodities__Rule__2011_dated_06.03.2023.json")
    chunks = __import__("app.extraction.chunker", fromlist=["chunk_pages"]).chunk_pages(
        [{"page": y["page"], "text": y["text"]} for y in __import__("json").loads(f.read_text(encoding="utf-8"))]
    )
    return chunks[0]


def test_1696_chunk_candidates_cover_exclusion_and_permission():
    from app.extraction.llm_rule_generator import extract_normative_candidates

    chunk = _smoke_chunk_1696()
    assert len(chunk["text"]) == 1696
    candidates = extract_normative_candidates(chunk["text"])
    texts = [c["text"] for c in candidates]
    assert any("25 kilogram" in t for t in texts), "25kg exclusion candidate missing"
    assert any("are covered" in t for t in texts), "50kg coverage candidate missing"
    assert any("allowed to pack" in t for t in texts), "50kg permission candidate missing"


def test_candidates_are_exact_substrings_with_deterministic_numbering():
    from app.extraction.llm_rule_generator import extract_normative_candidates

    chunk = _smoke_chunk_1696()
    first = extract_normative_candidates(chunk["text"])
    second = extract_normative_candidates(chunk["text"])
    assert [c["index"] for c in first] == list(range(1, len(first) + 1))
    assert first == second
    for candidate in first:
        assert chunk["text"][candidate["start"]:candidate["end"]] == candidate["text"]
        assert candidate["text"] in chunk["text"]


def test_evidence_equal_to_presented_candidate_is_grounded(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    chunk = _agri_chunk()
    candidates = generator.extract_normative_candidates(chunk["text"])
    assert len(candidates) == 1
    evidence = candidates[0]["text"]
    body = {
        "rules": [
            {
                "rule_id": "LM_QTY_050",
                "parameter": "net_quantity",
                "condition": "must_be_less_than",
                "requirement": "Packages containing agricultural farm produce must not exceed a net quantity of 50 kg.",
                "evidence_text": evidence,
                "expected_value": 50,
                "expected_unit": "kg",
            }
        ]
    }
    monkeypatch.setattr(generator.httpx, "post", lambda *a, **k: _Response({"response": json.dumps(body)}))
    rules = generate_rules_from_chunk(
        {"text": chunk["text"], "source_pages": chunk["source_pages"]}, "Smoke"
    )
    assert len(rules) == 1
    # Grounding returns the exact source span for the alphanumeric content
    # (trailing punctuation excluded); the evidence comes from the candidate.
    assert rules[0].evidence_text in chunk["text"]
    assert rules[0].evidence_text in evidence


def test_50kg_rule_cannot_use_25kg_candidate(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    chunk = _smoke_chunk_1696()
    candidates = generator.extract_normative_candidates(chunk["text"])
    exclusion = next(c["text"] for c in candidates if "25 kilogram" in c["text"])
    body = {
        "rules": [
            {
                "rule_id": "rule_1",
                "parameter": "net_quantity",
                "condition": "must_be_less_than",
                "requirement": "Packages of agricultural farm produce must have a net quantity less than 50 kilograms",
                "evidence_text": exclusion,
                "expected_value": "50",
                "expected_unit": "kilograms",
            }
        ]
    }
    monkeypatch.setattr(generator.httpx, "post", lambda *a, **k: _Response({"response": json.dumps(body)}))
    assert generate_rules_from_chunk({"text": chunk["text"], "source_pages": chunk["source_pages"]}, "Smoke") == []


def test_prompt_presents_full_chunk_and_numbered_candidates():
    chunk = _smoke_chunk_1696()
    text = _prompt({"text": chunk["text"], "source_pages": chunk["source_pages"]})
    assert chunk["text"] in text
    assert "CANDIDATE 1:" in text
    assert "verbatim" in text.lower()


def test_chunk_without_normative_signals_yields_no_candidates():
    from app.extraction.llm_rule_generator import extract_normative_candidates

    chunk = {"text": "Government of India\nMinistry of Consumer Affairs\nPhone 011-23389489", "source_pages": [1]}
    assert extract_normative_candidates(chunk["text"]) == []
    low = _prompt(chunk).lower()
    assert "none found" in low


def test_exemption_with_applies_to_is_accepted(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    chunk = {"text": "Nothing contained in these rules shall apply to a package containing a commodity meant for export.",
             "source_pages": [1]}
    body = {"rules": [{
        "rule_id": "LM_EXP_001", "parameter": "export_exemption", "condition": "must_not_exist",
        "requirement": "Declaration requirements do not apply to an export package.",
        "evidence_text": chunk["text"],
        "applies_to": {"package_type": "export_package"}}]}
    monkeypatch.setattr(generator.httpx, "post", lambda *a, **k: _Response({"response": json.dumps(body)}))
    rules = generate_rules_from_chunk(chunk, "Rules")
    assert len(rules) == 1
    assert rules[0].applies_to == {"package_type": "export_package"}


def test_exemption_without_applies_to_is_rejected(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    chunk = {"text": "Nothing contained in these rules shall apply to a package containing a commodity meant for export.",
             "source_pages": [1]}
    body = {"rules": [{
        "rule_id": "LM_EXP_001", "parameter": "export_exemption", "condition": "must_not_exist",
        "requirement": "Declaration requirements do not apply to an export package.",
        "evidence_text": chunk["text"]}]}
    monkeypatch.setattr(generator.httpx, "post", lambda *a, **k: _Response({"response": json.dumps(body)}))
    assert generate_rules_from_chunk(chunk, "Rules") == []


def test_supplementary_quantity_declaration_is_accepted(monkeypatch):
    import app.extraction.llm_rule_generator as generator

    chunk = {"text": "Every package shall declare the supplementary quantity declaration in addition to the principal declaration.",
             "source_pages": [7]}
    body = {"rules": [{
        "rule_id": "LM_SUP_001", "parameter": "supplementary quantity declaration",
        "condition": "must_exist",
        "requirement": "The package must declare the supplementary quantity declaration.",
        "evidence_text": chunk["text"]}]}
    monkeypatch.setattr(generator.httpx, "post", lambda *a, **k: _Response({"response": json.dumps(body)}))
    rules = generate_rules_from_chunk(chunk, "2015 Amendment")
    assert len(rules) == 1
    assert rules[0].parameter == "supplementary quantity declaration"


def test_requirement_normalization_repairs_safe_artifacts():
    from app.extraction.llm_rule_generator import _normalize_requirement_text

    source = "The package must declare the quantity declaration for the prepackage contents."
    assert _normalize_requirement_text("Pack the quantitydeclaration now", source) == "Pack the quantity declaration now"
    assert _normalize_requirement_text("Contents of the prepackage,the wrapper", source) == "Contents of the prepackage, the wrapper"
    assert _normalize_requirement_text("Net quantity 1,000 g", source) == "Net quantity 1,000 g"
    assert _normalize_requirement_text("Net quantity 50 kg", source) == "Net quantity 50 kg"


def test_prompt_requires_applies_to_for_exemptions():
    low = _prompt(_chunk()).lower()
    assert "must include applies_to" in low
    assert "do not leave it null" in low
