from app.extraction.rule_pipeline import (
    process_chunk,
    save_processed_rules,
)
from app.repository.rule_repository import load_rules
from app.extraction.rule_pipeline import process_chunks


def test_process_chunk():
    chunk = {
        "chunk_id": "chunk_0001",
        "text": """
        Declarations to be made on every package.
        Every package shall bear the name and address of the manufacturer.
        The retail sale price of the package shall be declared.
        """,
        "source_pages": [42, 44],
    }

    rules = process_chunk(
        chunk,
        source_document="Legal Metrology (Packaged Commodities) Rules, 2011",
    )

    assert len(rules) == 2

    for rule in rules:
        assert rule.rule_id
        assert rule.parameter
        assert rule.condition
        assert rule.source_document
        assert rule.source_pages
        assert rule.evidence_text


def test_save_processed_rules(tmp_path, monkeypatch):
    import app.repository.rule_repository as repository

    rules_file = tmp_path / "compliance_rules.json"

    monkeypatch.setattr(repository, "RULES_FILE", rules_file)
    monkeypatch.setattr(repository, "RULES_DIR", tmp_path)

    chunk = {
        "chunk_id": "chunk_0001",
        "text": """
        The retail sale price of the package shall be declared.
        """,
        "source_pages": [44],
    }

    rules = process_chunk(
        chunk,
        source_document="Legal Metrology (Packaged Commodities) Rules, 2011",
    )

    save_processed_rules(rules)

    saved_rules = load_rules()

    assert len(saved_rules) == 1
    assert saved_rules[0].rule_id == "LM_MRP_001"
    assert saved_rules[0].status == "active"


def test_duplicate_rules_merge_pages_and_evidence():
    chunks = [
        {"chunk_id": "one", "text": "The retail sale price shall be declared.", "source_pages": [4]},
        {"chunk_id": "two", "text": "Every retail sale price on a package shall be declared.", "source_pages": [5]},
    ]

    rules = process_chunks(chunks, "Rules, 2011")

    assert len(rules) == 1
    assert rules[0].source_pages == [4, 5]
    assert "retail sale price" in rules[0].evidence_text.lower()


def test_failed_document_extraction_keeps_existing_rules(tmp_path, monkeypatch):
    import app.extraction.rule_pipeline as pipeline
    import app.repository.rule_repository as repository
    from app.extraction.rule_schema import ComplianceRule
    from app.repository.rule_repository import add_rule

    monkeypatch.setattr(repository, "RULES_FILE", tmp_path / "compliance_rules.json")
    monkeypatch.setattr(repository, "RULES_DIR", tmp_path)
    old_rule = ComplianceRule(rule_id="LM_OLD", parameter="mrp", condition="must_exist",
        requirement="The package must declare MRP.", source_document="Old Rules", source_pages=[1], evidence_text="MRP shall be declared.")
    add_rule(old_rule)
    monkeypatch.setattr(
        pipeline,
        "extract_pdf_text",
        lambda path, progress_callback=None: (_ for _ in ()).throw(ValueError("bad PDF")),
    )

    import pytest
    with pytest.raises(ValueError, match="bad PDF"):
        pipeline.process_pdf_document("bad.pdf", "New Rules", "new-version")

    assert repository.load_rules() == [old_rule]


def test_pdf_extraction_progress_is_forwarded_to_service(monkeypatch):
    import app.extraction.rule_pipeline as pipeline

    callback_events = []
    monkeypatch.setattr(
        pipeline,
        "extract_pdf_text",
        lambda path, progress_callback: (
            progress_callback(1, 2),
            progress_callback(2, 2),
            [{"page": 1, "text": "Text", "method": "direct"}],
        )[-1],
    )
    monkeypatch.setattr(pipeline, "save_extracted_text", lambda *args: None)
    monkeypatch.setattr(pipeline, "process_chunks", lambda *args, **kwargs: [
        __import__("app.extraction.rule_schema", fromlist=["ComplianceRule"]).ComplianceRule(
            rule_id="LM_001", parameter="price", condition="must_exist",
            requirement="The package must declare price.", source_document="Rules", source_pages=[1], evidence_text="Every package shall declare price.",
        )
    ])

    rules = pipeline.process_pdf_document(
        "rules.pdf", "Rules", persist=False,
        progress_callback=lambda *event: callback_events.append(event),
    )

    assert callback_events[1] == ("pdf_extraction", {"chunk_id": "page_0001"}, 1, 2)
    assert callback_events[2] == ("pdf_extraction", {"chunk_id": "page_0002"}, 2, 2)
    assert rules[0].source_identity


def test_degraded_ocr_pages_are_not_sent_to_rule_generation(monkeypatch):
    import app.extraction.rule_pipeline as pipeline

    pages_seen = []
    monkeypatch.setattr(pipeline, "extract_pdf_text", lambda *args, **kwargs: [
        {"page": 1, "text": "Every package shall declare the retail sale price.", "method": "direct", "ocr_status": "GOOD"},
        {"page": 2, "text": "\ufffd\ufffd\ufffd", "method": "ocr", "ocr_status": "DEGRADED"},
    ])
    monkeypatch.setattr(pipeline, "save_extracted_text", lambda *args: None)
    monkeypatch.setattr(
        pipeline,
        "process_chunks",
        lambda chunks, *args, **kwargs: pages_seen.extend(chunks[0]["source_pages"]) or [],
    )

    pipeline.process_pdf_document("rules.pdf", "Rules", persist=False, require_rules=False)

    assert pages_seen == [1]


def test_all_degraded_ocr_pages_fail_document_processing(monkeypatch):
    import pytest
    import app.extraction.rule_pipeline as pipeline

    monkeypatch.setattr(pipeline, "extract_pdf_text", lambda *args, **kwargs: [
        {"page": 1, "text": "\ufffd\ufffd\ufffd", "method": "ocr", "ocr_status": "DEGRADED"},
    ])
    monkeypatch.setattr(pipeline, "save_extracted_text", lambda *args: None)

    with pytest.raises(pipeline.OcrQualityError, match="No trustworthy pages"):
        pipeline.process_pdf_document("rules.pdf", "Rules", persist=False, require_rules=False)


if __name__ == "__main__":
    test_process_chunk()
