import pymupdf
from app.extraction.text_cleaner import clean_text

from app.extraction.pdf_extractor import (
    extract_pdf_text,
    save_extracted_text,
)


def test_scanned_page_uses_ocr_fallback(tmp_path, monkeypatch):
    """OCR is called only when PyMuPDF cannot obtain usable text."""
    import app.extraction.pdf_extractor as extractor

    pdf_path = tmp_path / "scanned.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    language_calls = []
    monkeypatch.setattr(
        extractor,
        "_resolve_ocr_language_details",
        lambda languages: (
            language_calls.append(languages)
            or extractor.OcrLanguageResolution("eng", extractor.LANGUAGE_AVAILABLE, "eng", False)
        ),
    )
    monkeypatch.setattr(extractor, "_ocr_page", lambda page, languages: "OCR result")

    pages = extractor.extract_pdf_text(str(pdf_path))

    assert pages[0]["page"] == 1
    assert pages[0]["text"] == "OCR result"
    assert pages[0]["method"] == "ocr"
    assert pages[0]["extraction_method"] == "tesseract_ocr"
    assert pages[0]["ocr_status"] == extractor.OCR_GOOD
    assert language_calls == [None]


def test_ocr_languages_are_resolved_once_per_document(tmp_path, monkeypatch):
    import app.extraction.pdf_extractor as extractor

    pdf_path = tmp_path / "scanned.pdf"
    document = pymupdf.open()
    document.new_page()
    document.new_page()
    document.save(pdf_path)
    document.close()

    language_calls = []
    monkeypatch.setattr(
        extractor,
        "_resolve_ocr_language_details",
        lambda languages: (
            language_calls.append(languages)
            or extractor.OcrLanguageResolution("eng", extractor.LANGUAGE_AVAILABLE, languages, False)
        ),
    )
    monkeypatch.setattr(extractor, "_ocr_page", lambda page, languages: "OCR result")

    extractor.extract_pdf_text(str(pdf_path), "eng+hin")

    assert language_calls == ["eng+hin"]


def test_extraction_reports_each_page(tmp_path, monkeypatch):
    import app.extraction.pdf_extractor as extractor

    pdf_path = tmp_path / "text.pdf"
    document = pymupdf.open()
    for _ in range(2):
        page = document.new_page()
        page.insert_text((72, 72), "Retail sale price must be declared on every package.")
    document.save(pdf_path)
    document.close()

    reported_pages = []
    extractor.extract_pdf_text(
        str(pdf_path),
        progress_callback=lambda page, total: reported_pages.append((page, total)),
    )

    assert reported_pages == [(1, 2), (2, 2)]


def test_hindi_request_falls_back_to_english_when_unavailable(monkeypatch):
    import app.extraction.pdf_extractor as extractor

    monkeypatch.setattr(extractor, "available_ocr_languages", lambda: {"eng", "osd"})

    assert extractor.resolve_ocr_languages("eng+hin") == "eng"
    details = extractor._resolve_ocr_language_details("eng+hin")
    assert details.status == extractor.LANGUAGE_UNAVAILABLE
    assert details.languages == "eng"


def test_pdf_extraction_uses_small_text_fixture(tmp_path, monkeypatch):
    """Direct extraction is covered without processing the real 83-page PDF."""
    import app.extraction.pdf_extractor as extractor

    pdf_path = tmp_path / "text.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Retail sale price must be declared on every package.")
    document.save(pdf_path)
    document.close()

    pages = extract_pdf_text(str(pdf_path))

    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert pages[0]["method"] == "direct"
    assert pages[0]["extraction_method"] == "direct_text"
    assert pages[0]["ocr_status"] == extractor.OCR_GOOD
    assert "Retail sale price" in pages[0]["text"]

    monkeypatch.setattr(extractor, "EXTRACTED_DIR", tmp_path / "extracted")
    output_path = save_extracted_text(str(pdf_path), pages)

    assert output_path.exists()


def test_hindi_ocr_is_preserved_without_mojibake(tmp_path, monkeypatch):
    import app.extraction.pdf_extractor as extractor
    import app.repository.rule_repository as repository
    from app.extraction.rule_schema import ComplianceRule

    hindi = "पैकेज पर अधिकतम खुदरा मूल्य अंकित होना चाहिए"
    pdf_path = tmp_path / "scanned.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    monkeypatch.setattr(
        extractor,
        "_resolve_ocr_language_details",
        lambda languages: extractor.OcrLanguageResolution("eng+hin", extractor.LANGUAGE_AVAILABLE, "eng+hin", True),
    )
    monkeypatch.setattr(extractor, "_ocr_page", lambda page, languages: hindi)
    pages = extractor.extract_pdf_text(str(pdf_path))

    assert pages[0]["text"] == hindi
    assert pages[0]["ocr_status"] == extractor.OCR_GOOD
    assert clean_text(pages[0]["text"]) == hindi

    monkeypatch.setattr(repository, "RULES_DIR", tmp_path / "rules")
    monkeypatch.setattr(repository, "RULES_FILE", tmp_path / "rules" / "compliance_rules.json")
    repository.save_rules([ComplianceRule(
        rule_id="LM_HINDI_001", parameter="mrp", condition="must_exist",
        requirement="The package must declare the retail sale price.",
        source_document="Hindi Rules", source_pages=[1], evidence_text=pages[0]["text"],
    )])
    persisted = repository.RULES_FILE.read_text(encoding="utf-8")

    assert hindi in persisted
    assert "à¤" not in persisted


def test_mixed_english_hindi_ocr_text_is_not_automatically_degraded():
    import app.extraction.pdf_extractor as extractor

    text = "Every पैकेज shall display the अधिकतम खुदरा मूल्य."

    assert extractor.assess_ocr_quality(text) == extractor.OCR_GOOD


def test_clearly_malformed_ocr_text_is_detected_as_degraded():
    import app.extraction.pdf_extractor as extractor

    assert extractor.assess_ocr_quality("\ufffd\ufffd\ufffd ### !!!") == extractor.OCR_DEGRADED


def test_hindi_language_unavailability_is_recorded_on_ocr_pages(tmp_path, monkeypatch):
    import app.extraction.pdf_extractor as extractor

    pdf_path = tmp_path / "scanned.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    monkeypatch.setattr(
        extractor,
        "_resolve_ocr_language_details",
        lambda languages: extractor.OcrLanguageResolution("eng", extractor.LANGUAGE_UNAVAILABLE, "eng+hin", True),
    )
    monkeypatch.setattr(extractor, "_ocr_page", lambda page, languages: "English-only official notice")

    pages = extractor.extract_pdf_text(str(pdf_path))

    assert pages[0]["ocr_status"] == extractor.OCR_GOOD
    assert pages[0]["ocr_language_status"] == extractor.LANGUAGE_UNAVAILABLE
