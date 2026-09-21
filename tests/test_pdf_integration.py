from pathlib import Path

import pytest

from app.extraction.pdf_extractor import extract_pdf_text


@pytest.mark.integration
def test_real_legal_metrology_pdf_extraction():
    """Exercise the local 83-page government PDF, including OCR where needed."""
    pdf_path = Path("data/documents/Download_The_Legal_Metrology__Packaged_Commodities__Rules__2011.pdf")

    assert pdf_path.exists(), "The local Legal Metrology PDF fixture is required for integration tests."

    pages = extract_pdf_text(str(pdf_path))

    assert len(pages) == 83
    assert all(page["page"] >= 1 for page in pages)
    assert {page["method"] for page in pages} <= {"direct", "ocr"}
    assert {page["ocr_status"] for page in pages} <= {"GOOD", "DEGRADED", "OCR_UNAVAILABLE"}
    assert all(page["text"].strip() for page in pages if page["ocr_status"] == "GOOD")
