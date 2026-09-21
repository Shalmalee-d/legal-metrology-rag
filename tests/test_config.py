def test_check_interval_reads_environment(monkeypatch):
    from app.config import get_check_interval_days

    monkeypatch.setenv("RAG_CHECK_INTERVAL_DAYS", "3")
    assert get_check_interval_days() == 3


def test_ocr_timeout_must_be_positive(monkeypatch):
    import pytest
    from app.config import get_ocr_timeout_seconds

    monkeypatch.setenv("OCR_TIMEOUT_SECONDS", "0")

    with pytest.raises(ValueError, match="OCR_TIMEOUT_SECONDS"):
        get_ocr_timeout_seconds()
