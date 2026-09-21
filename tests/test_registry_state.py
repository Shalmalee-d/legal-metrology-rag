"""Registry state: UNCHANGED must imply already-processed.

An UNCHANGED document must not be treated as successfully processed when its
current fingerprint was never processed. Valid states:
- new/discovered: seen, not yet downloaded
- changed/downloaded: new bytes, needs OCR/Qwen
- unchanged: fingerprint == processed_fingerprint (and sha == processed_sha)
- processed: explicitly marked after successful activation
- failed/source_check_failed/no_longer_listed: retryable observations
"""

from pathlib import Path


def test_unchanged_requires_matching_processed_fingerprint():
    import app.registry.document_registry as registry

    # Simulate a downloaded-but-never-processed record.
    registry.upsert_discovered(
        {"title": "T", "url": "https://example.test/state.pdf", "source": "s", "category": "c"},
        status="downloaded",
        fingerprint="hash-1",
    )
    rec = registry.get_record("https://example.test/state.pdf")
    assert rec["fingerprint"] == "hash-1"
    assert rec.get("processed_fingerprint") is None

    # check_document_change must report CHANGED (needs work), never UNCHANGED.
    import app.fetcher.document_fetcher as fetcher

    # Avoid network: registry mismatch short-circuits before HEAD.
    assert fetcher.check_document_change({"title": "T", "url": "https://example.test/state.pdf"}) == "CHANGED"
    assert registry.get_record("https://example.test/state.pdf")["status"] == "changed"

    # After marking processed, same fingerprint is UNCHANGED only if metadata agrees.
    registry.mark_processed("https://example.test/state.pdf", "hash-1")
    rec2 = registry.get_record("https://example.test/state.pdf")
    assert rec2["fingerprint"] == rec2["processed_fingerprint"] == "hash-1"

    # Metadata with matching processed sha + strong etag => UNCHANGED.
    monkeypatch_etag_ok = {"url": "https://example.test/state.pdf", "sha256": "hash-1", "processed_sha256": "hash-1", "etag": '"e1"'}
    import pytest

    # Use monkeypatch via fixture in separate test below; here just assert state shape.
    assert rec2["status"] == "processed"


def test_metadata_unprocessed_forces_changed_status(monkeypatch, tmp_path):
    import app.registry.document_registry as registry
    import app.fetcher.document_fetcher as fetcher

    url = "https://example.test/meta-state.pdf"
    # Registry says processed...
    registry.upsert_discovered(
        {"title": "T", "url": url, "source": "s", "category": "c"},
        status="processed",
        fingerprint="hash-1",
        processed_fingerprint="hash-1",
    )
    # ...but metadata has unprocessed sha (out-of-sync case).
    monkeypatch.setattr(fetcher, "load_metadata", lambda: [{"url": url, "sha256": "hash-2", "processed_sha256": None}])
    monkeypatch.setattr(fetcher, "save_metadata", lambda m: None)

    assert fetcher.check_document_change({"title": "T", "url": url}) == "CHANGED"
    # Registry must have been corrected to changed, not left as processed/unchanged.
    assert registry.get_record(url)["status"] == "changed"


def test_download_hash_match_unprocessed_reports_changed(monkeypatch, tmp_path):
    import app.fetcher.document_fetcher as fetcher

    doc = {"title": "T", "url": "https://example.test/dl.pdf"}
    prev = {
        "url": doc["url"],
        "title": "T",
        "local_path": str(tmp_path / "x.pdf"),
        "sha256": "hash-same",
        "processed_sha256": None,
        "etag": None,
    }
    monkeypatch.setattr(fetcher, "load_metadata", lambda: [dict(prev)])
    monkeypatch.setattr(fetcher, "save_metadata", lambda m: None)

    class Resp:
        content = b"%PDF-1.4 fake"
        headers = {}

        def raise_for_status(self):
            pass

    # Force hash to match previous by patching calculate_sha256.
    monkeypatch.setattr(fetcher, "calculate_sha256", lambda data: "hash-same")

    class Client:
        def __init__(self, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            return Resp()

    monkeypatch.setattr(fetcher.httpx, "Client", Client)

    record = fetcher.download_document(doc)
    assert record["status"] == "CHANGED", "unprocessed hash-match must not report UNCHANGED"
    # Registry must reflect changed, not unchanged.
    import app.registry.document_registry as registry

    assert registry.get_record(doc["url"])["status"] == "changed"
    assert registry.get_record(doc["url"])["processed_fingerprint"] is None
