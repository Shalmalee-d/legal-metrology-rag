import httpx
import pytest


class _Response:
    content = b"%PDF-1.7 test"
    headers = {"etag": "test", "last-modified": "today", "content-length": "14"}

    def raise_for_status(self):
        return None


class _Client:
    calls = []
    failures_before_success = 0

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url):
        self.calls.append(url)
        if len(self.calls) <= self.failures_before_success:
            raise httpx.ReadTimeout("government server delayed response")
        return _Response()


def _configure(monkeypatch, tmp_path, failures_before_success):
    import app.fetcher.document_fetcher as fetcher

    _Client.calls = []
    _Client.failures_before_success = failures_before_success
    monkeypatch.setattr(fetcher, "DATA_DIR", tmp_path)
    monkeypatch.setattr(fetcher, "METADATA_FILE", tmp_path / "metadata.json")
    monkeypatch.setattr(fetcher.httpx, "Client", _Client)
    monkeypatch.setattr(fetcher.time, "sleep", lambda seconds: None)
    return fetcher


def test_download_retries_one_read_timeout_with_bounded_backoff(tmp_path, monkeypatch):
    fetcher = _configure(monkeypatch, tmp_path, failures_before_success=1)
    document = {"title": "Official Rules", "url": "https://example.gov.in/rules.pdf"}

    record = fetcher.download_document(document)

    assert record["status"] == "NEW"
    assert _Client.calls == [document["url"], document["url"]]


def test_download_error_records_timeout_type_and_alternate_urls(tmp_path, monkeypatch):
    fetcher = _configure(monkeypatch, tmp_path, failures_before_success=99)
    document = {"title": "Official Rules", "url": "http://example.gov.in/rules.pdf"}

    with pytest.raises(fetcher.DocumentDownloadError) as raised:
        fetcher.download_document(document)

    error = raised.value
    assert error.title == "Official Rules"
    assert len(error.attempts) == 4
    assert {item["url"] for item in error.attempts} == {
        "https://example.gov.in/rules.pdf", "http://example.gov.in/rules.pdf",
    }
    assert all(item["timeout_type"] == "read" for item in error.attempts)
    assert "Official Rules" in str(error)


def test_duplicate_titles_use_distinct_url_derived_cache_paths(tmp_path, monkeypatch):
    fetcher = _configure(monkeypatch, tmp_path, failures_before_success=0)
    monkeypatch.setattr(fetcher, "upsert_discovered", lambda *args, **kwargs: None)
    first = fetcher.download_document({"title": "Official Rules", "url": "https://example.gov.in/a.pdf"})
    second = fetcher.download_document({"title": "Official Rules", "url": "https://example.gov.in/b.pdf"})
    assert first["local_path"] != second["local_path"]
    assert first["local_path"].endswith(".pdf")


def _check_client(headers=None, error=None):
    class CheckClient:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def head(self, url):
            if error:
                raise error
            response = _Response()
            response.headers = headers or {}
            return response
    return CheckClient


def test_change_check_trusts_matching_strong_etag(tmp_path, monkeypatch):
    import app.fetcher.document_fetcher as fetcher
    document = {"title": "Rules", "url": "https://example.gov.in/rules.pdf"}
    previous = {"url": document["url"], "sha256": "hash", "processed_sha256": "hash", "etag": '"abc"'}
    monkeypatch.setattr(fetcher, "load_metadata", lambda: [previous])
    monkeypatch.setattr(fetcher, "save_metadata", lambda metadata: None)
    monkeypatch.setattr(fetcher, "get_registry_record", lambda url: {"fingerprint": "hash", "processed_fingerprint": "hash"})
    monkeypatch.setattr(fetcher, "upsert_discovered", lambda *args, **kwargs: None)
    monkeypatch.setattr(fetcher.httpx, "Client", _check_client({"etag": '"abc"', "last-modified": None, "content-length": None}))

    assert fetcher.check_document_change(document) == "UNCHANGED"


def test_change_check_uses_last_modified_and_length_without_etag(tmp_path, monkeypatch):
    import app.fetcher.document_fetcher as fetcher
    document = {"title": "Rules", "url": "https://example.gov.in/rules.pdf"}
    previous = {"url": document["url"], "sha256": "hash", "processed_sha256": "hash",
                "etag": None, "last_modified": "Mon, 01 Sep 2026 10:00:00 GMT", "content_length": "1234"}
    monkeypatch.setattr(fetcher, "load_metadata", lambda: [previous])
    monkeypatch.setattr(fetcher, "save_metadata", lambda metadata: None)
    monkeypatch.setattr(fetcher, "get_registry_record", lambda url: {"fingerprint": "hash", "processed_fingerprint": "hash"})
    monkeypatch.setattr(fetcher, "upsert_discovered", lambda *args, **kwargs: None)
    monkeypatch.setattr(fetcher.httpx, "Client", _check_client({"etag": None, "last-modified": previous["last_modified"], "content-length": "1234"}))

    assert fetcher.check_document_change(document) == "UNCHANGED"


def test_change_check_download_fallback_for_weak_or_missing_http_signature(tmp_path, monkeypatch):
    import app.fetcher.document_fetcher as fetcher
    document = {"title": "Rules", "url": "https://example.gov.in/rules.pdf"}
    previous = {"url": document["url"], "sha256": "hash", "processed_sha256": "hash",
                "etag": None, "last-modified": None, "content-length": None}
    monkeypatch.setattr(fetcher, "load_metadata", lambda: [previous])
    monkeypatch.setattr(fetcher, "save_metadata", lambda metadata: None)
    monkeypatch.setattr(fetcher, "get_registry_record", lambda url: {"fingerprint": "hash", "processed_fingerprint": "hash"})
    monkeypatch.setattr(fetcher, "upsert_discovered", lambda *args, **kwargs: None)
    monkeypatch.setattr(fetcher.httpx, "Client", _check_client({"etag": 'W/"abc"', "last-modified": None, "content-length": None}))

    assert fetcher.check_document_change(document) == "CHANGED"


def test_change_check_does_not_hide_head_failure_as_unchanged(tmp_path, monkeypatch):
    import app.fetcher.document_fetcher as fetcher
    document = {"title": "Rules", "url": "https://example.gov.in/rules.pdf"}
    previous = {"url": document["url"], "sha256": "hash", "processed_sha256": "hash", "etag": '"abc"'}
    monkeypatch.setattr(fetcher, "load_metadata", lambda: [previous])
    monkeypatch.setattr(fetcher, "save_metadata", lambda metadata: None)
    monkeypatch.setattr(fetcher, "get_registry_record", lambda url: {"fingerprint": "hash", "processed_fingerprint": "hash"})
    monkeypatch.setattr(fetcher, "upsert_discovered", lambda *args, **kwargs: None)
    monkeypatch.setattr(fetcher.httpx, "Client", _check_client(error=httpx.ReadTimeout("source unavailable")))

    with pytest.raises(fetcher.DocumentDownloadError):
        fetcher.check_document_change(document)
