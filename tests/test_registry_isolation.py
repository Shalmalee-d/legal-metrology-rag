"""Regression: tests can never pollute the real production registry.

- Fake example.gov.in records go only to the isolated tmp registry.
- Real data/registry/document_registry.json is byte-identical after the test.
- Registry CRUD still works inside the isolated registry.
"""

from pathlib import Path


PROD_REGISTRY = Path("data/registry/document_registry.json")


def _snapshot_prod():
    if not PROD_REGISTRY.exists():
        return None
    return (PROD_REGISTRY.stat().st_mtime_ns, PROD_REGISTRY.read_bytes())


def test_fake_example_gov_record_stays_in_isolated_registry(tmp_path, monkeypatch):
    import app.registry.document_registry as registry

    before = _snapshot_prod()

    # These are the exact fake URLs that previously polluted production.
    fakes = [
        {"title": "Official Rules", "url": "https://example.gov.in/rules.pdf"},
        {"title": "Official Rules", "url": "https://example.gov.in/a.pdf"},
        {"title": "Official Rules", "url": "https://example.gov.in/b.pdf"},
    ]
    for doc in fakes:
        registry.upsert_discovered(doc, status="downloaded", fingerprint="abc123")

    # Isolated registry (tmp) must contain the fakes and be usable.
    assert registry._resolve_registry_file() != PROD_REGISTRY.resolve() or (
        # When running without isolation env (should not happen under pytest),
        # at least ensure we are writing to tmp_path, not prod.
        str(registry._resolve_registry_file()).startswith(str(tmp_path))
    )
    records = registry.load_registry()
    urls = {r["url"] for r in records}
    assert "https://example.gov.in/rules.pdf" in urls
    assert "https://example.gov.in/a.pdf" in urls

    # get_record / mark_processed round-trip works in isolation.
    rec = registry.get_record("https://example.gov.in/rules.pdf")
    assert rec is not None
    assert rec["status"] == "downloaded"
    registry.mark_processed("https://example.gov.in/rules.pdf", "abc123")
    assert registry.get_record("https://example.gov.in/rules.pdf")["processed_fingerprint"] == "abc123"

    # Real production file must be untouched (never auto-deleted/modified).
    after = _snapshot_prod()
    assert before == after, "Test wrote to real production registry!"


def test_registry_operations_use_configured_location_consistently(tmp_path, monkeypatch):
    import app.registry.document_registry as registry

    # All operations must resolve to the same isolated file.
    first_file = registry._resolve_registry_file()
    registry.upsert_discovered({"title": "T", "url": "https://example.test/iso.pdf"}, status="new")
    second_file = registry._resolve_registry_file()
    assert first_file == second_file
    assert registry.get_record("https://example.test/iso.pdf") is not None

    # update_status must also be isolated (not prod).
    registry.save_update_status({"status": "up_to_date", "test": True})
    assert registry._resolve_update_status_file().exists()
    assert registry.load_update_status()["test"] is True

    # Prod files untouched.
    prod_status = Path("data/registry/update_status.json")
    # Do not assert prod_status absence (it exists in repo); assert our tmp write
    # did not leak: resolved file must not be prod when isolation is active.
    assert registry._resolve_update_status_file() != prod_status.resolve() or str(
        registry._resolve_update_status_file()
    ).startswith(str(tmp_path))

    # Other runtime dirs must also be isolated.
    import app.fetcher.document_fetcher as fetcher
    import app.repository.rule_repository as repository
    import app.extraction.pdf_extractor as extractor

    assert str(fetcher._resolve_metadata_file()).startswith(str(tmp_path))
    assert str(repository._resolve_rag_store()).startswith(str(tmp_path))
    assert str(extractor._resolve_extracted_dir()).startswith(str(tmp_path))
