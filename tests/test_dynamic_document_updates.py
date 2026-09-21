"""Unit coverage for dynamic source discovery and incremental update routing."""

from app.extraction.rule_schema import ComplianceRule


def _rule(document: str) -> ComplianceRule:
    return ComplianceRule(rule_id=f"LM_{document}", parameter="mrp", condition="must_exist",
        applies_to={"product_type": "packaged_commodity"}, requirement="The package must declare the retail sale price.",
        source_document=document, source_pages=[1], evidence_text="Every package shall declare the retail sale price.")


def test_discovery_uses_context_and_excludes_unrelated_categories(monkeypatch):
    import app.fetcher.document_fetcher as fetcher
    class Response:
        text = '''<h2>Legal Metrology (Packaged Commodities)</h2><ul><li><a href="pc-2026.pdf">Amendment, 2026</a></li></ul><h2>General Rules</h2><a href="general.pdf">Packaged Commodities Rules</a><h2>National Standards Rules</h2><a href="standards.pdf">Packaged Commodities amendment</a>'''
        def raise_for_status(self): pass
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def get(self, url): return Response()
    monkeypatch.setattr(fetcher.httpx, "Client", Client)
    documents = fetcher.discover_documents("https://example.test/legal")
    assert len(documents) == 1
    assert documents[0]["url"] == "https://example.test/pc-2026.pdf"
    assert documents[0]["category"] == "legal_metrology_packaged_commodities"


def test_registry_persists_stable_url_identity(tmp_path, monkeypatch):
    import app.registry.document_registry as registry
    monkeypatch.setattr(registry, "REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(registry, "REGISTRY_FILE", tmp_path / "document_registry.json")
    first = registry.upsert_discovered({"title": "Rules", "url": "HTTPS://Example.test/a.pdf#page=1", "source": "official", "category": "pc"}, status="new")
    second = registry.get_record("https://example.test/a.pdf")
    assert first["document_id"] == second["document_id"]
    assert second["title"] == "Rules"
    assert registry.REGISTRY_FILE.exists()


def test_dynamic_37th_and_38th_documents_are_processed_without_fixed_count(monkeypatch):
    import app.rag_update_service as service
    documents = [{"title": f"Document {number}", "url": f"https://example.test/{number}.pdf"} for number in range(1, 39)]
    processed = []
    monkeypatch.setattr(service, "OFFICIAL_SOURCES", [{"url": "https://example.test"}])
    monkeypatch.setattr(service, "discover_documents", lambda url: documents)
    monkeypatch.setattr(service, "mark_missing", lambda *args: [])
    monkeypatch.setattr(service, "load_rules", lambda: [_rule("existing")])
    monkeypatch.setattr(service, "get_active_rules", lambda: [])
    monkeypatch.setattr(service, "check_document_change", lambda doc: "NEW" if int(doc["title"].split()[-1]) > 36 else "UNCHANGED")
    monkeypatch.setattr(service, "download_document", lambda doc: {"title": doc["title"], "url": doc["url"], "local_path": doc["title"], "sha256": doc["title"], "status": "NEW"})
    monkeypatch.setattr(service, "process_pdf_document", lambda path, *args, **kwargs: processed.append(path) or [_rule(path)])
    monkeypatch.setattr(service, "update_rules", lambda rules: None)
    monkeypatch.setattr(service, "mark_document_processed", lambda *args: None)
    result = service.run_rag_update()
    assert processed == ["Document 37", "Document 38"]
    assert result["documents_discovered"] == 38
    assert result["new_documents"] == 2
    assert result["unchanged_documents"] == 36


def test_no_change_refresh_does_zero_expensive_processing(monkeypatch):
    import app.rag_update_service as service
    document = {"title": "Rules", "url": "https://example.test/rules.pdf"}
    monkeypatch.setattr(service, "OFFICIAL_SOURCES", [{"url": "https://example.test"}])
    monkeypatch.setattr(service, "discover_documents", lambda url: [document])
    monkeypatch.setattr(service, "mark_missing", lambda *args: [])
    monkeypatch.setattr(service, "load_rules", lambda: [_rule("existing")])
    monkeypatch.setattr(service, "check_document_change", lambda doc: "UNCHANGED")
    monkeypatch.setattr(service, "download_document", lambda doc: (_ for _ in ()).throw(AssertionError("download")))
    monkeypatch.setattr(service, "process_pdf_document", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Qwen")))
    result = service.run_rag_update()
    assert result["status"] == "up_to_date"
    assert result["unchanged_documents"] == 1


def test_discovery_excludes_unrelated_advisory_inside_packaged_section(monkeypatch):
    import app.fetcher.document_fetcher as fetcher
    class Response:
        text = '''<h2>The Legal Metrology (Packaged Commodities) Rules, 2011</h2>\n<a href="fuel.pdf">Advisory On Fuel Capacity of Car/Two Wheeler’s mention in the Service Manuals by Vehicle Manufacturers dated 06.03.2023</a>\n<a href="pkg.pdf">Advisory On Packages of agriculture farm produce upto 50kg under the Legal Metrology (Packaged Commodities) Rule, 2011</a>'''
        def raise_for_status(self): pass
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def get(self, url): return Response()
    monkeypatch.setattr(fetcher.httpx, "Client", Client)
    documents = fetcher.discover_documents("https://example.test/legal")
    assert [item["url"] for item in documents] == ["https://example.test/pkg.pdf"]
