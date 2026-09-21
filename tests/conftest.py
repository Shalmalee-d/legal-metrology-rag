"""Automatic test isolation for Legal Metrology RAG.

Production paths (data/registry/document_registry.json, data/documents/,
data/rules/, data/extracted/, update_status.json) must never be touched by
tests. This autouse fixture redirects all storage to tmp_path via env vars
AND patches legacy module globals to the same tmp locations, so both new
config-getter code and old monkeypatch-style code stay consistent.

No test needs to remember to patch REGISTRY_FILE manually. Explicit per-test
monkeypatching still works because resolvers prefer an explicitly patched
global that differs from the production default.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_rag_storage(tmp_path, monkeypatch):
    # Isolated roots per test (no hardcoded test registry path).
    registry_dir = tmp_path / "registry"
    documents_dir = tmp_path / "documents"
    rules_dir = tmp_path / "rules"
    extracted_dir = tmp_path / "extracted"

    registry_file = registry_dir / "document_registry.json"
    update_status_file = registry_dir / "update_status.json"
    metadata_file = documents_dir / "metadata.json"
    rules_file = rules_dir / "compliance_rules.json"

    # Env-var layer (used by app.config getters, no global mutable state).
    monkeypatch.setenv("RAG_DATA_ROOT", str(tmp_path / "data_root"))
    monkeypatch.setenv("RAG_REGISTRY_DIR", str(registry_dir))
    monkeypatch.setenv("RAG_REGISTRY_FILE", str(registry_file))
    monkeypatch.setenv("RAG_UPDATE_STATUS_FILE", str(update_status_file))
    monkeypatch.setenv("RAG_DOCUMENTS_DIR", str(documents_dir))
    monkeypatch.setenv("RAG_METADATA_FILE", str(metadata_file))
    monkeypatch.setenv("RAG_RULES_DIR", str(rules_dir))
    monkeypatch.setenv("RAG_RULES_FILE", str(rules_file))
    monkeypatch.setenv("RAG_EXTRACTED_DIR", str(extracted_dir))

    # Legacy-global layer (keeps old monkeypatch-style tests consistent).
    # Patch to the SAME tmp locations so resolvers never split-brain.
    import app.registry.document_registry as registry
    import app.fetcher.document_fetcher as fetcher
    import app.repository.rule_repository as repository
    import app.extraction.pdf_extractor as extractor

    monkeypatch.setattr(registry, "REGISTRY_DIR", registry_dir)
    monkeypatch.setattr(registry, "REGISTRY_FILE", registry_file)
    monkeypatch.setattr(registry, "UPDATE_STATUS_FILE", update_status_file)

    monkeypatch.setattr(fetcher, "DATA_DIR", documents_dir)
    monkeypatch.setattr(fetcher, "METADATA_FILE", metadata_file)

    monkeypatch.setattr(repository, "RULES_DIR", rules_dir)
    monkeypatch.setattr(repository, "RULES_FILE", rules_file)

    monkeypatch.setattr(extractor, "EXTRACTED_DIR", extracted_dir)

    # Reset in-memory scheduler/service progress so tests never leak state.
    try:
        import app.rag_update_service as service

        service.reset_update_state_for_tests()
    except Exception:
        pass

    yield

    # Ensure no leaked lock / progress after each test.
    try:
        import app.rag_update_service as service

        service.reset_update_state_for_tests()
    except Exception:
        pass
