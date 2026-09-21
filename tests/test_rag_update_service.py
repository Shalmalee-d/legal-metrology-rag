from app.extraction.rule_schema import ComplianceRule


def _rule():
    return ComplianceRule(rule_id="LM_001", parameter="mrp", condition="must_exist",
        requirement="The package must declare MRP.", source_document="Rules", source_pages=[1], evidence_text="MRP shall be declared.")


def _configure_service(monkeypatch, detected, record=None, rules=None):
    import app.rag_update_service as service
    document = {"title": "Rules", "url": "https://example.test/rules.pdf"}
    monkeypatch.setattr(service, "OFFICIAL_SOURCES", [{"url": "https://example.test"}])
    monkeypatch.setattr(service, "discover_documents", lambda url: [document])
    monkeypatch.setattr(service, "check_document_change", lambda item: detected)
    monkeypatch.setattr(service, "load_rules", lambda: [_rule()])
    if record is not None:
        monkeypatch.setattr(service, "download_document", lambda item: record)
    if rules is not None:
        monkeypatch.setattr(service, "process_pdf_document", lambda *args, **kwargs: rules)
    monkeypatch.setattr(service, "update_rules", lambda staged: None)
    monkeypatch.setattr(service, "mark_document_processed", lambda url, sha256: None)
    return service


def test_unchanged_document_does_not_process(monkeypatch):
    service = _configure_service(monkeypatch, "UNCHANGED")
    monkeypatch.setattr(service, "get_active_rules", lambda: [_rule()])
    monkeypatch.setattr(service, "download_document", lambda item: (_ for _ in ()).throw(AssertionError("downloaded")))

    result = service.run_rag_update()

    assert result["status"] == "up_to_date"
    assert result["rules_updated"] == 0


def test_new_document_runs_rag_processing(monkeypatch):
    record = {"title": "Rules", "local_path": "rules.pdf", "sha256": "hash", "status": "NEW"}
    service = _configure_service(monkeypatch, "NEW", record, [_rule()])

    result = service.run_rag_update()

    assert result["status"] == "updated"
    assert result["documents_changed"] == 1
    assert result["rules_activated"] == 1
    status = service.get_update_status()
    assert status["update_in_progress"] is False
    assert status["current_stage"] == "completed"
    assert status["documents_total"] == 1
    assert status["documents_checked"] == 1
    assert status["rules_updated"] == 1


def test_status_exposes_current_expensive_pipeline_stage(monkeypatch):
    import app.rag_update_service as service

    monkeypatch.setattr(service, "get_active_rules", lambda: [])
    service._set_progress(
        current_stage="source_check", current_document=None, current_chunk=None,
        documents_checked=0, documents_total=3, documents_changed=0,
        rules_updated=0, last_error=None,
    )
    report = service._pipeline_progress("Third Amendment Rules, 2011")
    report("ollama_rule_generation", {"chunk_id": "chunk_0002"}, 2, 4)

    status = service.get_update_status()

    assert {"update_in_progress", "current_stage", "current_document", "documents_checked",
            "documents_total", "documents_changed", "rules_updated", "last_successful_update",
            "last_error"}.issubset(status)
    assert status["current_stage"] == "ollama_rule_generation"
    assert status["current_document"] == "Third Amendment Rules, 2011"
    assert status["current_chunk"] == "chunk_0002 (2/4)"


def test_update_in_progress_is_not_run_twice(monkeypatch):
    import app.rag_update_service as service

    service._update_lock.acquire()
    try:
        assert service.run_rag_update()["status"] == "in_progress"
    finally:
        service._update_lock.release()


def test_partial_failure_activates_successful_staged_rules(monkeypatch):
    import app.rag_update_service as service
    first = {"title": "First", "url": "https://example.test/first.pdf"}
    second = {"title": "Second", "url": "https://example.test/second.pdf"}
    monkeypatch.setattr(service, "OFFICIAL_SOURCES", [{"url": "https://example.test"}])
    monkeypatch.setattr(service, "discover_documents", lambda url: [first, second])
    monkeypatch.setattr(service, "check_document_change", lambda item: "NEW")
    monkeypatch.setattr(service, "download_document", lambda item: {
        "title": item["title"], "local_path": "rules.pdf", "sha256": item["title"], "status": "NEW",
    })
    monkeypatch.setattr(service, "process_pdf_document", lambda path, *args, **kwargs: [_rule()] if path else [])
    # Simulate a later extraction failure, after the first document was staged.
    calls = {"count": 0}
    def process(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise ValueError("bad PDF")
        return [_rule()]
    monkeypatch.setattr(service, "process_pdf_document", process)
    activated = []
    processed = []
    monkeypatch.setattr(service, "update_rules", lambda rules: activated.extend(rules))
    monkeypatch.setattr(service, "mark_document_processed", lambda url, sha256: processed.append((url, sha256)))

    result = service.run_rag_update()

    assert result["status"] == "completed_with_errors"
    assert result["successful_documents"] == 1
    assert result["failed_documents"] == 1
    assert len(activated) == 1
    assert processed == [(first["url"], "First")]
    assert result["rules_preserved"] is True
    status = service.get_update_status()
    assert status["update_in_progress"] is False
    assert status["current_stage"] == "completed_with_errors"
    assert "bad PDF" in status["last_error"]


def _configure_repository_backed_update(monkeypatch, tmp_path, documents, process):
    import app.rag_update_service as service
    import app.repository.rule_repository as repository

    monkeypatch.setattr(repository, "RULES_DIR", tmp_path)
    monkeypatch.setattr(repository, "RULES_FILE", tmp_path / "compliance_rules.json")
    monkeypatch.setattr(service, "OFFICIAL_SOURCES", [{"url": "https://example.test"}])
    monkeypatch.setattr(service, "discover_documents", lambda url: documents)
    monkeypatch.setattr(service, "check_document_change", lambda document: "NEW")
    monkeypatch.setattr(service, "download_document", lambda document: {
        "title": document["title"],
        "url": document["url"],
        "local_path": document["title"],
        "sha256": f"{document['title']}-hash",
        "status": "NEW",
    })
    monkeypatch.setattr(service, "process_pdf_document", process)
    return service, repository


def test_bootstrap_partial_failure_persists_rules_and_leaves_failed_document_retryable(tmp_path, monkeypatch):
    successful = {"title": "Successful rules", "url": "https://example.test/success.pdf"}
    failed = {"title": "Failed rules", "url": "https://example.test/failed.pdf"}

    def process(path, *args, **kwargs):
        if path == failed["title"]:
            raise RuntimeError("Local Ollama rule generation failed")
        return [_rule().model_copy(update={"source_identity": "success-rule", "version": "success-v1"})]

    service, repository = _configure_repository_backed_update(
        monkeypatch, tmp_path, [successful, failed], process
    )
    processed = []
    monkeypatch.setattr(service, "mark_document_processed", lambda url, sha256: processed.append((url, sha256)))

    result = service.run_rag_update()

    assert result["status"] == "completed_with_errors"
    assert result["successful_documents"] == 1
    assert result["failed_documents"] == 1
    assert result["active_rule_count"] == 1
    assert len(repository.load_rules()) == 1
    assert processed == [(successful["url"], "Successful rules-hash")]
    assert "Failed rules" in result["failed_document_details"][0]


def test_all_document_failures_do_not_create_rule_repository(tmp_path, monkeypatch):
    document = {"title": "Broken rules", "url": "https://example.test/broken.pdf"}
    service, repository = _configure_repository_backed_update(
        monkeypatch,
        tmp_path,
        [document],
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Ollama unavailable")),
    )
    processed = []
    monkeypatch.setattr(service, "mark_document_processed", lambda url, sha256: processed.append((url, sha256)))

    result = service.run_rag_update()

    assert result["status"] == "failed"
    assert result["active_rule_count"] == 0
    assert repository.RULES_FILE.exists() is False
    assert processed == []


def test_incremental_partial_failure_replaces_success_without_duplicates_or_losing_other_rules(tmp_path, monkeypatch):
    import app.repository.rule_repository as repository

    changed = {"title": "Changed rules", "url": "https://example.test/changed.pdf"}
    failed = {"title": "Failed rules", "url": "https://example.test/failed.pdf"}
    service, repository = _configure_repository_backed_update(monkeypatch, tmp_path, [changed, failed], None)
    existing = [
        _rule().model_copy(update={"rule_id": "LM_CHANGED", "source_identity": "changed", "version": "v1"}),
        _rule().model_copy(update={"rule_id": "LM_FAILED", "source_identity": "failed", "version": "v1"}),
        _rule().model_copy(update={"rule_id": "LM_UNRELATED", "source_identity": "unrelated", "version": "v1"}),
    ]
    repository.save_rules(existing)

    def process(path, *args, **kwargs):
        if path == failed["title"]:
            raise RuntimeError("Ollama unavailable")
        return [_rule().model_copy(update={"rule_id": "LM_CHANGED", "source_identity": "changed", "version": "v2"})]

    monkeypatch.setattr(service, "process_pdf_document", process)
    processed = []
    monkeypatch.setattr(service, "mark_document_processed", lambda url, sha256: processed.append((url, sha256)))

    result = service.run_rag_update()
    active = {rule.rule_id: rule for rule in repository.get_active_rules()}

    assert result["status"] == "completed_with_errors"
    assert set(active) == {"LM_CHANGED", "LM_FAILED", "LM_UNRELATED"}
    assert active["LM_CHANGED"].version == "v2"
    assert active["LM_FAILED"].version == "v1"
    assert active["LM_UNRELATED"].version == "v1"
    assert processed == [(changed["url"], "Changed rules-hash")]

    service.run_rag_update()
    assert len([rule for rule in repository.load_rules() if rule.source_identity == "changed"]) == 2


def test_startup_check_defers_empty_repository_bootstrap(monkeypatch):
    import app.rag_update_service as service

    document = {"title": "Rules", "url": "https://example.test/rules.pdf"}
    monkeypatch.setattr(service, "OFFICIAL_SOURCES", [{"url": "https://example.test"}])
    monkeypatch.setattr(service, "discover_documents", lambda url: [document])
    monkeypatch.setattr(service, "load_rules", lambda: [])
    monkeypatch.setattr(service, "check_document_change", lambda item: "NEW")
    monkeypatch.setattr(
        service, "download_document",
        lambda item: (_ for _ in ()).throw(AssertionError("must not download")),
    )

    result = service.run_rag_update(allow_bootstrap=False)

    assert result["status"] == "up_to_date"
    assert result["documents_checked"] == 1


def test_rebuild_with_zero_extracted_rules_does_not_erase_existing_repository(tmp_path, monkeypatch):
    import app.repository.rule_repository as repository
    document = {"title": "Advisory", "url": "https://example.test/advisory.pdf"}
    service, repository = _configure_repository_backed_update(
        monkeypatch, tmp_path, [document], lambda *args, **kwargs: []
    )
    existing = [_rule().model_copy(update={"source_identity": "existing", "version": "v1"})]
    repository.save_rules(existing)

    result = service.run_rag_update(rebuild=True)

    assert result["status"] == "failed"
    assert repository.load_rules()[0].source_identity == "existing"


def test_rebuild_processes_unchanged_cached_documents_and_replaces_snapshot(tmp_path, monkeypatch):
    """A controlled rebuild must not let a matching cache bypass Qwen/pipeline."""
    import app.repository.rule_repository as repository
    import app.rag_update_service as service

    document = {"title": "Rules", "url": "https://example.test/rules.pdf"}
    monkeypatch.setattr(repository, "RULES_DIR", tmp_path)
    monkeypatch.setattr(repository, "RULES_FILE", tmp_path / "compliance_rules.json")
    repository.save_rules([_rule().model_copy(update={"source_identity": "old", "version": "v1"})])
    monkeypatch.setattr(service, "OFFICIAL_SOURCES", [{"url": "https://example.test"}])
    monkeypatch.setattr(service, "discover_documents", lambda url: [document])
    monkeypatch.setattr(service, "mark_missing", lambda *args: [])
    monkeypatch.setattr(service, "check_document_change", lambda item: "UNCHANGED")
    monkeypatch.setattr(service, "download_document", lambda item: {
        "title": item["title"], "url": item["url"], "local_path": "rules.pdf",
        "sha256": "cached-hash", "status": "UNCHANGED",
    })
    rebuilt = _rule().model_copy(update={"rule_id": "LM_REBUILT", "source_identity": "rebuilt", "version": "v2"})
    processed = []
    monkeypatch.setattr(service, "process_pdf_document", lambda *args, **kwargs: processed.append(args[0]) or [rebuilt])
    monkeypatch.setattr(service, "mark_document_processed", lambda url, sha256: None)

    result = service.run_rag_update(rebuild=True)

    assert processed == ["rules.pdf"]
    assert result["status"] == "updated"
    assert result["rebuild_documents_attempted"] == 1
    assert [rule.rule_id for rule in repository.load_rules()] == ["LM_REBUILT"]
