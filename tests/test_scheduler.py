from app.extraction.rule_schema import ComplianceRule


def _rule():
    return ComplianceRule(
        rule_id="LM_001", parameter="mrp", condition="must_exist",
        source_document="Rules", source_pages=[1], evidence_text="MRP shall be declared.",
    )


def test_scheduler_delegates_to_central_update_function(monkeypatch):
    import app.scheduler.scheduler as scheduler

    expected = {"status": "up_to_date", "documents_checked": 1}
    monkeypatch.setattr(scheduler, "run_rag_update", lambda: expected)

    assert scheduler.check_for_updates() == expected


def test_scheduler_uses_configured_days(monkeypatch):
    import app.scheduler.scheduler as scheduler

    monkeypatch.setattr(scheduler, "get_check_interval_days", lambda: 7)
    instance = scheduler.create_scheduler()
    try:
        job = instance.get_job("legal_metrology_regulatory_check")
        assert job.trigger.interval.total_seconds() == 7 * 24 * 60 * 60
    finally:
        # create_scheduler() intentionally returns a paused scheduler.
        # APScheduler only permits shutdown after start().
        if instance.running:
            instance.shutdown(wait=False)


def test_start_scheduler_queues_initial_check_without_running_it_synchronously(monkeypatch):
    import app.scheduler.scheduler as scheduler

    class FakeScheduler:
        def __init__(self):
            self.jobs = []
            self.started = False

        def add_job(self, function, **kwargs):
            self.jobs.append((function, kwargs))

        def start(self):
            self.started = True

    instance = FakeScheduler()
    monkeypatch.setattr(scheduler, "create_scheduler", lambda: instance)
    monkeypatch.setattr(
        scheduler,
        "check_for_updates",
        lambda: (_ for _ in ()).throw(AssertionError("must run in scheduler worker")),
    )

    assert scheduler.start_scheduler() is instance
    assert instance.started is True
    assert len(instance.jobs) == 1
    _, options = instance.jobs[0]
    assert options["trigger"] == "date"
    assert options["id"] == "legal_metrology_initial_check"


def test_fastapi_startup_only_starts_scheduler(monkeypatch):
    import app.main as main

    called = []
    monkeypatch.setattr(main, "start_scheduler", lambda: called.append(True))

    main.start_background_regulatory_updates()

    assert called == [True]
