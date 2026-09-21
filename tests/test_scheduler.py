from app.extraction.rule_schema import ComplianceRule


def _rule():
    return ComplianceRule(
        rule_id="LM_001", parameter="mrp", condition="must_exist",
        source_document="Rules", source_pages=[1], evidence_text="MRP shall be declared.",
    )


def test_scheduler_delegates_to_central_update_function(monkeypatch):
    import app.refresh as refresh
    import app.scheduler.scheduler as scheduler

    expected = {"status": "up_to_date", "documents_checked": 2}
    monkeypatch.setattr(refresh, "check_for_updates", lambda: expected)

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


def test_default_interval_is_seven_days(monkeypatch):
    from app.config import get_check_interval_days

    monkeypatch.delenv("RAG_CHECK_INTERVAL_DAYS", raising=False)
    assert get_check_interval_days() == 7


def test_weekly_and_initial_jobs_share_refresh_function(monkeypatch):
    import app.scheduler.scheduler as scheduler

    scheduler.reset_scheduler_for_tests()
    try:
        recurring = scheduler.create_scheduler()
        try:
            job = recurring.get_job("legal_metrology_regulatory_check")
            assert job.func is scheduler.check_for_updates
        finally:
            if recurring.running:
                recurring.shutdown(wait=False)

        class FakeScheduler:
            def __init__(self):
                self.jobs = []

            def add_job(self, function, **kwargs):
                self.jobs.append((function, kwargs))

            def start(self):
                pass

        instance = FakeScheduler()
        monkeypatch.setattr(scheduler, "create_scheduler", lambda: instance)
        assert scheduler.start_scheduler() is instance
        assert len(instance.jobs) == 1
        function, options = instance.jobs[0]
        assert function is scheduler.check_for_updates
        assert options["id"] == "legal_metrology_initial_check"
    finally:
        scheduler.reset_scheduler_for_tests()


def test_overlapping_run_returns_in_progress_without_work(monkeypatch):
    import app.refresh as refresh
    import app.scheduler.scheduler as scheduler

    calls = []
    monkeypatch.setattr(refresh, "discover_documents",
                        lambda url: calls.append(url) or (_ for _ in ()).throw(AssertionError("no work")))
    assert refresh._refresh_lock.acquire(blocking=False)
    try:
        result = scheduler.check_for_updates()
    finally:
        refresh._refresh_lock.release()
    assert result["status"] == "in_progress"
    assert calls == []


def test_restart_does_not_duplicate_jobs(monkeypatch):
    import app.scheduler.scheduler as scheduler

    class FakeScheduler:
        def __init__(self):
            self.jobs = []
            self.started = False

        def add_job(self, function, **kwargs):
            self.jobs.append((function, kwargs))

        def start(self):
            self.started = True

    scheduler.reset_scheduler_for_tests()
    try:
        first = FakeScheduler()
        monkeypatch.setattr(scheduler, "create_scheduler", lambda: first)
        assert scheduler.start_scheduler() is first
        assert scheduler.start_scheduler() is first
        # Only the single initial job: the restart added nothing.
        assert len(first.jobs) == 1
    finally:
        scheduler.reset_scheduler_for_tests()
