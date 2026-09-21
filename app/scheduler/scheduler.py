"""Independent, non-overlapping regulatory freshness checks."""

from apscheduler.schedulers.background import BackgroundScheduler

from app import refresh as refresh_service
from app.config import get_check_interval_days


def check_for_updates() -> dict:
    """Backward-compatible scheduler entry point (two-document refresh)."""
    return refresh_service.check_for_updates()


def create_scheduler() -> BackgroundScheduler:
    """Create, but do not start, the background regulatory update service."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        check_for_updates,
        trigger="interval",
        days=get_check_interval_days(),
        id="legal_metrology_regulatory_check",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    return scheduler


_scheduler_instance: BackgroundScheduler | None = None


def reset_scheduler_for_tests() -> None:
    """Forget the cached scheduler; test-only, never touches jobs or rules."""
    global _scheduler_instance
    _scheduler_instance = None


def start_scheduler() -> BackgroundScheduler:
    """Start weekly monitoring and queue a non-blocking initial check.

    ``BackgroundScheduler`` runs the one-shot job in its worker pool, allowing
    FastAPI's startup lifecycle to complete before network, OCR, or Ollama work
    begins. Both this job and the weekly job still use the shared update lock.

    Restart-safe within one process: a second call returns the already-started
    scheduler instead of creating duplicate jobs.
    """
    global _scheduler_instance
    if _scheduler_instance is not None:
        return _scheduler_instance
    scheduler = create_scheduler()
    scheduler.add_job(
        check_for_updates,
        trigger="date",
        id="legal_metrology_initial_check",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.start()
    _scheduler_instance = scheduler
    return scheduler
