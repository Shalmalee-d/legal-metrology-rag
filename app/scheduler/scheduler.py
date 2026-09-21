"""Independent, non-overlapping regulatory freshness checks."""

from apscheduler.schedulers.background import BackgroundScheduler

from app.config import get_check_interval_days
from app.rag_update_service import run_rag_update


def check_for_updates() -> dict:
    """Backward-compatible scheduler entry point."""
    return run_rag_update()


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


def start_scheduler() -> BackgroundScheduler:
    """Start weekly monitoring and queue a non-blocking initial check.

    ``BackgroundScheduler`` runs the one-shot job in its worker pool, allowing
    FastAPI's startup lifecycle to complete before network, OCR, or Ollama work
    begins. Both this job and the weekly job still use the shared update lock.
    """
    scheduler = create_scheduler()
    scheduler.add_job(
        lambda: run_rag_update(allow_bootstrap=False),
        trigger="date",
        id="legal_metrology_initial_check",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.start()
    return scheduler
