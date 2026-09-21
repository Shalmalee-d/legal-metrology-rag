"""HTTP contract for the RAG service and the separate Rule Engine."""

from threading import Lock, Thread

from fastapi import APIRouter

from app.refresh import check_for_updates, get_refresh_status, is_update_running
from app.repository.rule_repository import get_active_rules


router = APIRouter(prefix="/api")
_refresh_dispatch_lock = Lock()
_refresh_thread: Thread | None = None


def _queue_refresh() -> dict:
    """Queue the shared two-document refresh without occupying an API worker."""
    global _refresh_thread
    with _refresh_dispatch_lock:
        if is_update_running() or (_refresh_thread is not None and _refresh_thread.is_alive()):
            return {**get_refresh_status(), "status": "in_progress", "accepted": False}
        _refresh_thread = Thread(target=check_for_updates, name="manual-rag-refresh", daemon=True)
        _refresh_thread.start()
    return {**get_refresh_status(), "status": "running", "accepted": True}


@router.post("/rag/refresh")
def refresh_regulatory_knowledge() -> dict:
    """Queue the shared update service without occupying an API worker."""
    return _queue_refresh()


@router.post("/rag/check-updates")
def check_regulatory_updates() -> dict:
    """Frontend entry point for the shared two-document refresh."""
    return _queue_refresh()


@router.get("/rag/status")
def regulatory_knowledge_status() -> dict:
    return get_refresh_status()


@router.get("/rules/active")
def active_rules() -> list[dict]:
    """Provide active, validated RAG rules to API consumers."""
    return [rule.model_dump() for rule in get_active_rules()]
