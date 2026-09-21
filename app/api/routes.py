"""HTTP contract for the RAG service and the separate Rule Engine."""

from threading import Lock, Thread

from fastapi import APIRouter

from app.rag_update_service import get_update_status, is_update_running, run_rag_update
from app.repository.rule_repository import get_active_rules


router = APIRouter(prefix="/api")
_refresh_dispatch_lock = Lock()
_refresh_thread: Thread | None = None


@router.post("/rag/refresh")
def refresh_regulatory_knowledge() -> dict:
    """Queue the shared update service without occupying an API worker."""
    global _refresh_thread
    with _refresh_dispatch_lock:
        if is_update_running() or (_refresh_thread is not None and _refresh_thread.is_alive()):
            return {**get_update_status(), "status": "in_progress", "accepted": False}
        _refresh_thread = Thread(target=run_rag_update, name="manual-rag-refresh", daemon=True)
        _refresh_thread.start()
    return {**get_update_status(), "status": "running", "accepted": True}


@router.get("/rag/status")
def regulatory_knowledge_status() -> dict:
    return get_update_status()


@router.get("/rules/active")
def active_rules() -> list[dict]:
    """Provide active, validated RAG rules to API consumers."""
    return [rule.model_dump() for rule in get_active_rules()]
