"""HTTP contract for the RAG service and the separate Rule Engine."""

from threading import Lock, Thread

from fastapi import APIRouter, HTTPException

from app.refresh import check_for_updates, get_refresh_status, is_update_running
from app.repository.rule_repository import (
    CategoryNotFoundError,
    list_categories,
    load_applicable_rules,
    load_category_rules,
)


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
    """Provide active common product rules to API consumers."""
    return [rule.model_dump() for rule in load_applicable_rules(None)]


@router.get("/rules/categories")
def rule_categories() -> list[str]:
    """List product categories that have a rule file."""
    return list_categories()


@router.get("/rules/category/{category}")
def category_rules(category: str) -> list[dict]:
    """Provide active rules specific to one product category."""
    from app.repository.rule_repository import is_applicable

    try:
        rules = [rule for rule in load_category_rules(category) if is_applicable(rule)]
    except CategoryNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return [rule.model_dump() for rule in rules]


@router.get("/rules/applicable/{category}")
def applicable_rules(category: str) -> list[dict]:
    """Provide active common rules plus one category's active rules."""
    try:
        rules = load_applicable_rules(category)
    except CategoryNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return [rule.model_dump() for rule in rules]
