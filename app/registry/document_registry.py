"""Durable registry of official documents, separate from the rule repository."""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urlsplit, urlunsplit


REGISTRY_DIR = Path("data/registry")
REGISTRY_FILE = REGISTRY_DIR / "document_registry.json"
UPDATE_STATUS_FILE = REGISTRY_DIR / "update_status.json"

_DEFAULT_REGISTRY_DIR = Path("data/registry")
_DEFAULT_REGISTRY_FILE = _DEFAULT_REGISTRY_DIR / "document_registry.json"
_DEFAULT_UPDATE_STATUS_FILE = _DEFAULT_REGISTRY_DIR / "update_status.json"


def _resolve_registry_dir() -> Path:
    """Configured registry dir; explicit monkeypatch wins, else env/default."""
    # Direct monkeypatch of REGISTRY_DIR (legacy tests) must keep working
    # while the autouse test fixture redirects via env vars.
    if REGISTRY_DIR != _DEFAULT_REGISTRY_DIR:
        return REGISTRY_DIR
    from app.config import get_registry_dir

    return get_registry_dir()


def _resolve_registry_file() -> Path:
    """Configured registry file; explicit monkeypatch wins, else env/default."""
    if REGISTRY_FILE != _DEFAULT_REGISTRY_FILE:
        return REGISTRY_FILE
    from app.config import get_registry_file

    return get_registry_file()


def _resolve_update_status_file() -> Path:
    """Configured update-status file; explicit monkeypatch wins, else env/default."""
    if UPDATE_STATUS_FILE != _DEFAULT_UPDATE_STATUS_FILE:
        return UPDATE_STATUS_FILE
    from app.config import get_update_status_file

    return get_update_status_file()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_url(url: str) -> str:
    """Make a stable identity without treating URL fragments as documents."""
    url, _ = urldefrag(url.strip())
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


def document_id(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()


def load_registry() -> list[dict]:
    registry_file = _resolve_registry_file()
    if not registry_file.exists():
        return []
    with open(registry_file, encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError("Document registry must contain a JSON list.")
    return records


def save_registry(records: list[dict]) -> None:
    registry_dir = _resolve_registry_dir()
    registry_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=registry_dir,
                                     prefix=".document_registry.", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        json.dump(records, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    try:
        _replace_registry_file(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_registry_file(temporary: Path) -> None:
    """Persist the registry safely on Windows/OneDrive-managed paths.

    Prefer an atomic replacement. If Windows temporarily denies replacement
    because OneDrive is syncing the existing file, retry briefly. As a final
    report the write failure while retaining the prior complete registry.
    """
    import time

    registry_file = _resolve_registry_file()
    last_error: PermissionError | None = None
    for _ in range(5):
        try:
            os.replace(temporary, registry_file)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.5)

    # Do not fall back to opening the destination in ``wb``: a second lock or
    # disk failure could truncate the only valid registry snapshot.  The
    # temporary, fsynced file is cleaned by the caller and the next run can
    # retry the safe replacement.
    assert last_error is not None
    raise last_error


def get_record(url: str) -> dict | None:
    identity = document_id(url)
    return next((record for record in load_registry() if record.get("document_id") == identity), None)


def upsert_discovered(document: dict, *, status: str | None = None, **fields) -> dict:
    records = load_registry()
    identity = document_id(document["url"])
    previous = next((record for record in records if record.get("document_id") == identity), {})
    record = {
        **previous,
        "document_id": identity,
        "title": document["title"],
        "url": normalize_url(document["url"]),
        "source": document.get("source", document.get("category", "legal_metrology_packaged_commodities")),
        "category": document.get("category", "legal_metrology_packaged_commodities"),
        "last_checked": now(),
        **fields,
    }
    if status is not None:
        record["status"] = status
    record.setdefault("status", "discovered")
    records = [item for item in records if item.get("document_id") != identity]
    records.append(record)
    save_registry(records)
    return record


def mark_processed(url: str, fingerprint: str) -> None:
    record = get_record(url)
    if record is None:
        raise ValueError("Cannot mark an unknown registry document as processed.")
    upsert_discovered(record, status="processed", fingerprint=fingerprint,
                     processed_fingerprint=fingerprint, last_processed=now())


def mark_missing(source: str, discovered_urls: set[str]) -> list[dict]:
    """Track an absence as a source observation, never as a legal withdrawal."""
    records = load_registry()
    changed = []
    for record in records:
        if record.get("source") == source and normalize_url(record["url"]) not in discovered_urls:
            if record.get("status") != "no_longer_listed":
                record["status"] = "no_longer_listed"
                record["last_checked"] = now()
                changed.append(record)
    if changed:
        save_registry(records)
    return changed


def load_update_status() -> dict:
    """Read the durable, non-secret summary of the latest completed update."""
    status_file = _resolve_update_status_file()
    if not status_file.exists():
        return {}
    with open(status_file, encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Update status must contain a JSON object.")
    return data


def save_update_status(status: dict) -> None:
    """Persist update observability using the registry's safe write pattern."""
    registry_dir = _resolve_registry_dir()
    status_file = _resolve_update_status_file()
    registry_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=registry_dir,
                                     prefix=".update_status.", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        json.dump(status, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    try:
        _replace_file(temporary, status_file)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_file(temporary: Path, destination: Path) -> None:
    import time
    last_error: PermissionError | None = None
    for _ in range(5):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.5)
    assert last_error is not None
    raise last_error
