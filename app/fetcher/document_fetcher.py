import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import httpx
import truststore
from bs4 import BeautifulSoup
from app.registry.document_registry import (
    get_record as get_registry_record,
    mark_processed as mark_registry_processed,
    document_id,
    normalize_url,
    upsert_discovered,
)


# Use Windows' trusted certificates
truststore.inject_into_ssl()


DATA_DIR = Path("data/documents")
METADATA_FILE = DATA_DIR / "metadata.json"
_DEFAULT_DATA_DIR = Path("data/documents")
_DEFAULT_METADATA_FILE = _DEFAULT_DATA_DIR / "metadata.json"


def _resolve_data_dir() -> Path:
    if DATA_DIR != _DEFAULT_DATA_DIR:
        return DATA_DIR
    from app.config import get_documents_dir

    return get_documents_dir()


def _resolve_metadata_file() -> Path:
    if METADATA_FILE != _DEFAULT_METADATA_FILE:
        return METADATA_FILE
    from app.config import get_metadata_file

    return get_metadata_file()
logger = logging.getLogger(__name__)
# One retry is enough to absorb the common transient government-server read
# timeout without turning a scheduled check into a long-running retry storm.
# Each attempt tries the deterministic URL candidates (HTTPS before legacy
# HTTP), so an old HTTP URL has at most four network calls.
DOWNLOAD_ATTEMPTS_PER_URL = 2
SOURCE_CHECK_TIMEOUT = httpx.Timeout(connect=5.0, read=8.0, write=8.0, pool=8.0)


class DocumentDownloadError(RuntimeError):
    """A document could not be retrieved, with every bounded attempt recorded."""

    def __init__(self, title: str, attempts: list[dict]) -> None:
        self.title = title
        self.attempts = attempts
        details = "; ".join(
            f"url={item['url']} attempt={item['attempt']} error={item['error_type']}"
            f" timeout={item.get('timeout_type')} status={item.get('http_status')}"
            for item in attempts
        )
        super().__init__(f"Download failed for {title}: {details}")


def _checked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def calculate_sha256(data: bytes) -> str:
    """Calculate SHA-256 hash of downloaded data."""

    return hashlib.sha256(data).hexdigest()


def _download_error_details(error: Exception, url: str, attempt: int) -> dict:
    timeout_type = None
    if isinstance(error, httpx.ConnectTimeout):
        timeout_type = "connect"
    elif isinstance(error, httpx.ReadTimeout):
        timeout_type = "read"
    elif isinstance(error, httpx.WriteTimeout):
        timeout_type = "write"
    elif isinstance(error, httpx.PoolTimeout):
        timeout_type = "pool"
    status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
    return {"url": url, "attempt": attempt, "error_type": type(error).__name__,
            "timeout_type": timeout_type, "http_status": status, "message": str(error)}


def load_metadata() -> list:
    """Load metadata from previous runs."""

    metadata_file = _resolve_metadata_file()
    if not metadata_file.exists():
        return []

    try:
        with open(metadata_file, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Document metadata is corrupted: {metadata_file}") from error

    if not isinstance(metadata, list):
        raise ValueError("Document metadata must contain a JSON list.")
    return metadata


def save_metadata(metadata: list) -> None:
    """Save document metadata."""

    data_dir = _resolve_data_dir()
    metadata_file = _resolve_metadata_file()
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=data_dir,
        prefix=f".{metadata_file.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary_path = Path(file.name)
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    try:
        _replace_file_safely(temporary_path, metadata_file)
    finally:
        temporary_path.unlink(missing_ok=True)


def _replace_file_safely(temporary_path: Path, destination: Path) -> None:
    """Retry atomic replacement on Windows without risking a truncated file."""
    last_error: PermissionError | None = None
    for _ in range(5):
        try:
            os.replace(temporary_path, destination)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.25)
    assert last_error is not None
    raise last_error


def discover_documents(source_url: str) -> list[dict]:
    """
    Discover Legal Metrology documents
    from the official Department of Consumer Affairs page.
    """

    timeout = httpx.Timeout(
        connect=15.0,
        read=30.0,
        write=30.0,
        pool=30.0,
    )

    headers = {
        "User-Agent": (
            "Legal-Metrology-RAG/1.0 "
            "(Regulatory research and compliance monitoring)"
        )
    }

    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers=headers,
    ) as client:

        response = client.get(source_url)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    documents = []
    seen_urls = set()

    for link in soup.find_all("a", href=True):

        title = link.get_text(" ", strip=True)
        href = link["href"]

        context = _link_context(link)
        if not _is_packaged_commodities_document(title, context):
            continue
        url = normalize_url(urljoin(source_url, href))
        if url in seen_urls:
            continue
        seen_urls.add(url)
        documents.append({"title": title, "url": url, "source": source_url,
                          "category": "legal_metrology_packaged_commodities",
                          "source_context": context})

    return documents


_EXCLUDED_CATEGORIES = ("general rules", "national standards", "approval of models", "numeration rules")
_RELEVANT_TERMS = ("packaged commodit", "package commodity")
_DOCUMENT_TERMS = ("rule", "amendment", "corrigendum", "advisory", "guideline", "notification")


def _link_context(link) -> str:
    """Use nearby heading/list context when a link title is abbreviated."""
    parts = []
    heading = link.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
    if heading:
        parts.append(heading.get_text(" ", strip=True))
    parent = link.find_parent(["li", "tr", "p", "div"])
    if parent:
        parts.append(parent.get_text(" ", strip=True))
    return " ".join(parts)


def _is_packaged_commodities_document(title: str, context: str) -> bool:
    title_text = title.casefold()
    text = f"{title} {context}".casefold()
    if any(term in text for term in _EXCLUDED_CATEGORIES):
        return False
    if not any(term in text for term in _RELEVANT_TERMS):
        return False
    if not any(term in text for term in _DOCUMENT_TERMS):
        return False

    # A page section can contain unrelated advisories interleaved with the
    # Packaged Commodities material.  Unlike a terse amendment link, an
    # advisory/guideline must itself identify a package/commodity subject.
    # This prevents section context alone from admitting (for example)
    # vehicle-service-manual advisories.
    if any(term in title_text for term in ("advisory", "guideline")):
        return any(term in title_text for term in _RELEVANT_TERMS)
    return True


def check_document_change(document: dict) -> str:
    """Check stable HTTP metadata without unnecessarily waiting on dead HTTP links.

    HTTPS is preferred for legacy government URLs. If metadata cannot be
    obtained, the document is conservatively treated as changed so the bounded
    download path can either fetch it or use a verified cache during bootstrap.
    """
    registry_record = get_registry_record(document["url"])
    if registry_record is None:
        upsert_discovered(document, status="new")
        return "NEW"
    if registry_record.get("fingerprint") != registry_record.get("processed_fingerprint"):
        upsert_discovered(document, status="changed")
        return "CHANGED"

    metadata = load_metadata()
    previous = next((item for item in metadata if item.get("url") == document["url"]), None)
    if previous is None:
        upsert_discovered(document, status="changed")
        return "CHANGED"
    if previous.get("sha256") != previous.get("processed_sha256"):
        # Metadata shows a downloaded version that was never activated.
        # Keep registry status consistent so UNCHANGED is never reported
        # for a fingerprint that has no successful processing history.
        upsert_discovered(document, status="changed")
        return "CHANGED"

    urls_to_try = [document["url"]]
    if document["url"].startswith("http://"):
        urls_to_try.insert(0, document["url"].replace("http://", "https://", 1))

    failures: list[dict] = []
    for url in urls_to_try:
        try:
            with httpx.Client(timeout=SOURCE_CHECK_TIMEOUT, follow_redirects=True) as client:
                response = client.head(url)
                response.raise_for_status()
            signature = {
                "etag": response.headers.get("etag"),
                "last_modified": response.headers.get("last-modified"),
                "content_length": response.headers.get("content-length"),
            }
            old_signature = {key: previous.get(key) for key in signature}
            etag = signature["etag"]
            strong_etag_match = bool(
                etag and not etag.startswith("W/") and etag == old_signature["etag"]
            )
            metadata_match = bool(
                signature["last_modified"]
                and signature["content_length"]
                and signature["last_modified"] == old_signature["last_modified"]
                and signature["content_length"] == old_signature["content_length"]
            )
            unchanged = strong_etag_match or metadata_match
            previous.update({"last_checked": _checked_at(), **signature})
            save_metadata(metadata)
            upsert_discovered(
                document, status="unchanged" if unchanged else "changed", **signature
            )
            return "UNCHANGED" if unchanged else "CHANGED"
        except httpx.HTTPError as error:
            failures.append(_download_error_details(error, url, 1))
            logger.warning("SOURCE_CHECK_FAILED title=%s url=%s error_type=%s error=%s",
                           document["title"], url, type(error).__name__, error)

    # A metadata failure is neither evidence of a changed document nor proof
    # that it is unchanged.  Preserve it as a retryable source-check failure;
    # the orchestrator records it and keeps the prior valid rules intact.
    error_message = failures[-1]["message"] if failures else "No metadata response received."
    upsert_discovered(
        document,
        status="source_check_failed",
        source_check_error=error_message,
        source_check_failures=failures,
    )
    raise DocumentDownloadError(document["title"], failures)


def _download_document_attempt(document: dict, attempt: int) -> dict:
    """
    Download one document.

    The document is downloaded into memory first.
    Its hash is calculated before replacing any existing file.
    """

    data_dir = _resolve_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata()

    previous = next(
        (
            item
            for item in metadata
            if item.get("url") == document["url"]
        ),
        None,
    )

    # Create a safe filename
    safe_name = "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in document["title"]
    )

    # Titles are display metadata and are not unique.  URL-derived identity
    # avoids one official document overwriting another with the same title.
    file_path = data_dir / f"{document_id(document['url'])}.pdf"

    timeout = httpx.Timeout(
        connect=6.0,
        read=30.0,
        write=30.0,
        pool=30.0,
    )

    headers = {
        "User-Agent": (
            "Legal-Metrology-RAG/1.0 "
            "(Regulatory research and compliance monitoring)"
        )
    }

    urls_to_try = [document["url"]]

    # Some old government links use HTTP.
    # Try HTTPS first when possible.
    if document["url"].startswith("http://"):
        https_url = document["url"].replace(
            "http://",
            "https://",
            1,
        )

        urls_to_try.insert(0, https_url)

    failures: list[dict] = []

    for url in urls_to_try:

        try:
            logger.info("DOWNLOAD_ATTEMPT title=%s url=%s attempt=%s/%s", document["title"], url, attempt, DOWNLOAD_ATTEMPTS_PER_URL)

            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                headers=headers,
            ) as client:

                response = client.get(url)
                response.raise_for_status()

            content = response.content

            # Make sure we actually received a PDF
            if not content.startswith(b"%PDF"):
                raise ValueError(
                    "Downloaded content is not a valid PDF."
                )

            file_hash = calculate_sha256(content)

            # Content matches the previously downloaded bytes.
            if previous and previous.get("sha256") == file_hash:
                # Store newly observed HTTP metadata so future checks remain
                # lightweight even when older metadata lacked these fields.
                previous.update({
                    "etag": response.headers.get("etag"),
                    "last_modified": response.headers.get("last-modified"),
                    "content_length": response.headers.get("content-length"),
                    "last_checked": _checked_at(),
                })
                save_metadata(metadata)
                already_processed = previous.get("processed_sha256") == file_hash
                if already_processed:
                    upsert_discovered(
                        document, status="unchanged", fingerprint=file_hash,
                        processed_fingerprint=previous.get("processed_sha256"),
                        etag=response.headers.get("etag"),
                        last_modified=response.headers.get("last-modified"),
                        content_length=response.headers.get("content-length"),
                    )
                    return {**previous, "status": "UNCHANGED"}
                # Same bytes were downloaded before but never successfully
                # processed (e.g. Ollama/validation failed). Do NOT report
                # UNCHANGED: the fingerprint still requires OCR/Qwen work.
                # Keep registry honest and force the orchestrator to process.
                upsert_discovered(
                    document, status="changed", fingerprint=file_hash,
                    processed_fingerprint=previous.get("processed_sha256"),
                    etag=response.headers.get("etag"),
                    last_modified=response.headers.get("last-modified"),
                    content_length=response.headers.get("content-length"),
                )
                return {**previous, "status": "CHANGED"}

            # Save only after successful validation
            file_path.write_bytes(content)

            if previous is None:
                status = "NEW"
            else:
                status = "CHANGED"

            record = {
                "title": document["title"],
                "url": document["url"],
                "download_url": url,
                "local_path": str(file_path),
                "sha256": file_hash,
                "processed_sha256": previous.get("processed_sha256") if previous else None,
                "status": status,
                "etag": response.headers.get("etag"),
                "last_modified": response.headers.get("last-modified"),
                "content_length": response.headers.get("content-length"),
                "first_seen": previous.get("first_seen", _checked_at()) if previous else _checked_at(),
                "last_checked": _checked_at(),
            }

            # Replace previous metadata for this URL
            metadata = [
                item
                for item in metadata
                if item.get("url") != document["url"]
            ]

            metadata.append(record)

            save_metadata(metadata)
            upsert_discovered(
                document, status="downloaded", fingerprint=file_hash,
                etag=response.headers.get("etag"),
                last_modified=response.headers.get("last-modified"),
                content_length=response.headers.get("content-length"),
            )

            return record

        except Exception as error:
            details = _download_error_details(error, url, attempt)
            failures.append(details)
            logger.warning(
                "DOWNLOAD_FAILED title=%s url=%s attempt=%s/%s error_type=%s timeout_type=%s http_status=%s error=%s",
                document["title"], url, attempt, DOWNLOAD_ATTEMPTS_PER_URL,
                details["error_type"], details["timeout_type"], details["http_status"], details["message"],
            )

    raise DocumentDownloadError(document["title"], failures)


def download_document(document: dict) -> dict:
    """Download with one bounded retry for transient timeout failures only."""
    failures: list[dict] = []
    for attempt in range(1, DOWNLOAD_ATTEMPTS_PER_URL + 1):
        try:
            return _download_document_attempt(document, attempt)
        except DocumentDownloadError as error:
            failures.extend(error.attempts)
            timed_out = any(item["timeout_type"] is not None for item in error.attempts)
            if timed_out and attempt < DOWNLOAD_ATTEMPTS_PER_URL:
                logger.info("DOWNLOAD_RETRY title=%s delay_seconds=%s", document["title"], 0.5 * attempt)
                time.sleep(0.5 * attempt)
                continue
            raise DocumentDownloadError(document["title"], failures) from error

    raise DocumentDownloadError(document["title"], failures)


def get_verified_cached_document(document: dict) -> dict | None:
    """Return a previously downloaded PDF only when its recorded hash matches.

    This is intentionally a bootstrap recovery path, not a replacement for a
    live source check.  It lets an initial build process the official files
    already present on disk when a government host is temporarily unavailable,
    without accepting an unverified or partially written cache entry.
    """
    metadata = load_metadata()
    previous = next(
        (item for item in metadata if item.get("url") == document["url"]),
        None,
    )
    if previous is None:
        return None

    local_path = previous.get("local_path")
    expected_hash = previous.get("sha256")
    if not local_path or not expected_hash:
        return None

    path = Path(local_path)
    if not path.is_file():
        return None

    try:
        actual_hash = calculate_sha256(path.read_bytes())
    except OSError:
        return None
    if actual_hash != expected_hash:
        return None

    return {
        **previous,
        "title": previous.get("title", document["title"]),
        "local_path": str(path),
        "status": "CACHED",
    }


def mark_document_processed(url: str, sha256: str) -> None:
    """Record a hash only after its rules are safely activated."""
    metadata = load_metadata()
    for record in metadata:
        if record.get("url") == url and record.get("sha256") == sha256:
            record["processed_sha256"] = sha256
            record["last_processed"] = _checked_at()
            save_metadata(metadata)
            mark_registry_processed(url, sha256)
            return
    raise ValueError("Cannot mark an unknown document version as processed.")
