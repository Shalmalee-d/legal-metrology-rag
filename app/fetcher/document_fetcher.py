import hashlib
import json
from pathlib import Path
from urllib.parse import urljoin

import httpx
import truststore
from bs4 import BeautifulSoup


# Use Windows' trusted certificates
truststore.inject_into_ssl()


DATA_DIR = Path("data/documents")
METADATA_FILE = DATA_DIR / "metadata.json"


def calculate_sha256(data: bytes) -> str:
    """Calculate SHA-256 hash of downloaded data."""

    return hashlib.sha256(data).hexdigest()


def load_metadata() -> list:
    """Load metadata from previous runs."""

    if not METADATA_FILE.exists():
        return []

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_metadata(metadata: list) -> None:
    """Save document metadata."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


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

    for link in soup.find_all("a", href=True):

        title = link.get_text(" ", strip=True)
        href = link["href"]

        title_lower = title.lower()

        if (
            "packaged commodities" in title_lower
            and (
                "rule" in title_lower
                or "amendment" in title_lower
                or "corrigendum" in title_lower
                or "advisory" in title_lower
                or "guideline" in title_lower
            )
        ):

            url = urljoin(source_url, href)

            documents.append(
                {
                    "title": title,
                    "url": url,
                }
            )

    return documents


def download_document(document: dict) -> dict:
    """
    Download one document.

    The document is downloaded into memory first.
    Its hash is calculated before replacing any existing file.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata()

    previous = next(
        (
            item
            for item in metadata
            if item["url"] == document["url"]
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

    file_path = DATA_DIR / f"{safe_name}.pdf"

    timeout = httpx.Timeout(
        connect=10.0,
        read=20.0,
        write=20.0,
        pool=20.0,
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

    last_error = None

    for url in urls_to_try:

        try:

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

            # Nothing changed
            if previous and previous["sha256"] == file_hash:

                return {
                    **previous,
                    "status": "UNCHANGED",
                }

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
                "status": status,
            }

            # Replace previous metadata for this URL
            metadata = [
                item
                for item in metadata
                if item["url"] != document["url"]
            ]

            metadata.append(record)

            save_metadata(metadata)

            return record

        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"Download failed after trying all URLs: {last_error}"
    )