from app.config import get_rag_chunk_size
from app.extraction.text_cleaner import clean_text


def chunk_pages(
    pages: list[dict],
    max_characters: int | None = None,
) -> list[dict]:
    """
    Convert page-level extracted text into
    manageable chunks while preserving source pages.
    """

    max_characters = max_characters or get_rag_chunk_size()
    if max_characters <= 0:
        raise ValueError("max_characters must be greater than zero.")
    chunks = []

    current_text = ""
    current_pages = []

    for page in pages:

        page_number = page["page"]
        text = clean_text(page["text"])

        if not text:
            continue

        # Split an unusually dense page at a natural boundary.  Previously a
        # single page larger than the configured limit was passed to Ollama
        # whole, making the limit ineffective for real regulatory PDFs.
        for part in _split_text(text, max_characters):
            if current_text and len(current_text) + len(part) > max_characters:
                chunks.append({
                    "chunk_id": f"chunk_{len(chunks) + 1:04d}",
                    "text": current_text.strip(),
                    "source_pages": current_pages.copy(),
                })
                current_text = ""
                current_pages = []

            current_text += part + "\n\n"
            if page_number not in current_pages:
                current_pages.append(page_number)

            # A split part fills a request on its own; emit it immediately so
            # the next part cannot exceed the configured model input budget.
            if len(current_text.strip()) >= max_characters:
                chunks.append({
                    "chunk_id": f"chunk_{len(chunks) + 1:04d}",
                    "text": current_text.strip(),
                    "source_pages": current_pages.copy(),
                })
                current_text = ""
                current_pages = []

    # Save final chunk
    if current_text.strip():

        chunks.append(
            {
                "chunk_id": f"chunk_{len(chunks) + 1:04d}",
                "text": current_text.strip(),
                "source_pages": current_pages.copy(),
            }
        )

    return chunks


def _split_text(text: str, max_characters: int) -> list[str]:
    """Return complete, bounded spans without discarding regulatory wording."""
    parts: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_characters:
        window = remaining[:max_characters]
        # Prefer paragraph, line, then sentence/word boundaries in the latter
        # half of the window so the model retains enough legal context.
        candidates = (
            window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "),
            window.rfind("; "), window.rfind(" "),
        )
        split_at = max(candidates)
        if split_at < max_characters // 2:
            split_at = max_characters
        else:
            split_at += 1
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        parts.append(remaining)
    return parts
