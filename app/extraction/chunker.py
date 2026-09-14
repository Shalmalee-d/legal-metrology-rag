from app.extraction.text_cleaner import clean_text


def chunk_pages(
    pages: list[dict],
    max_characters: int = 4000,
) -> list[dict]:
    """
    Convert page-level extracted text into
    manageable chunks while preserving source pages.
    """

    chunks = []

    current_text = ""
    current_pages = []

    for page in pages:

        page_number = page["page"]
        text = clean_text(page["text"])

        if not text:
            continue

        # If adding this page exceeds the limit,
        # save the current chunk first.
        if (
            current_text
            and len(current_text) + len(text) > max_characters
        ):
            chunks.append(
                {
                    "chunk_id": f"chunk_{len(chunks) + 1:04d}",
                    "text": current_text.strip(),
                    "source_pages": current_pages.copy(),
                }
            )

            current_text = ""
            current_pages = []

        current_text += text + "\n\n"
        current_pages.append(page_number)

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