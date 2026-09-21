import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving
    the actual regulatory wording.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Join only unambiguous print-layout hyphenation; do not rewrite wording.
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Fix only unambiguous spacing artefacts.
    text = re.sub(r"\s+([,.;:])", r"\1", text)

    return text.strip()
