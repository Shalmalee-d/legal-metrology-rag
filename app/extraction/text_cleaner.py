import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving
    the actual regulatory wording.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Fix spaces before punctuation
    text = re.sub(r"\s+([,.;:])", r"\1", text)

    return text.strip()