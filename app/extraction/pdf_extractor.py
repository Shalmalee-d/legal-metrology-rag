import json
from pathlib import Path

import pymupdf


EXTRACTED_DIR = Path("data/extracted")


def extract_pdf_text(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF while preserving page numbers.
    """

    pdf_path = Path(pdf_path)

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text").strip()

        pages.append(
            {
                "page": page_number,
                "text": text,
            }
        )

    document.close()

    return pages


def save_extracted_text(
    pdf_path: str,
    pages: list[dict],
) -> Path:
    """
    Save extracted page-level text as JSON.
    """

    EXTRACTED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_name = Path(pdf_path).stem

    output_path = EXTRACTED_DIR / f"{pdf_name}.json"

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            pages,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path