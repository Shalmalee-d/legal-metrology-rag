import json
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image


EXTRACTED_DIR = Path("data/extracted")

# Windows Tesseract installation
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_pdf_text(pdf_path: str) -> list[dict]:
    pdf_path = Path(pdf_path)
    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        # First try normal PDF text extraction
        text = page.get_text("text").strip()

        # If no text exists, use OCR
        if not text:
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
            image = Image.frombytes(
                "RGB",
                [pixmap.width, pixmap.height],
                pixmap.samples,
            )

            text = pytesseract.image_to_string(image).strip()

        pages.append({
            "page": page_number,
            "text": text,
        })

    document.close()

    return pages


def save_extracted_text(pdf_path: str, pages: list[dict]) -> Path:
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_name = Path(pdf_path).stem
    output_path = EXTRACTED_DIR / f"{pdf_name}.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            pages,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path