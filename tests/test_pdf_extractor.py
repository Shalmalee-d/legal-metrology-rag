from pathlib import Path

from app.extraction.pdf_extractor import (
    extract_pdf_text,
    save_extracted_text,
)


def test_pdf_extraction():

    pdf_files = list(
        Path("data/documents").glob("*.pdf")
    )

    assert pdf_files, "No PDF files found."

    pdf_path = pdf_files[0]

    pages = extract_pdf_text(
        str(pdf_path)
    )

    assert pages, "No pages were extracted."

    for page in pages:
       assert "page" in page
       assert "text" in page
       assert page["text"].strip(), f"Page {page['page']} has no extracted text"

    output_path = save_extracted_text(
        str(pdf_path),
        pages,
    )

    assert output_path.exists()

    print()
    print(f"PDF tested: {pdf_path.name}")
    print(f"Pages extracted: {len(pages)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    test_pdf_extraction()