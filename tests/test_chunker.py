from pathlib import Path
import json

from app.extraction.chunker import chunk_pages


def test_chunking():

    json_files = list(
        Path("data/extracted").glob("*.json")
    )

    assert json_files, "No extracted JSON files found."

    with open(
        json_files[0],
        "r",
        encoding="utf-8",
    ) as file:

        pages = json.load(file)

    chunks = chunk_pages(pages)

    assert chunks, "No chunks were created."

    for chunk in chunks:

        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "source_pages" in chunk

        assert chunk["text"]
        assert chunk["source_pages"]

    print()
    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")

    for chunk in chunks[:3]:

        print()
        print(
            f"{chunk['chunk_id']} "
            f"(pages {chunk['source_pages']})"
        )

        print(chunk["text"][:500])


if __name__ == "__main__":
    test_chunking()