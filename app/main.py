from app.fetcher.document_fetcher import (
    discover_documents,
    download_document,
)
from app.sources.source_registry import OFFICIAL_SOURCES


def main():
    source = OFFICIAL_SOURCES[0]

    print(f"Checking: {source['name']}")
    print(f"URL: {source['url']}")
    print()

    documents = discover_documents(source["url"])

    print(f"Documents discovered: {len(documents)}")
    print()

    # Test only the first 3 documents for now
    test_documents = documents[:3]

    print(f"Testing download of {len(test_documents)} documents...")
    print()

    for document in test_documents:

        print(f"Processing: {document['title']}")

        try:
            result = download_document(document)

            print(f"Status: {result['status']}")
            print(f"Saved: {result['local_path']}")
            print(f"SHA-256: {result['sha256']}")

        except Exception as error:
            print(f"SKIPPED: {error}")

        print("-" * 80)


if __name__ == "__main__":
    main()