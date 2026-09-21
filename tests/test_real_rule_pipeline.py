import json
from pathlib import Path

from app.extraction.chunker import chunk_pages
from app.extraction.rule_pipeline import process_chunks


def test_real_rule_pipeline():
    json_path = Path(
        "data/extracted/Download_The_Legal_Metrology__Packaged_Commodities__Rules__2011.json"
    )
    assert json_path.exists(), "The base Legal Metrology extracted fixture is required."

    with open(json_path, "r", encoding="utf-8") as file:
        pages = json.load(file)

    chunks = chunk_pages(pages)

    assert chunks, "No chunks were created."

    all_rules = process_chunks(
        chunks,
        source_document=json_path.stem,
    )

    print()
    print(f"Document: {json_path.name}")
    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Unique rules extracted: {len(all_rules)}")

    for rule in all_rules:
        print()
        print(f"Rule ID: {rule.rule_id}")
        print(f"Parameter: {rule.parameter}")
        print(f"Condition: {rule.condition}")
        print(f"Pages: {rule.source_pages}")

    assert all_rules, "No valid rules were extracted."

    rule_ids = [rule.rule_id for rule in all_rules]

    assert len(rule_ids) == len(set(rule_ids)), (
        "Duplicate rule IDs were found."
    )


if __name__ == "__main__":
    test_real_rule_pipeline()
