from app.extraction.rule_extractor import extract_rules_from_chunk


def test_extract_rules_from_chunk():
    chunk = {
        "chunk_id": "chunk_0001",
        "text": """
        Declarations to be made on every package.
        Every package shall bear the name and address of the manufacturer.
        The retail sale price of the package shall be declared.
        """,
        "source_pages": [42, 44],
    }

    rules = extract_rules_from_chunk(
        chunk,
        source_document="Legal Metrology (Packaged Commodities) Rules, 2011",
    )

    assert len(rules) == 2

    rule_ids = {rule.rule_id for rule in rules}

    assert "LM_MRP_001" in rule_ids
    assert "LM_MANUFACTURER_001" in rule_ids

    mrp_rule = next(
        rule for rule in rules
        if rule.rule_id == "LM_MRP_001"
    )

    manufacturer_rule = next(
        rule for rule in rules
        if rule.rule_id == "LM_MANUFACTURER_001"
    )

    assert mrp_rule.parameter == "mrp"
    assert mrp_rule.condition == "must_exist"

    assert manufacturer_rule.parameter == "manufacturer"
    assert manufacturer_rule.condition == "must_exist"


if __name__ == "__main__":
    test_extract_rules_from_chunk()