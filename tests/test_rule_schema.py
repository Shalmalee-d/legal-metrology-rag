from app.extraction.rule_schema import ComplianceRule


def test_compliance_rule_schema():
    rule = ComplianceRule(
        rule_id="LM_001",
        parameter="mrp",
        condition="must_exist",
        applies_to={"product_type": "packaged_commodity"},
        requirement="MRP must be declared",
        source_document="Packaged Commodities Rules, 2011",
        source_pages=[25],
        evidence_text="Maximum Retail Price shall be declared.",
    )

    assert rule.rule_id == "LM_001"
    assert rule.parameter == "mrp"
    assert rule.condition == "must_exist"
    assert rule.source_pages == [25]
    assert rule.evidence_text


if __name__ == "__main__":
    test_compliance_rule_schema()
