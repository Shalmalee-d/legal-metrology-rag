from app.extraction.rule_schema import ComplianceRule
from app.validation.rule_validator import validate_rule


def _semantic_rule(**updates):
    values = {
        "rule_id": "LM_PRODUCT_001",
        "parameter": "manufacturer_name_and_address",
        "condition": "must_exist",
        "applies_to": {"product_type": "packaged_commodity"},
        "requirement": "The package must declare the manufacturer name and address.",
        "source_document": "Packaged Commodities Rules, 2011",
        "source_pages": [25],
        "evidence_text": "Every package shall bear the name and address of the manufacturer.",
    }
    values.update(updates)
    return ComplianceRule(**values)


def test_covered_under_the_rules_counts_as_normative_evidence():
    rule = _semantic_rule(
        rule_id="LM_QTY_050",
        parameter="net_quantity",
        condition="must_be_less_than",
        requirement="Packages containing agricultural farm produce must not exceed a net quantity of 50 kg.",
        expected_value=50,
        expected_unit="kg",
        evidence_text="The packages of agriculture farm produce upto 50 kg are covered under the Legal Metrology (Packaged Commodities) Rules, 2011.",
    )
    assert validate_rule(rule) == []


def test_allowed_permission_counts_as_normative_evidence():
    rule = _semantic_rule(
        rule_id="LM_QTY_051",
        parameter="net_quantity",
        condition="must_exist",
        requirement="The package must declare the net quantity that manufacturers are allowed to pack.",
        evidence_text="Importers of these packages are allowed to pack in any quantity upto 50 kg as per rules.",
    )
    assert validate_rule(rule) == []


def test_covered_cross_reference_without_product_context_stays_rejected():
    rule = _semantic_rule(
        rule_id="LM_X",
        parameter="company_definition",
        condition="must_exist",
        requirement="The company must declare its incorporation status.",
        evidence_text="A foreign company covered under clause (42) of section 2 of the Companies Act, 2013.",
    )
    assert validate_rule(rule)


def test_bare_covered_without_regulatory_context_is_not_enough():
    from app.validation.rule_validator import _NORMATIVE_EVIDENCE

    assert not _NORMATIVE_EVIDENCE.search("The field was covered yesterday.")
    assert _NORMATIVE_EVIDENCE.search("Packages are covered under the Rules.")


def test_metadata_date_title_evidence_remains_rejected():
    for evidence in [
        "REGD. No. D. L.-33004/99",
        "These rules shall come into force on 1 January 2026.",
        "These rules may be called the Packaged Commodities Rules, 2025.",
    ]:
        rule = _semantic_rule(evidence_text=evidence)
        assert validate_rule(rule), evidence


def test_existing_shall_must_prohibited_evidence_still_passes():
    assert validate_rule(_semantic_rule()) == []
    assert validate_rule(_semantic_rule(evidence_text="Every package must declare the retail sale price.")) == []


def test_valid_rule():
    rule = ComplianceRule(
        rule_id="LM_001",
        parameter="mrp",
        condition="must_exist",
        applies_to={"product_type": "packaged_commodity"},
        requirement="MRP must be declared",
        source_document="Packaged Commodities Rules, 2011",
        source_pages=[25],
        evidence_text="MRP shall be declared.",
    )

    errors = validate_rule(rule)

    assert errors == []


def test_invalid_condition():
    rule = ComplianceRule(
        rule_id="LM_002",
        parameter="mrp",
        condition="maybe_required",
        source_document="Packaged Commodities Rules, 2011",
        source_pages=[25],
        evidence_text="MRP shall be declared.",
    )

    errors = validate_rule(rule)

    assert "Invalid condition: maybe_required" in errors


def test_invalid_page_number():
    rule = ComplianceRule(
        rule_id="LM_003",
        parameter="mrp",
        condition="must_exist",
        source_document="Packaged Commodities Rules, 2011",
        source_pages=[0],
        evidence_text="MRP shall be declared.",
    )

    errors = validate_rule(rule)

    assert "source_pages must contain positive page numbers." in errors


def test_comparison_rule_requires_explicit_expected_value():
    rule = ComplianceRule(
        rule_id="LM_004",
        parameter="quantity",
        condition="must_equal",
        source_document="Rules",
        source_pages=[1],
        evidence_text="Quantity shall be 500 g.",
    )

    assert "comparison and match rules require expected_value." in validate_rule(rule)


def test_rejects_document_metadata_and_definition_only_candidates():
    candidates = [
        _semantic_rule(
            parameter="rule_name", condition="must_equal",
            requirement="The rules must have this name.", expected_value="Rules, 2025",
            evidence_text="These rules may be called the Packaged Commodities Rules, 2025.",
        ),
        _semantic_rule(
            parameter="regulatory_document", requirement="The regulatory document must exist.",
            evidence_text="REGD. No. D. L.-33004/99",
        ),
        _semantic_rule(
            parameter="gazette_identifier", requirement="The Gazette identifier must be displayed.",
            evidence_text="The Gazette of India, Extraordinary, REGD. No. D. L.-33004/99",
        ),
        _semantic_rule(
            parameter="effective_date", requirement="The rules must come into force on this date.",
            evidence_text="These rules shall come into force on 1 January 2026.",
        ),
        _semantic_rule(
            parameter="package", requirement="A package must exist.",
            evidence_text="A package means a commodity placed in a wrapper.",
        ),
        _semantic_rule(
            parameter="group_package", requirement="A group package must exist.",
            evidence_text="Group package means a package containing two or more packages.",
        ),
        _semantic_rule(
            parameter="multi_piece_package", requirement="A multi-piece package must exist.",
            evidence_text="Multi-piece package means a package containing multiple pieces.",
        ),
    ]

    for rule in candidates:
        assert validate_rule(rule), rule.parameter


def test_accepts_concrete_product_facing_requirements():
    rules = [
        _semantic_rule(),
        _semantic_rule(
            parameter="mrp", requirement="The package must declare the retail sale price inclusive of taxes.",
            evidence_text="Every package shall declare the retail sale price inclusive of all taxes.",
        ),
        _semantic_rule(
            parameter="net_quantity", requirement="The package must declare the net quantity in grams.",
            expected_value="500", expected_unit="g",
            evidence_text="Every package shall declare the net quantity of 500 g.",
        ),
        _semantic_rule(
            parameter="letter_height", condition="must_be_greater_than", expected_value=2,
            expected_unit="mm", requirement="The package label must use letters at least 2 mm high.",
            evidence_text="The height of every letter on the package shall not be less than 2 mm.",
        ),
        _semantic_rule(
            parameter="declaration_exemption", condition="must_not_exist",
            applies_to={"package_type": "export_package"},
            requirement="Declaration requirements do not apply to an export package.",
            evidence_text="Nothing contained in these rules shall apply to a package containing a commodity meant for export.",
        ),
    ]

    for rule in rules:
        assert validate_rule(rule) == []


if __name__ == "__main__":
    test_valid_rule()
    test_invalid_condition()
    test_invalid_page_number()
