"""Quality regression: small high-quality rule set over many weak rules."""

from app.extraction.rule_schema import ComplianceRule
from app.validation.rule_validator import validate_rule


def test_qwen_prompt_requires_actionable_evidence_backed_json_only():
    from app.extraction.llm_rule_generator import _prompt

    text = _prompt({"text": "sample", "chunk_id": "c", "source_pages": [1]}).lower()
    for required in [
        "only",
        "actionable",
        "packaged commodit",
        "do not invent",
        "exact",
        "evidence_text",
        "return []",
        "fewer precise",
        "json only",
        "amendment",
        "ignore",
        "gazette",
        "commencement",
    ]:
        assert required in text, required


def _base(**updates):
    values = {
        "rule_id": "LM_TEST",
        "parameter": "mrp",
        "condition": "must_exist",
        "applies_to": {"product_type": "packaged_commodity"},
        "requirement": "The package must declare the retail sale price.",
        "source_document": "Packaged Commodities Rules, 2011",
        "source_pages": [1],
        "evidence_text": "Every package shall declare the retail sale price.",
    }
    values.update(updates)
    return ComplianceRule(**values)


def test_rejects_metadata_only_rule():
    rule = _base(
        parameter="rule_name",
        condition="must_equal",
        requirement="The rules must have this name.",
        expected_value="Packaged Commodities Rules, 2025",
        evidence_text="These rules may be called the Packaged Commodities Rules, 2025.",
    )
    assert validate_rule(rule)


def test_rejects_gazette_information():
    rule = _base(
        parameter="gazette_identifier",
        requirement="The Gazette identifier must be displayed.",
        evidence_text="The Gazette of India, Extraordinary, REGD. No. D. L.-33004/99",
    )
    assert validate_rule(rule)


def test_rejects_commencement_effective_date_only_rule():
    rule = _base(
        parameter="effective_date",
        requirement="The rules must come into force on this date.",
        evidence_text="These rules shall come into force on 1 January 2026.",
    )
    assert validate_rule(rule)


def test_rejects_document_must_exist():
    rule = _base(
        parameter="regulatory_document",
        requirement="The regulatory document must exist.",
        evidence_text="REGD. No. D. L.-33004/99",
    )
    assert validate_rule(rule)


def test_rejects_package_must_exist():
    rule = _base(
        parameter="package",
        requirement="A package must exist.",
        evidence_text="A package means a commodity placed in a wrapper.",
    )
    assert validate_rule(rule)


def test_rejects_generic_definition():
    rule = _base(
        parameter="group_package",
        requirement="A group package must exist.",
        evidence_text="Group package means a package containing two or more packages.",
    )
    assert validate_rule(rule)


def test_rejects_vague_non_actionable_requirement():
    rule = _base(
        parameter="package",
        requirement="The package must comply with the rules.",
        evidence_text="Every package shall comply with these rules in all respects.",
    )
    # "comply" alone is not an observable declaration/measurement check.
    assert validate_rule(rule)


def test_rejects_missing_evidence():
    import pytest

    # Blank evidence is rejected at the schema boundary.
    with pytest.raises(Exception, match="blank"):
        _base(evidence_text="   ")
    # Whitespace-only evidence constructed bypassing validation is still
    # rejected by deterministic validation.
    raw = _base()
    raw = raw.model_construct(
        rule_id=raw.rule_id,
        parameter=raw.parameter,
        condition=raw.condition,
        applies_to=raw.applies_to,
        requirement=raw.requirement,
        source_document=raw.source_document,
        source_pages=raw.source_pages,
        evidence_text="   ",
        status=raw.status,
    )
    assert validate_rule(raw)


def test_rejects_missing_meaningful_requirement():
    raw = _base()
    raw = raw.model_copy(update={"requirement": None})
    assert validate_rule(raw)
    raw2 = _base()
    raw2 = raw2.model_copy(update={"requirement": "  "})
    # Blank requirement is rejected by schema; validator also rejects None.
    assert validate_rule(raw)


def test_rejects_unrelated_evidence():
    rule = _base(
        parameter="mrp",
        requirement="The package must declare the retail sale price.",
        evidence_text="The height of letters on the package shall not be less than 2 mm.",
    )
    # Both mention package, but no shared content tokens for price/declaration.
    # Parameter mrp is absent from evidence, requirement tokens do not overlap.
    assert validate_rule(rule)


def test_rejects_amendment_metadata_without_product_effect():
    rule = _base(
        parameter="amendment",
        requirement="Rule 6 was amended on 1 January 2022.",
        evidence_text="In rule 6, for sub-rule (6), the amendment shall come into force at once.",
    )
    assert validate_rule(rule)


def test_accepts_retail_sale_price_declaration():
    rule = _base(
        rule_id="LM_MRP_001",
        parameter="mrp",
        requirement="The package must declare the retail sale price inclusive of all taxes.",
        evidence_text="Every package shall declare the retail sale price inclusive of all taxes.",
    )
    assert validate_rule(rule) == []


def test_accepts_net_quantity_declaration():
    rule = _base(
        rule_id="LM_QTY_001",
        parameter="net_quantity",
        condition="must_exist",
        requirement="The package must declare the net quantity in grams.",
        evidence_text="Every package shall declare the net quantity of 500 g.",
    )
    assert validate_rule(rule) == []


def test_accepts_package_label_mandatory_declaration():
    rule = _base(
        rule_id="LM_MAN_001",
        parameter="manufacturer",
        requirement="The package must declare the name and address of the manufacturer.",
        evidence_text="Every package shall bear the name and address of the manufacturer.",
    )
    assert validate_rule(rule) == []


def test_accepts_measurable_letter_height_requirement():
    rule = _base(
        rule_id="LM_DIM_001",
        parameter="letter_height",
        condition="must_be_greater_than",
        expected_value=2,
        expected_unit="mm",
        requirement="The package label must use letters at least 2 mm high.",
        evidence_text="The height of every letter on the package shall not be less than 2 mm.",
    )
    assert validate_rule(rule) == []


def test_accepts_importer_declaration_for_imported_package():
    rule = _base(
        rule_id="LM_IMP_001",
        parameter="importer",
        requirement="An imported package must declare the name and address of the importer.",
        evidence_text="Every imported package shall declare the name and address of the importer.",
    )
    assert validate_rule(rule) == []
