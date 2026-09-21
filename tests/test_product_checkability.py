"""Product-checkability layer: package-observable rules pass, procedures fail."""

from app.extraction.rule_schema import ComplianceRule
from app.validation.product_checkability import (
    check_numeric_grounding,
    check_product_checkable,
    validate_product_rule,
)


def _rule(**updates):
    values = {
        "rule_id": "LM_T",
        "parameter": "mrp",
        "condition": "must_exist",
        "requirement": "The package must declare the retail sale price.",
        "source_document": "Rules",
        "source_pages": [1],
        "evidence_text": "Every package shall declare the retail sale price.",
    }
    values.update(updates)
    return ComplianceRule(**values)


def test_accepts_mrp_declaration():
    assert validate_product_rule(_rule()) == []


def test_accepts_manufacturer_declaration():
    assert validate_product_rule(_rule(
        parameter="manufacturer",
        requirement="The package must declare the name and address of the manufacturer.",
        evidence_text="Every package shall bear the name and address of the manufacturer.",
    )) == []


def test_accepts_consumer_care_declaration():
    assert validate_product_rule(_rule(
        parameter="consumer_care",
        requirement="The package must declare the name, address and Customer Care Number.",
        evidence_text="Every package shall declare the name, address and Customer Care Number.",
    )) == []


def test_accepts_net_quantity_threshold():
    assert validate_product_rule(_rule(
        parameter="net_quantity", condition="must_be_less_than",
        requirement="Packages must not exceed a net quantity of 50 kg.",
        evidence_text="Every package shall not exceed a net quantity of 50 kg.",
        expected_value=50, expected_unit="kg",
    )) == []


def test_accepts_qualitative_exemption_with_applies_to():
    assert validate_product_rule(_rule(
        parameter="export_exemption", condition="must_not_exist",
        requirement="Declaration requirements do not apply to an export package.",
        evidence_text="Nothing contained in these rules shall apply to an export package.",
        applies_to={"package_type": "export_package"},
    )) == []


def test_rejects_officer_powers():
    assert check_product_checkable(_rule(
        parameter="enforcement", condition="must_exist",
        requirement="Officers may seize non-compliant packages during inspection.",
        evidence_text="The officer may seize any package that contravenes the rules.",
    ))


def test_rejects_seizure_and_penalties():
    for requirement in [
        "Any package contravening these provisions shall be liable to seizure and penalty.",
        "The offender shall be punished with a penalty for the violation.",
    ]:
        assert check_product_checkable(_rule(
            parameter="penalty", requirement=requirement,
            evidence_text="Any contravention shall attract seizure and penalty.",
        )), requirement


def test_rejects_sampling_and_lab_procedures():
    assert check_product_checkable(_rule(
        parameter="sampling", condition="must_exist",
        requirement="Samples shall be drawn from each lot for laboratory testing.",
        evidence_text="Samples shall be drawn and tested in the laboratory.",
    ))


def test_rejects_tare_weight_calculation_procedure():
    assert check_product_checkable(_rule(
        parameter="tare_weight", condition="must_exist",
        requirement="The net weight shall be obtained by subtracting the tare weight.",
        evidence_text="The net weight shall be obtained by subtracting the tare weight.",
    ))


def test_rejects_amendment_history_and_definitions():
    assert check_product_checkable(_rule(
        parameter="amendment", requirement="Rule 6 was amended in 2017.",
        evidence_text="Rule 6 was amended in 2017.",
    ))
    assert check_product_checkable(_rule(
        parameter="group_package",
        requirement="A group package means two or more packages.",
        evidence_text="Group package means two or more packages.",
    ))


def test_rejects_bad_numeric_threshold_without_evidence_numbers():
    rule = _rule(
        parameter="net_quantity", condition="must_be_greater_than",
        requirement="Packages must have a net quantity greater than 35 units.",
        evidence_text="The equipment table lists the testing apparatus dimensions.",
        expected_value=35, expected_unit="35",
    )
    errors = check_numeric_grounding(rule)
    assert any("unit" in error for error in errors)
    assert any("threshold" in error for error in errors)
    assert validate_product_rule(rule)


def test_rejects_mismatched_numeric_threshold():
    rule = _rule(
        parameter="net_quantity", condition="must_be_less_than",
        requirement="Packages must not exceed a net quantity of 50 kilograms.",
        evidence_text="Packages containing more than 25 kilogram are excluded.",
        expected_value=50, expected_unit="kilograms",
    )
    assert any("threshold" in error for error in check_numeric_grounding(rule))


def test_accepts_matching_threshold_with_unit_alias():
    rule = _rule(
        parameter="net_quantity", condition="must_be_less_than",
        requirement="Packages must not exceed a net quantity of 50 kilograms.",
        evidence_text="Packages shall not exceed 50 kilogram under the Rules.",
        expected_value=50, expected_unit="kilograms",
    )
    assert check_numeric_grounding(rule) == []
