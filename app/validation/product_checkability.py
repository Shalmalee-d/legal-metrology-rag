"""Product-checkability layer: is this rule checkable from package information?

Runs AFTER evidence grounding, schema validation, and the normative
validator. It answers whether a downstream product scanner could evaluate
the rule against product/package/label data, rejecting enforcement
procedures, methodologies, histories, and other non-observable material
that may still carry normative wording.
"""

import re

_COMPARISON_CONDITIONS = frozenset({
    "must_equal",
    "must_be_greater_than",
    "must_be_less_than",
    "must_match",
    "must_not_exceed",
    "must_be_at_least",
})

# Something observable on the package, label, or product itself.
_PACKAGE_ANCHOR = re.compile(
    r"\b(package|packaged|commodity|product|label|declaration|marking|mrp|"
    r"retail sale price|net quantity|quantity|manufacturer|packer|importer|"
    r"consumer|care|contact|helpline|address|dimension|letter|measurement|"
    r"weight|volume|price)\b|"
    r"पैकेज|पैक|वस्तु|लेबल|मात्रा|कीमत|मूल्य|निर्माता|पैकर|आयातक|उपभोक्ता",
    re.IGNORECASE,
)

# Verbs stating an observable package/label state or declaration. "Weigh"
# covers weighed/weighing/weighs but not the noun "weight", so a tare-weight
# calculation procedure does not qualify as a declaration.
_DECLARATIVE_VERB = re.compile(
    r"\b(declar\w*|bear|display\w*|mark\w*|label\w*|print\w*|stat\w*|mention\w*|"
    r"specif\w*|indicat\w*|show\w*|appear\w*|contain\w*|provid\w*|describ\w*|"
    r"measur\w*|weigh(?:ed|ing|s)?|apply|exempt\w*)\b|"
    r"घोषित|अंकित|उल्लेख|प्रदर्शित",
    re.IGNORECASE,
)

_UNIT_ALIASES = {
    "kg": ("kg", "kilogram", "kilograms"),
    "g": ("g", "gram", "grams", "gm", "gms"),
    "mg": ("mg", "milligram", "milligrams"),
    "ml": ("ml", "millilitre", "millilitres", "milliliter", "milliliters"),
    "l": ("l", "litre", "litres", "liter", "liters"),
    "mm": ("mm", "millimetre", "millimetres", "millimeter", "millimeters"),
    "cm": ("cm", "centimetre", "centimetres", "centimeter", "centimeters"),
    "m": ("m", "metre", "metres", "meter", "meters"),
}


def _numbers(text: str) -> list[float]:
    values = []
    for token in re.findall(r"\d+(?:\.\d+)?", text.replace(",", "")):
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def check_numeric_grounding(rule) -> list[str]:
    """Require comparison thresholds to come from the evidence itself."""
    errors: list[str] = []
    if getattr(rule, "condition", None) not in _COMPARISON_CONDITIONS:
        return errors
    value = getattr(rule, "expected_value", None)
    unit = getattr(rule, "expected_unit", None)
    evidence = getattr(rule, "evidence_text", "") or ""
    try:
        numeric = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return ["expected_value must be numeric for a comparison rule."]
    if unit is not None:
        unit_text = str(unit).strip()
        if not re.search(r"[A-Za-z\u0900-\u097f]", unit_text):
            errors.append("expected_unit must name a real unit, not a bare number.")
        else:
            lowered = evidence.lower()
            normalized = unit_text.lower().rstrip(".")
            variants = {normalized}
            for key, aliases in _UNIT_ALIASES.items():
                forms = (key,) + aliases
                if normalized in forms or normalized.rstrip("s") in forms:
                    variants = set(forms)
                    break
            if not any(variant in lowered for variant in variants):
                errors.append("evidence must contain the same unit as the comparison rule.")
    if not any(abs(number - numeric) < 1e-9 for number in _numbers(evidence)):
        errors.append("evidence must contain the same numeric threshold as the comparison rule.")
    return errors


def check_product_checkable(rule) -> list[str]:
    """Require a package-observable declaration, measurement, or threshold."""
    errors: list[str] = []
    requirement = (getattr(rule, "requirement", None) or "").strip()
    parameter = (getattr(rule, "parameter", None) or "").strip()
    if not _PACKAGE_ANCHOR.search(requirement) and not _PACKAGE_ANCHOR.search(
        parameter.replace("_", " ")
    ):
        errors.append(
            "requirement/parameter must describe something checkable on the package or label."
        )
        return errors
    condition = getattr(rule, "condition", None)
    expected_value = getattr(rule, "expected_value", None)
    if (
        condition in _COMPARISON_CONDITIONS
        and expected_value is not None
        and (_PACKAGE_ANCHOR.search(requirement) or _PACKAGE_ANCHOR.search(parameter.replace("_", " ")))
    ):
        return errors
    if not _DECLARATIVE_VERB.search(requirement):
        errors.append(
            "requirement must state an observable package/label declaration, marking, or measurement."
        )
    return errors


def validate_product_rule(rule) -> list[str]:
    """Combined product-checkability gate for fully validated candidates."""
    return check_product_checkable(rule) + check_numeric_grounding(rule)
