from datetime import date
import re

from app.extraction.rule_schema import ComplianceRule


VALID_CONDITIONS = {
    "must_exist",
    "must_not_exist",
    "must_equal",
    "must_be_greater_than",
    "must_be_less_than",
    "must_match",
}


# These patterns assess the role a passage plays rather than enumerating
# disallowed parameter names. They deliberately cover both the English and
# Hindi terminology found in the official source material.
_PRODUCT_CONTEXT = re.compile(
    r"\b(package|packaged|commodity|product|label|declaration|mrp|retail sale "
    r"price|net quantity|quantity|manufacturer|packer|importer|consumer|"
    r"dimension|numeral|letter|measurement|weight|volume|price)\b|"
    r"पैकेज|पैक|वस्तु|लेबल|मात्रा|कीमत|मूल्य|निर्माता|पैकर|आयातक|उपभोक्ता",
    re.IGNORECASE,
)
_NORMATIVE_EVIDENCE = re.compile(
    r"\b(shall|must|required|prohibited|shall not|must not|may not|"
    r"not apply|does not apply|exempt(?:ed|ion)?|covered\s+under\s+the|allow(?:ed|s)?|only)\b|"
    r"होगा|होंगे|चाहिए|लागू नहीं|अपेक्षित|निषिद्ध",
    re.IGNORECASE,
)
_ACTIONABLE_REQUIREMENT = re.compile(
    r"\b(declar\w*|bear|display\w*|mark\w*|label\w*|print\w*|contain\w*|"
    r"provid\w*|stat\w*|mention\w*|measur\w*|weigh\w*|price|quantity|"
    r"dimension\w*|apply|exempt\w*|prohibit\w*|"
    r"must not|shall not|not apply)\b|"
    r"घोषित|अंकित|उल्लेख|प्रदर्शित|लागू नहीं|अपेक्षित|निषिद्ध",
    re.IGNORECASE,
)
_EXEMPTION_EVIDENCE = re.compile(
    r"\b(not apply|does not apply|exempt(?:ed|ion)?|nothing contained)\b|"
    r"लागू नहीं|छूट",
    re.IGNORECASE,
)

_STOPWORDS = frozenset(
    {
        "the", "and", "are", "was", "were", "been", "have", "has", "had",
        "will", "would", "should", "could", "shall", "must", "every", "each",
        "such", "than", "then", "them", "they", "this", "that", "with",
        "from", "into", "upon", "under", "over", "after", "before",
        "between", "through", "during", "about", "also", "only", "very",
        "can", "may", "might", "shall", "being", "does", "doing", "done",
    }
)


def _content_tokens(text: str) -> set[str]:
    parts = re.findall(r"[a-z0-9]+|[\u0900-\u097f]+", text.lower())
    return {p for p in parts if len(p) >= 3 and p not in _STOPWORDS}


def _contains_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097f]", text))


def _evidence_supports_requirement(requirement: str, evidence: str, parameter: str) -> bool:
    # Bilingual regulatory material: Hindi evidence with an English requirement
    # (or vice versa) cannot share tokens. Product-context + normative checks
    # already gate those cases; do not force cross-script token overlap.
    if _contains_devanagari(requirement) != _contains_devanagari(evidence):
        return True
    req_tokens = _content_tokens(requirement)
    ev_tokens = _content_tokens(evidence)
    shared = req_tokens & ev_tokens
    # Two shared content words indicate the same requirement; a single generic
    # word like "package" alone does not prove the evidence supports this rule.
    if len(shared) >= 2:
        return True
    # Short but decisive parameters (mrp, g, mm) may not survive token
    # filtering; a direct parameter mention in evidence still grounds the rule.
    param_norm = parameter.replace("_", " ").lower()
    for part in re.findall(r"[a-z0-9]+|[\u0900-\u097f]+", param_norm):
        if len(part) >= 2 and part not in _STOPWORDS and part in evidence.lower():
            return True
    return False


def _semantic_errors(rule: ComplianceRule) -> list[str]:
    """Reject grounded text that is not a product-facing compliance check."""
    errors: list[str] = []
    requirement = (rule.requirement or "").strip()
    evidence = rule.evidence_text.strip()
    parameter = (rule.parameter or "").strip()
    is_exemption = bool(_EXEMPTION_EVIDENCE.search(evidence))

    if not requirement:
        errors.append("rule must include a meaningful product-facing requirement.")
        return errors
    if len(requirement) < 12:
        errors.append("requirement must be a complete product-facing statement, not a fragment.")
    if len(evidence) < 12:
        errors.append("evidence_text must quote a complete supporting passage.")
    if not _ACTIONABLE_REQUIREMENT.search(requirement):
        errors.append("requirement must describe an actionable package or product check.")
    # The requirement itself (or its parameter) must name what the Rule Engine
    # checks on the package. Evidence context alone is not enough: it would
    # otherwise admit vague "must comply / must exist" rules whose evidence
    # merely mentions a package elsewhere.
    param_spaced = parameter.replace("_", " ")
    if not _PRODUCT_CONTEXT.search(requirement) and not _PRODUCT_CONTEXT.search(param_spaced):
        errors.append("requirement/parameter must identify a packaged-commodity check (package, label, declaration, quantity, price, dimensions, etc.).")

    if not _NORMATIVE_EVIDENCE.search(evidence):
        errors.append("evidence_text must support a normative requirement or exemption.")

    if not _PRODUCT_CONTEXT.search(evidence):
        errors.append("evidence_text must identify a product, package, label, or commodity context.")

    # Evidence must actually support this requirement, not just be any
    # product-related sentence from the same chunk.
    if requirement and evidence and not _evidence_supports_requirement(requirement, evidence, parameter):
        errors.append("evidence_text must directly support the stated requirement.")

    if is_exemption:
        if not isinstance(rule.applies_to, dict) or not rule.applies_to:
            errors.append("an exemption rule requires a structured applicability condition.")
    elif rule.condition == "must_exist" and not _ACTIONABLE_REQUIREMENT.search(requirement):
        errors.append("must_exist requires a concrete declaration or product-facing action.")

    return errors


def validate_rule(rule: ComplianceRule) -> list[str]:
    errors = []

    if not rule.rule_id.strip():
        errors.append("rule_id cannot be empty.")

    if not rule.parameter.strip():
        errors.append("parameter cannot be empty.")

    if rule.condition not in VALID_CONDITIONS:
        errors.append(
            f"Invalid condition: {rule.condition}"
        )

    if rule.condition in {
        "must_equal",
        "must_be_greater_than",
        "must_be_less_than",
        "must_match",
    } and rule.expected_value is None:
        errors.append("comparison and match rules require expected_value.")

    if not rule.source_document.strip():
        errors.append("source_document cannot be empty.")

    if not rule.source_pages:
        errors.append("source_pages cannot be empty.")

    if any(page <= 0 for page in rule.source_pages):
        errors.append("source_pages must contain positive page numbers.")

    if not rule.evidence_text.strip():
        errors.append("evidence_text cannot be empty.")

    if rule.status not in {"active", "inactive", "superseded"}:
        errors.append(
            f"Invalid status: {rule.status}"
        )

    for field_name in ("effective_from", "effective_until"):
        value = getattr(rule, field_name)
        if value is not None:
            try:
                date.fromisoformat(value)
            except ValueError:
                errors.append(f"{field_name} must use ISO YYYY-MM-DD format.")

    if rule.effective_from and rule.effective_until:
        if rule.effective_from > rule.effective_until:
            errors.append("effective_from must not be after effective_until.")

    if not errors:
        errors.extend(_semantic_errors(rule))

    return errors
