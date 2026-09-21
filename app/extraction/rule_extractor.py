from app.extraction.rule_schema import ComplianceRule


def _evidence_for(text: str, phrase: str) -> str:
    """Keep an actual source sentence whenever the page text provides one."""
    start = text.lower().find(phrase.lower())
    if start < 0:
        return phrase
    sentence_start = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    sentence_end = text.find(".", start)
    return text[sentence_start: sentence_end + 1 if sentence_end >= 0 else len(text)].strip()


def extract_rules_from_chunk(
    chunk: dict,
    source_document: str,
    diagnostics: dict | None = None,
) -> list[ComplianceRule]:
    """
    Extract basic compliance rules from a regulatory text chunk.

    This first version uses rule patterns for clearly identifiable
    mandatory declarations. LLM-based extraction can be added later.
    """

    text = chunk["text"]
    source_pages = chunk["source_pages"]

    rules = []

    text_lower = text.lower()

    if "retail sale price" in text_lower:
        rules.append(
            ComplianceRule(
                rule_id="LM_MRP_001",
                parameter="mrp",
                condition="must_exist",
                applies_to={"product_type": "packaged_commodity"},
                requirement="Retail sale price must be declared on the package.",
                source_document=source_document,
                source_pages=source_pages,
                evidence_text=_evidence_for(text, "retail sale price"),
            )
        )

    if "name and address of the manufacturer" in text_lower:
        rules.append(
            ComplianceRule(
                rule_id="LM_MANUFACTURER_001",
                parameter="manufacturer",
                condition="must_exist",
                applies_to={"product_type": "packaged_commodity"},
                requirement="Name and address of the manufacturer must be declared.",
                source_document=source_document,
                source_pages=source_pages,
                evidence_text=_evidence_for(text, "name and address of the manufacturer"),
            )
        )

    if "dimensions of the commodity" in text_lower:
        rules.append(
            ComplianceRule(
                rule_id="LM_DIMENSIONS_001",
                parameter="dimensions",
                condition="must_exist",
                applies_to={"product_type": "packaged_commodity"},
                requirement="Dimensions must be declared where relevant.",
                source_document=source_document,
                source_pages=source_pages,
                evidence_text=_evidence_for(text, "dimensions of the commodity"),
            )
        )

    if diagnostics is not None:
        diagnostics.update(proposed=len(rules), empty_response=not rules,
                           validation_rejected=0, rejection_reasons=[])

    return rules
