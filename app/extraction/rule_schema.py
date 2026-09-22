from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator


class ComplianceRule(BaseModel):
    """A traceable, service-friendly representation of one logical requirement."""

    model_config = ConfigDict(extra="forbid")
    rule_id: str
    parameter: str
    condition: str
    applies_to: dict[str, Any] | None = None
    requirement: Optional[str] = None
    expected_value: Any = None
    expected_unit: Optional[str] = None

    source_document: str
    source_pages: list[int]
    evidence_text: str

    effective_from: Optional[str] = None
    effective_until: Optional[str] = None
    status: str = Field(default="active")
    version: Optional[str] = None
    source_identity: Optional[str] = None
    # Stable discovered document URL.  This is intentionally separate from
    # the human-readable title so a title correction does not orphan rules.
    source_document_id: Optional[str] = None
    supersedes_source_identity: Optional[str] = None

    # Product-catalog overlay supplied by the model alongside the extraction
    # fields (category, title, applies_when, check_type, ...). Private: never
    # validated, serialized, or compared as part of the extraction schema.
    # The repository conversion reads it when building a ProductRule.
    _product_overlay: dict[str, Any] | None = PrivateAttr(default=None)

    @field_validator("rule_id", "parameter", "condition", "source_document", "evidence_text")
    @classmethod
    def required_text_cannot_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value.strip()

    @field_validator("requirement")
    @classmethod
    def requirement_if_present_must_be_meaningful(cls, value: Optional[str]) -> Optional[str]:
        # Transport stays Optional so legacy snapshots still load and are then
        # filtered by deterministic validation. New producers must supply a
        # non-blank actionable requirement; None is rejected by validate_rule().
        if value is None:
            return None
        import re as _re

        collapsed = _re.sub(r"\s+", " ", value).strip()
        if not collapsed:
            raise ValueError("requirement cannot be blank")
        return collapsed
