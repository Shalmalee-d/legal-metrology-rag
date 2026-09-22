from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProductSource(BaseModel):
    """One cited source for a finalized product rule."""

    model_config = ConfigDict(extra="ignore")
    document: str
    rule_or_section: Optional[str] = None
    page: Optional[int] = None

    @field_validator("document")
    @classmethod
    def document_cannot_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document cannot be blank")
        return value.strip()


class ProductRule(BaseModel):
    """A finalized Legal Metrology product rule (common or category-specific).

    This is the curated Rule Engine contract. It is intentionally separate
    from :class:`ComplianceRule`, which remains the RAG pipeline's internal
    extraction format. All fields present in the finalized JSON are preserved.
    """

    model_config = ConfigDict(extra="forbid")
    rule_id: str
    category: str
    title: str
    requirement: str
    applies_when: str
    check_type: str
    check_parameters: dict[str, Any] = Field(default_factory=dict)
    evidence_required: list[str] = Field(default_factory=list)
    source: list[ProductSource]
    source_text: str
    effective_from: Optional[str] = None
    status: str = Field(default="active")
    confidence: Optional[str] = None

    @field_validator("rule_id", "category", "title", "requirement", "check_type",
                      "source_text", "status")
    @classmethod
    def required_text_cannot_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value.strip()

    @field_validator("source")
    @classmethod
    def source_must_be_non_empty(cls, value: list[ProductSource]) -> list[ProductSource]:
        if not value:
            raise ValueError("source must list at least one cited source")
        return value

    @field_validator("effective_from")
    @classmethod
    def effective_from_must_be_iso_date(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        from datetime import date

        try:
            date.fromisoformat(value.strip())
        except ValueError as error:
            raise ValueError("effective_from must use ISO YYYY-MM-DD format.") from error
        return value.strip()
