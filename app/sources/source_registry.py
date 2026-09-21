OFFICIAL_SOURCES = [
    {
        "name": "Department of Consumer Affairs - Legal Metrology",
        "url": "https://consumeraffairs.gov.in/pages/legal-metrology-act",
        "organization": "Department of Consumer Affairs",
        "source_type": "official_government",
        "document_category": "legal_metrology_packaged_commodities",
    }
]

SOURCE_URL = next(
    source["url"] for source in OFFICIAL_SOURCES if "legal-metrology-act" in source["url"]
)


class TargetNotFoundError(RuntimeError):
    """Raised when a required production document cannot be discovered."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(
            "Production target(s) not discovered, refusing to substitute: "
            + "; ".join(missing)
        )


def _normalize_title(title: str) -> str:
    import re

    return re.sub(r"\s+", " ", title).strip().casefold()


_BASE_EXCLUDE_TERMS = (
    "amend",
    "corrigendum",
    "guideline",
    "advisor",
    "hosier",
    "readymade",
    "medical device",
    "sop",
    "operating procedure",
)


def _is_base_rules(document: dict) -> bool:
    title = _normalize_title(document.get("title", ""))
    return (
        "packaged commodit" in title
        and "2011" in title
        and not any(term in title for term in _BASE_EXCLUDE_TERMS)
    )


def _is_garments_advisory(document: dict) -> bool:
    title = _normalize_title(document.get("title", ""))
    return "readymade" in title and "hosier" in title


def select_targets(documents: list[dict]) -> tuple[dict, dict]:
    """Pick exactly the two production documents; never substitute.

    Returns ``(base_2011_rules, garments_advisory)``. Raises
    :class:`TargetNotFoundError` naming whichever target is missing. When
    several links match one target, the shortest title wins
    deterministically (the base/advisory titles are the minimal forms).
    """
    base = sorted(
        (item for item in documents if _is_base_rules(item)),
        key=lambda item: (len(_normalize_title(item.get("title", ""))), item.get("url", "")),
    )
    garments = sorted(
        (item for item in documents if _is_garments_advisory(item)),
        key=lambda item: (len(_normalize_title(item.get("title", ""))), item.get("url", "")),
    )
    missing = []
    if not base:
        missing.append("Legal Metrology (Packaged Commodities) Rules, 2011")
    if not garments:
        missing.append("Legal Metrology (Packaged Commodities) Rules, 2011 - Advisory for Readymade Garments/Hosiery products")
    if missing:
        raise TargetNotFoundError(missing)
    if base[0]["url"] == garments[0]["url"]:
        raise TargetNotFoundError(["targets resolved to the same document; refusing to proceed"])
    return base[0], garments[0]