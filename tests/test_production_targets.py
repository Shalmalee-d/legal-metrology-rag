"""Production two-document target selection (migrated from experiment harness)."""

import pytest

from app.sources.source_registry import (
    SOURCE_URL,
    TargetNotFoundError,
    select_targets,
)


def _doc(title, url):
    return {"title": title, "url": url, "source": "https://example.test",
            "category": "legal_metrology_packaged_commodities"}


BASE = _doc("Download The Legal Metrology (Packaged Commodities) Rules, 2011",
            "https://example.test/base.pdf")
GARMENTS = _doc("Download The Legal Metrology (Packaged Commodities) Rules,2011- Advisory "
                "for enforcement of provisions of Rules for Readymade Garments/ Hosiery products",
                "https://example.test/garments.pdf")
AMEND1 = _doc("Download The Legal Metrology (Packaged Commodities) Amendment Rules, 2017",
              "https://example.test/amend2017.pdf")
AMEND2 = _doc("Download The Legal Metrology (Packaged Commodities) Second Amendment Rules, 2012",
              "https://example.test/amend2012.pdf")
GUIDE = _doc("Download Guidelines For Implementation of the Legal Metrology Act, 2009",
             "https://example.test/guide.pdf")


def test_selects_exact_targets_not_first_two():
    base, garments = select_targets([AMEND1, AMEND2, GUIDE, GARMENTS, BASE])
    assert base["url"] == BASE["url"]
    assert garments["url"] == GARMENTS["url"]


def test_missing_base_reports_which_target():
    with pytest.raises(TargetNotFoundError) as raised:
        select_targets([AMEND1, GARMENTS])
    assert "Rules, 2011" in str(raised.value)


def test_missing_garments_reports_which_target():
    with pytest.raises(TargetNotFoundError) as raised:
        select_targets([AMEND1, BASE])
    assert "Garments" in str(raised.value)


def test_production_source_url_is_official_dca():
    assert SOURCE_URL == "https://consumeraffairs.gov.in/pages/legal-metrology-act"


def test_refresh_uses_production_selection():
    import app.refresh as refresh
    import app.sources.source_registry as registry

    assert refresh.select_targets is registry.select_targets
    assert refresh.SOURCE_URL == SOURCE_URL
