"""Unit tests for DrugKnowledgeBase with respx HTTP mocking."""
from __future__ import annotations

import httpx
import pytest
import respx

from medrag_toolkit.knowledge.drug_kb import OpenFDAClient, RxNormClient, _extract_drug_names, _parse_severity, InteractionSeverity


@respx.mock
@pytest.mark.asyncio
async def test_rxnorm_get_rxcui_found():
    respx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json").mock(
        return_value=httpx.Response(
            200,
            json={"idGroup": {"rxnormId": ["12345"]}},
        )
    )
    async with httpx.AsyncClient() as client:
        rxnorm = RxNormClient(client)
        rxcui = await rxnorm.get_rxcui("aspirin")
    assert rxcui == "12345"


@respx.mock
@pytest.mark.asyncio
async def test_rxnorm_validate_drug_exists_true():
    respx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json").mock(
        return_value=httpx.Response(
            200,
            json={"idGroup": {"rxnormId": ["1191"]}},
        )
    )
    async with httpx.AsyncClient() as client:
        rxnorm = RxNormClient(client)
        exists = await rxnorm.validate_drug_exists("aspirin")
    assert exists is True


@respx.mock
@pytest.mark.asyncio
async def test_rxnorm_validate_drug_not_found():
    respx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json").mock(
        return_value=httpx.Response(
            200,
            json={"idGroup": {"rxnormId": []}},
        )
    )
    async with httpx.AsyncClient() as client:
        rxnorm = RxNormClient(client)
        exists = await rxnorm.validate_drug_exists("fakedrugxyz")
    assert exists is False


@respx.mock
@pytest.mark.asyncio
async def test_openfda_404_returns_none():
    respx.get("https://api.fda.gov/drug/label.json").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    async with httpx.AsyncClient() as client:
        openfda = OpenFDAClient(client)
        result = await openfda.get_drug_label("unknowndrug")
    assert result is None


def test_extract_drug_names_from_query():
    query = "What are the interactions between warfarin and aspirin?"
    names = _extract_drug_names(query)
    assert "warfarin" in names


def test_parse_severity_contraindicated():
    text = "This drug is contraindicated in patients with renal failure."
    severity = _parse_severity(text)
    assert severity == InteractionSeverity.CONTRAINDICATED


def test_parse_severity_high():
    text = "Avoid concurrent use as it may cause serious adverse events."
    severity = _parse_severity(text)
    assert severity == InteractionSeverity.HIGH


def test_parse_severity_moderate():
    text = "Monitor INR closely when starting this medication."
    severity = _parse_severity(text)
    assert severity == InteractionSeverity.MODERATE


def test_parse_severity_unknown():
    text = "This drug has been approved for use in adults."
    severity = _parse_severity(text)
    assert severity == InteractionSeverity.UNKNOWN
