"""Unit tests for PubMed client with respx HTTP mocking."""
from __future__ import annotations


import httpx
import pytest
import respx

from medrag_toolkit.knowledge.pubmed import PubMedClient, _parse_text_abstracts


@respx.mock
@pytest.mark.asyncio
async def test_pubmed_search_returns_pmids():
    respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
        return_value=httpx.Response(
            200,
            json={"esearchresult": {"idlist": ["12345", "67890"]}},
        )
    )
    async with httpx.AsyncClient() as client:
        pubmed = PubMedClient(client)
        pmids = await pubmed.search("myocardial infarction")
    assert pmids == ["12345", "67890"]


@respx.mock
@pytest.mark.asyncio
async def test_pubmed_search_failure_returns_empty():
    respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
        side_effect=httpx.ConnectError("timeout")
    )
    async with httpx.AsyncClient() as client:
        pubmed = PubMedClient(client)
        pmids = await pubmed.search("any query")
    assert pmids == []


@respx.mock
@pytest.mark.asyncio
async def test_pubmed_fetch_summaries():
    respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi").mock(
        return_value=httpx.Response(
            200,
            json={
                "result": {
                    "12345": {
                        "title": "Aspirin for MI Prevention",
                        "source": "JAMA",
                        "pubdate": "2022 Jan",
                    }
                }
            },
        )
    )
    async with httpx.AsyncClient() as client:
        pubmed = PubMedClient(client)
        articles = await pubmed.fetch_summaries(["12345"])
    assert len(articles) == 1
    assert articles[0].pmid == "12345"
    assert articles[0].title == "Aspirin for MI Prevention"
    assert articles[0].year == "2022"


def test_parse_text_abstracts_basic():
    raw = """1. Aspirin for Prevention

Author information:...

Abstract
Aspirin reduces platelet aggregation. This is well established.

PMID: 12345

2. Ibuprofen Study

Abstract
Ibuprofen is an NSAID.

PMID: 67890"""
    articles = _parse_text_abstracts(["12345", "67890"], raw)
    assert len(articles) == 2
    assert articles[0].pmid == "12345"
    assert "Aspirin" in articles[0].title


def test_parse_text_abstracts_empty():
    articles = _parse_text_abstracts([], "")
    assert articles == []
