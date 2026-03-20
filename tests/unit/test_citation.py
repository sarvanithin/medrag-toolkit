"""Unit tests for citation extraction and verification."""
from __future__ import annotations


from medrag_toolkit.citation.extractor import extract
from medrag_toolkit.citation.verifier import verify
from medrag_toolkit.retrieval.base import RetrievedDocument


def test_extract_pmid_citations():
    text = "Aspirin reduces MI risk [PMID:12345678] and [PMID:87654321]."
    citations = extract(text)
    assert len(citations) == 2
    assert citations[0].type == "pmid"
    assert citations[0].id == "12345678"
    assert citations[1].id == "87654321"


def test_extract_drug_citations():
    text = "Warfarin [Drug:warfarin] interacts with aspirin [Drug:aspirin]."
    citations = extract(text)
    drug_cites = [c for c in citations if c.type == "drug"]
    assert len(drug_cites) == 2
    assert drug_cites[0].id == "warfarin"


def test_extract_mixed_citations():
    text = "Study shows [PMID:11111111] that [Drug:metformin] lowers HbA1c."
    citations = extract(text)
    assert len(citations) == 2
    types = {c.type for c in citations}
    assert types == {"pmid", "drug"}


def test_extract_no_citations():
    text = "This answer has no citations at all."
    assert extract(text) == []


def test_verify_grounded_pmid():
    text = "Aspirin is effective [PMID:12345]."
    citations = extract(text)
    doc = RetrievedDocument(
        id="pubmed_12345",
        content="Aspirin study",
        source="pubmed",
        metadata={"pmid": "12345"},
        score=0.9,
    )
    report = verify(text, citations, [doc])
    assert len(report.grounded_citations) == 1
    assert len(report.hallucinated_citations) == 0


def test_verify_hallucinated_pmid():
    text = "Some claim [PMID:99999]."
    citations = extract(text)
    doc = RetrievedDocument(
        id="pubmed_12345",
        content="Different study",
        source="pubmed",
        metadata={"pmid": "12345"},
        score=0.9,
    )
    report = verify(text, citations, [doc])
    assert len(report.hallucinated_citations) == 1


def test_verify_coverage_with_factual_claim():
    text = "Ibuprofen 400 mg is recommended [PMID:11111]."
    citations = extract(text)
    doc = RetrievedDocument(
        id="pubmed_11111",
        content="Ibuprofen study",
        source="pubmed",
        metadata={"pmid": "11111"},
        score=0.8,
    )
    report = verify(text, citations, [doc])
    assert report.coverage > 0.0
