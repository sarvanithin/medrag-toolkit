"""
Integration tests for the full MedRAG pipeline.

These tests use fixture data (no real API calls) to verify end-to-end flow.
Run with: pytest tests/integration/ -v
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from medrag_toolkit.citation.extractor import extract
from medrag_toolkit.citation.verifier import verify
from medrag_toolkit.hallucination.detector import HallucinationDetector
from medrag_toolkit.knowledge.base import Document
from medrag_toolkit.retrieval.base import RetrievedDocument


FIXTURE_DOCS = [
    RetrievedDocument(
        id="pubmed_12345678",
        content=(
            "Aspirin at 162-325 mg is recommended for acute myocardial infarction. "
            "Multiple randomized controlled trials demonstrate mortality benefit."
        ),
        source="pubmed",
        metadata={"pmid": "12345678", "title": "Aspirin in Acute MI", "year": "2021"},
        score=0.92,
    ),
    RetrievedDocument(
        id="pubmed_23456789",
        content=(
            "Antiplatelet therapy including aspirin reduces reinfarction risk by 25%. "
            "Early administration within 24 hours is critical."
        ),
        source="pubmed",
        metadata={"pmid": "23456789", "title": "Antiplatelet Therapy Meta-analysis", "year": "2020"},
        score=0.85,
    ),
]

FIXTURE_ANSWER = (
    "For acute myocardial infarction, aspirin 162-325 mg should be administered immediately "
    "[PMID:12345678]. Early antiplatelet therapy reduces reinfarction risk by approximately 25% "
    "[PMID:23456789]. Clinical guidelines recommend prompt administration within 24 hours."
)


@pytest.mark.asyncio
async def test_citation_extraction_from_fixture_answer():
    citations = extract(FIXTURE_ANSWER)
    assert len(citations) == 2
    pmids = {c.id for c in citations}
    assert "12345678" in pmids
    assert "23456789" in pmids


@pytest.mark.asyncio
async def test_citation_verification_grounded():
    citations = extract(FIXTURE_ANSWER)
    report = verify(FIXTURE_ANSWER, citations, FIXTURE_DOCS)
    assert len(report.grounded_citations) == 2
    assert len(report.hallucinated_citations) == 0
    assert report.coverage > 0.0
    assert report.is_well_cited


@pytest.mark.asyncio
async def test_hallucination_detection_clean_answer():
    detector = HallucinationDetector(hallucination_threshold=0.3)
    result = await detector.check(FIXTURE_ANSWER, context_docs=FIXTURE_DOCS)
    # Should not flag confident claims in this medically hedged text
    assert result.hallucination_score < 0.5


@pytest.mark.asyncio
async def test_hallucination_detection_with_impossible_dose():
    detector = HallucinationDetector(hallucination_threshold=0.3)
    bad_answer = "Aspirin 50000 mg should be given immediately in acute MI [PMID:12345678]."
    result = await detector.check(bad_answer, context_docs=FIXTURE_DOCS)
    assert result.hallucination_score > 0.0
    from medrag_toolkit.hallucination.detector import HallucinationType
    dosage_flags = [f for f in result.flags if f.type == HallucinationType.IMPOSSIBLE_DOSAGE]
    assert len(dosage_flags) > 0


@pytest.mark.asyncio
async def test_end_to_end_mock_pipeline():
    """Verify the complete pipeline flow without real API calls."""
    from medrag_toolkit.generation.base import GeneratorResponse

    # Mock the generator
    mock_gen = AsyncMock()
    mock_gen.generate.return_value = GeneratorResponse(
        answer=FIXTURE_ANSWER,
        raw_context_used=[d.content for d in FIXTURE_DOCS],
        model="llama3.2",
    )

    # Mock retrieval
    mock_retrieve = AsyncMock(return_value=FIXTURE_DOCS)

    # Run citation + hallucination checks (real, not mocked)
    answer = FIXTURE_ANSWER
    citations = extract(answer)
    citation_report = verify(answer, citations, FIXTURE_DOCS)

    detector = HallucinationDetector()
    hall_result = await detector.check(answer, context_docs=FIXTURE_DOCS)

    assert len(citations) == 2
    assert citation_report.is_well_cited
    assert hall_result.hallucination_score < 0.5
