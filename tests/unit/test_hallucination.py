"""Unit tests for hallucination detection."""
from __future__ import annotations

import pytest

from medrag_toolkit.hallucination.detector import HallucinationDetector, HallucinationType


@pytest.fixture
def detector():
    return HallucinationDetector(hallucination_threshold=0.3)


@pytest.mark.asyncio
async def test_clean_text_no_flags(detector):
    text = "Based on clinical studies, aspirin 100mg daily may reduce cardiovascular risk."
    result = await detector.check(text)
    assert result.hallucination_score < 0.3
    assert not result.blocked


@pytest.mark.asyncio
async def test_impossible_dosage_flagged(detector):
    text = "Ibuprofen 10000 mg should be taken daily for pain relief."
    result = await detector.check(text)
    flags = [f for f in result.flags if f.type == HallucinationType.IMPOSSIBLE_DOSAGE]
    assert len(flags) > 0
    assert result.hallucination_score > 0.0


@pytest.mark.asyncio
async def test_overconfident_claim_flagged(detector):
    text = "This drug definitely cures all types of cancer without any risk."
    result = await detector.check(text)
    flags = [f for f in result.flags if f.type == HallucinationType.CONFIDENT_UNSUPPORTED_CLAIM]
    assert len(flags) > 0


@pytest.mark.asyncio
async def test_safe_dosage_not_flagged(detector):
    text = "Ibuprofen 400 mg every 6 hours is within normal dosing."
    result = await detector.check(text)
    dosage_flags = [f for f in result.flags if f.type == HallucinationType.IMPOSSIBLE_DOSAGE]
    assert len(dosage_flags) == 0


@pytest.mark.asyncio
async def test_annotated_text_has_warnings(detector):
    text = "Aspirin 50000 mg is the correct dose."
    result = await detector.check(text)
    if result.flags:
        assert "[WARNING:" in result.annotated_text
