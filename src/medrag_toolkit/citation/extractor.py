"""
Citation extractor: parses [PMID:xxx] and [Drug:xxx] patterns from LLM output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_PMID_RE = re.compile(r"\[PMID:(\d+)\]")
_DRUG_RE = re.compile(r"\[Drug:([^\]]+)\]")


@dataclass
class Citation:
    type: str  # "pmid" | "drug"
    id: str
    span_start: int
    span_end: int
    raw: str


def extract(text: str) -> list[Citation]:
    """Extract all citations from LLM-generated text."""
    citations = []

    for match in _PMID_RE.finditer(text):
        citations.append(Citation(
            type="pmid",
            id=match.group(1),
            span_start=match.start(),
            span_end=match.end(),
            raw=match.group(0),
        ))

    for match in _DRUG_RE.finditer(text):
        citations.append(Citation(
            type="drug",
            id=match.group(1).strip(),
            span_start=match.start(),
            span_end=match.end(),
            raw=match.group(0),
        ))

    return sorted(citations, key=lambda c: c.span_start)
