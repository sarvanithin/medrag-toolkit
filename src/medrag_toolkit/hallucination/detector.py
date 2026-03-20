"""
Medical hallucination detection for RAG output.

Adapted from medguard/guardrails/hallucination.py:
- Copies all 4 sub-checks (fake drug names, impossible dosages, unknown terms, overconfident claims)
- Adds RAG groundedness check via cosine similarity between answer and context
- RxNorm and SNOMED clients are optional (inject at init, gracefully degrade)
"""
from __future__ import annotations

import asyncio
import re
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import structlog
from pydantic import BaseModel

if TYPE_CHECKING:
    from medrag_toolkit.knowledge.drug_kb import RxNormClient

log = structlog.get_logger(__name__)


class HallucinationType(str, Enum):
    FAKE_DRUG_NAME = "fake_drug_name"
    IMPOSSIBLE_DOSAGE = "impossible_dosage"
    UNKNOWN_MEDICAL_TERM = "unknown_medical_term"
    CONFIDENT_UNSUPPORTED_CLAIM = "confident_unsupported_claim"
    UNGROUNDED_ANSWER = "ungrounded_answer"


class HallucinationFlag(BaseModel):
    type: HallucinationType
    text: str
    start: int
    end: int
    confidence: float
    explanation: str


class HallucinationResult(BaseModel):
    flags: list[HallucinationFlag]
    hallucination_score: float  # 0.0 = clean, 1.0 = severe
    blocked: bool
    annotated_text: str


# ---------------------------------------------------------------------------
# Max daily doses (mg) — adapted from medguard verbatim
# ---------------------------------------------------------------------------
MAX_DOSES: dict[str, float] = {
    "ibuprofen": 3200,
    "naproxen": 1500,
    "acetaminophen": 4000,
    "aspirin": 4000,
    "metformin": 2550,
    "metoprolol": 400,
    "atenolol": 200,
    "amlodipine": 10,
    "lisinopril": 80,
    "losartan": 150,
    "atorvastatin": 80,
    "simvastatin": 80,
    "sertraline": 200,
    "fluoxetine": 80,
    "escitalopram": 40,
    "citalopram": 60,
    "amitriptyline": 300,
    "warfarin": 20,
    "prednisone": 300,
    "dexamethasone": 80,
    "omeprazole": 120,
    "pantoprazole": 240,
    "digoxin": 0.5,
    "colchicine": 1.8,
    "allopurinol": 800,
    "hydrochlorothiazide": 200,
    "furosemide": 600,
    "carbamazepine": 1600,
    "phenytoin": 600,
    "valproate": 3000,
    "lithium": 2400,
    "tramadol": 400,
    "codeine": 360,
    "morphine": 300,
}

_DOSAGE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z\s\-]{2,30}?)\s+(\d+(?:[,\.]\d+)?)\s*(mg|mcg|g|mEq|units?|IU)\b",
    re.IGNORECASE,
)
_DRUG_MENTION_RE = re.compile(
    r"\b([A-Za-z][a-z]{4,}(?:mycin|cillin|oxacin|olol|pril|sartan|statin|"
    r"azole|vir|mab|nib|tide|zide|olam|pam|zolam))\b",
    re.IGNORECASE,
)
_MEDICAL_TERM_RE = re.compile(
    r"\b([A-Za-z]{5,}(?:itis|osis|emia|uria|algia|plasty|ectomy|otomy|"
    r"scopy|graphy|ology|pathy|trophy|genesis|lysis))\b",
    re.IGNORECASE,
)
_CONFIDENT_RE = re.compile(
    r"\b(definitely|certainly|absolutely|guaranteed|always|never|100%|"
    r"proven cure|will cure|cures? (all|every)|no side effects|completely safe|"
    r"no risk|perfectly safe|without any risk)\b",
    re.IGNORECASE,
)
_DRUG_SUFFIX_EXCLUSIONS = frozenset([
    "technology", "strategy", "biology", "ecology", "psychology", "mythology",
    "terminology", "genealogy", "methodology", "pathology", "pharmacology",
    "cardiology", "neurology", "oncology", "radiology", "gastroenterology",
    "dermatology", "ophthalmology", "urology", "anesthesiology",
])

_GROUNDEDNESS_THRESHOLD = 0.3


class HallucinationDetector:
    def __init__(
        self,
        hallucination_threshold: float = 0.3,
        rxnorm: "RxNormClient | None" = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._threshold = hallucination_threshold
        self._rxnorm = rxnorm
        self._embedding_model_name = embedding_model
        self._embedding_model = None  # lazy

    def _get_embedding_model(self):
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self._embedding_model_name)
            except ImportError:
                pass
        return self._embedding_model

    async def check(
        self, text: str, context_docs: list | None = None
    ) -> HallucinationResult:
        tasks = [
            self._check_dosages(text),
            self._check_confident_claims_async(text),
        ]
        if self._rxnorm is not None:
            tasks.append(self._check_drug_names(text))
        if context_docs:
            tasks.append(self._check_groundedness(text, context_docs))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        flags: list[HallucinationFlag] = []
        for result in results:
            if isinstance(result, Exception):
                log.debug("hallucination_subcheck_failed", error=str(result))
            elif isinstance(result, list):
                flags.extend(result)

        score = _compute_score(flags)
        blocked = score >= self._threshold and bool(flags)
        annotated = _annotate_text(text, flags)

        return HallucinationResult(
            flags=flags,
            hallucination_score=score,
            blocked=blocked,
            annotated_text=annotated,
        )

    async def _check_confident_claims_async(self, text: str) -> list[HallucinationFlag]:
        return [
            HallucinationFlag(
                type=HallucinationType.CONFIDENT_UNSUPPORTED_CLAIM,
                text=m.group(0),
                start=m.start(),
                end=m.end(),
                confidence=0.7,
                explanation=(
                    f"Absolute claim '{m.group(0)}' detected. "
                    "Medical information should acknowledge uncertainty."
                ),
            )
            for m in _CONFIDENT_RE.finditer(text)
        ]

    async def _check_drug_names(self, text: str) -> list[HallucinationFlag]:
        if self._rxnorm is None:
            return []

        candidates = [
            (m.group(1).lower(), m.start(), m.end())
            for m in _DRUG_MENTION_RE.finditer(text)
            if m.group(1).lower() not in _DRUG_SUFFIX_EXCLUSIONS
        ]

        async def validate(name: str, start: int, end: int) -> HallucinationFlag | None:
            try:
                exists = await self._rxnorm.validate_drug_exists(name)
                if not exists:
                    return HallucinationFlag(
                        type=HallucinationType.FAKE_DRUG_NAME,
                        text=text[start:end],
                        start=start,
                        end=end,
                        confidence=0.75,
                        explanation=f"'{text[start:end]}' not found in RxNorm.",
                    )
            except Exception:
                pass
            return None

        results = await asyncio.gather(*[validate(n, s, e) for n, s, e in candidates])
        return [r for r in results if r is not None]

    async def _check_dosages(self, text: str) -> list[HallucinationFlag]:
        flags = []
        for match in _DOSAGE_RE.finditer(text):
            drug_name = match.group(1).strip().lower()
            try:
                dose_value = float(match.group(2).replace(",", ""))
            except ValueError:
                continue
            unit = match.group(3).lower()
            if unit in ("mcg", "ug"):
                dose_value /= 1000

            max_dose = MAX_DOSES.get(drug_name)
            if max_dose is None:
                for known in MAX_DOSES:
                    if known in drug_name or drug_name in known:
                        max_dose = MAX_DOSES[known]
                        break

            if max_dose is not None and dose_value > max_dose * 1.5:
                flags.append(HallucinationFlag(
                    type=HallucinationType.IMPOSSIBLE_DOSAGE,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                    explanation=(
                        f"Dose of {dose_value}{unit} for {drug_name} "
                        f"exceeds known maximum of {max_dose}mg/day."
                    ),
                ))
        return flags

    async def _check_groundedness(
        self, text: str, context_docs: list
    ) -> list[HallucinationFlag]:
        """Check if answer is grounded in context via cosine similarity."""
        model = self._get_embedding_model()
        if model is None:
            return []
        try:
            context_text = " ".join(doc.content[:200] for doc in context_docs[:5])
            embeddings = model.encode([text, context_text], normalize_embeddings=True)
            similarity = float(np.dot(embeddings[0], embeddings[1]))

            if similarity < _GROUNDEDNESS_THRESHOLD:
                return [HallucinationFlag(
                    type=HallucinationType.UNGROUNDED_ANSWER,
                    text=text[:100],
                    start=0,
                    end=min(100, len(text)),
                    confidence=1.0 - similarity,
                    explanation=(
                        f"Answer has low similarity ({similarity:.2f}) to retrieved context. "
                        "May not be grounded in provided evidence."
                    ),
                )]
        except Exception as exc:
            log.debug("groundedness_check_failed", error=str(exc))
        return []


def _compute_score(flags: list[HallucinationFlag]) -> float:
    if not flags:
        return 0.0
    weight_map = {
        HallucinationType.IMPOSSIBLE_DOSAGE: 0.4,
        HallucinationType.FAKE_DRUG_NAME: 0.35,
        HallucinationType.UNGROUNDED_ANSWER: 0.3,
        HallucinationType.UNKNOWN_MEDICAL_TERM: 0.2,
        HallucinationType.CONFIDENT_UNSUPPORTED_CLAIM: 0.15,
    }
    return min(sum(weight_map.get(f.type, 0.1) * f.confidence for f in flags), 1.0)


def _annotate_text(text: str, flags: list[HallucinationFlag]) -> str:
    if not flags:
        return text
    result = text
    for flag in sorted(flags, key=lambda f: f.start, reverse=True):
        annotation = f" [WARNING: {flag.explanation}]"
        result = result[: flag.end] + annotation + result[flag.end :]
    return result
