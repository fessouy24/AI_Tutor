from __future__ import annotations

import re
from typing import Any


LEAKAGE_PATTERNS = [
    r"\bthe answer is\b",
    r"\bfinal answer\b",
    r"\bsolution is\b",
    r"\bso the answer\b",
    r"\btherefore\s+[a-z]\s*=",
    r"\bhence\s+[a-z]\s*=",
    r"\bthe value of\s+[a-z]\s+is\b",
]

WORKED_SOLUTION_PATTERNS = [
    r"\bstep\s*1\b.*\bstep\s*2\b.*\bstep\s*3\b",
    r"\bsolve for\b.*\btherefore\b",
    r"\bsubstitute\b.*\bsimplify\b.*\banswer\b",
]


def detect_answer_leakage(
    response: str,
    final_answer: str | None = None,
    hint_mode: bool = True,
) -> dict[str, Any]:
    text = response or ""
    lower = text.lower()
    flagged: list[str] = []
    reasons: list[str] = []

    if final_answer:
        answer = str(final_answer).strip()
        if answer and len(answer) <= 80:
            pattern = re.escape(answer.lower())
            if re.search(rf"(^|[^a-z0-9]){pattern}([^a-z0-9]|$)", lower):
                flagged.append(answer)
                reasons.append("Exact final answer appears in tutor response.")

    for pattern in LEAKAGE_PATTERNS:
        match = re.search(pattern, lower, flags=re.DOTALL)
        if match:
            flagged.append(match.group(0))
            reasons.append("Direct-answer phrase detected.")

    if hint_mode:
        for pattern in WORKED_SOLUTION_PATTERNS:
            match = re.search(pattern, lower, flags=re.DOTALL)
            if match:
                flagged.append(match.group(0)[:80])
                reasons.append("Response looks like an overly direct worked solution.")

    leakage = int(bool(flagged))
    return {
        "leakage_score": leakage,
        "leakage_reason": "; ".join(dict.fromkeys(reasons)) if reasons else "",
        "flagged_phrases": list(dict.fromkeys(flagged)),
    }


def repair_hint_only(response: str, final_answer: str | None = None) -> str:
    text = response or ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept: list[str] = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(re.search(pattern, lower) for pattern in LEAKAGE_PATTERNS):
            continue
        if final_answer and str(final_answer).strip() and str(final_answer).strip().lower() in lower:
            continue
        kept.append(sentence)
    repaired = " ".join(kept).strip()
    if len(repaired) < 40:
        repaired = (
            "Try focusing on the relationship between the known quantities before calculating. "
            "What smaller step can you complete first, and how would that change the expression?"
        )
    return repaired


def rag_faithfulness_check(response: str, retrieved_chunks: list[Any]) -> dict[str, Any]:
    if not retrieved_chunks:
        return {"supported": False, "average_overlap": 0.0, "chunks_used": 0}
    response_terms = {t for t in re.findall(r"[a-zA-Z]{4,}", response.lower()) if len(t) > 3}
    overlaps: list[float] = []
    chunks_used = 0
    for chunk in retrieved_chunks:
        text = getattr(chunk, "text", str(chunk))
        terms = {t for t in re.findall(r"[a-zA-Z]{4,}", text.lower()) if len(t) > 3}
        if not terms:
            overlaps.append(0.0)
            continue
        overlap = len(response_terms & terms) / max(1, len(response_terms))
        overlaps.append(overlap)
        if overlap >= 0.08:
            chunks_used += 1
    avg = sum(overlaps) / max(1, len(overlaps))
    return {"supported": chunks_used > 0, "average_overlap": avg, "chunks_used": chunks_used}

