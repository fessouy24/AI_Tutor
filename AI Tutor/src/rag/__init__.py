"""Retrieval augmented grounding modules."""

from src.rag.index import EducationalRAG, RetrievedChunk

__all__ = [
    "AgenticRAGTutor",
    "EducationalRAG",
    "ResponsePlan",
    "RetrievalDecision",
    "RetrievedChunk",
    "StudentState",
]


def __getattr__(name: str):
    if name in {"AgenticRAGTutor", "ResponsePlan", "RetrievalDecision", "StudentState"}:
        from src.rag.agentic import AgenticRAGTutor, ResponsePlan, RetrievalDecision, StudentState

        return {
            "AgenticRAGTutor": AgenticRAGTutor,
            "ResponsePlan": ResponsePlan,
            "RetrievalDecision": RetrievalDecision,
            "StudentState": StudentState,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
