from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.rag.index import EducationalRAG, RetrievedChunk
from src.tutor.generator import SocraticTutor
from src.tutor.leakage import detect_answer_leakage, rag_faithfulness_check, repair_hint_only
from src.tutor.student_twin import StudentTwinProfile, StudentTwinStore
from src.utils.config import get_path


@dataclass
class StudentState:
    student_id: str
    weak_skill: str
    mastery_probability: float
    recent_mistakes: list[str] = field(default_factory=list)
    learning_trend: float = 0.0


@dataclass
class RetrievalDecision:
    retrieval_types: list[str]
    queries: list[str]
    rationale: str
    top_k_per_query: int


@dataclass
class ResponsePlan:
    mode: str
    allow_direct_answer: bool
    hint_depth: int
    rationale: str
    prompt_constraints: list[str]


class AgenticRAGTutor:
    """Agentic tutoring pipeline built on the baseline RAG and Socratic tutor.

    The "agents" are deterministic planner components. That makes experiments
    reproducible and easy to inspect, while still changing the baseline from a
    single retrieve-then-answer call into a state-aware retrieval and response
    planning loop.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.research_cfg = cfg.get("research", {})
        self.rag = EducationalRAG(cfg)
        self.tutor = SocraticTutor(cfg)
        self.twins = StudentTwinStore(cfg)
        outputs_dir = get_path(cfg, "outputs_dir")
        configured = cfg.get("paths", {}).get("agentic_tutor_log")
        self.log_path = Path(configured) if configured else outputs_dir / "agentic_tutor_logs.jsonl"
        if not self.log_path.is_absolute():
            self.log_path = outputs_dir / self.log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def analyze_student_state(
        self,
        explanation: dict[str, Any] | None = None,
        student_id: str = "demo_student",
        weak_skill: str = "unknown skill",
        mastery_probability: float = 0.5,
        recent_mistakes: list[str] | None = None,
    ) -> StudentState:
        explanation = explanation or {}
        student_id = str(explanation.get("student_id", student_id))
        weak_skill = str(explanation.get("weak_skill", weak_skill))
        mastery_probability = float(explanation.get("mastery_probability", mastery_probability))
        recent = explanation.get("recent_interactions", [])
        mistakes = recent_mistakes or [
            str(item.get("skill", weak_skill))
            for item in recent
            if int(item.get("correct", 1)) == 0
        ]
        trend = self._trend_from_recent(recent)
        return StudentState(
            student_id=student_id,
            weak_skill=weak_skill,
            mastery_probability=mastery_probability,
            recent_mistakes=mistakes[-5:],
            learning_trend=trend,
        )

    @staticmethod
    def _trend_from_recent(recent: list[dict[str, Any]]) -> float:
        if len(recent) < 4:
            return 0.0
        values = [float(item.get("correct", 0)) for item in recent]
        mid = max(1, len(values) // 2)
        return (sum(values[mid:]) / max(1, len(values[mid:]))) - (sum(values[:mid]) / max(1, len(values[:mid])))

    def decide_retrieval(self, state: StudentState, student_question: str, twin: StudentTwinProfile) -> RetrievalDecision:
        types: list[str] = []
        queries: list[str] = []
        weak = state.weak_skill
        if state.mastery_probability < 0.35:
            types.extend(["prerequisite concept", "beginner explanation", "worked example"])
            queries.extend(
                [
                    f"prerequisite concepts for {weak}",
                    f"beginner explanation of {weak}",
                    f"worked example for {weak} without revealing final answer",
                ]
            )
        elif state.mastery_probability < 0.65:
            types.extend(["conceptual explanation", "scaffolded example", "practice problem"])
            queries.extend(
                [
                    f"conceptual explanation of {weak}",
                    f"scaffolded example for {weak}",
                    f"practice problem about {weak}",
                ]
            )
        else:
            types.extend(["challenge practice", "common misconception check"])
            queries.extend([f"challenge practice for {weak}", f"common misconceptions in {weak}"])

        if state.recent_mistakes:
            mistake = state.recent_mistakes[-1]
            types.append("misconception repair")
            queries.append(f"why students confuse {mistake} when learning {weak}")
        if twin.hint_dependence > 0.55:
            types.append("motivational hint")
            queries.append(f"short motivational hint for {weak}")

        queries.append(f"{weak} {student_question}")
        top_k = max(1, int(self.cfg.get("rag", {}).get("top_k", 4)) // max(1, min(3, len(queries))))
        rationale = (
            f"mastery={state.mastery_probability:.2f}, trend={state.learning_trend:.2f}, "
            f"twin_style={twin.best_explanation_style}"
        )
        return RetrievalDecision(types, list(dict.fromkeys(queries)), rationale, top_k)

    def retrieve_agentic(self, decision: RetrievalDecision) -> tuple[list[RetrievedChunk], float]:
        allow_build = bool(self.research_cfg.get("build_rag_index_if_missing", False))
        if not allow_build and not self._rag_has_existing_chunks():
            return [], 0.0

        seen: set[str] = set()
        chunks: list[RetrievedChunk] = []
        total_latency = 0.0
        for query in decision.queries:
            try:
                retrieved, latency = self.rag.retrieve(query, top_k=decision.top_k_per_query)
            except Exception as exc:
                print(f"Agentic retrieval skipped for query={query!r}: {exc}")
                continue
            total_latency += latency
            for chunk in retrieved:
                key = chunk.chunk_id or f"{chunk.source}:{chunk.text[:80]}"
                if key in seen:
                    continue
                seen.add(key)
                meta = dict(chunk.metadata or {})
                meta["agentic_query"] = query
                meta["retrieval_rationale"] = decision.rationale
                chunk.metadata = meta
                chunks.append(chunk)
        chunks.sort(key=lambda c: float(c.score or 0.0), reverse=True)
        return chunks[: int(self.cfg.get("rag", {}).get("top_k", 4))], total_latency

    def _rag_has_existing_chunks(self) -> bool:
        return self.rag.has_existing_chunks()

    def plan_response(self, state: StudentState, twin: StudentTwinProfile, chunks: list[RetrievedChunk]) -> ResponsePlan:
        if state.mastery_probability < 0.35 or twin.needs_step_by_step:
            mode = "scaffolded"
            depth = 3
        elif state.mastery_probability < 0.65:
            mode = "socratic"
            depth = 2
        else:
            mode = "hint_only"
            depth = 1
        if state.learning_trend < -0.2:
            mode = "motivational_hint"
        constraints = [
            "do not reveal the final answer",
            "ask exactly one guiding question",
            "cite retrieved context when used",
            "prefer hints over full worked solutions",
        ]
        if not chunks:
            constraints.append("state uncertainty because no retrieved grounding was available")
        rationale = (
            f"mode={mode} chosen from mastery={state.mastery_probability:.2f}, "
            f"trend={state.learning_trend:.2f}, confidence={twin.confidence:.2f}"
        )
        return ResponsePlan(mode=mode, allow_direct_answer=False, hint_depth=depth, rationale=rationale, prompt_constraints=constraints)

    def build_agentic_prompt(
        self,
        student_question: str,
        state: StudentState,
        twin: StudentTwinProfile,
        decision: RetrievalDecision,
        plan: ResponsePlan,
        chunks: list[RetrievedChunk],
        stricter: bool = False,
    ) -> str:
        context = "\n\n".join(
            f"[{i + 1}] score={float(c.score or 0.0):.3f} source={Path(c.source).name if c.source else 'retrieved'}\n{c.text}"
            for i, c in enumerate(chunks)
        )
        strict = (
            "Repair mode: rewrite as a hint-only response. Remove final answers, equations that complete the solution, and direct answer phrases."
            if stricter
            else ""
        )
        return f"""You are an adaptive Socratic AI tutor using Agentic RAG.

Student state:
- weak skill: {state.weak_skill}
- mastery probability: {state.mastery_probability:.3f}
- learning trend: {state.learning_trend:.3f}
- recent mistakes: {", ".join(state.recent_mistakes) if state.recent_mistakes else "none recorded"}

{twin.to_prompt_context()}

Retrieval decision:
- retrieval types: {", ".join(decision.retrieval_types)}
- rationale: {decision.rationale}

Response plan:
- mode: {plan.mode}
- hint depth: {plan.hint_depth}
- rationale: {plan.rationale}
- constraints: {"; ".join(plan.prompt_constraints)}

Retrieved grounding context:
{context if context else "[No retrieved context available.]"}

Student question:
{student_question}

Write a concise tutoring response. It must:
- adapt to the twin profile
- avoid direct final answers
- use scaffolded hints appropriate to the response plan
- ask exactly one guiding question
- include a next practice recommendation only if useful
{strict}
"""

    def generate(
        self,
        student_question: str,
        explanation: dict[str, Any] | None = None,
        final_answer: str | None = None,
        student_id: str = "demo_student",
        weak_skill: str = "unknown skill",
        mastery_probability: float = 0.5,
    ) -> dict[str, Any]:
        state = self.analyze_student_state(explanation, student_id, weak_skill, mastery_probability)
        twin = self.twins.load(state.student_id)
        decision = self.decide_retrieval(state, student_question, twin)
        chunks, retrieval_latency_ms = self.retrieve_agentic(decision)
        plan = self.plan_response(state, twin, chunks)

        prompt = self.build_agentic_prompt(student_question, state, twin, decision, plan, chunks)
        start = time.perf_counter()
        response, mode_used = self.tutor._generate_once(prompt, state.weak_skill, state.mastery_probability, chunks)
        tutor_latency_ms = (time.perf_counter() - start) * 1000

        original = response
        leakage = detect_answer_leakage(response, final_answer=final_answer, hint_mode=True)
        repaired = False
        if leakage["leakage_score"]:
            strict_prompt = self.build_agentic_prompt(student_question, state, twin, decision, plan, chunks, stricter=True)
            try:
                response, mode_used = self.tutor._generate_once(strict_prompt, state.weak_skill, state.mastery_probability, chunks)
            except Exception:
                response = repair_hint_only(original, final_answer=final_answer)
            second = detect_answer_leakage(response, final_answer=final_answer, hint_mode=True)
            if second["leakage_score"]:
                response = repair_hint_only(response, final_answer=final_answer)
            leakage = second
            repaired = True

        faithfulness = rag_faithfulness_check(response, chunks)
        updated_twin = self.twins.update(
            student_id=state.student_id,
            weak_skill=state.weak_skill,
            mastery_probability=state.mastery_probability,
            recent_mistakes=state.recent_mistakes,
            response_mode=plan.mode,
            tutor_result={"repaired": repaired, "faithfulness": faithfulness},
        )
        result = {
            "student_question": student_question,
            "student_state": asdict(state),
            "retrieval_decision": asdict(decision),
            "response_plan": asdict(plan),
            "retrieved_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "score": c.score,
                    "metadata": c.metadata or {},
                    "preview": c.text[:240],
                }
                for c in chunks
            ],
            "response": response,
            "original_response": original if repaired else None,
            "mode_used": mode_used,
            "retrieval_latency_ms": retrieval_latency_ms,
            "tutor_latency_ms": tutor_latency_ms,
            "leakage": leakage,
            "repaired": repaired,
            "faithfulness": faithfulness,
            "updated_twin": asdict(updated_twin),
        }
        self._log(result)
        return result

    def _log(self, result: dict[str, Any]) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
