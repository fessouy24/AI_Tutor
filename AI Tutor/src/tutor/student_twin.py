from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.utils.config import get_path


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _ema(old: float, new: float, alpha: float = 0.25) -> float:
    return _clamp((1.0 - alpha) * float(old) + alpha * float(new))


@dataclass
class StudentTwinProfile:
    """A compact, editable digital twin for tutor personalization.

    This is intentionally heuristic for the research baseline: it gives us a
    reproducible personalization state that can later be replaced by learned
    profile updates or an LLM-based reflection module.
    """

    student_id: str
    interactions: int = 0
    confidence: float = 0.5
    hint_dependence: float = 0.25
    difficulty_tolerance: float = 0.5
    speed_of_mastery: float = 0.5
    needs_step_by_step: bool = True
    best_explanation_style: str = "scaffolded hints"
    learning_style: str = "mixed"
    weak_topics: dict[str, float] = field(default_factory=dict)
    common_mistakes: dict[str, int] = field(default_factory=dict)
    mastery_history: list[dict[str, Any]] = field(default_factory=list)
    response_mode_history: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def top_weak_topics(self, k: int = 5) -> list[tuple[str, float]]:
        return sorted(self.weak_topics.items(), key=lambda item: item[1], reverse=True)[:k]

    def learning_trend(self, window: int = 6) -> float:
        recent = self.mastery_history[-window:]
        if len(recent) < 2:
            return 0.0
        first = float(recent[0].get("mastery_probability", 0.5))
        last = float(recent[-1].get("mastery_probability", 0.5))
        return last - first

    def to_prompt_context(self) -> str:
        weak = ", ".join(f"{name} ({score:.2f})" for name, score in self.top_weak_topics(3)) or "none recorded"
        mistakes = ", ".join(name for name, _ in sorted(self.common_mistakes.items(), key=lambda x: x[1], reverse=True)[:3])
        mistakes = mistakes or "none recorded"
        return (
            f"Student twin profile: confidence={self.confidence:.2f}, "
            f"hint_dependence={self.hint_dependence:.2f}, "
            f"difficulty_tolerance={self.difficulty_tolerance:.2f}, "
            f"preferred_style={self.best_explanation_style}, "
            f"needs_step_by_step={self.needs_step_by_step}, "
            f"top_weak_topics={weak}, common_mistakes={mistakes}."
        )


class StudentTwinStore:
    """Persist and update student digital twins under outputs/explanations."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        outputs_dir = get_path(cfg, "outputs_dir")
        configured = cfg.get("paths", {}).get("student_twins_dir")
        self.profiles_dir = Path(configured) if configured else outputs_dir / "student_twins"
        if not self.profiles_dir.is_absolute():
            self.profiles_dir = outputs_dir / self.profiles_dir
        self.snapshots_path = self.profiles_dir / "twin_snapshots.jsonl"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, student_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(student_id))
        return self.profiles_dir / f"{safe}.json"

    def load(self, student_id: str) -> StudentTwinProfile:
        path = self._path(student_id)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return StudentTwinProfile(**data)
        return StudentTwinProfile(student_id=str(student_id))

    def save(self, profile: StudentTwinProfile) -> None:
        profile.updated_at = time.time()
        with open(self._path(profile.student_id), "w", encoding="utf-8") as f:
            json.dump(asdict(profile), f, indent=2)

    def update(
        self,
        student_id: str,
        weak_skill: str,
        mastery_probability: float,
        correct: bool | None = None,
        recent_mistakes: list[str] | None = None,
        response_mode: str | None = None,
        tutor_result: dict[str, Any] | None = None,
    ) -> StudentTwinProfile:
        profile = self.load(student_id)
        mastery = _clamp(mastery_probability)
        profile.interactions += 1
        profile.mastery_history.append(
            {
                "t": profile.interactions,
                "weak_skill": weak_skill,
                "mastery_probability": mastery,
                "correct": correct,
            }
        )
        profile.mastery_history = profile.mastery_history[-100:]

        weakness = 1.0 - mastery
        if weak_skill:
            profile.weak_topics[weak_skill] = _ema(profile.weak_topics.get(weak_skill, weakness), weakness)
        for mistake in recent_mistakes or []:
            if mistake:
                profile.common_mistakes[mistake] = int(profile.common_mistakes.get(mistake, 0)) + 1

        outcome_signal = mastery if correct is None else (0.7 * mastery + 0.3 * float(bool(correct)))
        profile.confidence = _ema(profile.confidence, outcome_signal, alpha=0.2)
        profile.speed_of_mastery = _ema(profile.speed_of_mastery, 0.5 + profile.learning_trend(), alpha=0.2)
        profile.needs_step_by_step = profile.confidence < 0.62 or mastery < 0.55
        profile.difficulty_tolerance = _clamp(0.35 + 0.5 * profile.confidence - 0.25 * profile.hint_dependence)

        if response_mode:
            profile.response_mode_history.append(response_mode)
            profile.response_mode_history = profile.response_mode_history[-50:]
        result = tutor_result or {}
        if result.get("repaired") or response_mode in {"scaffolded", "hint_only", "socratic"}:
            profile.hint_dependence = _ema(profile.hint_dependence, 0.65 if mastery < 0.55 else 0.45, alpha=0.2)
        else:
            profile.hint_dependence = _ema(profile.hint_dependence, 0.25, alpha=0.1)

        if mastery < 0.35:
            profile.best_explanation_style = "worked examples"
            profile.learning_style = "concrete-first"
        elif profile.needs_step_by_step:
            profile.best_explanation_style = "step-by-step scaffolding"
            profile.learning_style = "guided practice"
        elif profile.difficulty_tolerance > 0.65:
            profile.best_explanation_style = "concise conceptual hints"
            profile.learning_style = "challenge-ready"
        else:
            profile.best_explanation_style = "Socratic hints"
            profile.learning_style = "mixed"

        self.save(profile)
        self._log_snapshot(profile)
        return profile

    def _log_snapshot(self, profile: StudentTwinProfile) -> None:
        with open(self.snapshots_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(profile), ensure_ascii=False) + "\n")


def profile_from_explanation(explanation: dict[str, Any]) -> tuple[str, str, float, list[str]]:
    student_id = str(explanation.get("student_id", "unknown"))
    weak_skill = str(explanation.get("weak_skill", "unknown skill"))
    mastery = float(explanation.get("mastery_probability", 0.5))
    mistakes = [
        str(item.get("skill", "unknown skill"))
        for item in explanation.get("recent_interactions", [])
        if int(item.get("correct", 1)) == 0
    ]
    return student_id, weak_skill, mastery, mistakes
