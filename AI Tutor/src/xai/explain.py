from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

from src.models.training import move_batch_to_device
from src.utils.config import get_path


def _reverse_vocab(vocab: dict[str, int]) -> dict[int, str]:
    return {int(v): str(k) for k, v in vocab.items()}


def safe_explanation_filename(value: Any, fallback: str = "sample") -> str:
    text = str(value or fallback)
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text).strip(" ._")
    return (text or fallback)[:140]


def explain_akt_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    mappings: dict[str, Any],
    device: torch.device,
    output_path: str | Path | None = None,
    top_k_attention: int = 5,
) -> dict[str, Any]:
    model.eval()
    batch = move_batch_to_device(batch, device)
    with torch.no_grad():
        out = model(
            batch["question_ids"],
            batch["skill_ids"],
            batch["correct_ids"],
            batch["target_question_ids"],
            batch["target_skill_ids"],
            input_mask=batch["mask"],
            return_attention=True,
        )
    logits = out["logits"]
    probs = torch.sigmoid(logits)
    mask = batch["mask"].bool()[0]
    valid_probs = probs[0][mask]
    valid_target_skills = batch["target_skill_ids"][0][mask]
    valid_target_questions = batch["target_question_ids"][0][mask]
    if valid_probs.numel() == 0:
        raise ValueError("Cannot explain an empty masked sequence.")

    weak_pos = int(torch.argmin(valid_probs).item())
    mastery_probability = float(valid_probs[weak_pos].detach().cpu().item())
    skill_rev = _reverse_vocab(mappings.get("skill_vocab", {}))
    question_rev = _reverse_vocab(mappings.get("question_vocab", {}))
    weak_skill_id = int(valid_target_skills[weak_pos].detach().cpu().item())
    weak_question_id = int(valid_target_questions[weak_pos].detach().cpu().item())

    valid_len = int(mask.sum().item())
    recent = []
    start = max(0, valid_len - 10)
    q_hist = batch["question_ids"][0].detach().cpu().tolist()
    s_hist = batch["skill_ids"][0].detach().cpu().tolist()
    c_hist = batch["correct_ids"][0].detach().cpu().tolist()
    for i in range(start, valid_len):
        recent.append(
            {
                "position": i,
                "question_id": q_hist[i],
                "question": question_rev.get(q_hist[i], str(q_hist[i])),
                "skill_id": s_hist[i],
                "skill": skill_rev.get(s_hist[i], str(s_hist[i])),
                "correct": c_hist[i],
            }
        )

    top_attended = []
    attentions = out.get("attentions", [])
    if attentions:
        last_attn = attentions[-1][0].detach().cpu()
        target_history_pos = min(weak_pos, last_attn.shape[-1] - 1)
        scores = last_attn[:, target_history_pos, : valid_len].mean(dim=0)
        top = torch.topk(scores, k=min(top_k_attention, scores.numel()))
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            top_attended.append(
                {
                    "position": int(idx),
                    "attention": float(score),
                    "question": question_rev.get(q_hist[idx], str(q_hist[idx])),
                    "skill": skill_rev.get(s_hist[idx], str(s_hist[idx])),
                    "correct": c_hist[idx],
                }
            )

    explanation = {
        "student_id": batch.get("student_id", ["unknown"])[0] if isinstance(batch.get("student_id"), list) else str(batch.get("student_id", "unknown")),
        "weak_skill_id": weak_skill_id,
        "weak_skill": skill_rev.get(weak_skill_id, str(weak_skill_id)),
        "weak_question_id": weak_question_id,
        "weak_question": question_rev.get(weak_question_id, str(weak_question_id)),
        "mastery_probability": mastery_probability,
        "recent_interactions": recent,
        "top_attended_interactions": top_attended,
        "note": "Attention is an explanation hook, not a causal proof of model reasoning.",
    }
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(explanation, f, indent=2)
    return explanation


def save_sample_explanations(
    model: torch.nn.Module,
    loader: Any,
    mappings: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
    max_examples: int = 5,
) -> list[dict[str, Any]]:
    out_dir = get_path(cfg, "explanations_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    explanations = []
    for i, batch in enumerate(loader):
        if i >= max_examples:
            break
        student_id = batch.get("student_id", [f"sample_{i}"])[0]
        path = out_dir / f"student_{safe_explanation_filename(student_id, f'sample_{i}')}.json"
        explanations.append(explain_akt_batch(model, batch, mappings, device, output_path=path))
    return explanations
