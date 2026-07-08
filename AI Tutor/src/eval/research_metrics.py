from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn

from src.data.knowledge_tracing import build_dataloaders
from src.models.checkpointing import load_checkpoint
from src.models.factory import build_student_model
from src.models.training import masked_bce_loss, move_batch_to_device
from src.utils.config import get_path
from src.utils.reproducibility import get_device, set_seed


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    labels = labels.astype(np.float32)
    probabilities = probabilities.astype(np.float32)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lo) & (probabilities < hi if hi < 1.0 else probabilities <= hi)
        if not np.any(mask):
            continue
        confidence = float(np.mean(probabilities[mask]))
        accuracy = float(np.mean(labels[mask]))
        ece += float(np.mean(mask)) * abs(confidence - accuracy)
    return float(ece)


@torch.no_grad()
def evaluate_student_modeling(
    cfg: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """Research-grade student-model metrics beyond AUC and accuracy."""

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(prefer_cuda=True)
    loaders, payload = build_dataloaders(cfg)
    checkpoint = Path(checkpoint_path) if checkpoint_path else get_path(cfg, "checkpoints_dir") / "best_model.pt"
    ckpt = load_checkpoint(checkpoint, map_location=device)
    model_cfg = ckpt["model_config"]
    model = build_student_model(
        ckpt.get("model_type", "akt"),
        n_questions=int(model_cfg["n_questions"]),
        n_skills=int(model_cfg["n_skills"]),
        max_seq_len=int(model_cfg["max_seq_len"]),
        model_cfg=model_cfg,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    all_logits: list[float] = []
    all_labels: list[float] = []
    total_loss = 0.0
    total_tokens = 0
    for batch in loaders[split]:
        batch = move_batch_to_device(batch, device)
        out = model(
            batch["question_ids"],
            batch["skill_ids"],
            batch["correct_ids"],
            batch["target_question_ids"],
            batch["target_skill_ids"],
            input_mask=batch["mask"],
        )
        logits = out["logits"]
        loss = masked_bce_loss(logits, batch["labels"], batch["mask"], criterion)
        mask = batch["mask"].bool()
        n = int(mask.sum().item())
        total_loss += float(loss.item()) * n
        total_tokens += n
        all_logits.extend(logits[mask].detach().cpu().tolist())
        all_labels.extend(batch["labels"][mask].detach().cpu().tolist())

    y = np.asarray(all_labels, dtype=np.int32)
    logits_np = np.asarray(all_logits, dtype=np.float32)
    probabilities = 1.0 / (1.0 + np.exp(-logits_np))
    preds = (probabilities >= 0.5).astype(np.int32)
    metrics = {
        "student_model_name": "AKT-style attention KT" if ckpt.get("model_type", "akt") == "akt" else ckpt.get("model_type"),
        "dataset_path": payload.get("dataset_path", ""),
        "split": split,
        "loss": total_loss / max(total_tokens, 1),
        "auc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "ece": expected_calibration_error(y, probabilities),
        "n_tokens": int(total_tokens),
        "n_questions": int(payload.get("n_questions", 0)),
        "n_skills": int(payload.get("n_skills", 0)),
    }
    return metrics


def summarize_tutoring_logs(paths: list[str | Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    if not rows:
        return {
            "n_tutor_events": 0,
            "leakage_rate": 0.0,
            "hallucination_rate": 0.0,
            "regeneration_rate": 0.0,
            "retrieval_grounding_score": 0.0,
            "pedagogical_score": 0.0,
            "socratic_question_frequency": 0.0,
            "average_hint_depth": 0.0,
            "avg_tutor_latency_ms": 0.0,
        }

    leakage = []
    unsupported = []
    repaired = []
    grounding = []
    questions = []
    hint_depth = []
    latencies = []
    for row in rows:
        leakage.append(float(row.get("leakage", {}).get("leakage_score", 0)))
        faith = row.get("faithfulness", {}) or {}
        unsupported.append(0.0 if faith.get("supported", False) else 1.0)
        grounding.append(float(faith.get("average_overlap", 0.0)))
        repaired.append(float(bool(row.get("repaired", False))))
        text = str(row.get("response", ""))
        questions.append(float("?" in text))
        plan = row.get("response_plan", {}) or {}
        hint_depth.append(float(plan.get("hint_depth", 1)))
        latencies.append(float(row.get("tutor_latency_ms", row.get("latency_ms", 0.0))))

    grounding_score = float(np.mean(grounding))
    leakage_rate = float(np.mean(leakage))
    hallucination_rate = float(np.mean(unsupported))
    socratic = float(np.mean(questions))
    pedagogical = float(np.clip(0.40 * socratic + 0.35 * (1.0 - leakage_rate) + 0.25 * (1.0 - hallucination_rate), 0.0, 1.0))
    return {
        "n_tutor_events": len(rows),
        "leakage_rate": leakage_rate,
        "hallucination_rate": hallucination_rate,
        "regeneration_rate": float(np.mean(repaired)),
        "retrieval_grounding_score": grounding_score,
        "pedagogical_score": pedagogical,
        "socratic_question_frequency": socratic,
        "average_hint_depth": float(np.mean(hint_depth)),
        "avg_tutor_latency_ms": float(np.mean(latencies)),
    }


def summarize_retrieval_logs(paths: list[str | Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    scores: list[float] = []
    latencies: list[float] = []
    sources: dict[str, int] = {}
    for row in rows:
        latencies.append(float(row.get("retrieval_latency_ms", row.get("latency_ms", 0.0))))
        chunks = row.get("retrieved_chunks", row.get("chunks", [])) or []
        for chunk in chunks:
            score = chunk.get("score")
            if score is not None:
                scores.append(float(score))
            source = str(chunk.get("source", "unknown") or "unknown")
            sources[source] = sources.get(source, 0) + 1
    return {
        "n_retrieval_events": len(rows),
        "avg_retrieval_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "avg_similarity_score": float(np.mean(scores)) if scores else 0.0,
        "p50_similarity_score": float(np.median(scores)) if scores else 0.0,
        "retrieval_source_count": len(sources),
        "top_retrieval_sources": dict(sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]),
    }


def summarize_twin_profiles(profiles_dir: str | Path) -> dict[str, Any]:
    profiles_dir = Path(profiles_dir)
    profiles = []
    if profiles_dir.exists():
        for path in profiles_dir.glob("*.json"):
            if path.name == "twin_snapshots.jsonl":
                continue
            try:
                profiles.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
    if not profiles:
        return {"n_student_twins": 0, "avg_confidence": 0.0, "avg_hint_dependence": 0.0, "top_weak_topics": {}}
    topic_scores: dict[str, float] = {}
    for profile in profiles:
        for topic, score in (profile.get("weak_topics") or {}).items():
            topic_scores[topic] = max(topic_scores.get(topic, 0.0), float(score))
    return {
        "n_student_twins": len(profiles),
        "avg_confidence": float(np.mean([float(p.get("confidence", 0.0)) for p in profiles])),
        "avg_hint_dependence": float(np.mean([float(p.get("hint_dependence", 0.0)) for p in profiles])),
        "top_weak_topics": dict(sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:10]),
    }


def write_final_results(outputs_dir: str | Path, results: dict[str, Any]) -> tuple[Path, Path]:
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    json_path = outputs_dir / "final_results.json"
    csv_path = outputs_dir / "final_results.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    flat: dict[str, Any] = {}
    for section, payload in results.items():
        if isinstance(payload, dict):
            for key, value in payload.items():
                flat[f"{section}.{key}"] = json.dumps(value) if isinstance(value, (dict, list)) else value
        else:
            flat[section] = payload
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
    return csv_path, json_path
