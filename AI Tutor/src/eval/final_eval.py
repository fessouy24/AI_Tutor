from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.knowledge_tracing import build_dataloaders
from src.models.checkpointing import load_checkpoint
from src.models.factory import build_student_model
from src.models.training import evaluate_model
from src.utils.config import ensure_output_dirs, get_path, load_config
from src.utils.reproducibility import get_device, set_seed

import numpy as np
import torch


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _tutor_metrics(path: Path) -> dict[str, float]:
    rows = _read_jsonl(path)
    if not rows:
        return {"leakage_rate": 0.0, "repair_rate": 0.0, "avg_tutor_latency_ms": 0.0}
    leakage = [r.get("leakage", {}).get("leakage_score", 0) for r in rows]
    repairs = [1 if r.get("repaired") else 0 for r in rows]
    latencies = [float(r.get("latency_ms", 0.0)) for r in rows]
    return {
        "leakage_rate": float(np.mean(leakage)),
        "repair_rate": float(np.mean(repairs)),
        "avg_tutor_latency_ms": float(np.mean(latencies)),
    }


def _retrieval_metrics(path: Path) -> dict[str, float]:
    rows = _read_jsonl(path)
    if not rows:
        return {"avg_retrieval_latency_ms": 0.0, "average_retrieved_chunks_used": 0.0}
    return {
        "avg_retrieval_latency_ms": float(np.mean([float(r.get("latency_ms", 0.0)) for r in rows])),
        "average_retrieved_chunks_used": float(np.mean([float(r.get("n_chunks", 0.0)) for r in rows])),
    }


def _onnx_latency(cfg: dict[str, Any]) -> float:
    model_path = get_path(cfg, "outputs_dir") / "onnx" / "akt_model.onnx"
    if not model_path.exists():
        return 0.0
    try:
        import onnxruntime as ort

        seq_len = int(cfg.get("data", {}).get("max_seq_len", 200))
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = {
            "question_ids": np.ones((1, seq_len), dtype=np.int64),
            "skill_ids": np.ones((1, seq_len), dtype=np.int64),
            "correct_ids": np.zeros((1, seq_len), dtype=np.int64),
            "target_question_ids": np.ones((1, seq_len), dtype=np.int64),
            "target_skill_ids": np.ones((1, seq_len), dtype=np.int64),
            "input_mask": np.ones((1, seq_len), dtype=bool),
        }
        for _ in range(5):
            session.run(["logits"], inputs)
        start = time.perf_counter()
        for _ in range(30):
            session.run(["logits"], inputs)
        return ((time.perf_counter() - start) / 30) * 1000
    except Exception as exc:
        print(f"ONNX latency skipped: {exc}")
        return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final AI Tutor evaluation report.")
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path; defaults to best_model.pt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(prefer_cuda=True)

    loaders, payload = build_dataloaders(cfg)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else get_path(cfg, "checkpoints_dir") / "best_model.pt"
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model_cfg = ckpt["model_config"]
    model = build_student_model(
        ckpt.get("model_type", "akt"),
        n_questions=int(model_cfg["n_questions"]),
        n_skills=int(model_cfg["n_skills"]),
        max_seq_len=int(model_cfg["max_seq_len"]),
        model_cfg=model_cfg,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    test = evaluate_model(model, loaders["test"], device)
    history = ckpt.get("history", [])
    last = history[-1] if history else {}
    tutor = _tutor_metrics(get_path(cfg, "tutor_log"))
    retrieval = _retrieval_metrics(get_path(cfg, "retrieval_log"))
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    row = {
        "student_model_name": "AKT-style attention KT" if ckpt.get("model_type", "akt") == "akt" else ckpt.get("model_type"),
        "dataset_name": cfg.get("data", {}).get("dataset_name", "unknown"),
        "train_loss": last.get("train_loss", ""),
        "val_loss": last.get("val_loss", ""),
        "test_loss": test.get("loss", ""),
        "test_auc": test.get("auc", ""),
        "test_accuracy": test.get("accuracy", ""),
        "test_precision": test.get("precision", ""),
        "test_recall": test.get("recall", ""),
        "test_f1": test.get("f1", ""),
        "test_pr_auc": test.get("pr_auc", ""),
        "test_tn": test.get("tn", ""),
        "test_fp": test.get("fp", ""),
        "test_fn": test.get("fn", ""),
        "test_tp": test.get("tp", ""),
        "test_tokens": test.get("n_tokens", ""),
        "leakage_rate": tutor["leakage_rate"],
        "repair_rate": tutor["repair_rate"],
        "rag_top_k": cfg.get("rag", {}).get("top_k", cfg.get("evaluation", {}).get("rag_top_k", "")),
        "avg_retrieval_latency_ms": retrieval["avg_retrieval_latency_ms"],
        "avg_tutor_latency_ms": tutor["avg_tutor_latency_ms"],
        "peak_vram_mb": peak_vram,
        "onnx_latency_ms": _onnx_latency(cfg),
        "notes": cfg.get("evaluation", {}).get("notes", ""),
    }
    outputs_dir = get_path(cfg, "outputs_dir")
    csv_path = outputs_dir / "final_eval.csv"
    json_path = outputs_dir / "final_eval.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    print(f"Saved final evaluation: {csv_path}")
    print(f"Saved final evaluation: {json_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
