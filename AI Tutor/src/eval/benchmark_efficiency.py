from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.models.checkpointing import load_checkpoint
from src.models.factory import build_student_model
from src.tutor.generator import SocraticTutor
from src.utils.config import ensure_output_dirs, get_path, load_config
from src.utils.memory import clear_gpu_memory
from src.utils.reproducibility import get_device, set_seed


def _peak_vram() -> float:
    return torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0


def _dummy_inputs(model_cfg: dict[str, Any], batch: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    seq_len = int(model_cfg["max_seq_len"])
    n_questions = int(model_cfg["n_questions"])
    n_skills = int(model_cfg["n_skills"])
    return (
        torch.randint(1, n_questions + 1, (batch, seq_len), dtype=torch.long, device=device),
        torch.randint(1, n_skills + 1, (batch, seq_len), dtype=torch.long, device=device),
        torch.randint(0, 2, (batch, seq_len), dtype=torch.long, device=device),
        torch.randint(1, n_questions + 1, (batch, seq_len), dtype=torch.long, device=device),
        torch.randint(1, n_skills + 1, (batch, seq_len), dtype=torch.long, device=device),
        torch.ones(batch, seq_len, dtype=torch.bool, device=device),
    )


def benchmark_akt(cfg: dict[str, Any], checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model_cfg = ckpt["model_config"]
    model = build_student_model("akt", model_cfg["n_questions"], model_cfg["n_skills"], model_cfg["max_seq_len"], model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    inputs = _dummy_inputs(model_cfg, batch=1, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(*inputs[:5], input_mask=inputs[5])
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(int(cfg.get("benchmark", {}).get("akt_batches", 20))):
            model(*inputs[:5], input_mask=inputs[5])
        if device.type == "cuda":
            torch.cuda.synchronize()
    latency = ((time.perf_counter() - start) / int(cfg.get("benchmark", {}).get("akt_batches", 20))) * 1000
    row = {
        "model_name": "AKT-style KT",
        "mode": f"pytorch-{device.type}",
        "precision": "fp16-mixed" if device.type == "cuda" else "fp32",
        "latency_ms": latency,
        "peak_vram_mb": _peak_vram(),
        "tokens_per_sec": "",
        "notes": "Sequence-level student model inference.",
    }
    clear_gpu_memory(model, label="AKT benchmark cleanup")
    return row


def benchmark_onnx(cfg: dict[str, Any]) -> dict[str, Any] | None:
    model_path = get_path(cfg, "outputs_dir") / "onnx" / "akt_model.onnx"
    if not model_path.exists():
        return None
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
    n = int(cfg.get("benchmark", {}).get("onnx_batches", 50))
    start = time.perf_counter()
    for _ in range(n):
        session.run(["logits"], inputs)
    return {
        "model_name": "AKT-style KT",
        "mode": "onnxruntime-cpu",
        "precision": "fp32",
        "latency_ms": ((time.perf_counter() - start) / n) * 1000,
        "peak_vram_mb": 0.0,
        "tokens_per_sec": "",
        "notes": str(model_path),
    }


def benchmark_tutor(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    local_cfg = {**cfg, "tutor": {**cfg.get("tutor", {}), "mode": mode}}
    tutor = SocraticTutor(local_cfg)
    start = time.perf_counter()
    result = tutor.generate(
        student_question="How do I solve a simple linear equation?",
        weak_skill="linear equations",
        mastery_probability=0.42,
        retrieved_chunks=[],
    )
    elapsed = time.perf_counter() - start
    text = result["response"]
    tokens = max(1, len(text.split()))
    return {
        "model_name": local_cfg.get("tutor", {}).get("openai_model" if mode == "api" else "local_model", mode),
        "mode": mode,
        "precision": local_cfg.get("tutor", {}).get("quantization", "api" if mode == "api" else "unknown"),
        "latency_ms": elapsed * 1000,
        "peak_vram_mb": _peak_vram(),
        "tokens_per_sec": tokens / max(elapsed, 1e-6),
        "notes": "Tutor generation benchmark.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark realistic AI Tutor efficiency options.")
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(prefer_cuda=True)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else get_path(cfg, "checkpoints_dir") / "best_model.pt"

    rows: list[dict[str, Any]] = []
    rows.append(benchmark_akt(cfg, checkpoint_path, device))
    onnx_row = benchmark_onnx(cfg)
    if onnx_row:
        rows.append(onnx_row)
    if cfg.get("benchmark", {}).get("run_local_llm", False):
        rows.append(benchmark_tutor(cfg, "local"))
    if cfg.get("benchmark", {}).get("run_api_llm", False) and os.environ.get("OPENAI_API_KEY"):
        rows.append(benchmark_tutor(cfg, "api"))

    out_path = get_path(cfg, "outputs_dir") / "efficiency_benchmark.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fields = ["model_name", "mode", "precision", "latency_ms", "peak_vram_mb", "tokens_per_sec", "notes"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved efficiency benchmark: {out_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
