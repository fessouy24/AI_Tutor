from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.models.checkpointing import load_checkpoint
from src.models.factory import build_student_model
from src.rag.index import RetrievedChunk
from src.tutor.generator import SocraticTutor
from src.utils.config import get_path
from src.utils.memory import clear_gpu_memory


def _peak_vram() -> float:
    return torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0


def _model_memory_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)


def _dummy_inputs(model_cfg: dict[str, Any], batch: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, ...]:
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


def _load_akt(cfg: dict[str, Any], checkpoint_path: Path, device: torch.device, dtype: torch.dtype | None = None) -> tuple[nn.Module, dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model_cfg = ckpt["model_config"]
    model = build_student_model("akt", model_cfg["n_questions"], model_cfg["n_skills"], model_cfg["max_seq_len"], model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    if dtype is not None and device.type == "cuda":
        model = model.to(dtype=dtype)
    model.eval()
    return model, model_cfg


def _benchmark_torch_model(
    model: nn.Module,
    model_cfg: dict[str, Any],
    device: torch.device,
    precision: str,
    seq_len: int | None = None,
    n_runs: int = 50,
) -> dict[str, Any]:
    seq_len = int(seq_len or model_cfg["max_seq_len"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    inputs = _dummy_inputs(model_cfg, batch=1, seq_len=seq_len, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(*inputs[:5], input_mask=inputs[5])
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            model(*inputs[:5], input_mask=inputs[5])
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / max(1, n_runs)
    return {
        "model_name": "AKT-style KT",
        "mode": f"pytorch-{device.type}",
        "precision": precision,
        "latency_ms": elapsed * 1000,
        "peak_vram_mb": _peak_vram(),
        "tokens_per_sec": seq_len / max(elapsed, 1e-9),
        "quality_proxy": "",
        "leakage_rate": "",
        "retrieval_quality": "",
        "hallucination_rate": "",
        "notes": f"seq_len={seq_len}; model_memory_mb={_model_memory_mb(model):.2f}",
    }


def _benchmark_onnx(cfg: dict[str, Any], seq_len: int, n_runs: int) -> dict[str, Any] | None:
    model_path = get_path(cfg, "outputs_dir") / "onnx" / "akt_model.onnx"
    if not model_path.exists():
        return {
            "model_name": "AKT-style KT",
            "mode": "onnxruntime-cpu",
            "precision": "fp32",
            "latency_ms": "",
            "peak_vram_mb": 0.0,
            "tokens_per_sec": "",
            "quality_proxy": "",
            "leakage_rate": "",
            "retrieval_quality": "",
            "hallucination_rate": "",
            "notes": "skipped: ONNX model not found; run src/deployment/export_onnx.py first",
        }
    try:
        import onnxruntime as ort

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
        for _ in range(n_runs):
            session.run(["logits"], inputs)
        elapsed = (time.perf_counter() - start) / max(1, n_runs)
        return {
            "model_name": "AKT-style KT",
            "mode": "onnxruntime-cpu",
            "precision": "fp32-optimized",
            "latency_ms": elapsed * 1000,
            "peak_vram_mb": 0.0,
            "tokens_per_sec": seq_len / max(elapsed, 1e-9),
            "quality_proxy": "same logits target as PyTorch export",
            "leakage_rate": "",
            "retrieval_quality": "",
            "hallucination_rate": "",
            "notes": str(model_path),
        }
    except Exception as exc:
        return {
            "model_name": "AKT-style KT",
            "mode": "onnxruntime-cpu",
            "precision": "fp32-optimized",
            "latency_ms": "",
            "peak_vram_mb": 0.0,
            "tokens_per_sec": "",
            "quality_proxy": "",
            "leakage_rate": "",
            "retrieval_quality": "",
            "hallucination_rate": "",
            "notes": f"skipped: {exc}",
        }


def _benchmark_context_modes(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    tutor_cfg = {**cfg, "tutor": {**cfg.get("tutor", {}), "mode": "template"}}
    tutor = SocraticTutor(tutor_cfg)
    long_text = (
        "A linear equation keeps both sides balanced. A useful hint is to isolate the variable by undoing operations "
        "in reverse order. The student should identify the variable term, constants, and the operation that can be undone. "
    ) * 12
    rows: list[dict[str, Any]] = []
    for label, chars in [("full_context", 1600), ("reduced_context", 420), ("compressed_chunks", 260)]:
        chunk = RetrievedChunk(text=long_text[:chars], source=f"{label}.txt", chunk_id=label, score=0.85)
        start = time.perf_counter()
        result = tutor.generate(
            student_question="How do I solve this equation without being told the final answer?",
            weak_skill="linear equations",
            mastery_probability=0.42,
            retrieved_chunks=[chunk],
        )
        elapsed = time.perf_counter() - start
        words = max(1, len(result.get("response", "").split()))
        faith = result.get("faithfulness", {})
        rows.append(
            {
                "model_name": "Tutor prompt/RAG context",
                "mode": label,
                "precision": "context-ablation",
                "latency_ms": elapsed * 1000,
                "peak_vram_mb": _peak_vram(),
                "tokens_per_sec": words / max(elapsed, 1e-9),
                "quality_proxy": float(faith.get("average_overlap", 0.0)),
                "leakage_rate": float(result.get("leakage", {}).get("leakage_score", 0)),
                "retrieval_quality": float(chunk.score or 0.0),
                "hallucination_rate": 0.0 if faith.get("supported", False) else 1.0,
                "notes": f"context_chars={chars}; template tutor benchmark",
            }
        )
    return rows


def _skipped_llm_rows() -> list[dict[str, Any]]:
    base = {
        "model_name": "Local tutor LLM",
        "latency_ms": "",
        "peak_vram_mb": "",
        "tokens_per_sec": "",
        "quality_proxy": "",
        "leakage_rate": "",
        "retrieval_quality": "",
        "hallucination_rate": "",
    }
    return [
        {**base, "mode": "turboquant", "precision": "turboquant-kv", "notes": "skipped by default; enable benchmark.run_local_llm to load the local TurboQuant cache backend, or configure tutor.turboquant_base_url for a server"},
        {**base, "mode": "local", "precision": "8-bit", "notes": "skipped by default; enable benchmark.run_local_llm for a heavy local LLM run"},
        {**base, "mode": "local", "precision": "4-bit-nf4", "notes": "skipped by default; requires bitsandbytes + compatible CUDA GPU"},
        {**base, "mode": "kv-cache-stress", "precision": "fp16/turboquant", "notes": "skipped by default; enable when a local or TurboQuant LLM backend is available"},
    ]


def benchmark_research_efficiency(
    cfg: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    research_cfg = cfg.get("research", {})
    n_runs = int(research_cfg.get("benchmark_runs", 30))
    seq_len = int(cfg.get("data", {}).get("max_seq_len", 200))
    checkpoint = Path(checkpoint_path) if checkpoint_path else get_path(cfg, "checkpoints_dir") / "best_model.pt"
    rows: list[dict[str, Any]] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = _load_akt(cfg, checkpoint, device)
    rows.append(_benchmark_torch_model(model, model_cfg, device, "fp32" if device.type == "cpu" else "fp32/autocast-off", n_runs=n_runs))
    clear_gpu_memory(model, label="research fp32 benchmark")

    if torch.cuda.is_available():
        model, model_cfg = _load_akt(cfg, checkpoint, torch.device("cuda"), dtype=torch.float16)
        rows.append(_benchmark_torch_model(model, model_cfg, torch.device("cuda"), "fp16", n_runs=n_runs))
        clear_gpu_memory(model, label="research fp16 benchmark")
    else:
        rows.append(
            {
                "model_name": "AKT-style KT",
                "mode": "pytorch-cuda",
                "precision": "fp16",
                "latency_ms": "",
                "peak_vram_mb": 0.0,
                "tokens_per_sec": "",
                "quality_proxy": "",
                "leakage_rate": "",
                "retrieval_quality": "",
                "hallucination_rate": "",
                "notes": "skipped: CUDA unavailable",
            }
        )

    try:
        cpu_model, model_cfg = _load_akt(cfg, checkpoint, torch.device("cpu"))
        q_model = torch.quantization.quantize_dynamic(cpu_model, {nn.Linear}, dtype=torch.qint8)
        rows.append(_benchmark_torch_model(q_model, model_cfg, torch.device("cpu"), "dynamic-int8", n_runs=max(10, n_runs // 2)))
        clear_gpu_memory(q_model, label="research int8 benchmark")
    except Exception as exc:
        rows.append(
            {
                "model_name": "AKT-style KT",
                "mode": "pytorch-cpu",
                "precision": "dynamic-int8",
                "latency_ms": "",
                "peak_vram_mb": 0.0,
                "tokens_per_sec": "",
                "quality_proxy": "",
                "leakage_rate": "",
                "retrieval_quality": "",
                "hallucination_rate": "",
                "notes": f"skipped: {exc}",
            }
        )

    rows.append(_benchmark_onnx(cfg, seq_len=seq_len, n_runs=max(10, n_runs)))
    rows.extend(_benchmark_context_modes(cfg))
    if not cfg.get("benchmark", {}).get("run_local_llm", False):
        rows.extend(_skipped_llm_rows())

    out = Path(output_path) if output_path else get_path(cfg, "outputs_dir") / "research_efficiency.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_name",
        "mode",
        "precision",
        "latency_ms",
        "peak_vram_mb",
        "tokens_per_sec",
        "quality_proxy",
        "leakage_rate",
        "retrieval_quality",
        "hallucination_rate",
        "notes",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows
