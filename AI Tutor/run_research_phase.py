from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.eval.research_efficiency import benchmark_research_efficiency
from src.eval.research_metrics import (
    evaluate_student_modeling,
    summarize_retrieval_logs,
    summarize_tutoring_logs,
    summarize_twin_profiles,
    write_final_results,
)
from src.rag.agentic import AgenticRAGTutor
from src.utils.config import ensure_output_dirs, get_path, load_config
from src.xai.research_visualizations import generate_research_graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run research-improvement experiments for the AI Tutor.")
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--agentic-samples", type=int, default=None)
    parser.add_argument("--skip-agentic", action="store_true")
    parser.add_argument("--skip-efficiency", action="store_true")
    parser.add_argument("--skip-student-metrics", action="store_true")
    parser.add_argument("--build-rag-index-if-missing", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_config_snapshot(cfg: dict[str, Any], config_path: str | Path) -> Path:
    outputs_dir = get_path(cfg, "outputs_dir")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "config": cfg,
    }
    path = outputs_dir / "research_config_snapshot.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _sample_explanations(cfg: dict[str, Any], n: int) -> list[dict[str, Any]]:
    explanations_dir = get_path(cfg, "explanations_dir")
    files = sorted(explanations_dir.glob("student_*.json")) if explanations_dir.exists() else []
    examples: list[dict[str, Any]] = []
    for path in files[:n]:
        try:
            examples.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    if examples:
        return examples[:n]
    return [
        {
            "student_id": f"research_demo_{i + 1}",
            "weak_skill": skill,
            "mastery_probability": mastery,
            "recent_interactions": [
                {"skill": skill, "correct": 0},
                {"skill": skill, "correct": 1 if mastery > 0.45 else 0},
                {"skill": skill, "correct": 0},
            ],
        }
        for i, (skill, mastery) in enumerate(
            [
                ("linear equations", 0.32),
                ("algebra word problems", 0.44),
                ("fractions and ratios", 0.58),
            ][:n]
        )
    ]


def _run_agentic_samples(cfg: dict[str, Any], n: int) -> dict[str, Any]:
    pipeline = AgenticRAGTutor(cfg)
    examples = _sample_explanations(cfg, n)
    results = []
    for i, explanation in enumerate(examples, start=1):
        weak_skill = str(explanation.get("weak_skill", "the weak skill"))
        question = f"I am stuck on {weak_skill}. Can you help me without giving away the final answer?"
        result = pipeline.generate(student_question=question, explanation=explanation)
        results.append(result)
        print(
            f"Agentic sample {i}/{len(examples)}: mode={result['response_plan']['mode']} "
            f"leakage={result['leakage']['leakage_score']} repaired={result['repaired']}"
        )
    return {
        "n_agentic_samples": len(results),
        "agentic_log": str(pipeline.log_path),
        "avg_retrieved_chunks": sum(len(r.get("retrieved_chunks", [])) for r in results) / max(1, len(results)),
        "modes": sorted({r.get("response_plan", {}).get("mode", "unknown") for r in results}),
    }


def _markdown_table(rows: list[tuple[str, Any]]) -> str:
    text = ["| Metric | Value |", "|---|---|"]
    for key, value in rows:
        if isinstance(value, float):
            value = f"{value:.4f}"
        text.append(f"| {key} | {value} |")
    return "\n".join(text)


def _write_summary(cfg: dict[str, Any], results: dict[str, Any], graph_paths: list[str]) -> Path:
    outputs_dir = get_path(cfg, "outputs_dir")
    path = outputs_dir / "research_summary.md"
    student = results.get("student_modeling", {})
    tutor = results.get("tutoring_quality", {})
    retrieval = results.get("retrieval_quality", {})
    efficiency = results.get("efficiency_summary", {})
    twin = results.get("student_twin", {})
    lines = [
        "# AI Tutor Research Improvement Summary",
        "",
        "This report summarizes the research layer added on top of the AKT-first baseline: Agentic RAG, student digital twins, leakage/hallucination controls, efficiency experiments, and explainability graphs.",
        "",
        "## Student Modeling",
        _markdown_table(
            [
                ("Model", student.get("student_model_name", "AKT-style attention KT")),
                ("AUC", student.get("auc", "")),
                ("Accuracy", student.get("accuracy", "")),
                ("F1", student.get("f1", "")),
                ("Precision", student.get("precision", "")),
                ("Recall", student.get("recall", "")),
                ("ECE calibration error", student.get("ece", "")),
                ("Evaluated tokens", student.get("n_tokens", "")),
            ]
        ),
        "",
        "## Agentic RAG And Tutor Quality",
        _markdown_table(
            [
                ("Agentic samples", results.get("agentic_rag", {}).get("n_agentic_samples", 0)),
                ("Leakage rate", tutor.get("leakage_rate", 0.0)),
                ("Hallucination/unsupported rate", tutor.get("hallucination_rate", 0.0)),
                ("Regeneration rate", tutor.get("regeneration_rate", 0.0)),
                ("Retrieval grounding score", tutor.get("retrieval_grounding_score", 0.0)),
                ("Pedagogical score", tutor.get("pedagogical_score", 0.0)),
                ("Socratic question frequency", tutor.get("socratic_question_frequency", 0.0)),
                ("Average hint depth", tutor.get("average_hint_depth", 0.0)),
            ]
        ),
        "",
        "## Retrieval",
        _markdown_table(
            [
                ("Retrieval events", retrieval.get("n_retrieval_events", 0)),
                ("Average retrieval latency ms", retrieval.get("avg_retrieval_latency_ms", 0.0)),
                ("Average similarity", retrieval.get("avg_similarity_score", 0.0)),
                ("Retrieval source count", retrieval.get("retrieval_source_count", 0)),
            ]
        ),
        "",
        "## Student Twin Personalization",
        _markdown_table(
            [
                ("Student twins", twin.get("n_student_twins", 0)),
                ("Average confidence", twin.get("avg_confidence", 0.0)),
                ("Average hint dependence", twin.get("avg_hint_dependence", 0.0)),
            ]
        ),
        "",
        "## Efficiency",
        _markdown_table(
            [
                ("Benchmark rows", efficiency.get("n_rows", 0)),
                ("Best latency ms", efficiency.get("best_latency_ms", "")),
                ("Best throughput tokens/sec", efficiency.get("best_tokens_per_sec", "")),
                ("Lowest VRAM MB", efficiency.get("lowest_vram_mb", "")),
            ]
        ),
        "",
        "## Graphs",
    ]
    for graph in graph_paths:
        rel = Path(graph)
        try:
            rel = rel.relative_to(outputs_dir)
        except Exception:
            pass
        lines.append(f"- `{rel}`")
    lines.extend(
        [
            "",
            "## Key Conclusions",
            "- The baseline is now evaluated with research-grade metrics beyond AUC, including F1, precision, recall, and calibration error.",
            "- Agentic RAG records retrieval decisions, response plans, repair behavior, and grounding checks for each tutoring event.",
            "- The student twin gives personalization a persistent state that can be measured over time instead of being hidden in prompts.",
            "- TurboQuant is tracked as the target LLM KV-cache backend; the tutor can use a local TurboQuant cache path or an OpenAI-compatible TurboQuant server, while efficiency outputs keep PyTorch/ONNX, TurboVec retrieval, and LLM-cache measurements separate.",
            "- Explainability artifacts now cover AKT attention, retrieval ranking/source attribution, tutor repair decisions, and twin evolution.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _efficiency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def nums(key: str) -> list[float]:
        out = []
        for row in rows:
            try:
                if row.get(key) != "":
                    out.append(float(row.get(key)))
            except Exception:
                pass
        return out

    lat = nums("latency_ms")
    tok = nums("tokens_per_sec")
    vram = nums("peak_vram_mb")
    return {
        "n_rows": len(rows),
        "best_latency_ms": min(lat) if lat else "",
        "best_tokens_per_sec": max(tok) if tok else "",
        "lowest_vram_mb": min(vram) if vram else "",
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("research", {})
    if args.build_rag_index_if_missing:
        cfg["research"]["build_rag_index_if_missing"] = True
    ensure_output_dirs(cfg)
    outputs_dir = get_path(cfg, "outputs_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _save_config_snapshot(cfg, args.config)

    checkpoint = Path(args.checkpoint) if args.checkpoint else get_path(cfg, "checkpoints_dir") / "best_model.pt"
    baseline_eval = _load_json(outputs_dir / "final_eval.json")
    results: dict[str, Any] = {
        "run": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": str(args.config),
            "checkpoint": str(checkpoint),
            "seed": int(cfg.get("seed", 42)),
        },
        "baseline": baseline_eval,
    }

    if args.skip_student_metrics or not checkpoint.exists():
        results["student_modeling"] = baseline_eval
        if not checkpoint.exists():
            results["student_modeling_note"] = f"checkpoint not found: {checkpoint}"
    else:
        print("Evaluating research student-model metrics...")
        results["student_modeling"] = evaluate_student_modeling(cfg, checkpoint_path=checkpoint)

    if args.skip_agentic:
        results["agentic_rag"] = {"n_agentic_samples": 0, "note": "skipped by CLI flag"}
    else:
        n = args.agentic_samples if args.agentic_samples is not None else int(cfg.get("research", {}).get("agentic_samples", 3))
        print(f"Running {n} Agentic RAG tutor sample(s)...")
        results["agentic_rag"] = _run_agentic_samples(cfg, n)

    if args.skip_efficiency:
        efficiency_rows: list[dict[str, Any]] = []
        results["efficiency_summary"] = {"n_rows": 0, "note": "skipped by CLI flag"}
    else:
        print("Running research efficiency benchmarks...")
        efficiency_rows = benchmark_research_efficiency(cfg, checkpoint_path=checkpoint)
        results["efficiency_summary"] = _efficiency_summary(efficiency_rows)

    agentic_log = outputs_dir / "agentic_tutor_logs.jsonl"
    results["tutoring_quality"] = summarize_tutoring_logs([get_path(cfg, "tutor_log"), agentic_log])
    results["retrieval_quality"] = summarize_retrieval_logs([get_path(cfg, "retrieval_log"), agentic_log])
    results["student_twin"] = summarize_twin_profiles(outputs_dir / "student_twins")

    print("Generating publication graphs...")
    graph_paths = generate_research_graphs(cfg)
    results["graphs"] = graph_paths

    csv_path, json_path = write_final_results(outputs_dir, results)
    summary_path = _write_summary(cfg, results, graph_paths)
    print(f"Saved final research CSV: {csv_path}")
    print(f"Saved final research JSON: {json_path}")
    print(f"Saved research summary: {summary_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
