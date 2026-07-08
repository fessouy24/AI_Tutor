from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.research_metrics import read_jsonl
from src.utils.config import get_path


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def _save(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path)


def _blank(path: Path, title: str, message: str) -> str:
    plt.figure(figsize=(7, 3.5))
    plt.axis("off")
    plt.title(title)
    plt.text(0.02, 0.55, message, wrap=True, va="center")
    return _save(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _training_graphs(outputs_dir: Path, graphs_dir: Path) -> list[str]:
    paths: list[str] = []
    df = _read_csv(outputs_dir / "metrics.csv")
    if df.empty or "epoch" not in df.columns:
        return [_blank(graphs_dir / "training_metrics_missing.png", "Training Metrics", "No metrics.csv file was available.")]

    plt.figure(figsize=(8, 4.5))
    if "train_loss" in df:
        plt.plot(df["epoch"], df["train_loss"], marker="o", label="train loss")
    if "val_loss" in df:
        plt.plot(df["epoch"], df["val_loss"], marker="o", label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("AKT Training And Validation Loss")
    plt.legend()
    paths.append(_save(graphs_dir / "training_loss_curves.png"))

    plt.figure(figsize=(8, 4.5))
    if "val_auc" in df:
        plt.plot(df["epoch"], df["val_auc"], marker="o", label="validation AUC")
    if "val_accuracy" in df:
        plt.plot(df["epoch"], df["val_accuracy"], marker="o", label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.title("AKT Validation Quality")
    plt.legend()
    paths.append(_save(graphs_dir / "training_auc_accuracy.png"))
    return paths


def _efficiency_graphs(outputs_dir: Path, graphs_dir: Path) -> list[str]:
    paths: list[str] = []
    df = _read_csv(outputs_dir / "research_efficiency.csv")
    if df.empty:
        df = _read_csv(outputs_dir / "efficiency_benchmark.csv")
    if df.empty:
        return [_blank(graphs_dir / "efficiency_missing.png", "Efficiency Benchmarks", "No efficiency benchmark CSV was available.")]

    label = (df.get("mode", pd.Series([""] * len(df))).astype(str) + "\n" + df.get("precision", pd.Series([""] * len(df))).astype(str)).str.slice(0, 34)
    for column, title, filename, ylabel in [
        ("latency_ms", "Inference Latency Comparison", "efficiency_latency.png", "Latency (ms)"),
        ("peak_vram_mb", "Peak VRAM Comparison", "efficiency_vram.png", "Peak VRAM (MB)"),
        ("tokens_per_sec", "Throughput Comparison", "efficiency_tokens_per_sec.png", "Tokens/sec"),
    ]:
        values = pd.to_numeric(df.get(column, pd.Series(dtype=float)), errors="coerce")
        valid = values.notna()
        if not valid.any():
            paths.append(_blank(graphs_dir / filename, title, f"No numeric {column} values were available."))
            continue
        plt.figure(figsize=(9, 4.8))
        plt.bar(label[valid], values[valid])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel(ylabel)
        plt.title(title)
        paths.append(_save(graphs_dir / filename))
    return paths


def _retrieval_graphs(cfg: dict[str, Any], outputs_dir: Path, graphs_dir: Path) -> list[str]:
    paths: list[str] = []
    logs = read_jsonl(get_path(cfg, "retrieval_log"))
    agentic_path = outputs_dir / "agentic_tutor_logs.jsonl"
    logs.extend(read_jsonl(agentic_path))
    scores: list[float] = []
    source_counts: dict[str, int] = {}
    for row in logs:
        chunks = row.get("retrieved_chunks", row.get("chunks", [])) or []
        for chunk in chunks:
            score = chunk.get("score")
            if score is not None:
                scores.append(float(score))
            source = Path(str(chunk.get("source", "unknown"))).name or "unknown"
            source_counts[source] = source_counts.get(source, 0) + 1

    if scores:
        plt.figure(figsize=(7, 4.2))
        plt.hist(scores, bins=12, color="#4777b3", edgecolor="white")
        plt.xlabel("Similarity Score")
        plt.ylabel("Chunk Count")
        plt.title("Retrieval Similarity Distribution")
        paths.append(_save(graphs_dir / "retrieval_similarity_distribution.png"))
    else:
        paths.append(_blank(graphs_dir / "retrieval_similarity_distribution.png", "Retrieval Similarity Distribution", "No retrieval scores were logged."))

    if source_counts:
        top = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:12]
        plt.figure(figsize=(9, 4.8))
        plt.bar([k[:30] for k, _ in top], [v for _, v in top])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Retrieved Chunks")
        plt.title("Retrieval Source Frequency")
        paths.append(_save(graphs_dir / "retrieval_source_frequency.png"))
    else:
        paths.append(_blank(graphs_dir / "retrieval_source_frequency.png", "Retrieval Source Frequency", "No retrieval source counts were logged."))
    return paths


def _leakage_graphs(outputs_dir: Path, graphs_dir: Path) -> list[str]:
    rows = read_jsonl(outputs_dir / "agentic_tutor_logs.jsonl")
    rows.extend(read_jsonl(outputs_dir / "tutor_logs.jsonl"))
    if not rows:
        return [_blank(graphs_dir / "leakage_hallucination_rates.png", "Leakage And Hallucination", "No tutor logs were available.")]
    leakage = np.mean([float(r.get("leakage", {}).get("leakage_score", 0)) for r in rows])
    repair = np.mean([float(bool(r.get("repaired", False))) for r in rows])
    unsupported = np.mean([0.0 if (r.get("faithfulness", {}) or {}).get("supported", False) else 1.0 for r in rows])
    regen = repair
    plt.figure(figsize=(7, 4.2))
    plt.bar(["Leakage", "Repair", "Regeneration", "Unsupported"], [leakage, repair, regen, unsupported], color=["#b34d4d", "#4f8a5b", "#6f79b8", "#c49a45"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Rate")
    plt.title("Leakage, Repair, And Unsupported Response Rates")
    return [_save(graphs_dir / "leakage_hallucination_rates.png")]


def _twin_graphs(outputs_dir: Path, graphs_dir: Path) -> list[str]:
    snapshots = read_jsonl(outputs_dir / "student_twins" / "twin_snapshots.jsonl")
    if not snapshots:
        return [_blank(graphs_dir / "student_twin_evolution.png", "Student Twin Evolution", "No student twin snapshots were available.")]
    df = pd.DataFrame(snapshots)
    if "interactions" not in df:
        return [_blank(graphs_dir / "student_twin_evolution.png", "Student Twin Evolution", "Twin snapshots were incomplete.")]
    plt.figure(figsize=(8, 4.5))
    for column in ["confidence", "hint_dependence", "difficulty_tolerance", "speed_of_mastery"]:
        if column in df:
            plt.plot(df["interactions"], pd.to_numeric(df[column], errors="coerce"), marker="o", label=column.replace("_", " "))
    plt.xlabel("Interaction")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.title("Student Twin Profile Evolution")
    plt.legend()
    return [_save(graphs_dir / "student_twin_evolution.png")]


def _xai_graphs(outputs_dir: Path, graphs_dir: Path) -> list[str]:
    explanations_dir = outputs_dir / "explanations"
    files = sorted(explanations_dir.glob("student_*.json")) if explanations_dir.exists() else []
    if not files:
        return [_blank(graphs_dir / "xai_attention_top_interactions.png", "AKT Attention Explanation", "No explanation JSON files were available.")]

    def _usable_label(value: Any) -> str:
        text = str(value or "").strip()
        return "" if text in {"", "__missing__", "nan", "None"} else text

    def _label(item: dict[str, Any]) -> str:
        position = item.get("position")
        skill = _usable_label(item.get("skill"))
        question = _usable_label(item.get("question"))
        label = skill or question or f"position {position}"
        return f"pos {position} | {label}"[:42] if position is not None else label[:42]

    best_data: dict[str, Any] | None = None
    best_score: tuple[int, int, float] | None = None
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            continue
        top = data.get("top_attended_interactions", [])
        labels = [_label(item) for item in top if isinstance(item, dict)]
        score = (sum("__missing__" not in label for label in labels), len(set(labels)), file.stat().st_mtime)
        if best_score is None or score > best_score:
            best_data = data
            best_score = score

    if best_data is None:
        return [_blank(graphs_dir / "xai_attention_top_interactions.png", "AKT Attention Explanation", "Could not read explanation JSON.")]

    data = best_data
    top = data.get("top_attended_interactions", [])
    if not top:
        return [_blank(graphs_dir / "xai_attention_top_interactions.png", "AKT Attention Explanation", "No attention weights were stored in the explanation.")]
    labels = [_label(item) for item in top]
    values = [float(item.get("attention", 0.0)) for item in top]
    plt.figure(figsize=(8, 4.2))
    plt.barh(labels[::-1], values[::-1])
    plt.xlabel("Attention Weight")
    plt.title("Top AKT Attended Previous Interactions")
    return [_save(graphs_dir / "xai_attention_top_interactions.png")]


def generate_research_graphs(cfg: dict[str, Any]) -> list[str]:
    _style()
    outputs_dir = get_path(cfg, "outputs_dir")
    graphs_dir = outputs_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    paths.extend(_training_graphs(outputs_dir, graphs_dir))
    paths.extend(_efficiency_graphs(outputs_dir, graphs_dir))
    paths.extend(_retrieval_graphs(cfg, outputs_dir, graphs_dir))
    paths.extend(_leakage_graphs(outputs_dir, graphs_dir))
    paths.extend(_twin_graphs(outputs_dir, graphs_dir))
    paths.extend(_xai_graphs(outputs_dir, graphs_dir))
    manifest = graphs_dir / "graphs_manifest.json"
    manifest.write_text(json.dumps({"graphs": paths}, indent=2), encoding="utf-8")
    return paths
