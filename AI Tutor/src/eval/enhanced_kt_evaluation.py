from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.eval.kt_metrics import (
    calculate_binary_metrics,
    plot_confusion_matrix,
    plot_model_metric_comparison,
    plot_pr_curve,
    plot_roc_curve,
    plot_unavailable,
    sigmoid,
)
from src.utils.config import ensure_output_dirs, get_path, load_config


SOURCE_STANDARDS: list[dict[str, str]] = [
    {
        "source": "Deep Knowledge Tracing.pdf",
        "standard": "Treats KT as next-response binary prediction and reports AUC.",
        "reported_metrics": "DKT AUC values include Khan Math 0.85 and Assistments 0.86; BKT on Assistments is reported around 0.67.",
        "comparability": "Not directly comparable: different dataset/setup from the current EdNet multi-source run.",
    },
    {
        "source": "A Self-Attentive model for Knowledge Tracing.pdf",
        "standard": "Uses binary classification evaluation and reports AUC for KT models.",
        "reported_metrics": "SAKT reports AUC 0.848 on ASSIST2009, 0.854 on ASSIST2015, and 0.824 average across listed datasets.",
        "comparability": "Not directly comparable: ASSIST/STATICS benchmark splits differ from the current EdNet configuration.",
    },
    {
        "source": "Context-Aware Attentive Knowledge Tracing.pdf",
        "standard": "Uses AUC for future learner response prediction and reports mean/std across folds.",
        "reported_metrics": "AKT reports AUC values including ASSISTments2009 AKT-R 0.8346, ASSISTments2015 AKT-NR 0.7828, and ASSISTments2017 AKT-R 0.7702.",
        "comparability": "Not directly comparable: the paper uses benchmark ASSISTments/Statics folds rather than this EdNet aggregation.",
    },
    {
        "source": "RKT Relation-Aware Self-Attention for Knowledge Tracing.pdf",
        "standard": "Frames prediction as binary correct/incorrect classification and reports AUC plus accuracy (ACC).",
        "reported_metrics": "The paper reports AUC and ACC tables for KT baselines and RKT.",
        "comparability": "Not directly comparable unless the same dataset split and preprocessing are reproduced.",
    },
    {
        "source": "BKT-LSTM Efficient Student Modeling for knowledge tracing and student performance prediction.pdf",
        "standard": "Reports AUC and RMSE over multiple datasets with cross-validation.",
        "reported_metrics": "Reports ASS-09 AUC: DKT 0.72, DKVMN 0.71, BKT-LSTM 0.80; Algebra AUC: DKT 0.78, BKT-LSTM 0.85.",
        "comparability": "Not directly comparable: different datasets and a different model family.",
    },
    {
        "source": "Enhancing Attention-Based Knowledge Tracing with Digital Textbook Interaction.pdf",
        "standard": "Reports AUC, accuracy, MAE, RMSE, and F1 on an 8:1:1 split with repeated test splits.",
        "reported_metrics": "Reports SAKT AUC 0.6885, ACC 0.6276, F1 0.6758; proposed model AUC 0.7016, ACC 0.6380, F1 0.6877.",
        "comparability": "Not directly comparable: small course-specific dataset with textbook interaction features.",
    },
    {
        "source": "Towards Modeling Learner Performance with Large Language Models.pdf",
        "standard": "Defines balanced accuracy, precision, recall, F1, AUC, and RMSE for learner-performance prediction.",
        "reported_metrics": "Uses classification and ranking metrics to evaluate learner prediction, including AUC and F1-style measures.",
        "comparability": "Useful for metric selection, but not directly comparable without matching datasets and prompts.",
    },
    {
        "source": "EdNet A Large-Scale Hierarchical Dataset in Education.pdf",
        "standard": "Introduces the EdNet dataset used for educational prediction/knowledge tracing research.",
        "reported_metrics": "Dataset source rather than a direct AKT/DKT metric baseline in this project report.",
        "comparability": "Relevant dataset context, but not a model-performance comparison by itself.",
    },
]

PREDICTION_CACHE_NAME = "akt_test_predictions.npz"
PREDICTION_CACHE_SUMMARY_NAME = "akt_test_predictions_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate enhanced KT evaluation artifacts and markdown report.")
    parser.add_argument("--config", default="configs/local_notebook.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--full-eval", action="store_true", help="Recompute prediction-level metrics on the full test loader.")
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Do not cache prediction-level labels/probabilities after --full-eval.",
    )
    parser.add_argument("--report-path", default=None, help="Defaults to ../evaluation_report.txt from the AI Tutor folder.")
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc)}


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _final_eval_metric_updates(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "test_loss": metrics.get("loss"),
        "test_auc": metrics.get("auc"),
        "test_accuracy": metrics.get("accuracy"),
        "test_precision": metrics.get("precision"),
        "test_recall": metrics.get("recall"),
        "test_f1": metrics.get("f1"),
        "test_pr_auc": metrics.get("pr_auc"),
        "test_tn": metrics.get("tn"),
        "test_fp": metrics.get("fp"),
        "test_fn": metrics.get("fn"),
        "test_tp": metrics.get("tp"),
        "test_tokens": metrics.get("n_tokens"),
        "test_threshold": metrics.get("threshold"),
    }


def update_final_eval_payload(outputs_dir: Path, final_eval: dict[str, Any], summary: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    final_eval_payload = dict(final_eval)
    final_eval_payload.setdefault("student_model_name", summary.get("model", "AKT-style attention KT"))
    final_eval_payload.setdefault("dataset_name", summary.get("dataset_path", "unknown"))
    final_eval_payload.update(_final_eval_metric_updates(metrics))
    (outputs_dir / "final_eval.json").write_text(json.dumps(final_eval_payload, indent=2), encoding="utf-8")
    write_csv(outputs_dir / "final_eval.csv", [final_eval_payload])
    return final_eval_payload


def save_prediction_cache(
    evaluation_dir: Path,
    labels: np.ndarray,
    probabilities: np.ndarray,
    metrics: dict[str, Any],
) -> Path:
    cache_path = evaluation_dir / PREDICTION_CACHE_NAME
    np.savez_compressed(
        cache_path,
        labels=np.asarray(labels, dtype=np.uint8),
        probabilities=np.asarray(probabilities, dtype=np.float32),
    )
    summary_path = evaluation_dir / PREDICTION_CACHE_SUMMARY_NAME
    summary_path.write_text(
        json.dumps(
            {
                "cache_path": str(cache_path),
                "n_predictions": int(labels.size),
                "metrics": _final_eval_metric_updates(metrics),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return cache_path


def load_prediction_cache(evaluation_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Path] | None:
    cache_path = evaluation_dir / PREDICTION_CACHE_NAME
    if not cache_path.exists():
        return None
    with np.load(cache_path) as data:
        labels = np.asarray(data["labels"], dtype=np.int32)
        probabilities = np.asarray(data["probabilities"], dtype=np.float64)
    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError(
            f"Prediction cache length mismatch: labels={labels.shape[0]} probabilities={probabilities.shape[0]}"
        )
    metrics = calculate_binary_metrics(labels, probabilities, n_tokens=labels.size)
    return labels, probabilities, metrics, cache_path


def fmt(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return "not available"
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if math.isnan(numeric):
        return "not available"
    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}"
    return f"{numeric:.{digits}f}"


def _status_row(model: str, status: str, reason: str) -> dict[str, Any]:
    return {
        "model": model,
        "status": status,
        "reason": reason,
        "auc": "",
        "accuracy": "",
        "precision": "",
        "recall": "",
        "f1": "",
        "loss": "",
    }


def load_akt_summary(outputs_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    summary = read_json(outputs_dir / "akt_training_summary.json")
    training_log = read_json(outputs_dir / "training_log.json")
    final_eval = read_json(outputs_dir / "final_eval.json")
    return summary, training_log, final_eval


def akt_metrics_from_saved(summary: dict[str, Any], training_log: dict[str, Any], final_eval: dict[str, Any]) -> dict[str, Any]:
    test_metrics = dict(summary.get("test_metrics") or training_log.get("test_metrics") or {})
    if final_eval:
        final_map = {
            "auc": final_eval.get("test_auc"),
            "accuracy": final_eval.get("test_accuracy"),
            "precision": final_eval.get("test_precision"),
            "recall": final_eval.get("test_recall"),
            "f1": final_eval.get("test_f1"),
            "pr_auc": final_eval.get("test_pr_auc"),
            "loss": final_eval.get("test_loss"),
            "n_tokens": final_eval.get("test_tokens"),
            "tn": final_eval.get("test_tn"),
            "fp": final_eval.get("test_fp"),
            "fn": final_eval.get("test_fn"),
            "tp": final_eval.get("test_tp"),
        }
        for key, value in final_map.items():
            if value not in ("", None):
                test_metrics[key] = value
    return {
        "model": "AKT",
        "status": "ok" if test_metrics else "missing",
        "dataset": summary.get("dataset_path", final_eval.get("dataset_name", "")),
        "auc": test_metrics.get("auc", ""),
        "accuracy": test_metrics.get("accuracy", ""),
        "precision": test_metrics.get("precision", ""),
        "recall": test_metrics.get("recall", ""),
        "f1": test_metrics.get("f1", ""),
        "pr_auc": test_metrics.get("pr_auc", ""),
        "loss": test_metrics.get("loss", ""),
        "n_tokens": test_metrics.get("n_tokens", ""),
        "tn": test_metrics.get("tn", ""),
        "fp": test_metrics.get("fp", ""),
        "fn": test_metrics.get("fn", ""),
        "tp": test_metrics.get("tp", ""),
    }


def discover_dkt_artifacts(project_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for root_name in ["outputs", "checkpoints", "logs"]:
        root = project_dir / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and "dkt" in path.name.lower():
                candidates.append(path)
    return sorted(candidates)


def collect_test_predictions(cfg: dict[str, Any], checkpoint_path: Path | None) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    import torch
    from torch import nn

    from src.data.knowledge_tracing import build_dataloaders
    from src.models.checkpointing import load_checkpoint
    from src.models.factory import build_student_model
    from src.models.training import masked_bce_loss, move_batch_to_device
    from src.utils.reproducibility import get_device, set_seed

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(prefer_cuda=True)
    loaders, _payload = build_dataloaders(cfg)
    checkpoint = checkpoint_path or get_path(cfg, "checkpoints_dir") / "best_model.pt"
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
    logits_all: list[float] = []
    labels_all: list[float] = []
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loaders["test"]:
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
            logits_all.extend(logits[mask].detach().cpu().tolist())
            labels_all.extend(batch["labels"][mask].detach().cpu().tolist())

    labels = np.asarray(labels_all, dtype=np.int32)
    probabilities = sigmoid(logits_all)
    metrics = calculate_binary_metrics(labels, probabilities, loss=total_loss / max(total_tokens, 1), n_tokens=total_tokens)
    return metrics, labels, probabilities


def source_standards_from_folder(sources_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    warnings: list[str] = []
    found: list[dict[str, str]] = []
    if not sources_dir.exists():
        return [], [f"Sources folder was not found: {sources_dir}"]

    existing = {path.name.lower(): path for path in sources_dir.glob("*.pdf")}
    for row in SOURCE_STANDARDS:
        source_name = row["source"]
        path = existing.get(source_name.lower())
        if path is None:
            warnings.append(f"Source not found: {source_name}")
            continue
        found.append({**row, "path": str(path)})
    return found, warnings


def dataset_statistics(summary: dict[str, Any], training_log: dict[str, Any]) -> dict[str, Any]:
    source_counts = summary.get("source_counts") or {}
    split_source_counts = summary.get("split_source_counts") or {}
    test_metrics = summary.get("test_metrics") or training_log.get("test_metrics") or {}
    return {
        "dataset_path": summary.get("dataset_path", "unknown"),
        "n_questions": summary.get("n_questions", "unknown"),
        "n_skills": summary.get("n_skills", "unknown"),
        "source_counts": source_counts,
        "split_source_counts": split_source_counts,
        "total_sequences": sum(int(v) for v in source_counts.values()) if source_counts else "",
        "test_tokens": test_metrics.get("n_tokens", ""),
        "dataloader_build_seconds": summary.get("dataloader_build_seconds", ""),
        "epochs_ran": summary.get("epochs_ran", len(training_log.get("history", []))),
    }


def create_evaluation_plots(
    evaluation_dir: Path,
    akt_row: dict[str, Any],
    model_rows: list[dict[str, Any]],
    labels: np.ndarray | None,
    probabilities: np.ndarray | None,
) -> list[Path]:
    graph_paths: list[Path] = []
    has_confusion = all(akt_row.get(key) not in ("", None) for key in ["tn", "fp", "fn", "tp"])
    if has_confusion:
        cm = np.array([[int(float(akt_row["tn"])), int(float(akt_row["fp"]))], [int(float(akt_row["fn"])), int(float(akt_row["tp"]))]])
        graph_paths.append(plot_confusion_matrix(cm, evaluation_dir / "akt_confusion_matrix.png", "AKT Final Test Confusion Matrix"))
    else:
        graph_paths.append(
            plot_unavailable(
                evaluation_dir / "akt_confusion_matrix.png",
                "AKT Final Test Confusion Matrix",
                "The current EdNet run saved AUC, accuracy, BCE loss, and token count, but not prediction-level labels. Rerun enhanced evaluation with --full-eval or rerun final_eval.py after the metric update to generate this plot.",
            )
        )

    if labels is not None and probabilities is not None:
        graph_paths.append(plot_roc_curve(labels, probabilities, evaluation_dir / "akt_roc_curve.png", "AKT Final Test ROC Curve"))
        graph_paths.append(plot_pr_curve(labels, probabilities, evaluation_dir / "akt_pr_curve.png", "AKT Final Test PR Curve"))
    else:
        graph_paths.append(
            plot_unavailable(
                evaluation_dir / "akt_roc_curve.png",
                "AKT Final Test ROC Curve",
                "ROC needs saved labels and predicted probabilities. They were not stored by the previous EdNet run.",
            )
        )
        graph_paths.append(
            plot_unavailable(
                evaluation_dir / "akt_pr_curve.png",
                "AKT Final Test PR Curve",
                "Precision-recall needs saved labels and predicted probabilities. They were not stored by the previous EdNet run.",
            )
        )

    graph_paths.append(plot_model_metric_comparison(model_rows, evaluation_dir / "model_metric_comparison.png"))
    return graph_paths


def write_report(
    report_path: Path,
    stats: dict[str, Any],
    akt_row: dict[str, Any],
    dkt_artifacts: list[Path],
    source_rows: list[dict[str, str]],
    graph_paths: list[Path],
    csv_paths: dict[str, Path],
    warnings: list[str],
) -> Path:
    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append("## Dataset Statistics")
    lines.append(f"- Dataset/run: `{stats['dataset_path']}`")
    lines.append(f"- Questions: {fmt(stats['n_questions'], 0)}")
    lines.append(f"- Skills: {fmt(stats['n_skills'], 0)}")
    lines.append(f"- Total source sequences: {fmt(stats['total_sequences'], 0)}")
    lines.append(f"- Final test tokens: {fmt(stats['test_tokens'], 0)}")
    lines.append(f"- Epochs completed: {fmt(stats['epochs_ran'], 0)}")
    lines.append(f"- Dataloader build time: {fmt(stats['dataloader_build_seconds'], 1)} seconds")
    if stats["source_counts"]:
        lines.append("")
        lines.append("| Source | Sequences |")
        lines.append("|---|---:|")
        for source, count in stats["source_counts"].items():
            lines.append(f"| {source} | {int(count):,} |")

    lines.append("")
    lines.append("## AKT Final Metrics")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key, label in [
        ("auc", "AUC"),
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
        ("pr_auc", "PR-AUC"),
        ("loss", "BCE loss"),
        ("n_tokens", "Evaluated tokens"),
    ]:
        lines.append(f"| {label} | {fmt(akt_row.get(key))} |")

    lines.append("")
    lines.append("## DKT Final Metrics")
    if dkt_artifacts:
        lines.append("DKT-looking artifacts were found, but automatic DKT evaluation is only run when a compatible DKT model implementation/checkpoint is present.")
        for path in dkt_artifacts:
            lines.append(f"- `{path}`")
    else:
        lines.append("No DKT training code, checkpoint, or final metrics were found in `AI Tutor`. The comparison therefore reports AKT only and does not invent DKT results.")

    lines.append("")
    lines.append("## Graph Paths")
    for path in graph_paths:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("CSV outputs:")
    for label, path in csv_paths.items():
        lines.append(f"- {label}: `{path}`")

    lines.append("")
    lines.append("## Confusion Matrix Explanation")
    lines.append("The confusion matrix counts true incorrect, false correct, false incorrect, and true correct predictions at the selected probability threshold, currently 0.5. It is useful because accuracy alone can hide whether the model mostly misses correct answers or mostly over-predicts correctness.")

    lines.append("")
    lines.append("## ROC/AUC Explanation")
    lines.append("The ROC curve plots true-positive rate against false-positive rate across thresholds. AUC summarizes ranking quality: 0.5 is random ranking and 1.0 is perfect ranking. It is a standard KT metric because KT is usually framed as predicting whether the next response will be correct.")

    lines.append("")
    lines.append("## Source Paper Comparison")
    lines.append("| Source | Evaluation standard / reported metrics | Fairness note |")
    lines.append("|---|---|---|")
    for row in source_rows:
        standard = row["standard"] + " " + row["reported_metrics"]
        lines.append(f"| `{row['source']}` | {standard} | {row['comparability']} |")

    lines.append("")
    lines.append("Current AKT AUC is " + fmt(akt_row.get("auc")) + " and accuracy is " + fmt(akt_row.get("accuracy")) + ". These can be discussed next to the papers as context, but most source results are not directly comparable because they use different datasets, preprocessing, folds, model variants, and feature sets.")

    lines.append("")
    lines.append("## Recommendations")
    lines.append("- Rerun `src/eval/final_eval.py` or `src/eval/enhanced_kt_evaluation.py --full-eval` after this update when you can afford the full EdNet dataloader time, so precision, recall, F1, PR-AUC, and confusion counts are computed on the exact final test split.")
    lines.append("- Add a real DKT baseline implementation/checkpoint if AKT-vs-DKT comparison is required in the thesis.")
    lines.append("- Store prediction-level labels/probabilities, or at least confusion counts, at the end of training to avoid expensive full-test recomputation.")
    lines.append("- Report comparisons against papers as not directly comparable unless the same dataset split and preprocessing are reproduced.")
    lines.append("- Consider threshold tuning on the validation set if recall or precision matters more for tutoring interventions.")

    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    outputs_dir = get_path(cfg, "outputs_dir")
    evaluation_dir = outputs_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    main_dir = ROOT.parent
    report_path = Path(args.report_path) if args.report_path else main_dir / "evaluation_report.txt"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None

    summary, training_log, final_eval = load_akt_summary(outputs_dir)
    warnings: list[str] = []
    if not summary:
        warnings.append("AKT training summary was not found; report uses whatever final_eval/training_log data exists.")
    if final_eval and any(key not in final_eval for key in ["test_precision", "test_recall", "test_f1"]):
        warnings.append("Current final_eval.json was created before precision/recall/F1 were added.")

    labels: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    if args.full_eval:
        try:
            full_metrics, labels, probabilities = collect_test_predictions(cfg, checkpoint_path)
            final_eval = update_final_eval_payload(outputs_dir, final_eval, summary, full_metrics)
            if not args.no_save_predictions:
                save_prediction_cache(evaluation_dir, labels, probabilities, full_metrics)
        except Exception as exc:
            warnings.append(f"Full prediction-level evaluation failed, so saved metrics were used instead: {exc}")

    if labels is None or probabilities is None:
        try:
            cached_predictions = load_prediction_cache(evaluation_dir)
        except Exception as exc:
            cached_predictions = None
            warnings.append(f"Cached prediction-level evaluation could not be loaded: {exc}")
        if cached_predictions is not None:
            labels, probabilities, cached_metrics, cache_path = cached_predictions
            final_eval = update_final_eval_payload(outputs_dir, final_eval, summary, cached_metrics)
            if not args.full_eval:
                warnings.append(
                    f"Loaded cached prediction-level labels/probabilities from {cache_path.name}; "
                    "run with --full-eval to refresh them after a new checkpoint."
                )
        elif not args.full_eval:
            warnings.append(
                "Full prediction-level EdNet evaluation was not rerun and no cached predictions were found; "
                "previous logs do not contain labels/probabilities for ROC/PR curves."
            )

    akt_row = akt_metrics_from_saved(summary, training_log, final_eval)
    dkt_artifacts = discover_dkt_artifacts(ROOT)
    model_rows = [akt_row]
    if dkt_artifacts:
        model_rows.append(_status_row("DKT", "unsupported", "DKT artifacts found but no compatible DKT model loader exists."))

    source_rows, source_warnings = source_standards_from_folder(main_dir / "Sources")
    warnings.extend(source_warnings)
    stats = dataset_statistics(summary, training_log)
    graph_paths = create_evaluation_plots(evaluation_dir, akt_row, model_rows, labels, probabilities)

    csv_paths = {
        "final test metrics": write_csv(evaluation_dir / "final_test_metrics.csv", [akt_row]),
        "model comparison": write_csv(evaluation_dir / "model_comparison_metrics.csv", model_rows),
        "source standards": write_csv(evaluation_dir / "source_evaluation_standards.csv", source_rows),
    }
    (evaluation_dir / "warnings.txt").write_text("\n".join(warnings) + "\n", encoding="utf-8")
    report = write_report(report_path, stats, akt_row, dkt_artifacts, source_rows, graph_paths, csv_paths, warnings)

    print(f"Saved enhanced evaluation report: {report}")
    print(f"Saved enhanced evaluation outputs: {evaluation_dir}")
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
