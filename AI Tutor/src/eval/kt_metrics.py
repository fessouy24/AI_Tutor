from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def sigmoid(logits: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def calculate_binary_metrics(
    labels: list[float] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    *,
    threshold: float = 0.5,
    loss: float | None = None,
    n_tokens: int | float | None = None,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int32)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.size == 0:
        return {
            "auc": float("nan"),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "pr_auc": float("nan"),
            "loss": float("nan") if loss is None else float(loss),
            "n_tokens": 0.0 if n_tokens is None else float(n_tokens),
            "threshold": float(threshold),
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    preds = (p >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    unique_labels = np.unique(y)
    auc = float(roc_auc_score(y, p)) if unique_labels.size > 1 else float("nan")
    pr_auc = float(average_precision_score(y, p)) if unique_labels.size > 1 else float("nan")
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "pr_auc": pr_auc,
        "loss": float("nan") if loss is None else float(loss),
        "n_tokens": float(y.size if n_tokens is None else n_tokens),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def metrics_from_logits(
    logits: list[float] | np.ndarray,
    labels: list[float] | np.ndarray,
    *,
    threshold: float = 0.5,
    loss: float | None = None,
    n_tokens: int | float | None = None,
) -> dict[str, Any]:
    return calculate_binary_metrics(
        labels,
        sigmoid(logits),
        threshold=threshold,
        loss=loss,
        n_tokens=n_tokens,
    )


def ensure_plot_dir(path: str | Path) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_or_close(fig: plt.Figure, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_unavailable(path: str | Path, title: str, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=12)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=10)
    return _save_or_close(fig, path)


def plot_confusion_matrix(cm: np.ndarray, path: str | Path, title: str = "Confusion Matrix") -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["Incorrect", "Correct"])
    ax.set_yticks([0, 1], labels=["Incorrect", "Correct"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return _save_or_close(fig, path)


def plot_roc_curve(labels: np.ndarray, probabilities: np.ndarray, path: str | Path, title: str = "ROC Curve") -> Path:
    if np.unique(labels).size < 2:
        return plot_unavailable(path, title, "ROC curve requires both positive and negative labels.")
    fpr, tpr, _ = roc_curve(labels, probabilities)
    auc = roc_auc_score(labels, probabilities)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="#2563eb", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    return _save_or_close(fig, path)


def plot_pr_curve(labels: np.ndarray, probabilities: np.ndarray, path: str | Path, title: str = "Precision-Recall Curve") -> Path:
    if np.unique(labels).size < 2:
        return plot_unavailable(path, title, "Precision-recall curve requires both positive and negative labels.")
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    ap = average_precision_score(labels, probabilities)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot(recall, precision, label=f"Average precision = {ap:.4f}", color="#0f766e", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    return _save_or_close(fig, path)


def plot_model_metric_comparison(
    rows: list[dict[str, Any]],
    path: str | Path,
    metrics: list[str] | None = None,
    title: str = "Final Test Metric Comparison",
) -> Path:
    metrics = metrics or ["auc", "accuracy", "precision", "recall", "f1"]
    available_rows = [row for row in rows if row.get("status", "ok") == "ok"]
    if not available_rows:
        return plot_unavailable(path, title, "No model metrics were available for comparison.")

    labels = [str(row.get("model", row.get("student_model_name", "model"))) for row in available_rows]
    values = np.array(
        [
            [float(row[m]) if row.get(m) not in ("", None) and not np.isnan(float(row[m])) else np.nan for m in metrics]
            for row in available_rows
        ],
        dtype=float,
    )
    x = np.arange(len(metrics))
    width = min(0.8 / max(len(labels), 1), 0.32)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for idx, label in enumerate(labels):
        offset = (idx - (len(labels) - 1) / 2) * width
        ax.bar(x + offset, values[idx], width=width, label=label)
    ax.set_title(title)
    ax.set_xticks(x, labels=[m.upper() if m == "auc" else m.title() for m in metrics])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    if np.isnan(values).any():
        ax.text(
            0.5,
            -0.20,
            "Blank bars mean that metric was not stored for that model run.",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
        )
    return _save_or_close(fig, path)
