from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.eval.kt_metrics import metrics_from_logits
from src.models.checkpointing import checkpoint_model_mismatches, load_checkpoint, restore_training_state, save_checkpoint
from src.models.optim import EagerAdamW, build_scheduler
from src.utils.config import get_path
from src.utils.memory import print_gpu_memory


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    raw = criterion(logits, labels)
    weights = mask.float()
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def _metrics_from_logits(logits: list[float], labels: list[float]) -> dict[str, float]:
    if not labels:
        return {
            "auc": float("nan"),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "pr_auc": float("nan"),
        }
    return metrics_from_logits(logits, labels)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: Any, device: torch.device) -> dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    total_loss = 0.0
    total_tokens = 0
    all_logits: list[float] = []
    all_labels: list[float] = []
    for batch in loader:
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
    metrics = _metrics_from_logits(all_logits, all_labels)
    metrics["loss"] = total_loss / max(total_tokens, 1)
    metrics["n_tokens"] = float(total_tokens)
    return metrics


def _write_metrics_csv(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    keys = sorted({key for row in history for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def _write_training_json(path: Path, history: list[dict[str, Any]], extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"history": history, **extra}, f, indent=2)


def _write_resume_skip(path: Path, resume_path: str | Path, reason: str, details: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "resume_path": str(resume_path),
        "reason": reason,
        "details": details or [],
        "action": "started fresh training run",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def train_model(
    model: nn.Module,
    loaders: dict[str, Any],
    cfg: dict[str, Any],
    mappings: dict[str, Any],
    device: torch.device,
    resume: str | Path | None = None,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, float]]:
    train_cfg = cfg.get("training", {})
    seed = int(cfg.get("seed", 42))
    checkpoints_dir = get_path(cfg, "checkpoints_dir")
    outputs_dir = get_path(cfg, "outputs_dir")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    optimizer = EagerAdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(train_cfg.get("epochs", 20))
    scheduler = build_scheduler(optimizer, train_cfg.get("scheduler", "cosine"), epochs)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    use_amp = bool(train_cfg.get("mixed_precision", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_val_auc = float("-inf")
    history: list[dict[str, Any]] = []
    bad_epochs = 0
    if resume:
        try:
            checkpoint = load_checkpoint(resume, map_location="cpu")
            mismatches = checkpoint_model_mismatches(checkpoint, model)
            if mismatches:
                preview = "; ".join(mismatches[:6])
                print(
                    f"Skipping incompatible resume checkpoint: {resume}. "
                    f"{len(mismatches)} mismatch(es): {preview}"
                )
                _write_resume_skip(outputs_dir / "resume_skipped.json", resume, "model_state_mismatch", mismatches)
            else:
                start_epoch, best_val_auc, history, bad_epochs = restore_training_state(
                    checkpoint, model, optimizer, scheduler
                )
                print(f"Resumed from {resume} at epoch {start_epoch}; best_val_auc={best_val_auc:.5f}")
        except Exception as exc:
            print(f"Skipping resume checkpoint because it could not be restored: {resume}. Reason: {exc}")
            _write_resume_skip(outputs_dir / "resume_skipped.json", resume, "restore_failed", [str(exc)])

    patience = int(train_cfg.get("patience", 5))
    grad_clip = train_cfg.get("grad_clip", 1.0)
    last_path = checkpoints_dir / "last_checkpoint.pt"
    best_path = checkpoints_dir / "best_model.pt"
    metrics_path = outputs_dir / "metrics.csv"
    log_path = outputs_dir / "training_log.json"

    print(
        f"Training started: entering epoch loop at epoch {start_epoch}; "
        f"epoch logs below include training plus validation time."
    )
    print_gpu_memory("before training")
    for epoch in range(start_epoch, epochs):
        model.train()
        start = time.perf_counter()
        running_loss = 0.0
        running_tokens = 0

        for batch in loaders["train"]:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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

            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()

            n = int(batch["mask"].sum().item())
            running_loss += float(loss.item()) * n
            running_tokens += n

        train_loss = running_loss / max(running_tokens, 1)
        val_metrics = evaluate_model(model, loaders["val"], device)
        if scheduler is not None:
            if scheduler.__class__.__name__ == "PlateauScheduler":
                scheduler.step(val_metrics["auc"] if not math.isnan(val_metrics["auc"]) else -1.0)
            else:
                scheduler.step()

        elapsed = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} time={elapsed:.1f}s"
        )

        val_auc = val_metrics["auc"]
        improved = not math.isnan(val_auc) and val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_val_auc, cfg, seed, mappings, history, bad_epochs)
            print(f"  saved best checkpoint: {best_path}")
        else:
            bad_epochs += 1

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_val_auc, cfg, seed, mappings, history, bad_epochs)
        _write_metrics_csv(metrics_path, history)
        _write_training_json(log_path, history, {"best_val_auc": best_val_auc, "last_epoch": epoch})
        print(f"  saved last checkpoint: {last_path}")

        if bad_epochs >= patience:
            print(f"Early stopping after {bad_epochs} epochs without validation AUC improvement.")
            break

    test_metrics = evaluate_model(model, loaders["test"], device) if "test" in loaders else {}
    _write_training_json(log_path, history, {"best_val_auc": best_val_auc, "test_metrics": test_metrics})
    print_gpu_memory("after training")
    return model, history, test_metrics
