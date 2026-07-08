from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models.factory import model_config_dict


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    epoch: int,
    best_val_auc: float,
    config: dict[str, Any],
    seed: int,
    mappings: dict[str, Any],
    history: list[dict[str, Any]],
    bad_epochs: int = 0,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_val_auc": best_val_auc,
        "config": config,
        "random_seed": seed,
        "vocabulary_mappings": mappings,
        "model_config": model_config_dict(model),
        "model_type": config.get("model", {}).get("type", "akt"),
        "history": history,
        "bad_epochs": bad_epochs,
        "rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["rng_cuda"] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def checkpoint_model_mismatches(checkpoint: dict[str, Any], model: nn.Module) -> list[str]:
    """Return state-dict incompatibilities that make a checkpoint unsafe to resume."""
    checkpoint_state = checkpoint.get("model_state_dict", {})
    model_state = model.state_dict()
    mismatches: list[str] = []
    for key, current_value in model_state.items():
        saved_value = checkpoint_state.get(key)
        if saved_value is None:
            mismatches.append(f"missing key in checkpoint: {key}")
            continue
        if tuple(saved_value.shape) != tuple(current_value.shape):
            mismatches.append(
                f"size mismatch for {key}: checkpoint {tuple(saved_value.shape)} vs current {tuple(current_value.shape)}"
            )
    for key in checkpoint_state.keys() - model_state.keys():
        mismatches.append(f"unexpected checkpoint key: {key}")
    return mismatches


def restore_training_state(
    checkpoint: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    strict: bool = True,
) -> tuple[int, float, list[dict[str, Any]], int]:
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("rng_state") is not None:
        rng_state = checkpoint["rng_state"]
        if torch.is_tensor(rng_state):
            rng_state = rng_state.cpu()
        torch.set_rng_state(rng_state)
    if torch.cuda.is_available() and checkpoint.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all([state.cpu() if torch.is_tensor(state) else state for state in checkpoint["rng_cuda"]])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_val_auc = float(checkpoint.get("best_val_auc", float("-inf")))
    history = list(checkpoint.get("history", []))
    bad_epochs = int(checkpoint.get("bad_epochs", 0))
    return start_epoch, best_val_auc, history, bad_epochs
