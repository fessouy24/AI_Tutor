from __future__ import annotations

from typing import Any

from torch import nn

from src.models.akt import AKTModel, AKTModelConfig


def build_student_model(
    model_type: str,
    n_questions: int,
    n_skills: int,
    max_seq_len: int,
    model_cfg: dict[str, Any],
) -> nn.Module:
    model_type = model_type.lower()
    common = {
        "n_questions": n_questions,
        "n_skills": n_skills,
        "max_seq_len": max_seq_len,
        "d_model": int(model_cfg.get("d_model", 128)),
        "n_layers": int(model_cfg.get("n_layers", 3)),
        "dropout": float(model_cfg.get("dropout", 0.2)),
    }
    if model_type == "akt":
        cfg = AKTModelConfig(
            **common,
            n_heads=int(model_cfg.get("n_heads", 8)),
            ff_multiplier=int(model_cfg.get("ff_multiplier", 4)),
        )
        return AKTModel(cfg)
    raise ValueError(f"Unknown student model type: {model_type}")


def model_config_dict(model: nn.Module) -> dict[str, Any]:
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        return {}
    return dict(cfg.__dict__)
