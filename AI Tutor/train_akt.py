from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from src.data.knowledge_tracing import build_dataloaders
from src.models.factory import build_student_model
from src.models.training import train_model
from src.utils.config import ensure_output_dirs, get_path, load_config
from src.utils.logging import setup_logging
from src.utils.memory import clear_gpu_memory, print_gpu_memory
from src.utils.reproducibility import get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AKT-style Knowledge Tracing baseline.")
    parser.add_argument("--config", default="configs/local.yaml", help="Path to config YAML.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("model", {})["type"] = "akt"
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size

    ensure_output_dirs(cfg)
    logger = setup_logging(get_path(cfg, "logs_dir"), name="train_akt")
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(prefer_cuda=True)
    logger.info("Training AKT baseline on device=%s", device)
    print_gpu_memory("startup")

    logger.info(
        "Training has not started yet. Building KT dataloaders and preprocessing sources now; "
        "[multi-KT] logs below are data loading, not epoch training."
    )
    dataloader_start = time.perf_counter()
    loaders, payload = build_dataloaders(cfg)
    dataloader_seconds = time.perf_counter() - dataloader_start
    logger.info(
        "KT preprocessing/dataloader build finished in %.1fs (%.2f min). "
        "Training is starting now. train_batches=%s val_batches=%s test_batches=%s dataset=%s",
        dataloader_seconds,
        dataloader_seconds / 60.0,
        len(loaders.get("train", [])),
        len(loaders.get("val", [])),
        len(loaders.get("test", [])),
        payload.get("dataset_path"),
    )
    model = build_student_model(
        "akt",
        n_questions=int(payload["n_questions"]),
        n_skills=int(payload["n_skills"]),
        max_seq_len=int(cfg.get("data", {}).get("max_seq_len", 200)),
        model_cfg=cfg.get("model", {}),
    )
    logger.info("AKT parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    _, history, test_metrics = train_model(model, loaders, cfg, payload["mappings"], device, resume=args.resume)
    outputs_dir = get_path(cfg, "outputs_dir")
    summary_path = outputs_dir / "akt_training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "AKT-style attention KT",
                "dataset_path": payload["dataset_path"],
                "n_questions": payload["n_questions"],
                "n_skills": payload["n_skills"],
                "source_counts": payload.get("source_counts", {}),
                "split_source_counts": payload.get("split_source_counts", {}),
                "source_reports": payload.get("source_reports", []),
                "dataloader_build_seconds": dataloader_seconds,
                "epochs_ran": len(history),
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
        )
    logger.info("Saved AKT training summary to %s", summary_path)
    clear_gpu_memory(model, label="AKT training cleanup")


if __name__ == "__main__":
    main()
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
