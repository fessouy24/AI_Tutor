from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.knowledge_tracing import _find_column, _first_token, build_dataloaders
from src.models.checkpointing import load_checkpoint
from src.models.factory import build_student_model
from src.utils.config import ensure_output_dirs, first_existing_path, get_path, load_config
from src.utils.reproducibility import get_device, set_seed
from src.xai.explain import explain_akt_batch, safe_explanation_filename, save_sample_explanations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AKT explanation JSON files.")
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument(
        "--full-dataloader",
        action="store_true",
        help="Use the original full processed-cache dataloader path instead of lightweight raw-file sampling.",
    )
    return parser.parse_args()


def _source_enabled(source: dict[str, Any]) -> bool:
    return bool(source.get("enabled", True))


def _source_path(source: dict[str, Any]) -> Path | None:
    candidates: list[str | Path] = []
    if source.get("path"):
        candidates.append(source["path"])
    candidates.extend(source.get("path_candidates") or [])
    return first_existing_path(candidates)


def _find_ednet_metadata(source_path: Path, source: dict[str, Any]) -> Path | None:
    candidates: list[str | Path] = []
    if source.get("correct_answer_path"):
        candidates.append(source["correct_answer_path"])
    candidates.extend(source.get("correct_answer_path_candidates") or [])
    found = first_existing_path(candidates)
    if found:
        return found

    for root in [source_path, source_path.parent, source_path.parent.parent]:
        for name in ["questions.csv", "contents/questions.csv", "EdNet-Contents/contents/questions.csv"]:
            candidate = root / name
            if candidate.exists():
                return candidate
    return None


def _load_ednet_answer_maps(metadata_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    meta = pd.read_csv(metadata_path, low_memory=False)
    q_col = _find_column(list(meta.columns), ["question_id", "item_id", "question"])
    a_col = _find_column(list(meta.columns), ["correct_answer", "answer", "correct"])
    s_col = _find_column(list(meta.columns), ["tags", "skill_id", "concept_id", "kc_id"])
    if not q_col or not a_col:
        raise ValueError(f"EdNet metadata is missing question_id/correct_answer columns: {metadata_path}")

    correct_map = dict(zip(meta[q_col].astype(str), meta[a_col].astype(str).str.strip().str.lower()))
    skill_map = dict(zip(meta[q_col].astype(str), meta[s_col].fillna("").astype(str).map(_first_token))) if s_col else {}
    return correct_map, skill_map


def _iter_ednet_student_files(path: Path, source: dict[str, Any]):
    if path.is_file():
        if path.suffix.lower() == ".csv":
            yield path
        return

    patterns = source.get("file_patterns") or ["*.csv"]
    seen: set[Path] = set()
    for pattern in patterns:
        if "csv" not in str(pattern).lower():
            continue
        for file_path in path.rglob(str(pattern)):
            if not file_path.is_file() or file_path in seen:
                continue
            if file_path.name.lower() == "questions.csv":
                continue
            seen.add(file_path)
            yield file_path


def _parse_ednet_student_file(
    file_path: Path,
    source_name: str,
    question_vocab: dict[str, int],
    skill_vocab: dict[str, int],
    correct_map: dict[str, str],
    skill_map: dict[str, str],
    min_seq_len: int,
) -> dict[str, Any] | None:
    df = pd.read_csv(file_path, low_memory=False)
    columns = list(df.columns)
    item_col = _find_column(columns, ["item_id", "question_id"])
    ans_col = _find_column(columns, ["user_answer", "answer"])
    time_col = _find_column(columns, ["timestamp", "time"])
    action_col = _find_column(columns, ["action_type"])
    if not item_col or not ans_col:
        return None

    work = df
    if action_col:
        work = work[work[action_col].astype(str).str.lower().eq("respond")]
    work = work.dropna(subset=[item_col, ans_col])
    if time_col:
        work = work.sort_values(time_col, kind="mergesort")

    question_ids: list[int] = []
    skill_ids: list[int] = []
    corrects: list[int] = []
    for _, row in work.iterrows():
        raw_q = str(row[item_col])
        correct_answer = correct_map.get(raw_q)
        if correct_answer is None:
            continue

        question_key = f"{source_name}::{raw_q}"
        question_id = question_vocab.get(question_key)
        if question_id is None:
            continue

        skill = skill_map.get(raw_q, raw_q) or raw_q
        skill_key = f"{source_name}::{skill}"
        skill_id = skill_vocab.get(skill_key) or skill_vocab.get(f"{source_name}::{raw_q}")
        if skill_id is None:
            continue

        question_ids.append(int(question_id))
        skill_ids.append(int(skill_id))
        corrects.append(1 if str(row[ans_col]).strip().lower() == correct_answer else 0)

    if len(question_ids) < min_seq_len + 1:
        return None
    return {
        "student_id": f"{source_name}::{file_path.stem}",
        "questions": question_ids,
        "skills": skill_ids,
        "corrects": corrects,
    }


def _sequence_to_batch(sequence: dict[str, Any], max_seq_len: int) -> dict[str, Any]:
    questions = list(sequence["questions"])[-(max_seq_len + 1) :]
    skills = list(sequence["skills"])[-(max_seq_len + 1) :]
    corrects = list(sequence["corrects"])[-(max_seq_len + 1) :]
    length = min(len(questions) - 1, max_seq_len)
    if length <= 0:
        raise ValueError("Cannot explain an empty sequence.")

    q_in = torch.zeros((1, max_seq_len), dtype=torch.long)
    s_in = torch.zeros((1, max_seq_len), dtype=torch.long)
    c_in = torch.zeros((1, max_seq_len), dtype=torch.long)
    q_tgt = torch.zeros((1, max_seq_len), dtype=torch.long)
    s_tgt = torch.zeros((1, max_seq_len), dtype=torch.long)
    labels = torch.zeros((1, max_seq_len), dtype=torch.float32)
    mask = torch.zeros((1, max_seq_len), dtype=torch.bool)

    q_in[0, :length] = torch.tensor(questions[:length], dtype=torch.long)
    s_in[0, :length] = torch.tensor(skills[:length], dtype=torch.long)
    c_in[0, :length] = torch.tensor(corrects[:length], dtype=torch.long)
    q_tgt[0, :length] = torch.tensor(questions[1 : length + 1], dtype=torch.long)
    s_tgt[0, :length] = torch.tensor(skills[1 : length + 1], dtype=torch.long)
    labels[0, :length] = torch.tensor(corrects[1 : length + 1], dtype=torch.float32)
    mask[0, :length] = True

    return {
        "student_id": [str(sequence.get("student_id", "sample"))],
        "question_ids": q_in,
        "skill_ids": s_in,
        "correct_ids": c_in,
        "target_question_ids": q_tgt,
        "target_skill_ids": s_tgt,
        "labels": labels,
        "mask": mask,
    }


def _save_lightweight_ednet_explanations(
    model: torch.nn.Module,
    mappings: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
    max_examples: int,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    question_vocab = mappings.get("question_vocab") or {}
    skill_vocab = mappings.get("skill_vocab") or {}
    if not question_vocab or not skill_vocab:
        return []

    out_dir = get_path(cfg, "explanations_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg.get("data", {})
    min_seq_len = int(data_cfg.get("min_seq_len", 3))
    scan_limit = int(data_cfg.get("xai_scan_limit", 5000) or 5000)
    explanations: list[dict[str, Any]] = []

    for source in data_cfg.get("kt_sources", []):
        if len(explanations) >= max_examples:
            break
        if not isinstance(source, dict) or not _source_enabled(source):
            continue

        source_name = str(source.get("name", "kt_source"))
        source_path = _source_path(source)
        if source_path is None:
            print(f"[xai] skipping {source_name}: source path not found", flush=True)
            continue

        metadata_path = _find_ednet_metadata(source_path, source)
        if metadata_path is None:
            print(f"[xai] skipping {source_name}: EdNet questions metadata not found", flush=True)
            continue

        print(f"[xai] sampling {source_name} directly from raw files", flush=True)
        correct_map, skill_map = _load_ednet_answer_maps(metadata_path)
        scanned = 0
        for file_path in _iter_ednet_student_files(source_path, source):
            if len(explanations) >= max_examples:
                break
            scanned += 1
            if scanned > scan_limit:
                print(f"[xai] reached scan limit for {source_name}: {scan_limit} files", flush=True)
                break
            try:
                sequence = _parse_ednet_student_file(
                    file_path,
                    source_name,
                    question_vocab,
                    skill_vocab,
                    correct_map,
                    skill_map,
                    min_seq_len,
                )
            except Exception as exc:
                print(f"[xai] skipping unreadable file {file_path}: {exc}", flush=True)
                continue
            if sequence is None:
                continue

            batch = _sequence_to_batch(sequence, max_seq_len)
            student_id = batch["student_id"][0]
            output_path = out_dir / f"student_{safe_explanation_filename(student_id)}.json"
            explanations.append(explain_akt_batch(model, batch, mappings, device, output_path=output_path))

    return explanations


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(prefer_cuda=True)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else get_path(cfg, "checkpoints_dir") / "best_model.pt"
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model_cfg = ckpt["model_config"]
    model = build_student_model("akt", model_cfg["n_questions"], model_cfg["n_skills"], model_cfg["max_seq_len"], model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    explanations: list[dict[str, Any]]
    checkpoint_mappings = ckpt.get("vocabulary_mappings") or {}
    is_multi_source = bool(cfg.get("data", {}).get("use_multi_source") or cfg.get("data", {}).get("kt_sources"))
    if is_multi_source and not args.full_dataloader:
        explanations = _save_lightweight_ednet_explanations(
            model,
            checkpoint_mappings,
            cfg,
            device,
            args.max_examples,
            int(model_cfg["max_seq_len"]),
        )
        if not explanations:
            raise RuntimeError(
                "Could not create lightweight XAI samples from raw EdNet files. "
                "Check that the checkpoint contains vocabulary_mappings, or rerun with --full-dataloader."
            )
    else:
        loaders, payload = build_dataloaders(cfg)
        explanations = save_sample_explanations(model, loaders[args.split], payload["mappings"], cfg, device, args.max_examples)

    print(f"Saved {len(explanations)} explanations to {get_path(cfg, 'explanations_dir')}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
