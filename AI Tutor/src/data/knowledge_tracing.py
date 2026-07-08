from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.config import first_existing_path, get_path, project_root, resolve_path

try:
    pd.options.mode.string_storage = "python"
    if hasattr(pd.options, "future") and hasattr(pd.options.future, "infer_string"):
        pd.options.future.infer_string = False
except Exception:
    pass


STUDENT_COLS = ["student_id", "studentId", "ITEST_id", "user_id", "uid", "student", "anon_student_id"]
QUESTION_COLS = ["question_id", "problem_id", "problemId", "item_id", "assistment_id", "assistmentId", "questions", "problem_log_id"]
SKILL_COLS = ["skill_id", "concept_id", "kc_id", "concepts", "list_skill_ids", "skill", "Skill Name", "KC(SubSkills)", "kc", "tags"]
CORRECT_COLS = ["correct", "Correct First Attempt", "is_correct", "answer_correct", "responses", "correctness"]
TIME_COLS = ["timestamp", "timestamps", "order_id", "action_num", "actionId", "start_time", "startTime", "time", "position", "problem_log_id"]
MASK_COLS = ["selectmasks", "select_mask", "mask"]


@dataclass
class ColumnSpec:
    student: str
    question: str
    skill: str
    correct: str
    timestamp: str | None = None
    mask: str | None = None
    presequenced: bool = False


class KTSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: list[dict[str, Any]],
        max_seq_len: int,
        min_seq_len: int = 3,
        stride: int | None = None,
        name: str = "dataset",
    ) -> None:
        self.max_seq_len = max_seq_len
        self.name = name
        self.windows: list[dict[str, Any]] = []
        stride = stride or max_seq_len
        for seq in sequences:
            q = seq["questions"]
            s = seq["skills"]
            c = seq["corrects"]
            if len(q) < min_seq_len + 1:
                continue
            for start in range(0, max(1, len(q) - 1), stride):
                end = min(start + max_seq_len + 1, len(q))
                if end - start < min_seq_len + 1:
                    continue
                self.windows.append(
                    {
                        "student_id": str(seq.get("student_id", "unknown")),
                        "questions": q[start:end],
                        "skills": s[start:end],
                        "corrects": c[start:end],
                    }
                )

        if not self.windows:
            raise ValueError(f"{name} has no valid sequences after min_seq_len/window filtering.")
        print(f"{name}: {len(self.windows):,} training windows from {len(sequences):,} students/sequences")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.windows[idx]
        q = item["questions"]
        s = item["skills"]
        c = item["corrects"]
        length = min(len(q) - 1, self.max_seq_len)

        q_in = torch.zeros(self.max_seq_len, dtype=torch.long)
        s_in = torch.zeros(self.max_seq_len, dtype=torch.long)
        c_in = torch.zeros(self.max_seq_len, dtype=torch.long)
        q_tgt = torch.zeros(self.max_seq_len, dtype=torch.long)
        s_tgt = torch.zeros(self.max_seq_len, dtype=torch.long)
        y = torch.zeros(self.max_seq_len, dtype=torch.float32)
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)

        q_in[:length] = torch.tensor(q[:length], dtype=torch.long)
        s_in[:length] = torch.tensor(s[:length], dtype=torch.long)
        c_in[:length] = torch.tensor(c[:length], dtype=torch.long)
        q_tgt[:length] = torch.tensor(q[1 : length + 1], dtype=torch.long)
        s_tgt[:length] = torch.tensor(s[1 : length + 1], dtype=torch.long)
        y[:length] = torch.tensor(c[1 : length + 1], dtype=torch.float32)
        mask[:length] = True

        return {
            "student_id": item["student_id"],
            "question_ids": q_in,
            "skill_ids": s_in,
            "correct_ids": c_in,
            "target_question_ids": q_tgt,
            "target_skill_ids": s_tgt,
            "labels": y,
            "mask": mask,
        }


def _source_enabled(source: dict[str, Any]) -> bool:
    return bool(source.get("enabled", True))


def _source_required(source: dict[str, Any]) -> bool:
    return bool(source.get("required", True))


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalised = {_normalise_name(c): c for c in columns}
    for candidate in candidates:
        key = _normalise_name(candidate)
        if key in normalised:
            return normalised[key]
    for col in columns:
        col_key = _normalise_name(col)
        if any(_normalise_name(c) in col_key for c in candidates):
            return col
    return None


def detect_columns(df: pd.DataFrame, overrides: dict[str, str] | None = None) -> ColumnSpec:
    overrides = overrides or {}
    columns = list(df.columns)
    student = overrides.get("student") or _find_column(columns, STUDENT_COLS)
    question = overrides.get("question") or _find_column(columns, QUESTION_COLS)
    skill = overrides.get("skill") or _find_column(columns, SKILL_COLS)
    correct = overrides.get("correct") or _find_column(columns, CORRECT_COLS)
    timestamp = overrides.get("timestamp") or _find_column(columns, TIME_COLS)
    mask = overrides.get("mask") or _find_column(columns, MASK_COLS)

    missing = [name for name, value in [("student", student), ("question", question), ("correct", correct)] if not value]
    if missing:
        raise ValueError(f"Could not detect required columns {missing}. Columns available: {columns}")

    if skill is None:
        skill = question
        print("No skill column detected; using question IDs as skill IDs.")

    presequenced = question in {"questions"} or correct in {"responses"} or skill in {"concepts"}
    spec = ColumnSpec(student=student, question=question, skill=skill, correct=correct, timestamp=timestamp, mask=mask, presequenced=presequenced)
    print("Detected columns:", spec)
    return spec


def _split_cell(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [p.strip() for p in re.split(r"[,;|]", text) if p.strip()]


def _first_token(value: Any) -> str:
    parts = _split_cell(value)
    if not parts:
        return "__missing__"
    return parts[0]


def _correct_to_int(value: Any) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "t", "yes", "y", "correct"}:
            return 1
        if text in {"false", "f", "no", "n", "incorrect"}:
            return 0
    try:
        return 1 if float(value) > 0 else 0
    except Exception:
        return 0


def _read_dataframe(path: Path, max_rows: int | None = None, usecols: list[str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        kwargs: dict[str, Any] = {"nrows": max_rows, "low_memory": False}
        if usecols:
            wanted = {str(col) for col in usecols if col}
            kwargs["usecols"] = lambda col: col in wanted
        return pd.read_csv(path, **kwargs)
    if suffix == ".feather":
        df = pd.read_feather(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported KT data file extension: {path}")
    return df.head(max_rows) if max_rows else df


def _build_vocab(values: list[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for value in values:
        key = str(value)
        if key not in vocab:
            vocab[key] = len(vocab) + 1
    return vocab


def _map_or_add(vocab: dict[str, int], value: Any) -> int:
    key = str(value)
    if key not in vocab:
        vocab[key] = len(vocab) + 1
    return vocab[key]


def _parse_presequenced(df: pd.DataFrame, spec: ColumnSpec) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    question_vocab: dict[str, int] = {}
    skill_vocab: dict[str, int] = {}
    sequences: list[dict[str, Any]] = []

    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        questions = _split_cell(row_dict[spec.question])
        skills = _split_cell(row_dict[spec.skill])
        corrects = _split_cell(row_dict[spec.correct])
        masks = _split_cell(row_dict[spec.mask]) if spec.mask else ["1"] * len(questions)
        n = min(len(questions), len(skills), len(corrects), len(masks))

        q_ids: list[int] = []
        s_ids: list[int] = []
        c_ids: list[int] = []
        for i in range(n):
            if questions[i] in {"-1", ""} or skills[i] in {"-1", ""} or masks[i] in {"0", "-1"}:
                continue
            q_ids.append(_map_or_add(question_vocab, questions[i]))
            s_ids.append(_map_or_add(skill_vocab, skills[i]))
            c_ids.append(_correct_to_int(corrects[i]))

        if q_ids:
            sequences.append(
                {
                    "student_id": row_dict.get(spec.student, len(sequences)),
                    "questions": q_ids,
                    "skills": s_ids,
                    "corrects": c_ids,
                }
            )
    return sequences, {"question_vocab": question_vocab, "skill_vocab": skill_vocab}


def _parse_row_interactions(df: pd.DataFrame, spec: ColumnSpec) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    work = df[[c for c in [spec.student, spec.question, spec.skill, spec.correct, spec.timestamp] if c]].copy()
    work = work.dropna(subset=[spec.student, spec.question, spec.correct])
    work["_skill_norm"] = work[spec.skill].map(_first_token)
    work["_correct_bin"] = work[spec.correct].map(_correct_to_int)
    work["_question_norm"] = work[spec.question].astype(str)

    if spec.timestamp:
        sort_cols = [spec.student, spec.timestamp]
        work = work.sort_values(sort_cols, kind="mergesort")
    else:
        work = work.sort_values([spec.student], kind="mergesort")

    question_vocab = _build_vocab(work["_question_norm"].astype(str).tolist())
    skill_vocab = _build_vocab(work["_skill_norm"].astype(str).tolist())

    sequences: list[dict[str, Any]] = []
    for student_id, group in work.groupby(spec.student, sort=False):
        q_ids = [question_vocab[str(v)] for v in group["_question_norm"].tolist()]
        s_ids = [skill_vocab[str(v)] for v in group["_skill_norm"].tolist()]
        c_ids = [int(v) for v in group["_correct_bin"].tolist()]
        sequences.append({"student_id": str(student_id), "questions": q_ids, "skills": s_ids, "corrects": c_ids})

    return sequences, {"question_vocab": question_vocab, "skill_vocab": skill_vocab}


def _reverse_vocab(vocab: dict[str, int]) -> dict[int, str]:
    return {int(v): str(k) for k, v in vocab.items()}


def _merge_source_sequences(
    sequences: list[dict[str, Any]],
    mappings: dict[str, dict[str, int]],
    source_name: str,
    global_question_vocab: dict[str, int],
    global_skill_vocab: dict[str, int],
) -> list[dict[str, Any]]:
    q_rev = _reverse_vocab(mappings.get("question_vocab", {}))
    s_rev = _reverse_vocab(mappings.get("skill_vocab", {}))
    merged: list[dict[str, Any]] = []
    for seq in sequences:
        q_ids: list[int] = []
        s_ids: list[int] = []
        for q in seq["questions"]:
            q_key = f"{source_name}::{q_rev.get(int(q), str(q))}"
            q_ids.append(_map_or_add(global_question_vocab, q_key))
        for s in seq["skills"]:
            s_key = f"{source_name}::{s_rev.get(int(s), str(s))}"
            s_ids.append(_map_or_add(global_skill_vocab, s_key))
        merged.append(
            {
                "student_id": f"{source_name}::{seq.get('student_id', 'unknown')}",
                "questions": q_ids,
                "skills": s_ids,
                "corrects": list(seq["corrects"]),
                "source": source_name,
            }
        )
    return merged


def _resolve_source_path(source: dict[str, Any]) -> Path | None:
    candidates: list[str | Path] = []
    if source.get("path"):
        candidates.append(source["path"])
    candidates.extend(source.get("path_candidates") or [])
    return first_existing_path(candidates)


def _source_search_paths(source: dict[str, Any]) -> list[str]:
    candidates: list[str | Path] = []
    if source.get("path"):
        candidates.append(source["path"])
    candidates.extend(source.get("path_candidates") or [])
    return [str(resolve_path(candidate, project_root())) for candidate in candidates]


def _format_source_report(report: dict[str, Any]) -> str:
    parts = [f"{report.get('name', 'kt_source')}: status={report.get('status', 'unknown')}"]
    if report.get("required") is False:
        parts.append("required=false")
    if report.get("sequences") is not None:
        parts.append(f"sequences={report['sequences']}")
    if report.get("path"):
        parts.append(f"path={report['path']}")
    if report.get("searched"):
        parts.append(f"searched={report['searched']}")
    if report.get("reason"):
        parts.append(f"reason={report['reason']}")
    if report.get("error"):
        parts.append(f"error={report['error']}")
    return " | ".join(parts)


def _raise_for_strict_source_failures(source_reports: list[dict[str, Any]]) -> None:
    failures = [report for report in source_reports if report.get("required", True) and report.get("status") != "ok"]
    if not failures:
        return
    details = "\n".join(f"- {_format_source_report(report)}" for report in failures)
    raise FileNotFoundError(
        "Strict KT source validation failed. Every enabled data.kt_sources entry must be present "
        "and required=true entry must produce at least one usable supervised sequence before training starts.\n"
        f"{details}\n"
        "Fix the missing paths/metadata, disable sources you do not intend to train on, "
        "or set data.require_all_kt_sources=false."
    )


def _parse_kt_file(path: Path, cfg: dict[str, Any], source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, Any]]:
    data_cfg = cfg.get("data", {})
    max_rows = source.get("max_rows", data_cfg.get("max_rows"))
    overrides = dict(data_cfg.get("column_overrides") or {})
    overrides.update(source.get("column_overrides") or {})
    usecols = [
        overrides.get("student"),
        overrides.get("question"),
        overrides.get("skill"),
        overrides.get("correct"),
        overrides.get("timestamp"),
        overrides.get("mask"),
    ]
    df = _read_dataframe(path, max_rows=max_rows, usecols=[col for col in usecols if col])
    spec = detect_columns(df, overrides)
    if spec.presequenced:
        sequences, mappings = _parse_presequenced(df, spec)
    else:
        sequences, mappings = _parse_row_interactions(df, spec)
    return sequences, mappings, {"path": str(path), "column_spec": spec.__dict__, "rows": int(len(df))}


def _find_ednet_metadata(source_path: Path, source: dict[str, Any]) -> Path | None:
    candidates: list[str | Path] = []
    if source.get("correct_answer_path"):
        candidates.append(source["correct_answer_path"])
    candidates.extend(source.get("correct_answer_path_candidates") or [])
    found = first_existing_path(candidates)
    if found:
        return found
    roots = [source_path, source_path.parent, source_path.parent.parent]
    for root in roots:
        for name in ["questions.csv", "contents/questions.csv", "EdNet-Contents/contents/questions.csv"]:
            candidate = root / name
            if candidate.exists():
                return candidate
    return None


def _parse_ednet_folder(path: Path, cfg: dict[str, Any], source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, Any]]:
    metadata_path = _find_ednet_metadata(path, source)
    if metadata_path is None:
        return [], {"question_vocab": {}, "skill_vocab": {}}, {
            "path": str(path),
            "skipped": True,
            "reason": "EdNet/KT2 folder needs questions.csv with correct_answer; user_answer alone is not a correctness label.",
        }

    meta = pd.read_csv(metadata_path, low_memory=False)
    q_col = _find_column(list(meta.columns), ["question_id", "item_id", "question"])
    a_col = _find_column(list(meta.columns), ["correct_answer", "answer", "correct"])
    s_col = _find_column(list(meta.columns), ["tags", "skill_id", "concept_id", "kc_id"])
    if not q_col or not a_col:
        return [], {"question_vocab": {}, "skill_vocab": {}}, {
            "path": str(path),
            "metadata_path": str(metadata_path),
            "skipped": True,
            "reason": "questions metadata did not contain question_id and correct_answer columns.",
        }
    correct_map = dict(zip(meta[q_col].astype(str), meta[a_col].astype(str).str.strip().str.lower()))
    skill_map = dict(zip(meta[q_col].astype(str), meta[s_col].fillna("").astype(str).map(_first_token))) if s_col else {}

    patterns = source.get("file_patterns") or ["*.csv", "*.feather", "*.parquet", "*.pq"]
    max_files = int(source.get("max_files", 0) or 0)
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path.rglob(str(pattern)))
    files = sorted({fp for fp in files if fp.is_file()})
    if max_files:
        files = files[:max_files]
    source_name = str(source.get("name", path.name))
    print(
        f"[multi-KT] {source_name}: found {len(files):,} EdNet files under {path}; "
        "parsing now before epoch training starts."
    )
    question_vocab: dict[str, int] = {}
    skill_vocab: dict[str, int] = {}
    sequences: list[dict[str, Any]] = []
    skipped = 0
    progress_every = int(source.get("progress_every_files", cfg.get("data", {}).get("ednet_progress_every_files", 10000)) or 0)
    for file_index, file_path in enumerate(files, start=1):
        if progress_every and file_index % progress_every == 0:
            print(f"[multi-KT] {source_name}: parsed {file_index:,}/{len(files):,} EdNet files...")
        try:
            df = _read_dataframe(file_path)
        except Exception:
            skipped += 1
            continue
        columns = list(df.columns)
        item_col = _find_column(columns, ["item_id", "question_id"])
        ans_col = _find_column(columns, ["user_answer", "answer"])
        time_col = _find_column(columns, ["timestamp", "time"])
        action_col = _find_column(columns, ["action_type"])
        if not item_col or not ans_col:
            skipped += 1
            continue
        work = df.copy()
        if action_col:
            work = work[work[action_col].astype(str).str.lower().eq("respond")]
        work = work.dropna(subset=[item_col, ans_col])
        if time_col:
            work = work.sort_values(time_col, kind="mergesort")
        q_ids: list[int] = []
        s_ids: list[int] = []
        c_ids: list[int] = []
        for row in work.itertuples(index=False):
            row_dict = row._asdict()
            raw_q = str(row_dict[item_col])
            if raw_q not in correct_map:
                continue
            raw_answer = str(row_dict[ans_col]).strip().lower()
            skill = skill_map.get(raw_q, raw_q) or raw_q
            q_ids.append(_map_or_add(question_vocab, raw_q))
            s_ids.append(_map_or_add(skill_vocab, skill))
            c_ids.append(1 if raw_answer == correct_map[raw_q] else 0)
        if q_ids:
            sequences.append({"student_id": file_path.stem, "questions": q_ids, "skills": s_ids, "corrects": c_ids})
    return sequences, {"question_vocab": question_vocab, "skill_vocab": skill_vocab}, {
        "path": str(path),
        "metadata_path": str(metadata_path),
        "files_seen": len(files),
        "files_skipped": skipped,
        "file_patterns": patterns,
    }


def _parse_supervised_folder(path: Path, cfg: dict[str, Any], source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, Any]]:
    patterns = source.get("file_patterns") or ["*.csv", "*.feather", "*.parquet", "*.pq"]
    max_files = int(source.get("max_files", 12) or 0)
    max_file_mb = float(source.get("max_file_mb", 500))
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path.rglob(str(pattern)))
    files = sorted({fp for fp in files if fp.is_file()})
    if max_files:
        files = files[:max_files]

    folder_question_vocab: dict[str, int] = {}
    folder_skill_vocab: dict[str, int] = {}
    folder_sequences: list[dict[str, Any]] = []
    parsed_files = 0
    skipped: list[dict[str, str]] = []
    for file_path in files:
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > max_file_mb:
                skipped.append({"file": str(file_path), "reason": f"larger than max_file_mb={max_file_mb}"})
                continue
            sequences, mappings, _meta = _parse_kt_file(file_path, cfg, source)
            sequences = [s for s in sequences if len(s["questions"]) >= int(cfg.get("data", {}).get("min_seq_len", 3)) + 1]
            merged = _merge_source_sequences(
                sequences,
                mappings,
                file_path.stem,
                folder_question_vocab,
                folder_skill_vocab,
            )
            folder_sequences.extend(merged)
            parsed_files += 1
        except Exception as exc:
            skipped.append({"file": str(file_path), "reason": str(exc)[:300]})

    return folder_sequences, {"question_vocab": folder_question_vocab, "skill_vocab": folder_skill_vocab}, {
        "path": str(path),
        "files_seen": len(files),
        "files_parsed": parsed_files,
        "files_skipped": len(skipped),
        "skip_examples": skipped[:8],
    }


def _cache_path(cfg: dict[str, Any], dataset_path: Path) -> Path:
    processed_dir = get_path(cfg, "processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg.get("data", {})
    key_data = {
        "path": str(dataset_path.resolve()),
        "mtime": dataset_path.stat().st_mtime if dataset_path.exists() else None,
        "max_seq_len": data_cfg.get("max_seq_len"),
        "min_seq_len": data_cfg.get("min_seq_len"),
        "max_rows": data_cfg.get("max_rows"),
        "split": data_cfg.get("split"),
    }
    digest = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return processed_dir / f"kt_processed_{digest}.pt"


def _multi_cache_path(cfg: dict[str, Any], sources: list[dict[str, Any]]) -> Path:
    processed_dir = get_path(cfg, "processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg.get("data", {})
    configured_cache = data_cfg.get("processed_cache_path")
    if configured_cache:
        cache_path = resolve_path(configured_cache, project_root())
        if cache_path is None:
            raise ValueError("data.processed_cache_path resolved to None")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cache_path
    source_keys = []
    for source in sources:
        if not _source_enabled(source):
            continue
        path = _resolve_source_path(source)
        source_keys.append(
            {
                "name": source.get("name"),
                "type": source.get("type", "file"),
                "split_role": source.get("split_role", "auto"),
                "path": str(path.resolve()) if path else None,
                "mtime": path.stat().st_mtime if path and path.exists() and path.is_file() else None,
                "max_rows": source.get("max_rows", data_cfg.get("max_rows")),
                "max_files": source.get("max_files"),
                "max_file_mb": source.get("max_file_mb"),
            }
        )
    key_data = {
        "mode": "multi_source_v1",
        "sources": source_keys,
        "max_seq_len": data_cfg.get("max_seq_len"),
        "min_seq_len": data_cfg.get("min_seq_len"),
        "split": data_cfg.get("split"),
    }
    digest = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return processed_dir / f"kt_multi_processed_{digest}.pt"


def _source_cache_path(cfg: dict[str, Any], source: dict[str, Any], path: Path) -> Path:
    processed_dir = get_path(cfg, "processed_dir")
    source_cache_dir = resolve_path(cfg.get("data", {}).get("source_cache_dir"), project_root())
    cache_dir = source_cache_dir or (processed_dir / "kt_source_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg.get("data", {})
    source_type = str(source.get("type", "file")).lower()
    metadata_path = _find_ednet_metadata(path, source) if (path.is_dir() or source_type in {"ednet", "ednet_folder", "kt2", "kt_folder"}) else None
    try:
        path_stat = path.stat()
    except OSError:
        path_stat = None
    try:
        metadata_stat = metadata_path.stat() if metadata_path else None
    except OSError:
        metadata_stat = None
    key_data = {
        "mode": "kt_source_v2",
        "name": source.get("name"),
        "type": source.get("type", "file"),
        "split_role": source.get("split_role", "auto"),
        "path": str(path.resolve()),
        "path_mtime": path_stat.st_mtime if path_stat else None,
        "path_size": path_stat.st_size if path_stat and path.is_file() else None,
        "metadata_path": str(metadata_path.resolve()) if metadata_path else None,
        "metadata_mtime": metadata_stat.st_mtime if metadata_stat else None,
        "file_patterns": source.get("file_patterns"),
        "max_rows": source.get("max_rows", data_cfg.get("max_rows")),
        "max_files": source.get("max_files"),
        "max_file_mb": source.get("max_file_mb"),
        "column_overrides": source.get("column_overrides", data_cfg.get("column_overrides")),
    }
    digest = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    source_name = _normalise_name(str(source.get("name", "kt_source"))) or "kt_source"
    return cache_dir / f"kt_source_{source_name}_{digest}.pt"


def _load_torch_cache(path: Path, label: str) -> Any | None:
    try:
        return torch.load(path, weights_only=False)
    except Exception as exc:
        print(f"Ignoring unreadable {label} cache: {path}. Reason: {exc}")
        return None


def _save_torch_cache(payload: Any, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    print(f"Saved {label} cache: {path}")


def _load_or_parse_source(
    cfg: dict[str, Any],
    source: dict[str, Any],
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, Any], bool]:
    data_cfg = cfg.get("data", {})
    source_type = str(source.get("type", "file")).lower()
    use_source_cache = bool(data_cfg.get("cache_processed", True)) and bool(data_cfg.get("cache_sources", True))
    cache_path = _source_cache_path(cfg, source, path)
    rebuild = bool(data_cfg.get("rebuild_cache", False) or source.get("rebuild_cache", False))
    if use_source_cache and cache_path.exists() and not rebuild:
        print(f"[multi-KT] loading source cache for {source.get('name', path.name)}: {cache_path}")
        cached = _load_torch_cache(cache_path, "source")
        if cached is not None:
            return cached["sequences"], cached["mappings"], cached.get("meta", {}), True

    if source_type in {"supervised_folder", "kt_file_folder", "folder_files"}:
        sequences, mappings, meta = _parse_supervised_folder(path, cfg, source)
    elif source_type in {"ednet", "ednet_folder", "kt2", "kt_folder"} or path.is_dir():
        sequences, mappings, meta = _parse_ednet_folder(path, cfg, source)
    else:
        sequences, mappings, meta = _parse_kt_file(path, cfg, source)

    if use_source_cache and not meta.get("skipped") and sequences:
        _save_torch_cache({"sequences": sequences, "mappings": mappings, "meta": meta}, cache_path, "source")
    return sequences, mappings, meta, False


def locate_dataset(cfg: dict[str, Any]) -> Path:
    paths_cfg = cfg.get("paths", {})
    candidates: list[str | Path] = []
    if paths_cfg.get("dataset_path"):
        candidates.append(paths_cfg["dataset_path"])
    candidates.extend(paths_cfg.get("dataset_path_candidates") or [])
    found = first_existing_path(candidates)
    if found:
        return found
    searched = [str(resolve_path(p, project_root())) for p in candidates]
    raise FileNotFoundError(f"No dataset file found. Searched: {searched}")


def load_multi_sequences(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg.get("data", {})
    sources = [s for s in data_cfg.get("kt_sources", []) if _source_enabled(s)]
    if not sources:
        raise ValueError("data.use_multi_source is enabled but data.kt_sources has no enabled sources.")

    require_all_sources = bool(data_cfg.get("require_all_kt_sources", False))
    cache_path = _multi_cache_path(cfg, sources)
    if data_cfg.get("cache_processed", True) and cache_path.exists() and not data_cfg.get("rebuild_cache", False):
        print(f"Loading processed multi-source KT cache: {cache_path}")
        payload = _load_torch_cache(cache_path, "processed multi-source")
        if payload is not None:
            if require_all_sources:
                cached_reports = list(payload.get("source_reports") or [])
                cached_by_name = {str(report.get("name", "")): report for report in cached_reports}
                strict_reports: list[dict[str, Any]] = []
                for source in sources:
                    name = str(source.get("name", "kt_source"))
                    cached = dict(cached_by_name.get(name) or {})
                    if cached:
                        cached["required"] = _source_required(source)
                        strict_reports.append(cached)
                    else:
                        strict_reports.append(
                            {
                                "name": name,
                                "status": "missing_from_cache",
                                "required": _source_required(source),
                                "path": str(_resolve_source_path(source) or ""),
                            }
                        )
                _raise_for_strict_source_failures(strict_reports)
            return payload

    global_question_vocab: dict[str, int] = {}
    global_skill_vocab: dict[str, int] = {}
    all_sequences: list[dict[str, Any]] = []
    fixed_train: list[dict[str, Any]] = []
    fixed_val: list[dict[str, Any]] = []
    fixed_test: list[dict[str, Any]] = []
    train_val_pool: list[dict[str, Any]] = []
    auto_pool: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    min_seq_len = int(data_cfg.get("min_seq_len", 3))

    for source in sources:
        name = str(source.get("name", "kt_source"))
        required = _source_required(source)
        path = _resolve_source_path(source)
        if path is None:
            report = {"name": name, "status": "missing", "required": required, "searched": _source_search_paths(source)}
            source_reports.append(report)
            print(f"[multi-KT] missing {name}")
            continue
        try:
            sequences, mappings, meta, loaded_from_cache = _load_or_parse_source(cfg, source, path)
            sequences = [s for s in sequences if len(s["questions"]) >= min_seq_len + 1]
            merged = _merge_source_sequences(sequences, mappings, name, global_question_vocab, global_skill_vocab)
            all_sequences.extend(merged)
            split_role = str(source.get("split_role", "auto")).lower().replace("-", "_")
            if split_role in {"train", "training"}:
                fixed_train.extend(merged)
            elif split_role in {"val", "valid", "validation"}:
                fixed_val.extend(merged)
            elif split_role in {"test", "holdout", "eval", "evaluation"}:
                fixed_test.extend(merged)
            elif split_role in {"train_val", "train_valid", "train_validation"}:
                train_val_pool.extend(merged)
            else:
                auto_pool.extend(merged)
            report = {
                "name": name,
                "status": "ok" if merged else "skipped",
                "required": required,
                "sequences": len(merged),
                "split_role": split_role,
                "loaded_from_cache": loaded_from_cache,
                "path": str(path),
                **meta,
            }
            source_reports.append(report)
            cache_note = " cache=hit" if loaded_from_cache else ""
            required_note = "" if required else " required=false"
            print(f"[multi-KT] {report['status']} {name}: sequences={len(merged):,} split_role={split_role}{cache_note}{required_note} path={path}")
            if meta.get("reason"):
                print(f"           reason: {meta['reason']}")
        except Exception as exc:
            source_reports.append({"name": name, "status": "failed", "required": required, "path": str(path), "error": str(exc)})
            required_note = "" if required else " required=false"
            print(f"[multi-KT] failed {name}:{required_note} {exc}")

    if require_all_sources:
        _raise_for_strict_source_failures(source_reports)

    if not all_sequences:
        raise ValueError("No usable supervised KT sequences found from enabled data.kt_sources.")

    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    split_cfg = data_cfg.get("split", {"train": 0.8, "val": 0.1, "test": 0.1})
    train_sequences = list(fixed_train)
    val_sequences = list(fixed_val)
    test_sequences = list(fixed_test)

    def _split_pool(pool: list[dict[str, Any]], include_test: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        if not pool:
            return [], [], []
        indices = np.arange(len(pool))
        rng.shuffle(indices)
        if include_test:
            n_train = int(len(indices) * float(split_cfg.get("train", 0.8)))
            n_val = int(len(indices) * float(split_cfg.get("val", 0.1)))
            train_idx = indices[:n_train]
            val_idx = indices[n_train : n_train + n_val]
            test_idx = indices[n_train + n_val :]
        else:
            train_weight = float(split_cfg.get("train", 0.8))
            val_weight = float(split_cfg.get("val", 0.1))
            denom = max(train_weight + val_weight, 1e-9)
            n_train = int(len(indices) * (train_weight / denom))
            train_idx = indices[:n_train]
            val_idx = indices[n_train:]
            test_idx = np.array([], dtype=int)
        return [pool[i] for i in train_idx], [pool[i] for i in val_idx], [pool[i] for i in test_idx]

    auto_train, auto_val, auto_test = _split_pool(auto_pool, include_test=True)
    tv_train, tv_val, _ = _split_pool(train_val_pool, include_test=False)
    train_sequences.extend(auto_train)
    train_sequences.extend(tv_train)
    val_sequences.extend(auto_val)
    val_sequences.extend(tv_val)
    test_sequences.extend(auto_test)
    if not test_sequences and len(val_sequences) > 1:
        test_sequences = val_sequences[-1:]
        val_sequences = val_sequences[:-1]

    source_counts: dict[str, int] = {}
    for seq in all_sequences:
        source_counts[str(seq.get("source", "unknown"))] = source_counts.get(str(seq.get("source", "unknown")), 0) + 1
    split_source_counts: dict[str, dict[str, int]] = {}
    for split_name, split_sequences in [("train", train_sequences), ("val", val_sequences), ("test", test_sequences)]:
        split_source_counts[split_name] = {}
        for seq in split_sequences:
            source_name = str(seq.get("source", "unknown"))
            split_source_counts[split_name][source_name] = split_source_counts[split_name].get(source_name, 0) + 1

    payload = {
        "dataset_path": "multi_source:" + ",".join(r.get("name", "unknown") for r in source_reports if r.get("status") == "ok"),
        "source_reports": source_reports,
        "source_counts": source_counts,
        "split_source_counts": split_source_counts,
        "splits": {
            "train": train_sequences,
            "val": val_sequences,
            "test": test_sequences,
        },
        "mappings": {"question_vocab": global_question_vocab, "skill_vocab": global_skill_vocab},
        "n_questions": len(global_question_vocab),
        "n_skills": len(global_skill_vocab),
        "config_data": data_cfg,
    }
    print(
        "Processed multi-source KT data:",
        f"sequences={len(all_sequences):,}",
        f"questions={payload['n_questions']:,}",
        f"skills={payload['n_skills']:,}",
        f"sources={source_counts}",
        f"split_sources={split_source_counts}",
    )
    if data_cfg.get("cache_processed", True):
        _save_torch_cache(payload, cache_path, "processed multi-source")
    return payload


def load_sequences(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg.get("data", {})
    if data_cfg.get("use_multi_source", False) or data_cfg.get("kt_sources"):
        return load_multi_sequences(cfg)
    dataset_path = locate_dataset(cfg)
    cache_path = _cache_path(cfg, dataset_path)
    if data_cfg.get("cache_processed", True) and cache_path.exists() and not data_cfg.get("rebuild_cache", False):
        print(f"Loading processed KT cache: {cache_path}")
        payload = _load_torch_cache(cache_path, "processed")
        if payload is not None:
            return payload

    print(f"Loading KT dataset: {dataset_path}")
    max_rows = data_cfg.get("max_rows")
    df = pd.read_csv(dataset_path, nrows=max_rows, low_memory=False)
    spec = detect_columns(df, data_cfg.get("column_overrides") or {})
    if spec.presequenced:
        sequences, mappings = _parse_presequenced(df, spec)
    else:
        sequences, mappings = _parse_row_interactions(df, spec)

    sequences = [s for s in sequences if len(s["questions"]) >= data_cfg.get("min_seq_len", 3) + 1]
    if not sequences:
        raise ValueError("No usable student sequences found after preprocessing.")

    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    indices = np.arange(len(sequences))
    rng.shuffle(indices)
    split_cfg = data_cfg.get("split", {"train": 0.8, "val": 0.1, "test": 0.1})
    n_train = int(len(indices) * float(split_cfg.get("train", 0.8)))
    n_val = int(len(indices) * float(split_cfg.get("val", 0.1)))
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    if len(test_idx) == 0 and len(val_idx) > 1:
        test_idx = val_idx[-1:]
        val_idx = val_idx[:-1]

    payload = {
        "dataset_path": str(dataset_path),
        "column_spec": spec.__dict__,
        "splits": {
            "train": [sequences[i] for i in train_idx],
            "val": [sequences[i] for i in val_idx],
            "test": [sequences[i] for i in test_idx],
        },
        "mappings": mappings,
        "n_questions": len(mappings["question_vocab"]),
        "n_skills": len(mappings["skill_vocab"]),
        "config_data": data_cfg,
    }

    print(
        "Processed KT data:",
        f"students/sequences={len(sequences):,}",
        f"questions={payload['n_questions']:,}",
        f"skills={payload['n_skills']:,}",
    )
    if data_cfg.get("cache_processed", True):
        _save_torch_cache(payload, cache_path, "processed")
    return payload


def build_dataloaders(cfg: dict[str, Any]) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    payload = load_sequences(cfg)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    max_seq_len = int(data_cfg.get("max_seq_len", 200))
    min_seq_len = int(data_cfg.get("min_seq_len", 3))
    stride = int(data_cfg.get("stride", max_seq_len))

    datasets = {
        split: KTSequenceDataset(
            sequences,
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            stride=stride,
            name=split,
        )
        for split, sequences in payload["splits"].items()
        if sequences
    }
    loader_kwargs = {
        "batch_size": int(train_cfg.get("batch_size", 64)),
        "num_workers": int(train_cfg.get("num_workers", 0)),
        "pin_memory": bool(train_cfg.get("pin_memory", True) and torch.cuda.is_available()),
    }
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, **loader_kwargs),
        "val": DataLoader(datasets["val"], shuffle=False, **loader_kwargs),
        "test": DataLoader(datasets["test"], shuffle=False, **loader_kwargs),
    }
    return loaders, payload
