from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort

from src.utils.config import get_path, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX AKT inference.")
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--model", default=None, help="Path to ONNX model.")
    parser.add_argument("--input-json", default=None, help="Optional JSON input with arrays matching ONNX input names.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_path = Path(args.model) if args.model else get_path(cfg, "outputs_dir") / "onnx" / "akt_model.onnx"
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = [x.name for x in session.get_inputs()]

    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        inputs = {name: np.asarray(payload[name]) for name in input_names}
    else:
        first_shape = session.get_inputs()[0].shape
        seq_len = first_shape[1] if len(first_shape) > 1 and isinstance(first_shape[1], int) else int(cfg.get("data", {}).get("max_seq_len", 200))
        inputs = {
            "question_ids": np.ones((1, seq_len), dtype=np.int64),
            "skill_ids": np.ones((1, seq_len), dtype=np.int64),
            "correct_ids": np.zeros((1, seq_len), dtype=np.int64),
            "target_question_ids": np.ones((1, seq_len), dtype=np.int64),
            "target_skill_ids": np.ones((1, seq_len), dtype=np.int64),
            "input_mask": np.ones((1, seq_len), dtype=bool),
        }
    logits = session.run(["logits"], inputs)[0]
    probs = 1.0 / (1.0 + np.exp(-logits))
    print(json.dumps({"model": str(model_path), "mean_probability": float(probs.mean()), "shape": list(probs.shape)}, indent=2))


if __name__ == "__main__":
    main()
