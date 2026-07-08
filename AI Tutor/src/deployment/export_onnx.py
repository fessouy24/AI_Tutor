from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.models.akt import AKTONNXWrapper
from src.models.checkpointing import load_checkpoint
from src.models.factory import build_student_model
from src.utils.config import ensure_output_dirs, get_path, load_config
from src.utils.memory import clear_gpu_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained AKT checkpoint to ONNX.")
    parser.add_argument("--config", default="configs/local.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path; defaults to checkpoints/best_model.pt.")
    parser.add_argument("--output", default=None, help="Output ONNX path; defaults to outputs/onnx/akt_model.onnx.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else get_path(cfg, "checkpoints_dir") / "best_model.pt"
    output_path = Path(args.output) if args.output else get_path(cfg, "outputs_dir") / "onnx" / "akt_model.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    model_type = ckpt.get("model_type", "akt")
    if model_type != "akt":
        raise ValueError(f"ONNX export expects AKT checkpoint, got {model_type}")
    model_cfg = ckpt["model_config"]
    model = build_student_model(
        "akt",
        n_questions=int(model_cfg["n_questions"]),
        n_skills=int(model_cfg["n_skills"]),
        max_seq_len=int(model_cfg["max_seq_len"]),
        model_cfg=model_cfg,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    wrapper = AKTONNXWrapper(model).eval()

    batch = 2
    seq_len = int(model_cfg["max_seq_len"])
    n_questions = int(model_cfg["n_questions"])
    n_skills = int(model_cfg["n_skills"])
    dummy = (
        torch.randint(1, n_questions + 1, (batch, seq_len), dtype=torch.long),
        torch.randint(1, n_skills + 1, (batch, seq_len), dtype=torch.long),
        torch.randint(0, 2, (batch, seq_len), dtype=torch.long),
        torch.randint(1, n_questions + 1, (batch, seq_len), dtype=torch.long),
        torch.randint(1, n_skills + 1, (batch, seq_len), dtype=torch.long),
        torch.ones(batch, seq_len, dtype=torch.bool),
    )
    input_names = [
        "question_ids",
        "skill_ids",
        "correct_ids",
        "target_question_ids",
        "target_skill_ids",
        "input_mask",
    ]
    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes={name: {0: "batch"} for name in input_names} | {"logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Saved ONNX model: {output_path}")

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        ort_inputs = {name: tensor.detach().cpu().numpy() for name, tensor in zip(input_names, dummy)}
        with torch.no_grad():
            torch_out = wrapper(*dummy).detach().cpu().numpy()
        ort_out = session.run(["logits"], ort_inputs)[0]
        max_diff = float(np.max(np.abs(torch_out - ort_out)))
        print(f"ONNX verification max_abs_diff={max_diff:.6f}")
    except Exception as exc:
        print(f"ONNX Runtime verification skipped: {exc}")
    clear_gpu_memory(wrapper, label="ONNX export cleanup")


if __name__ == "__main__":
    main()
