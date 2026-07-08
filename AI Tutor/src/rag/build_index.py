from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.index import EducationalRAG
from src.utils.config import ensure_output_dirs, load_config
from src.utils.memory import clear_gpu_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or refresh the educational RAG index.")
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild the existing collection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    rag = EducationalRAG(cfg)
    rag.build_index(force_rebuild=args.force)
    clear_gpu_memory(label="RAG index build cleanup")


if __name__ == "__main__":
    main()
