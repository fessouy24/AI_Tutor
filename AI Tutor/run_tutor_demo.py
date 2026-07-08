from __future__ import annotations

import argparse
import json

from src.rag.index import EducationalRAG
from src.tutor.generator import SocraticTutor
from src.utils.config import ensure_output_dirs, load_config
from src.utils.memory import clear_gpu_memory, print_gpu_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG-grounded Socratic tutor demo.")
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--question", default="I am stuck. How should I start solving this problem?")
    parser.add_argument("--skill", default="linear equations")
    parser.add_argument("--mastery", type=float, default=0.42)
    parser.add_argument("--lesson", default=None)
    parser.add_argument("--final-answer", default=None)
    parser.add_argument("--allow-direct-answer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    print_gpu_memory("before tutor demo")

    rag = EducationalRAG(cfg)
    chunks, retrieval_latency_ms = rag.retrieve(
        student_question=args.question,
        weak_skill=args.skill,
        lesson=args.lesson,
        top_k=cfg.get("rag", {}).get("top_k", 4),
    )
    tutor = SocraticTutor(cfg)
    result = tutor.generate(
        student_question=args.question,
        weak_skill=args.skill,
        mastery_probability=args.mastery,
        retrieved_chunks=chunks,
        final_answer=args.final_answer,
        allow_direct_answer=args.allow_direct_answer,
    )
    print(
        json.dumps(
            {
                "question": args.question,
                "weak_skill": args.skill,
                "mastery_probability": args.mastery,
                "retrieval_latency_ms": retrieval_latency_ms,
                "retrieved_chunks": [
                    {"source": c.source, "score": c.score, "preview": c.text[:180]} for c in chunks
                ],
                "tutor": result,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    clear_gpu_memory(label="tutor demo cleanup")


if __name__ == "__main__":
    main()
