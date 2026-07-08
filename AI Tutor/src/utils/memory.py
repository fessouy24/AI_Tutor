from __future__ import annotations

import gc
from typing import Any

import torch


def _gpu_stats() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    return allocated, reserved


def print_gpu_memory(label: str = "") -> None:
    if not torch.cuda.is_available():
        print(f"[VRAM] {label} CUDA not available")
        return
    allocated, reserved = _gpu_stats()
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    prefix = f"[VRAM] {label} " if label else "[VRAM] "
    print(f"{prefix}allocated={allocated:.1f} MB reserved={reserved:.1f} MB peak={peak:.1f} MB")


def empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except Exception as exc:
                print(f"[VRAM] ipc_collect skipped: {exc}")


def clear_gpu_memory(obj: Any | None = None, label: str = "cleanup") -> None:
    print_gpu_memory(f"before {label}")
    if obj is not None:
        try:
            del obj
        except Exception as exc:
            print(f"[VRAM] object deletion skipped: {exc}")
    gc.collect()
    empty_cuda_cache()
    print_gpu_memory(f"after {label}")


def safe_del(obj: Any | None = None, label: str = "safe_del") -> None:
    clear_gpu_memory(obj, label=label)

