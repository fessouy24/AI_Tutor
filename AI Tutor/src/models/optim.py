from __future__ import annotations

import math
from typing import Any, Iterable

import torch


class EagerAdamW:
    """Small eager AdamW optimizer.

    This follows decoupled AdamW update math while avoiding torch.optim wrapper
    paths that can touch TorchDynamo on some Windows/Python builds.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.params = [p for p in params if p.requires_grad]
        self.param_groups = [
            {
                "params": self.params,
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": betas,
                "eps": eps,
            }
        ]
        self.state: dict[int, dict[str, Any]] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any:
        loss = closure() if closure is not None else None
        group = self.param_groups[0]
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])

        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("EagerAdamW does not support sparse gradients.")
            state = self.state.setdefault(
                id(p),
                {
                    "step": 0,
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p),
                },
            )
            state["step"] += 1
            if weight_decay:
                p.mul_(1.0 - lr * weight_decay)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            step = int(state["step"])
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            step_size = lr / bias_correction1
            p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

    def state_dict(self) -> dict[str, Any]:
        states = []
        for p in self.params:
            state = self.state.get(id(p))
            if state is None:
                states.append(None)
            else:
                states.append(
                    {
                        "step": state["step"],
                        "exp_avg": state["exp_avg"].detach().cpu(),
                        "exp_avg_sq": state["exp_avg_sq"].detach().cpu(),
                    }
                )
        group = {k: v for k, v in self.param_groups[0].items() if k != "params"}
        return {"param_groups": [group], "state": states}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("param_groups"):
            saved = state_dict["param_groups"][0]
            for key in ["lr", "weight_decay", "betas", "eps"]:
                if key in saved:
                    self.param_groups[0][key] = saved[key]
        self.state.clear()
        for p, saved_state in zip(self.params, state_dict.get("state", [])):
            if saved_state is None:
                continue
            self.state[id(p)] = {
                "step": int(saved_state["step"]),
                "exp_avg": saved_state["exp_avg"].to(device=p.device, dtype=p.dtype),
                "exp_avg_sq": saved_state["exp_avg_sq"].to(device=p.device, dtype=p.dtype),
            }


class CosineScheduler:
    def __init__(self, optimizer: EagerAdamW, t_max: int) -> None:
        self.optimizer = optimizer
        self.t_max = max(1, t_max)
        self.last_epoch = 0
        self.base_lrs = [float(g["lr"]) for g in optimizer.param_groups]

    def step(self, metric: float | None = None) -> None:
        self.last_epoch += 1
        factor = 0.5 * (1.0 + math.cos(math.pi * min(self.last_epoch, self.t_max) / self.t_max))
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def state_dict(self) -> dict[str, Any]:
        return {"last_epoch": self.last_epoch, "base_lrs": self.base_lrs, "t_max": self.t_max}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.last_epoch = int(state_dict.get("last_epoch", 0))
        self.base_lrs = list(state_dict.get("base_lrs", self.base_lrs))
        self.t_max = int(state_dict.get("t_max", self.t_max))


class PlateauScheduler:
    def __init__(self, optimizer: EagerAdamW, patience: int = 2, factor: float = 0.5) -> None:
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.best = float("-inf")
        self.bad = 0

    def step(self, metric: float | None = None) -> None:
        metric = float("-inf") if metric is None or math.isnan(metric) else float(metric)
        if metric > self.best:
            self.best = metric
            self.bad = 0
            return
        self.bad += 1
        if self.bad > self.patience:
            for group in self.optimizer.param_groups:
                group["lr"] = float(group["lr"]) * self.factor
            self.bad = 0

    def state_dict(self) -> dict[str, Any]:
        return {"best": self.best, "bad": self.bad, "patience": self.patience, "factor": self.factor}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.best = float(state_dict.get("best", self.best))
        self.bad = int(state_dict.get("bad", self.bad))
        self.patience = int(state_dict.get("patience", self.patience))
        self.factor = float(state_dict.get("factor", self.factor))


def build_scheduler(optimizer: EagerAdamW, name: str, epochs: int):
    name = str(name or "").lower()
    if name == "cosine":
        return CosineScheduler(optimizer, epochs)
    if name == "plateau":
        return PlateauScheduler(optimizer)
    if name in {"", "none", "null"}:
        return None
    raise ValueError(f"Unsupported scheduler: {name}")

