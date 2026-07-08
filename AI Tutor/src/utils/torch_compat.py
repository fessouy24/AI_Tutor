from __future__ import annotations


def disable_torch_dynamo_eval_frame() -> None:
    """Avoid TorchDynamo eval-frame crashes seen on some Windows/Python builds.

    This baseline does not rely on torch.compile. Optimizer constructors in
    PyTorch 2.x pass through TorchDynamo wrappers, and on some local Windows
    environments that can fail at C eval-frame setup after pandas preprocessing.
    Returning the callback directly keeps eager PyTorch behavior.
    """

    try:
        import torch._dynamo.eval_frame as eval_frame

        if getattr(eval_frame, "_ai_tutor_eval_frame_disabled", False):
            return
        eval_frame._maybe_set_eval_frame = lambda callback: callback
        eval_frame._ai_tutor_eval_frame_disabled = True
    except Exception:
        pass

