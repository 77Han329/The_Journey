"""
Implementation of the AdanW optimizer (Adaptive Nesterov Momentum + decoupled WD).

PyTorch nightly now ships an ``Adan`` optimizer, but this hand-written version
keeps the project self-contained and mirrors the equations from the original
paper: https://arxiv.org/abs/2208.06677.  The ``W`` suffix indicates that the
weight decay term is decoupled (à la AdamW).
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch


class AdanW(torch.optim.Optimizer):
    """
    Minimal AdanW optimizer.

    Args:
        params: iterable of parameters to optimize.
        lr: learning rate.
        betas: (beta1, beta2, beta3) coefficients for the moving averages.
        eps: numerical stability term.
        weight_decay: decoupled weight decay coefficient.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.02,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad

                if grad.is_sparse:
                    raise RuntimeError("AdanW does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_diff"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    state["prev_grad"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_diff = state["exp_avg_diff"]
                exp_avg_sq = state["exp_avg_sq"]
                prev_grad = state["prev_grad"]

                state["step"] += 1
                step = state["step"]

                grad_diff = grad - prev_grad

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_diff.mul_(beta2).add_(grad_diff, alpha=1 - beta2)
                exp_avg_sq.mul_(beta3).addcmul_(grad, grad, value=1 - beta3)
                state["prev_grad"] = grad.detach().clone()

                # Bias corrections keep the early steps stable.
                bias_correction1 = 1 - beta1**step
                bias_correction3 = 1 - beta3**step

                m = exp_avg + (1 - beta1) * exp_avg_diff
                m_hat = m / bias_correction1
                denom = (exp_avg_sq / bias_correction3).sqrt().add_(eps)
                update = m_hat / denom

                if wd != 0:
                    param.data.mul_(1 - lr * wd)

                param.add_(update, alpha=-lr)

        return loss

