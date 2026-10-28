import math
from typing import Optional, Callable

import torch


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (SAM) optimizer.

    This is a self-contained implementation with AdamW support.

    Two-step update:
      1) first_step(): ascent step to move to worst-case neighborhood
      2) second_step(): return and perform the descent update

    Supports:
      - adaptive (ASAM) mode via `adaptive` flag
      - weight decay (passed to base optimizer via defaults)
      - GPU tensors
      - mixed precision compatibility (works with external AMP via GradScaler)

    Note:
      - This optimizer expects gradients to be set (via loss.backward())
        before calling first_step/second_step.
      - For AMP compatibility you should use GradScaler externally in
        the training loop (this file doesn't require it).
    """

    def __init__(
        self,
        params,
        base_optimizer: Callable,
        rho: float = 0.05,
        adaptive: bool = False,
        **base_optimizer_kwargs,
    ):
        if rho < 0.0:
            raise ValueError(f"rho must be >= 0, got {rho}")

        # We keep SAM's own defaults minimal; most hyperparams belong to base optimizer.
        defaults = dict(rho=rho, adaptive=adaptive, **base_optimizer_kwargs)
        super().__init__(params, defaults)

        self.rho = rho
        self.adaptive = adaptive

        # Create base optimizer from the same parameter groups.
        # `base_optimizer` must be a class/constructor like torch.optim.AdamW
        self.base_optimizer = base_optimizer(
            self.param_groups,
            **base_optimizer_kwargs,
        )

        # Cache for epsilon (perturbation) per parameter.
        self._eps: Optional[list[torch.Tensor]] = None

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Compute perturbation and apply it to parameters."""
        grad_norm = self._grad_norm()
        if grad_norm == 0.0:
            # If gradients are zero, perturbation is zero; still call base optimizer
            # step in second_step to keep consistent behavior.
            scale = 0.0
        else:
            scale = self.rho / (grad_norm + 1e-12)

        eps_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    eps = torch.zeros_like(p)
                else:
                    if self.adaptive:
                        # ASAM uses |w| * grad
                        eps = (torch.pow(p, 2) if p.dtype.is_floating_point else torch.abs(p))
                        eps = eps * p.grad
                    else:
                        eps = p.grad
                    eps = eps * scale

                p.add_(eps)
                eps_list.append(eps)

        self._eps = eps_list

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Restore parameters and apply base optimizer step."""
        if self._eps is None:
            raise RuntimeError("SAM.second_step() called before first_step().")

        # Restore parameters by subtracting perturbation.
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                eps = self._eps[idx]
                p.sub_(eps)
                idx += 1

        # Perform descent step with base optimizer.
        self.base_optimizer.step()

        self._eps = None

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self) -> float:
        """Compute L2 norm of gradients (optionally adaptive)."""
        device = None
        norms = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if device is None:
                    device = p.grad.device

                if self.adaptive:
                    # ASAM: grad * |w|
                    w = torch.abs(p)
                    norms.append((w * p.grad).norm(p=2))
                else:
                    norms.append(p.grad.norm(p=2))

        if not norms:
            return 0.0

        # Combine norms via sqrt(sum(norm_i^2)).
        total = torch.norm(torch.stack(norms), p=2)
        return float(total.item())

    def zero_grad(self, set_to_none: bool = True):
        return self.base_optimizer.zero_grad(set_to_none=set_to_none)

