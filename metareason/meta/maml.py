"""Model-Agnostic Meta-Learning (Finn et al., 2017) for any ``nn.Module``.

The inner loop runs on a *functional* copy of the parameters
(:func:`torch.func.functional_call`), so any architecture works. Two modes:

* second-order MAML (``first_order=False``): the outer gradient goes through
  the inner updates (``create_graph=True``);
* first-order MAML / FOMAML (``first_order=True``): inner gradients are
  detached, which is cheaper and often almost as accurate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.func import functional_call

LossFn = Callable[[Tensor, Tensor], Tensor]
Params = dict[str, Tensor]


@dataclass(frozen=True)
class Task:
    """One few-shot task: a support set to adapt on, a query set to evaluate on."""

    support_x: Tensor
    support_y: Tensor
    query_x: Tensor
    query_y: Tensor


class MetaModel(nn.Module):
    """Two-layer MLP kept for backward compatibility with the 1.0 API.

    ``forward(x, params)`` accepts an optional dict of parameter overrides
    named ``weight1, bias1, weight2, bias2``.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weight1 = nn.Parameter(torch.randn(hidden_dim, input_dim) * (2.0 / input_dim) ** 0.5)
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
        self.weight2 = nn.Parameter(torch.randn(output_dim, hidden_dim) * (1.0 / hidden_dim) ** 0.5)
        self.bias2 = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: Tensor, params: Mapping[str, Tensor] | None = None) -> Tensor:
        p: Mapping[str, Tensor] = params if params is not None else dict(self.named_parameters())
        h = torch.relu(nn.functional.linear(x, p["weight1"], p["bias1"]))
        return nn.functional.linear(h, p["weight2"], p["bias2"])


class MAML:
    """MAML / FOMAML meta-learner.

    Args:
        model: any module; adapted functionally, never mutated in the inner loop.
        inner_lr: SGD step size of the inner loop.
        inner_steps: number of inner SGD steps.
        first_order: use the first-order approximation.
        loss_fn: default loss for :meth:`adapt` and :meth:`meta_loss`.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 1,
        first_order: bool = False,
        loss_fn: LossFn | None = None,
    ) -> None:
        if inner_steps < 0:
            raise ValueError("inner_steps must be >= 0")
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.loss_fn: LossFn = loss_fn or nn.functional.mse_loss

    def params(self) -> Params:
        """Current meta-parameters (trainable only)."""
        return {k: v for k, v in self.model.named_parameters() if v.requires_grad}

    def forward(self, x: Tensor, params: Mapping[str, Tensor] | None = None) -> Tensor:
        """Run the model with optional parameter overrides."""
        if params is None:
            return self.model(x)  # type: ignore[no-any-return]
        return functional_call(self.model, dict(params), (x,))  # type: ignore[no-any-return]

    def adapt(
        self,
        support_x: Tensor,
        support_y: Tensor,
        loss_fn: LossFn | None = None,
        steps: int | None = None,
        create_graph: bool | None = None,
    ) -> Params:
        """Return parameters after ``steps`` inner SGD steps on the support set."""
        fn = loss_fn or self.loss_fn
        n_steps = self.inner_steps if steps is None else steps
        graph = (not self.first_order) if create_graph is None else create_graph
        params = self.params()
        # enable_grad: adaptation must work even inside a caller's no_grad block
        # (e.g. evaluation); the outer graph is only kept when grad is enabled.
        with torch.enable_grad():
            for _ in range(n_steps):
                loss = fn(self.forward(support_x, params), support_y)
                grads = torch.autograd.grad(loss, list(params.values()), create_graph=graph, allow_unused=True)
                params = {
                    k: (p if g is None else p - self.inner_lr * (g if graph else g.detach()))
                    for (k, p), g in zip(params.items(), grads, strict=True)
                }
        return params

    def adapt_to_task(self, loss_fn: LossFn, support_x: Tensor, support_y: Tensor) -> Params:
        """Backward-compatible alias of :meth:`adapt` (one call, same semantics)."""
        return self.adapt(support_x, support_y, loss_fn)

    def meta_loss(self, tasks: Iterable[Task], loss_fn: LossFn | None = None) -> Tensor:
        """Mean query loss after adaptation, differentiable w.r.t. meta-parameters."""
        fn = loss_fn or self.loss_fn
        losses = [fn(self.forward(t.query_x, self.adapt(t.support_x, t.support_y, fn)), t.query_y) for t in tasks]
        if not losses:
            raise ValueError("meta_loss needs at least one task")
        return torch.stack(losses).mean()

    def meta_step(
        self, tasks: Iterable[Task], optimizer: torch.optim.Optimizer, loss_fn: LossFn | None = None
    ) -> float:
        """One outer-loop update. Returns the meta-loss before the update."""
        optimizer.zero_grad(set_to_none=True)
        loss = self.meta_loss(tasks, loss_fn)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        return float(loss.detach())
