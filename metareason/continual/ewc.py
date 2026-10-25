"""Elastic Weight Consolidation (Kirkpatrick et al., 2017), with online EWC.

The penalty is ``(lambda / 2) * sum_i F_i (theta_i - theta*_i)^2`` with a
diagonal Fisher estimate ``F``. Two estimators are offered:

* ``"empirical"``: squared gradients of the loss on the true labels;
* ``"exact"`` (classification only): expectation over labels sampled from
  the model's own predictive distribution, i.e. the true Fisher.

Several tasks can be registered: penalties accumulate per task, or, with
``online=True`` (Schwarz et al., 2018), one running Fisher is decayed by
``gamma`` and anchored at the latest optimum.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import torch
from torch import Tensor, nn

LossFn = Callable[[Tensor, Tensor], Tensor]


class ElasticWeightConsolidation:
    """Diagonal-Fisher EWC regulariser.

    Args:
        model: model being trained sequentially.
        importance_weight: lambda.
        online: use online EWC (single decayed Fisher).
        gamma: decay of the running Fisher in online mode.
    """

    def __init__(self, model: nn.Module, importance_weight: float = 400.0, online: bool = False, gamma: float = 1.0):
        self.model = model
        self.importance = importance_weight
        self.online = online
        self.gamma = gamma
        self.tasks: list[tuple[dict[str, Tensor], dict[str, Tensor]]] = []  # (fisher, optimum)

    # Back-compat views of the most recent task.
    @property
    def fisher_matrix(self) -> dict[str, Tensor]:
        return self.tasks[-1][0] if self.tasks else {}

    @property
    def optimal_weights(self) -> dict[str, Tensor]:
        return self.tasks[-1][1] if self.tasks else {}

    def estimate_fisher(
        self,
        dataset: Iterable[tuple[Tensor, Tensor]],
        loss_fn: LossFn,
        mode: str = "empirical",
        batched: bool = False,
    ) -> dict[str, Tensor]:
        """Diagonal Fisher averaged over examples.

        ``dataset`` yields ``(x, y)`` pairs. If ``batched`` is False each pair is a
        single example (a batch dimension is added); otherwise each pair is a
        batch and per-example gradients are taken one row at a time.
        """
        if mode not in {"empirical", "exact"}:
            raise ValueError("mode must be 'empirical' or 'exact'")
        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        fisher = {n: torch.zeros_like(p) for n, p in params.items()}
        was_training = self.model.training
        self.model.eval()
        count = 0
        try:
            for x, y in dataset:
                xs, ys = (x, y) if batched else (x.unsqueeze(0), y.reshape(1) if y.numel() == 1 else y.unsqueeze(0))
                for i in range(xs.size(0)):
                    out = self.model(xs[i : i + 1])
                    if mode == "exact":
                        target = torch.distributions.Categorical(logits=out.detach()).sample()
                    else:
                        target = ys[i : i + 1]
                    grads = torch.autograd.grad(loss_fn(out, target), list(params.values()), allow_unused=True)
                    for (n, _), g in zip(params.items(), grads, strict=True):
                        if g is not None:
                            fisher[n] += g.detach() ** 2
                    count += 1
        finally:
            self.model.train(was_training)
        if count == 0:
            raise ValueError("empty dataset")
        return {n: f / count for n, f in fisher.items()}

    def register_previous_task(
        self,
        dataset: Sequence[tuple[Tensor, Tensor]] | Iterable[tuple[Tensor, Tensor]],
        loss_fn: LossFn,
        mode: str = "empirical",
        batched: bool = False,
    ) -> None:
        """Estimate the Fisher at the current weights and anchor them."""
        fisher = self.estimate_fisher(dataset, loss_fn, mode, batched)
        optimum = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
        if self.online and self.tasks:
            old = self.tasks[0][0]
            fisher = {n: self.gamma * old[n] + f for n, f in fisher.items()}
            self.tasks = [(fisher, optimum)]
        else:
            self.tasks.append((fisher, optimum))

    def compute_penalty_loss(self) -> Tensor:
        """Differentiable EWC penalty (a zero tensor before any task is registered)."""
        params = dict(self.model.named_parameters())
        device = next(iter(params.values())).device if params else None
        penalty = torch.zeros((), device=device)
        for fisher, optimum in self.tasks:
            for n, f in fisher.items():
                penalty = penalty + (f * (params[n] - optimum[n]) ** 2).sum()
        return 0.5 * self.importance * penalty
