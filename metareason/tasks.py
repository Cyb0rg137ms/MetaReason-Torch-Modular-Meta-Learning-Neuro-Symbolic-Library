"""Seeded synthetic task distributions for meta-learning experiments."""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch import Tensor

from metareason.meta.maml import Task


def seed_everything(seed: int) -> torch.Generator:
    """Seed Python, NumPy and torch; return a torch generator seeded the same way."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


class SinusoidTasks:
    """Finn et al. (2017) sinusoid regression: ``y = A sin(x - phi)``.

    ``A ~ U[0.1, 5]``, ``phi ~ U[0, pi]``, ``x ~ U[-5, 5]``.
    """

    def __init__(self, k_shot: int = 10, k_query: int = 10, seed: int = 0) -> None:
        self.k_shot, self.k_query = k_shot, k_query
        self.gen = torch.Generator().manual_seed(seed)

    def _u(self, lo: float, hi: float, *shape: int) -> Tensor:
        return lo + (hi - lo) * torch.rand(*shape, generator=self.gen)

    def sample(self) -> Task:
        """One task with ``k_shot`` support and ``k_query`` query points."""
        amp, phase = float(self._u(0.1, 5.0, 1)), float(self._u(0.0, math.pi, 1))
        x = self._u(-5.0, 5.0, self.k_shot + self.k_query, 1)
        y = amp * torch.sin(x - phase)
        k = self.k_shot
        return Task(x[:k], y[:k], x[k:], y[k:])

    def batch(self, n: int) -> list[Task]:
        """``n`` independent tasks."""
        return [self.sample() for _ in range(n)]


class GaussianClassTasks:
    """N-way K-shot classification over random Gaussian class centres.

    Each task draws ``n_way`` centres from ``N(0, I_d * spread^2)`` and
    examples from ``N(centre, I_d)``; labels are shuffled per task so that
    no fixed class order can be memorised.
    """

    def __init__(
        self, n_way: int = 5, k_shot: int = 1, k_query: int = 15, dim: int = 16, spread: float = 1.5, seed: int = 0
    ):
        self.n_way, self.k_shot, self.k_query, self.dim, self.spread = n_way, k_shot, k_query, dim, spread
        self.gen = torch.Generator().manual_seed(seed)

    def sample(self) -> Task:
        """One episode; support and query labels lie in ``0..n_way-1``."""
        centres = torch.randn(self.n_way, self.dim, generator=self.gen) * self.spread

        def draw(k: int) -> tuple[Tensor, Tensor]:
            y = torch.arange(self.n_way).repeat_interleave(k)
            x = centres[y] + torch.randn(len(y), self.dim, generator=self.gen)
            perm = torch.randperm(len(y), generator=self.gen)
            return x[perm], y[perm]

        sx, sy = draw(self.k_shot)
        qx, qy = draw(self.k_query)
        return Task(sx, sy, qx, qy)
