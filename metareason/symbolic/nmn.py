"""Neural Module Networks (Andreas et al., 2016): programs routed through modules.

A program is a nested list ``[op, child1, child2, ...]``. Leaves are either a
bare op name (applied to the input features), a tensor (used as-is) or the
string ``"x"`` (the input features). Child outputs are summed before the
op's module is applied. Modules are a registry: add your own with
:meth:`NeuralModuleNetwork.register`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

Program = str | Tensor | list[Any]


class NeuralModule(nn.Module):
    """Default residual-free MLP module ``dim -> dim``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # type: ignore[no-any-return]


class NeuralModuleNetwork(nn.Module):
    """Executes symbolic programs over a registry of neural modules."""

    def __init__(self, dim: int, max_depth: int = 32) -> None:
        super().__init__()
        self.dim = dim
        self.max_depth = max_depth
        self.modules_dict = nn.ModuleDict(
            {
                "filter_color": NeuralModule(dim),
                "filter_shape": NeuralModule(dim),
                "logical_and": NeuralModule(dim),
                "count_objects": nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid()),
            }
        )

    def register(self, name: str, module: nn.Module) -> None:
        """Add or replace a module under ``name``."""
        self.modules_dict[name] = module

    def forward(self, x: Tensor, query_tree: Program) -> Tensor:
        """Evaluate a program on input features ``x``."""
        return self._eval(x, query_tree, 0)

    def _eval(self, x: Tensor, node: Program, depth: int) -> Tensor:
        if depth > self.max_depth:
            raise RecursionError(f"program deeper than max_depth={self.max_depth}")
        if isinstance(node, Tensor):
            return node
        if isinstance(node, str):
            return x if node == "x" else self._run_op(node, x)
        if not isinstance(node, (list, tuple)) or not node or not isinstance(node[0], str):
            raise ValueError(f"malformed program node: {node!r}")
        op, *args = node
        children = [self._eval(x, a, depth + 1) for a in args]
        combined = x if not children else children[0] if len(children) == 1 else torch.stack(children).sum(0)
        return self._run_op(op, combined)

    def _run_op(self, op: str, inp: Tensor) -> Tensor:
        if op not in self.modules_dict:
            raise ValueError(f"Unknown symbolic operation node: {op}")
        return self.modules_dict[op](inp)  # type: ignore[no-any-return]
