"""Verified task amplification for few-shot / meta-learning data.

Turns one solved task into many *semantically equivalent* variants and keeps
only variants that pass an exact invertibility check:

* grid tasks (ARC-style): the 8 dihedral transforms x colour permutations
  (colour 0 kept as background), applied consistently to every train pair,
  the test input and the gold output; a variant is kept only if inverting
  the transform recovers the original gold grid exactly;
* multiple-choice tasks: option permutations with the gold letter remapped
  and re-checked.

Deterministic for a given ``seed``.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

Grid = list[list[int]]


def _rot90(g: Grid) -> Grid:
    return [list(r) for r in zip(*g[::-1], strict=True)]


def _transpose(g: Grid) -> Grid:
    return [list(r) for r in zip(*g, strict=True)]


DIHEDRAL: dict[str, Callable[[Grid], Grid]] = {
    "id": lambda g: [r[:] for r in g],
    "r90": _rot90,
    "r180": lambda g: _rot90(_rot90(g)),
    "r270": lambda g: _rot90(_rot90(_rot90(g))),
    "fliph": lambda g: [r[::-1] for r in g],
    "flipv": lambda g: [r[:] for r in g[::-1]],
    "tr": _transpose,
    "atr": lambda g: _rot90(_rot90(_transpose(g))),
}
INVERSE = {
    "id": "id",
    "r90": "r270",
    "r180": "r180",
    "r270": "r90",
    "fliph": "fliph",
    "flipv": "flipv",
    "tr": "tr",
    "atr": "atr",
}


def recolor(g: Grid, perm: list[int]) -> Grid:
    """Map colour ``c`` to ``perm[c]``."""
    return [[perm[c] for c in row] for row in g]


def _valid(g: Grid) -> bool:
    return bool(g) and len({len(r) for r in g}) == 1 and all(0 <= c <= 9 for r in g for c in r)


@dataclass
class GridTask:
    """Train pairs plus one test pair."""

    train: list[tuple[Grid, Grid]]
    test_input: Grid
    test_output: Grid
    aug: dict[str, object] = field(default_factory=dict)


def grid_variants(task: GridTask, n: int = 32, seed: int = 0) -> list[GridTask]:
    """Up to ``n`` distinct verified variants of ``task`` (excluding the original)."""
    rnd = random.Random(seed)
    combos: list[tuple[str, list[int]]] = [(g, list(range(10))) for g in DIHEDRAL if g != "id"]
    for _ in range(n * 3):
        p = list(range(1, 10))
        rnd.shuffle(p)
        combos.append((rnd.choice(list(DIHEDRAL)), [0, *p]))
    seen = {repr((task.train, task.test_input))}
    out: list[GridTask] = []
    for geo, perm in combos:
        if len(out) >= n:
            break
        inv_perm = [perm.index(i) for i in range(10)]

        def tf(g: Grid, geo: str = geo, perm: list[int] = perm) -> Grid:
            return recolor(DIHEDRAL[geo](g), perm)

        v = GridTask(
            [(tf(a), tf(b)) for a, b in task.train],
            tf(task.test_input),
            tf(task.test_output),
            {"geo": geo, "perm": perm},
        )
        if recolor(DIHEDRAL[INVERSE[geo]](v.test_output), inv_perm) != task.test_output:
            continue
        if not (_valid(v.test_input) and _valid(v.test_output)):
            continue
        key = repr((v.train, v.test_input))
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def choice_variants(options: list[str], gold: int, n: int = 8, seed: int = 0) -> list[tuple[list[str], int]]:
    """Up to ``n`` option permutations, each with the remapped (and re-checked) gold index."""
    if not 0 <= gold < len(options) or len(options) < 2:
        raise ValueError("gold index out of range or fewer than two options")
    rnd = random.Random(seed)
    seen = {tuple(options)}
    out: list[tuple[list[str], int]] = []
    for _ in range(n * 8):
        if len(out) >= n:
            break
        o = options[:]
        rnd.shuffle(o)
        if tuple(o) in seen:
            continue
        seen.add(tuple(o))
        new_gold = o.index(options[gold])
        if o[new_gold] == options[gold]:
            out.append((o, new_gold))
    return out
