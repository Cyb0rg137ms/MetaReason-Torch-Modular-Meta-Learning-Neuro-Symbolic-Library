"""Small, dependable SAT utilities used by the step verifier.

A complete DPLL solver with unit propagation and pure-literal-free branching
on the most frequent variable. It is meant for the small constraint sets a
reasoning trace produces (tens to a few hundred variables), not for hard
industrial instances. Every SAT answer is checked against the clauses.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

Clause = list[int]


@dataclass(frozen=True)
class SatResult:
    """Outcome of :func:`solve`. ``model[v]`` is the value of variable ``v`` (1-based)."""

    sat: bool
    model: dict[int, bool] | None
    decisions: int


def evaluate(clauses: list[Clause], model: dict[int, bool]) -> bool:
    """True iff every clause has a literal made true by ``model`` (unassigned = false)."""
    return all(any(model.get(abs(lit), False) == (lit > 0) for lit in c) for c in clauses)


def solve(clauses: list[Clause], max_decisions: int = 1_000_000) -> SatResult:
    """Decide satisfiability. Raises ``RuntimeError`` if the decision budget runs out."""
    for c in clauses:
        if any(lit == 0 for lit in c):
            raise ValueError("literal 0 is not allowed")
    decisions = 0
    variables = {abs(lit) for c in clauses for lit in c}

    def propagate(cls: list[frozenset[int]], assign: dict[int, bool]) -> list[frozenset[int]] | None:
        while True:
            unit = None
            out: list[frozenset[int]] = []
            for c in cls:
                if any(assign.get(abs(lit)) == (lit > 0) for lit in c):
                    continue
                rest = frozenset(lit for lit in c if abs(lit) not in assign)
                if not rest:
                    return None
                if len(rest) == 1 and unit is None:
                    unit = next(iter(rest))
                out.append(rest)
            if unit is None:
                return out
            assign[abs(unit)] = unit > 0
            cls = out

    stack: list[tuple[list[frozenset[int]], dict[int, bool]]] = [([frozenset(c) for c in clauses], {})]
    while stack:
        cls, assign = stack.pop()
        reduced = propagate(cls, assign)
        if reduced is None:
            continue
        if not reduced:
            model = {v: assign.get(v, False) for v in variables}
            assert evaluate(clauses, model)
            return SatResult(True, model, decisions)
        decisions += 1
        if decisions > max_decisions:
            raise RuntimeError("decision budget exhausted")
        var, _ = Counter(abs(lit) for c in reduced for lit in c).most_common(1)[0]
        pos = sum(1 for c in reduced if var in c) >= sum(1 for c in reduced if -var in c)
        stack.append((reduced, {**assign, var: not pos}))
        stack.append((reduced, {**assign, var: pos}))
    return SatResult(False, None, decisions)


def unsat_core(clauses: list[Clause]) -> list[int]:
    """Indices of a minimal unsatisfiable subset (deletion-based, MUS).

    Removing any clause of the returned subset makes it satisfiable.
    Raises ``ValueError`` if the input is satisfiable.
    """
    if solve(clauses).sat:
        raise ValueError("formula is satisfiable; no core")
    core = list(range(len(clauses)))
    i = 0
    while i < len(core):
        trial = core[:i] + core[i + 1 :]
        if not solve([clauses[j] for j in trial]).sat:
            core = trial
        else:
            i += 1
    return core
