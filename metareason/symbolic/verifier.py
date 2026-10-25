"""Neurosymbolic step verification with failure diagnosis.

A (neural) reasoner emits a trace of natural-language steps. Each step may
assert propositional facts, either assignments (``x3 = true``) or clauses
(``(x1 OR ~x2)``, also ``∨``, ``|``, ``¬``, ``-``, ``not``). The verifier
compiles the problem's base clauses plus everything asserted by steps
``1..i`` to CNF and decides it with an exact solver:

* SAT: step ``i`` is consistent (the model is checked clause by clause);
* UNSAT: step ``i`` is the **first failing step**. A minimal unsatisfiable
  core is computed and mapped back to its sources, so the diagnosis says
  *which earlier steps and which problem constraints* conflict with it, and
  the earliest step a re-try should restart from.

This is a gate for consistency, not for truth: a step with no compilable
constraint is reported as ``unchecked`` rather than accepted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from metareason.symbolic.sat import Clause, evaluate, solve, unsat_core

_LIT = re.compile(r"(¬|~|!|not\s+|-)?\s*x\s*(\d+)", re.IGNORECASE)
_ASSIGN = re.compile(r"\bx\s*(\d+)\s*(?:=|:=|is)\s*(true|false|1|0|t|f)\b", re.IGNORECASE)
_GROUP = re.compile(r"\(([^()]*)\)")
_SPLIT = re.compile(r"\bOR\b|∨|\|", re.IGNORECASE)


def parse_step(text: str) -> list[Clause]:
    """Clauses asserted by one step: unit clauses for assignments, plus clause groups."""
    clauses: list[Clause] = []
    for m in _ASSIGN.finditer(text):
        v = int(m.group(1))
        if v >= 1:
            clauses.append([v if m.group(2).lower() in {"true", "1", "t"} else -v])
    for group in _GROUP.findall(text):
        lits: list[int] = []
        for part in _SPLIT.split(group):
            lm = _LIT.search(part)
            if lm and int(lm.group(2)) >= 1:
                v = int(lm.group(2))
                lits.append(-v if lm.group(1) else v)
        if lits and len(_SPLIT.findall(group)) + 1 == len(lits):
            clauses.append(sorted(set(lits)))
    return clauses


@dataclass
class StepReport:
    """Verification outcome of one step."""

    index: int
    status: str  # "consistent" | "inconsistent" | "unchecked"
    clauses: list[Clause] = field(default_factory=list)


@dataclass
class Diagnosis:
    """Why the first failing step failed."""

    failing_step: int
    conflicting_steps: list[int]
    conflicting_base_clauses: list[int]
    core: list[Clause]
    restart_from: int

    def explain(self) -> str:
        """One-paragraph human-readable diagnosis."""
        others = [s for s in self.conflicting_steps if s != self.failing_step]
        parts = [f"step {self.failing_step + 1} is inconsistent"]
        if others:
            parts.append("with step(s) " + ", ".join(str(s + 1) for s in others))
        if self.conflicting_base_clauses:
            parts.append(f"and problem constraint(s) {[i + 1 for i in self.conflicting_base_clauses]}")
        return " ".join(parts) + f"; minimal conflict {self.core}; restart from step {self.restart_from + 1}."


@dataclass
class VerificationResult:
    """Per-step reports plus an optional diagnosis and final model."""

    steps: list[StepReport]
    diagnosis: Diagnosis | None
    model: dict[int, bool] | None

    @property
    def ok(self) -> bool:
        """No step is inconsistent."""
        return self.diagnosis is None


class StepVerifier:
    """Checks a reasoning trace against a CNF problem, step by step."""

    def __init__(self, base_clauses: list[Clause] | None = None) -> None:
        self.base = [list(c) for c in (base_clauses or [])]

    def verify(self, steps: list[str]) -> VerificationResult:
        """Verify steps in order and stop at the first inconsistent one."""
        clauses = list(self.base)
        origin: list[int] = [-1] * len(clauses)  # -1 = base problem
        reports: list[StepReport] = []
        model: dict[int, bool] | None = solve(clauses).model if clauses else {}
        if clauses and model is None:
            core = unsat_core(clauses)
            diag = Diagnosis(-1, [], core, [clauses[i] for i in core], 0)
            return VerificationResult([], diag, None)
        for i, text in enumerate(steps):
            new = parse_step(text)
            if not new:
                reports.append(StepReport(i, "unchecked"))
                continue
            clauses += new
            origin += [i] * len(new)
            res = solve(clauses)
            if res.sat:
                assert res.model is not None and evaluate(clauses, res.model)
                model = res.model
                reports.append(StepReport(i, "consistent", new))
                continue
            reports.append(StepReport(i, "inconsistent", new))
            core = unsat_core(clauses)
            steps_in_core = sorted({origin[j] for j in core if origin[j] >= 0})
            base_in_core = sorted(j for j in core if origin[j] < 0)
            restart = min((s for s in steps_in_core if s != i), default=i)
            return VerificationResult(
                reports, Diagnosis(i, steps_in_core, base_in_core, [clauses[j] for j in core], restart), None
            )
        return VerificationResult(reports, None, model)

    def verify_answer(self, assignment: dict[int, bool]) -> bool:
        """True iff ``assignment`` satisfies every base clause."""
        return evaluate(self.base, assignment)
