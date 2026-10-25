"""
rule_engine.py
==============
First-order logic rule engine with forward and backward chaining.

Implements a Datalog-style production rule system where:
  - Facts are ground atoms: Predicate(arg1, arg2, ...)
  - Rules are Horn clauses: Head :- Body1, Body2, ...
  - Forward chaining derives all provable facts from a fact base.
  - Backward chaining proves goal queries with explanation traces.

Example::

    engine = RuleEngine()

    # Define facts
    engine.add_fact("parent", "alice", "bob")
    engine.add_fact("parent", "bob", "charlie")

    # Define rules
    engine.add_rule(
        head=("ancestor", "X", "Y"),
        body=[("parent", "X", "Y")]
    )
    engine.add_rule(
        head=("ancestor", "X", "Z"),
        body=[("parent", "X", "Y"), ("ancestor", "Y", "Z")]
    )

    derived = engine.forward_chain()
    # → includes ("ancestor", "alice", "charlie")

    proof = engine.backward_chain(("ancestor", "alice", "charlie"))
    # → [ProofNode explaining derivation]
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass, field

Atom = tuple[str, ...]  # (predicate, arg1, arg2, ...)
Substitution = dict[str, str]  # variable bindings: {"X": "alice"}


def is_variable(term: str) -> bool:
    """True for logic variables: terms starting with an uppercase letter or ``_``."""
    return bool(term) and (term[0].isupper() or term[0] == "_")


def _walk(term: str, s: Substitution) -> str:
    while is_variable(term) and term in s:
        term = s[term]
    return term


def unify(pattern: Atom, ground: Atom, bindings: Substitution | None = None) -> Substitution | None:
    """Unify two atoms (either side may contain variables).

    Returns the extended substitution, or ``None`` if they do not unify. The
    input substitution is never mutated.
    """
    if len(pattern) != len(ground):
        return None
    result: Substitution = dict(bindings or {})
    for a, b in zip(pattern, ground, strict=True):
        a, b = _walk(a, result), _walk(b, result)
        if a == b:
            continue
        if is_variable(a):
            result[a] = b
        elif is_variable(b):
            result[b] = a
        else:
            return None
    return result


def apply_substitution(atom: Atom, bindings: Substitution) -> Atom:
    """Apply (transitively) a substitution to an atom."""
    return tuple(_walk(t, bindings) for t in atom)


@dataclass
class ProofNode:
    """A node in a proof tree."""

    goal: Atom
    rule_head: Atom | None = None
    children: list[ProofNode] = field(default_factory=list)
    is_fact: bool = False

    def to_str(self, indent: int = 0) -> str:
        """Indented, human-readable proof."""
        kind = "[FACT]" if self.is_fact else "[RULE]"
        lines = [f"{'  ' * indent}{kind} {self.goal}"]
        lines += [c.to_str(indent + 1) for c in self.children]
        return "\n".join(lines)

    def depth(self) -> int:
        """Height of the proof tree."""
        return 1 + max((c.depth() for c in self.children), default=0)

    def __repr__(self) -> str:
        return f"ProofNode(goal={self.goal}, children={len(self.children)})"


@dataclass
class Rule:
    """A Horn clause ``head :- body1, body2, ...``."""

    head: Atom
    body: list[Atom]

    def __post_init__(self) -> None:
        head_vars = {t for t in self.head[1:] if is_variable(t)}
        body_vars = {t for a in self.body for t in a[1:] if is_variable(t)}
        if not head_vars <= body_vars:
            raise ValueError(f"unsafe rule {self}: head variables {head_vars - body_vars} do not occur in the body")

    def __repr__(self) -> str:
        return f"{self.head} :- {', '.join(str(b) for b in self.body)}"


class RuleEngine:
    """Datalog engine: semi-naive forward chaining and SLD backward chaining.

    Backward chaining standardises rule variables apart, so recursive rules
    work without forward chaining first (a bug in 1.0), and it detects loops
    on the current proof path, so left-recursive rules terminate.
    """

    def __init__(self) -> None:
        self._facts: set[Atom] = set()
        self._original_facts: set[Atom] = set()
        self._rules: list[Rule] = []
        self._fresh = itertools.count()

    def add_fact(self, predicate: str, *args: str) -> None:
        """Add a ground fact."""
        fact = (predicate, *args)
        if any(is_variable(a) for a in args):
            raise ValueError(f"facts must be ground: {fact}")
        self._facts.add(fact)
        self._original_facts.add(fact)

    def add_rule(self, head: Atom, body: list[Atom]) -> None:
        """Add a Horn rule (must be range-restricted)."""
        self._rules.append(Rule(head=tuple(head), body=[tuple(b) for b in body]))

    def facts(self) -> frozenset[Atom]:
        """Known facts, including any derived by :meth:`forward_chain`."""
        return frozenset(self._facts)

    # ---------------------------------------------------------------- forward
    def forward_chain(self, max_iterations: int = 1000) -> set[Atom]:
        """Semi-naive bottom-up evaluation to the least fixpoint."""
        derived = set(self._facts)
        delta = set(derived)
        for _ in range(max_iterations):
            new: set[Atom] = set()
            for rule in self._rules:
                for i in range(len(rule.body)):
                    # at least one body atom must come from the last round's delta
                    for s in self._match(rule.body[i : i + 1], delta, {}):
                        for s2 in self._match(rule.body[:i] + rule.body[i + 1 :], derived, s):
                            h = apply_substitution(rule.head, s2)
                            if h not in derived:
                                new.add(h)
            if not new:
                break
            derived |= new
            delta = new
        self._facts = derived
        return derived

    @staticmethod
    def _match(body: list[Atom], facts: set[Atom], s: Substitution) -> Iterator[Substitution]:
        if not body:
            yield s
            return
        first, rest = body[0], body[1:]
        for fact in facts:
            if fact[0] == first[0]:
                s2 = unify(first, fact, s)
                if s2 is not None:
                    yield from RuleEngine._match(rest, facts, s2)

    # --------------------------------------------------------------- backward
    def _rename(self, rule: Rule) -> Rule:
        k = next(self._fresh)
        ren = lambda a: tuple(f"{t}_{k}" if is_variable(t) else t for t in a)
        r = Rule.__new__(Rule)
        r.head, r.body = ren(rule.head), [ren(b) for b in rule.body]
        return r

    def solve(self, goal: Atom, max_depth: int = 50) -> Iterator[tuple[Substitution, ProofNode]]:
        """Yield every ``(bindings, proof)`` for a goal that may contain variables."""
        yield from self._prove(tuple(goal), {}, max_depth, ())

    def _prove(
        self, goal: Atom, s: Substitution, depth: int, path: tuple[Atom, ...]
    ) -> Iterator[tuple[Substitution, ProofNode]]:
        g = apply_substitution(goal, s)
        if depth == 0 or g in path:
            return
        for fact in self._original_facts:
            if fact[0] == g[0]:
                s2 = unify(g, fact, s)
                if s2 is not None:
                    yield s2, ProofNode(goal=fact, is_fact=True)
        for rule in self._rules:
            if rule.head[0] != g[0]:
                continue
            r = self._rename(rule)
            s2 = unify(r.head, g, s)
            if s2 is not None:
                for s3, kids in self._prove_all(r.body, s2, depth - 1, (*path, g)):
                    yield s3, ProofNode(goal=apply_substitution(g, s3), rule_head=rule.head, children=kids)

    def _prove_all(
        self, goals: list[Atom], s: Substitution, depth: int, path: tuple[Atom, ...]
    ) -> Iterator[tuple[Substitution, list[ProofNode]]]:
        if not goals:
            yield s, []
            return
        for s2, node in self._prove(goals[0], s, depth, path):
            for s3, rest in self._prove_all(goals[1:], s2, depth, path):
                yield s3, [node, *rest]

    def backward_chain(self, goal: Atom, max_depth: int = 50) -> ProofNode | None:
        """First proof of ``goal`` found by SLD resolution, or ``None``."""
        return next((p for _, p in self.solve(goal, max_depth)), None)

    def query(self, predicate: str, *args: str) -> bool:
        """True if the goal is provable."""
        return self.backward_chain((predicate, *args)) is not None

    def explain(self, predicate: str, *args: str) -> str:
        """Human-readable proof, or ``NOT PROVABLE``."""
        goal = (predicate, *args)
        proof = self.backward_chain(goal)
        return f"NOT PROVABLE: {goal}" if proof is None else proof.to_str()
