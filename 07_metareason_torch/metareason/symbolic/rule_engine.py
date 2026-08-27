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

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Generator, List, Optional, Set, Tuple


Atom = Tuple[str, ...]  # (predicate, arg1, arg2, ...)
Substitution = Dict[str, str]  # variable bindings: {"X": "alice", "Y": "bob"}


def is_variable(term: str) -> bool:
    """Returns True if the term is a logic variable (uppercase first letter)."""
    return term[0].isupper() if term else False


# ---------------------------------------------------------------------------
# Unification
# ---------------------------------------------------------------------------

def unify(
    pattern: Atom,
    ground: Atom,
    bindings: Optional[Substitution] = None,
) -> Optional[Substitution]:
    """
    Attempts to unify a pattern atom (may contain variables) with a ground atom.

    Args:
        pattern: Pattern tuple with possible variable terms.
        ground: Fully ground atom (no variables).
        bindings: Existing variable bindings to extend.

    Returns:
        Extended substitution if unification succeeds, else None.
    """
    if bindings is None:
        bindings = {}

    if len(pattern) != len(ground):
        return None

    result = dict(bindings)
    for p_term, g_term in zip(pattern, ground):
        if is_variable(p_term):
            if p_term in result:
                if result[p_term] != g_term:
                    return None  # Conflicting binding
            else:
                result[p_term] = g_term
        else:
            if p_term != g_term:
                return None  # Constant mismatch

    return result


def apply_substitution(atom: Atom, bindings: Substitution) -> Atom:
    """Applies variable substitutions to an atom, grounding variables."""
    return tuple(bindings.get(term, term) if is_variable(term) else term for term in atom)


# ---------------------------------------------------------------------------
# Proof tree
# ---------------------------------------------------------------------------

@dataclass
class ProofNode:
    """A node in the backward-chaining proof tree."""
    goal: Atom
    rule_head: Optional[Atom] = None
    children: List["ProofNode"] = field(default_factory=list)
    is_fact: bool = False

    def to_str(self, indent: int = 0) -> str:
        prefix = "  " * indent
        kind = "[FACT]" if self.is_fact else "[RULE]"
        lines = [f"{prefix}{kind} {self.goal}"]
        for child in self.children:
            lines.append(child.to_str(indent + 1))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ProofNode(goal={self.goal}, children={len(self.children)})"


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """A Horn clause: head :- body1, body2, ..."""
    head: Atom
    body: List[Atom]

    def __repr__(self) -> str:
        body_str = ", ".join(str(b) for b in self.body)
        return f"{self.head} :- {body_str}"


class RuleEngine:
    """
    Datalog rule engine supporting forward chaining and backward chaining.

    Variable convention: terms starting with uppercase are variables.
    Constants start with lowercase.
    """

    def __init__(self) -> None:
        self._facts: Set[Atom] = set()
        self._original_facts: Set[Atom] = set()
        self._rules: List[Rule] = []

    # ------------------------------------------------------------------
    # Knowledge base construction
    # ------------------------------------------------------------------

    def add_fact(self, predicate: str, *args: str) -> None:
        """Adds a ground fact to the knowledge base."""
        fact = (predicate,) + args
        self._facts.add(fact)
        self._original_facts.add(fact)

    def add_rule(self, head: Atom, body: List[Atom]) -> None:
        """
        Adds a Horn clause rule.

        Args:
            head: Head atom (may contain variables).
            body: List of body atoms (may contain variables).
        """
        self._rules.append(Rule(head=head, body=body))

    def facts(self) -> FrozenSet[Atom]:
        """Returns the current set of known facts (including derived)."""
        return frozenset(self._facts)

    # ------------------------------------------------------------------
    # Forward chaining (bottom-up)
    # ------------------------------------------------------------------

    def forward_chain(self, max_iterations: int = 100) -> Set[Atom]:
        """
        Derives all facts reachable from current facts via rules.

        Runs to fixpoint (no new facts derived) or max_iterations.

        Returns:
            Full derived fact set (including original facts).
        """
        derived = set(self._facts)

        for _ in range(max_iterations):
            new_facts: Set[Atom] = set()

            for rule in self._rules:
                for subst in self._match_body(rule.body, derived):
                    new_head = apply_substitution(rule.head, subst)
                    if new_head not in derived:
                        new_facts.add(new_head)

            if not new_facts:
                break  # Fixpoint reached
            derived |= new_facts

        self._facts = derived
        return derived

    def _match_body(
        self,
        body: List[Atom],
        facts: Set[Atom],
        bindings: Optional[Substitution] = None,
    ) -> Generator[Substitution, None, None]:
        """
        Recursively matches all body atoms against the fact base.

        Yields substitutions that satisfy all body atoms simultaneously.
        """
        if bindings is None:
            bindings = {}

        if not body:
            yield bindings
            return

        head_atom, *rest = body
        for fact in facts:
            subst = unify(head_atom, fact, bindings)
            if subst is not None:
                yield from self._match_body(rest, facts, subst)

    # ------------------------------------------------------------------
    # Backward chaining (top-down)
    # ------------------------------------------------------------------

    def backward_chain(
        self,
        goal: Atom,
        max_depth: int = 20,
    ) -> Optional[ProofNode]:
        """
        Proves a goal atom using backward chaining with proof-tree construction.

        Args:
            goal: The ground atom to prove.
            max_depth: Maximum proof depth (prevents infinite recursion).

        Returns:
            ProofNode if provable, else None.
        """
        return self._bc_prove(goal, max_depth)

    def _bc_prove(
        self,
        goal: Atom,
        depth: int,
    ) -> Optional[ProofNode]:
        if depth == 0:
            return None

        # Check if goal is directly a known base fact
        if goal in self._original_facts:
            return ProofNode(goal=goal, is_fact=True)

        # Try rules
        for rule in self._rules:
            subst = unify(rule.head, goal)
            if subst is None:
                continue

            # Find all satisfying groundings of the body from the fact database
            for body_subst in self._match_body(rule.body, self._facts, subst):
                children: List[ProofNode] = []
                success = True
                for body_atom in rule.body:
                    grounded_body = apply_substitution(body_atom, body_subst)
                    child_proof = self._bc_prove(grounded_body, depth - 1)
                    if child_proof is None:
                        success = False
                        break
                    children.append(child_proof)

                if success:
                    return ProofNode(
                        goal=goal,
                        rule_head=rule.head,
                        children=children,
                    )

    def query(self, predicate: str, *args: str) -> bool:
        """Convenience method: checks if a ground fact is provable."""
        goal = (predicate,) + args
        return self.backward_chain(goal) is not None

    def explain(self, predicate: str, *args: str) -> str:
        """Returns a human-readable proof explanation for a goal."""
        goal = (predicate,) + args
        proof = self.backward_chain(goal)
        if proof is None:
            return f"NOT PROVABLE: {goal}"
        return proof.to_str()
