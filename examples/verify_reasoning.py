"""Gate a reasoning trace with the neurosymbolic step verifier and get a failure diagnosis."""

from metareason import StepVerifier

# Problem: (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x3 OR NOT x2)
verifier = StepVerifier([[1, 2], [-1, 3], [-3, -2]])
trace = [
    "Assume x2 = true to satisfy the first clause.",
    "Then (NOT x3 OR NOT x2) forces x3 = false.",
    "To be safe, also take x1 = true.",  # conflicts: x1 -> x3, but x3 is false
    "Done.",
]
result = verifier.verify(trace)
for step in result.steps:
    print(f"step {step.index + 1}: {step.status}  {step.clauses}")
if result.diagnosis:
    print(result.diagnosis.explain())

fixed = [*trace[:2], "So x1 = false."]
ok = verifier.verify(fixed)
print("fixed trace ok:", ok.ok, "model:", ok.model, "answer verified:", verifier.verify_answer(ok.model or {}))
