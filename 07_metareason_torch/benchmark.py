"""
benchmark.py
============
Performance benchmark script for MetaReason-Torch.
Measures symbolic rule engine fact derivation and resolution speed.
"""

import time
from metareason.symbolic.rule_engine import RuleEngine

def benchmark_rule_engine():
    engine = RuleEngine()
    
    # Define rules: ancestor(X, Y) :- parent(X, Y)
    # ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)
    engine.add_rule("ancestor(?x, ?y)", ["parent(?x, ?y)"])
    engine.add_rule("ancestor(?x, ?z)", ["parent(?x, ?y)", "ancestor(?y, ?z)"])
    
    # Add parent facts
    for i in range(10):
        engine.add_fact(f"parent(p{i}, p{i+1})")
        
    t0 = time.perf_counter()
    iterations = 50
    for _ in range(iterations):
        proof = engine.backward_chain("ancestor(p0, p5)")
        assert proof is not None
    elapsed = (time.perf_counter() - t0) / iterations
    print(f"MetaReason Rule Engine Time Ms: {elapsed * 1000.0:.3f}")

if __name__ == "__main__":
    print("Running MetaReason-Torch benchmarks...")
    benchmark_rule_engine()
