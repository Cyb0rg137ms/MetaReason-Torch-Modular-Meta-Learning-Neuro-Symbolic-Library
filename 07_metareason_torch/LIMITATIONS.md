"""
LIMITATIONS.md — MetaReason-Torch
=================================

# Known Limitations and Future Work

## Meta-Learning (MAML / ANIL / ProtoNets)

### 1. Approximate Second-Order Gradients
Standard MAML requires calculating Hessian-vector products (second-order derivatives) during the outer meta-update. Our implementation uses PyTorch's default `create_graph=True` which tracks the full computational graph. This is:
- **Memory-heavy**: scale of $O(N)$ with number of inner steps.
- **Slow**: ~5× slower than first-order MAML.

In production, one should use **First-Order MAML (FOMAML)** or **Reptile**, which approximate the meta-gradient using only first-order gradients from the end of the inner loop, reducing memory and step time.

### 2. Inner-Loop Optimizer Stiffness
Our MAML and ANIL use standard SGD for the inner loop. In practice, SGD suffers from "stiffness" in the inner loop on complex loss landscapes. Modern solutions employ **meta-optimizers** (like Meta-SGD or ALFA) that learn a coordinate-wise step size or parameter-specific inner learning rates.

### 3. ProtoNets Embedding Collapse
Prototypical Networks assume a static metric space (Euclidean distance). If the embedding network `embedder` is not regularised (e.g. via weight decay or dropout), it can suffer from representation collapse, mapping different classes to overlapping regions.

---

## Continual Learning (EWC)

### 1. Diagonal Fisher Information Approximation
Elastic Weight Consolidation (EWC) assumes the Fisher Information Matrix (FIM) is diagonal. This ignores cross-parameter correlations:
- Crucial correlations between weights in consecutive layers are neglected.
- The quadratic penalty does not capture the true curvature of the parameter space.

Using K-FAC (Kronecker-factored Approximate Curvature) would resolve this but increases computational complexity.

### 2. Linear Memory Growth
EWC requires storing the optimal weights $\theta_A^*$ and the diagonal Fisher matrix $F_A$ for every single task trained. For a sequence of $T$ tasks, memory scales as $O(T \times P)$ where $P$ is the parameter count. For large models (millions of parameters), this becomes prohibitive.

---

## Symbolic Rule Engine

### 1. Naive Unification and Pattern Matching
Our Horn clause rule engine uses a simple recursive matcher (`_match_body`). It checks all ground facts against the rule body:
- **Complexity**: $O(F^B)$ where $F$ is the number of facts and $B$ is the number of body atoms.
- For $F > 1000$, forward-chaining fixpoint calculation becomes extremely slow.

Production engines (like CLIPS or Datalog systems) use the **Rete algorithm** or **semi-naive evaluation** to match rules only against newly derived facts rather than repeating matches on all facts.

### 2. No Infinite Recursion Protection in Backward Chaining
Our backward chainer uses a simple depth limit (`max_depth = 20`) to prevent infinite recursion. It does not perform loop detection (e.g., if rule is $A \rightarrow B$ and $B \rightarrow A$).
"""
