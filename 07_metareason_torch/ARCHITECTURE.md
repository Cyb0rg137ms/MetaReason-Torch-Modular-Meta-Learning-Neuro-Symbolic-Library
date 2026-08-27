# MetaReason-Torch — Architecture & Technical Reference

> **Full Project Name:** MetaReason-Torch — Meta-Learning & Continual Reasoning Engine
> **Category:** Advanced Deep Learning / Meta-Learning / Continual Learning
> **Language:** Python 3.9+, PyTorch
> **Test Coverage:** 3/3 unit tests passing ✅

---

## 1. Architecture Overview

```
07_metareason_torch/
├── metareason/
│   ├── maml.py       # MAML meta-learner (inner/outer loop, gradient surgery)
│   ├── ewc.py        # Elastic Weight Consolidation (Fisher information)
│   ├── nmn.py        # Neural Module Network (dynamic routing + soft selection)
│   └── __init__.py
├── tests/
│   └── test_metareason.py
├── train_demo.py     # End-to-end few-shot task demonstration
└── README.md
```

### Component Interaction

```
┌──────────────────────────────────────────────────────────────────┐
│                  METAREASON-TORCH PIPELINE                      │
│                                                                  │
│  TaskDistribution (support/query sets)                          │
│           │                                                      │
│           ▼                                                      │
│    ┌─────────────────┐                                          │
│    │   MAML Engine   │ ──► Fast inner loop gradient updates     │
│    │ (meta-learning) │ ──► Outer loop meta-parameter update     │
│    └─────────────────┘                                          │
│           │                                                      │
│           ▼                                                      │
│    ┌─────────────────┐                                          │
│    │  EWC Regularizer│ ──► Fisher Information Penalty          │
│    │ (anti-forgetting)│ ──► Penalizes drift from prior tasks    │
│    └─────────────────┘                                          │
│           │                                                      │
│           ▼                                                      │
│    ┌─────────────────┐                                          │
│    │   NMN Router    │ ──► Gating: select module per input      │
│    │ (compositional) │ ──► Forward: route through expert modules│
│    └─────────────────┘                                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Framework

### 2.1 MAML — Model-Agnostic Meta-Learning

MAML finds a single initialization `θ` that can be quickly adapted to any new task
using just a few gradient steps.

**Inner loop** — task-specific fast adaptation (k gradient steps per task):

```
θ_i'  =  θ  -  α × ∇_θ  Loss(task_i, model_θ)

α       = inner learning rate  (small, e.g. 0.01)
∇_θ     = gradient of the loss with respect to model parameters
θ_i'    = adapted parameters for task i  (task-specific, temporary)
```

**Outer loop** — update the shared initialization across all N tasks:

```
θ  ←  θ  -  β × ∇_θ  SUM over tasks T_i:  Loss(task_i, model_{θ_i'})

β       = outer (meta) learning rate
The outer gradient passes THROUGH the inner optimization steps
→ this requires computing second-order derivatives (Hessians)
```

The goal: find `θ` such that after just k=1 or k=5 inner steps on a new task,
the model already performs well on that task's test data.

### 2.2 MAML Gradient Surgery

When two tasks conflict (their gradients point in opposite directions), naively
averaging them degrades both. Gradient Surgery fixes this by projecting away conflict:

```
For tasks i, j where:  dot(grad_i, grad_j) < 0  (conflicting directions)

Corrected gradient for task i:
  grad_i'  =  grad_i  -  (dot(grad_i, grad_j) / ||grad_j||^2)  ×  grad_j

This removes the component of grad_i that opposes grad_j,
so both tasks can be updated without hurting each other.
```

### 2.3 EWC — Elastic Weight Consolidation

When a model learns task B after task A, it tends to forget task A (catastrophic forgetting).
EWC prevents this by adding a penalty for changing parameters that were important for task A.

```
Total loss when learning task B:

  L_total(θ)  =  L_B(θ)  +  (λ/2) × SUM over parameters i:
                               F_i × (θ_i  -  θ_i^A*)^2

  θ_i^A*   = optimal parameter values found when training on task A
  F_i       = Fisher information score for parameter i
  λ         = regularization strength  (how much to protect task A)
```

**Fisher information** `F_i` measures how sensitive task A's predictions are to
changes in parameter `i`:

```
F_i  =  Expected value of:  ( d/dθ_i  log P(y | x; θ) )^2

In practice, computed as: average squared gradient of log-likelihood
over the task A training data.

High F_i  →  parameter i is crucial for task A → large penalty if changed
Low F_i   →  parameter i barely matters for A → free to change for task B
```

### 2.4 NMN — Neural Module Network Routing

The NMN contains M specialized expert modules `{f_1, ..., f_M}` and a gating network
that routes each input to the most appropriate expert(s).

**Gating** — soft probability over which modules to use:

```
gate(x)  =  softmax( W_gate × x  +  b_gate )   →   vector of M probabilities

W_gate, b_gate  = learnable gating parameters
gate(x)[m]      = probability of routing input x to module m
```

**Output** — weighted mixture of expert module outputs:

```
y  =  SUM over m = 1..M:   gate(x)[m]  ×  f_m(x)

Optional: Top-K sparsification → only activate the K highest-probability modules
(e.g. K=2 out of M=8 experts), making the network cheaper to evaluate.
```

---

## 3. Workflow

```
Define task distribution  p(T)
        │
        ▼
Sample batch of N tasks from p(T)
        │
        ▼
For each task T_i:
    Run k inner gradient steps using support set
    → produces task-specific adapted params  θ_i'
        │
        ▼
Compute outer meta-loss:  test each θ_i' on query set
        │
        ▼
Apply gradient surgery:  remove conflicting gradient components
        │
        ▼
Meta-update shared init:  θ ← θ - β × ∇_θ(meta-loss)
        │
        ├── EWC active?  →  add Fisher penalty to loss
        │
        ▼
For inference, route input through NMN:
    gate(x) → select top-K expert modules
    output  = weighted sum of f_m(x) for active modules
```

---

## 4. System Design

| Component | Module | Role |
|-----------|--------|------|
| **Meta-Learner** | `maml.py` | Inner/outer loop gradient computation, gradient surgery |
| **Memory** | `ewc.py` | Fisher matrix computation, elastic penalty, parameter anchor |
| **Routing** | `nmn.py` | Module registry, gating network, MoE forward pass |
| **Demo** | `train_demo.py` | Sinusoidal few-shot regression demonstration |
| **Tests** | `test_metareason.py` | Adaptation correctness, EWC gradient bounds, routing shapes |

---

## 5. Key Advantages

| Advantage | Description |
|-----------|-------------|
| **Few-shot adaptation** | MAML adapts to new tasks in 1–5 gradient steps |
| **Anti-catastrophic forgetting** | EWC preserves critical parameters from prior tasks |
| **Compositional reasoning** | NMN routes sub-problems to specialized expert modules |
| **Gradient surgery** | Prevents task interference in multi-task meta-learning |
| **Model-agnostic** | Works with any differentiable PyTorch architecture |

---

## 6. Test Results

```
tests/test_metareason.py::test_maml_adaptation    PASSED
tests/test_metareason.py::test_ewc_regularization PASSED
tests/test_metareason.py::test_nmn_routing        PASSED
────────────────────────────────────────────────
3 passed in 0.59s
```

---

## 7. Quick Start

```bash
pip install torch numpy
pytest tests/
python train_demo.py
```

<div align="center">
  <a href="https://q.com"><img src="../../assets/https_q_com.png" width="80" /></a>
</div>
