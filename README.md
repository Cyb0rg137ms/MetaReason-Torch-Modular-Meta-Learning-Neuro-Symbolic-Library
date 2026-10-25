# MetaReason-Torch

**Meta-learning, continual learning and *verified* neurosymbolic reasoning in PyTorch.**

MetaReason-Torch combines gradient-based meta-learning with symbolic checks that can say *where* a chain of reasoning went wrong. It covers:

- **Meta-learning:**
  - MAML (second-order) and FOMAML for **any** `nn.Module`, built on `torch.func.functional_call`.
  - Prototypical Networks and ANIL.
- **Continual learning:** EWC with an empirical or exact Fisher, multiple tasks, and online EWC.
- **Neurosymbolic step verification:**
  - Each step of a reasoning trace is compiled to CNF and decided by an exact solver.
  - The *first failing step* is reported, along with a minimal unsatisfiable core that names the conflicting earlier steps and problem constraints and the step to restart from.
- **Verified task amplification:**
  - ARC-style grid tasks get 8 dihedral transforms × colour permutations.
  - Multiple-choice tasks get option permutations.
  - A variant is kept only if inverting the transform recovers the gold answer exactly.
- **Symbolic substrate:**
  - A Datalog engine with semi-naive forward chaining and SLD backward chaining that supports recursion and non-ground queries.
  - Neural Module Networks with a module registry.

Architecture diagrams are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and known limitations are in [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

> **CI:** the workflow lives in `ci/github-actions-ci.yml`. Move it to `.github/workflows/ci.yml` to enable GitHub Actions. The push token lacked the `workflow` scope.

## Install
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA/ROCm build
pip install -e ".[dev]"
```

## Quickstart
```python
import torch
from torch import nn
from metareason import MAML, StepVerifier
from metareason.tasks import SinusoidTasks

model = nn.Sequential(nn.Linear(1, 40), nn.ReLU(), nn.Linear(40, 1))
maml = MAML(model, inner_lr=0.01, inner_steps=1)  # first_order=True -> FOMAML
opt = torch.optim.Adam(model.parameters(), 1e-3)
for _ in range(100):
    maml.meta_step(SinusoidTasks().batch(8), opt)

v = StepVerifier([[1, 2], [-1, 3], [-3, -2]])  # CNF problem
r = v.verify(["Assume x2 = true.", "Then x3 = false.", "Also x1 = true."])
print(r.diagnosis.explain())
# step 3 is inconsistent with step(s) 2 and problem constraint(s) [2];
# minimal conflict [[-1, 3], [-3], [1]]; restart from step 2.
```
```bash
metareason verify examples/trace.json          # exit 1 + diagnosis on an inconsistent trace
metareason bench all --config examples/bench.toml
```
More examples are in [`examples/`](examples).

## Measured results
These are the results of `metareason bench all` with seed 0 and the defaults in `examples/bench.toml`. The run was on a CPU-only Intel Xeon VM with PyTorch 2.14.1 and took 32 s wall-clock. Confidence intervals are Student-t 95%.

**Sinusoid regression** (10-shot, 2000 meta-iterations, meta-batch 8, 300 held-out tasks; MSE on query points after adaptation):

| method | MSE after 1 step | MSE after 10 steps | ms / meta-iter |
|---|---|---|---|
| MAML (2nd order) | **1.180 ± 0.143** | **0.576 ± 0.084** | 5.3 |
| FOMAML | 1.783 ± 0.206 | 1.037 ± 0.158 | 3.7 |
| Joint pre-training + fine-tune | 2.489 ± 0.307 | 1.674 ± 0.247 | 1.3 |

**Few-shot classification** (synthetic 5-way 1-shot Gaussian classes, 16-d, 1000 training episodes, 600 test episodes):

| method | accuracy % | ms / episode |
|---|---|---|
| ProtoNet | **93.80 ± 0.43** | 1.0 |
| ANIL (5 head steps) | 90.29 ± 0.57 | 1.8 |
| Untrained embedder + nearest prototype | 79.41 ± 0.80 | — |

**Catastrophic forgetting with EWC** (exact Fisher; two 5-way tasks trained one after the other with the same output head; accuracy on task A):

| λ | A before B | A after B | B |
|---|---|---|---|
| 0 | 99.8 | 21.2 | 99.8 |
| 1e5 | 99.8 | 22.8 | 99.8 |
| 1e6 | 99.8 | 45.0 | 99.7 |
| 1e7 | 99.8 | **82.9** | 99.3 |

The ordering MAML > FOMAML > pre-training and ProtoNet > ANIL matches the literature qualitatively. These are synthetic tasks, so they are not comparable to Omniglot or miniImageNet numbers.

## Bugs fixed relative to 1.0
- **ANIL:** the feature extractor ran under `torch.no_grad()`, so the outer loop never trained it.
- **Rule engine:** backward chaining could not prove recursive goals unless `forward_chain()` had already run, because body variables were only bound against materialised facts. Rule variables were also never renamed apart. The engine is now SLD resolution with loop checking.
- **ProtoNet:**
  - The loss assumed queries were sorted by class.
  - Prototypes assumed balanced, sortable support sets.
  - Labels are now explicit; the old call style still works with a deprecation warning.
- **EWC:**
  - The penalty was a Python `float` before any task was registered.
  - The model was left in `eval()` mode.
  - Shape handling made `MSELoss` broadcast silently.
- **Confidence intervals:** they used a wrong t-value (2.045 for every n < 30).
- **NMN:** string leaves were indexed character by character.
- **MAML:**
  - It was limited to one hard-coded MLP and a single inner step, with no outer loop.
  - It now has a generic functional implementation with k steps, a first-order mode and `meta_step`.
- **Removed:** fabricated `BENCHMARK_RESULTS.md`, such as the Omniglot numbers that were never measured.

## Development
```bash
ruff check . && ruff format --check . && mypy metareason && pytest
```
There are 74 tests. They cover:
- MAML against hand-computed SGD and finite-difference second-order gradients
- SAT against brute force on 300 random formulas, and minimality of the unsat core
- forward and backward chaining agreement on random programs
- the amplification inverse round-trip
- the CLI

## Credits
The step verifier and amplification modules are clean re-implementations of ideas from the author's research prototypes (CNF-gated step verification with retry; verified dihedral/colour augmentation of ARC tasks). Method references: Finn et al. 2017 (MAML), Snell et al. 2017 (ProtoNets), Raghu et al. 2020 (ANIL), Kirkpatrick et al. 2017 (EWC), Schwarz et al. 2018 (online EWC), Andreas et al. 2016 (NMN).

## License
MIT
