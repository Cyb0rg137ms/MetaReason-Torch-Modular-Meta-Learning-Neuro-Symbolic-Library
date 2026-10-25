# Architecture

```mermaid
flowchart LR
    subgraph meta[metareason.meta]
        MAML["MAML / FOMAML<br/>functional_call, any nn.Module"]
        PROTO["PrototypicalNetwork<br/>label-indexed prototypes"]
        ANIL["ANIL<br/>head-only inner loop"]
        AMP["amplify<br/>verified dihedral x colour / option permutations"]
    end
    subgraph continual[metareason.continual]
        EWC["EWC<br/>empirical / exact Fisher, multi-task, online"]
    end
    subgraph symbolic[metareason.symbolic]
        RULES["RuleEngine<br/>semi-naive forward + SLD backward"]
        NMN["NeuralModuleNetwork<br/>program -> module registry"]
        SAT["sat: DPLL + MUS core"]
        VER["StepVerifier<br/>first failing step + diagnosis"]
    end
    TASKS["tasks: seeded Sinusoid / Gaussian N-way"] --> MAML & PROTO & ANIL
    VER --> SAT
    BENCH["benchmarks + BenchConfig (TOML)"] --> TASKS & EWC
    CLI["metareason CLI: bench | verify"] --> BENCH & VER
```

## Step verification flow

```mermaid
sequenceDiagram
    participant R as Reasoner (LLM / policy)
    participant V as StepVerifier
    participant S as DPLL solver
    R->>V: steps 1..n (text)
    loop each step i
        V->>V: parse assignments + clause groups
        alt no constraint
            V-->>R: step i "unchecked"
        else
            V->>S: base CNF + steps 1..i
            alt SAT (model re-checked)
                V-->>R: step i "consistent"
            else UNSAT
                V->>S: deletion-based minimal unsat core
                V-->>R: Diagnosis(failing step, conflicting steps, base clauses, restart_from)
            end
        end
    end
```

## Design choices

* **Functional inner loops.** MAML never mutates the model; adapted parameters are dicts fed to `torch.func.functional_call`, so any architecture, including ones with buffers, works, and evaluation can call `adapt` inside `torch.no_grad()`.
* **Exactness over speed in the symbolic layer.** The SAT solver is complete and every model is re-checked, and the unsat core is provably minimal (each clause is necessary). Traces are small, so the quadratic MUS search is cheap.
* **Verified augmentation.** A variant is kept only if inverting the transform reproduces the original gold answer. This guards against label-corrupting augmentations.
* **Back-compat.** The 1.0 public API (`MetaModel`, `MAML.adapt_to_task`, `compute_prototypes(z, n_way=, k_shot=)`, `fisher_matrix`, `optimal_weights`) still works. The 1.0 tests are kept and pass.
