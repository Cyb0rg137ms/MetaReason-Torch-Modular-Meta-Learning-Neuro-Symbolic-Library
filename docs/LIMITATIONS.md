# Limitations

* **Benchmarks are synthetic.** They use sinusoid regression and Gaussian N-way classification. No Omniglot or miniImageNet results are reported; the numbers in the 1.0 README for those datasets were not reproducible and have been removed. The `ConvEmbedder` backbone is provided but has not been benchmarked here.
* **EWC on a saturated classifier needs a huge λ.** When the model is ~100% confident on task A, the exact Fisher is tiny, so in our two-task benchmark λ had to reach 1e6–1e7 before forgetting was reduced substantially. This is a known property of the Fisher at confident optima, not a tuning trick.
* **The step verifier checks consistency, not correctness.** It catches steps that contradict the problem or earlier steps. Steps without compilable propositional content are marked `unchecked`. Natural-language parsing is regex-based (`x<k> = true`, `(x1 OR ~x2)`).
* **The SAT backend is DPLL.** It is fine for the tens-to-hundreds of variables typical of reasoning traces; use a CDCL solver (e.g. GraphSAT or PySAT) for large CNFs.
* **Backward chaining** is plain SLD resolution with loop checking on the current path and no tabling. It can be exponential on highly recursive programs; use `forward_chain` for full materialisation.
* Single-process CPU benchmarks only. GPU paths are standard PyTorch but are not tested in CI.
