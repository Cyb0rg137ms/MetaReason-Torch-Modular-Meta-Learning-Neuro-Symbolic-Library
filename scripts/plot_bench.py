"""Render docs/assets/bench.png from a measured `metareason bench all --json` result.

Usage: python scripts/plot_bench.py [docs/assets/bench.json] [docs/assets/bench.png]
The JSON is a dict with keys "sinusoid", "fewshot" and "ewc", each the row list printed by the CLI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def plot(data: dict, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    s = data["sinusoid"]
    names = [r["method"] for r in s]
    x = range(len(s))
    axes[0].bar([i - 0.2 for i in x], [r["mse_1step"] for r in s], 0.4, yerr=[r["ci_1step"] for r in s], label="1 step")
    axes[0].bar(
        [i + 0.2 for i in x], [r["mse_10step"] for r in s], 0.4, yerr=[r["ci_10step"] for r in s], label="10 steps"
    )
    axes[0].set_xticks(list(x), names)
    axes[0].set_ylabel("query MSE (lower is better)")
    axes[0].set_title("Sinusoid 10-shot regression")
    axes[0].legend()
    f = data["fewshot"]
    axes[1].bar([r["method"] for r in f], [r["acc"] for r in f], yerr=[r["ci95"] for r in f], color="tab:green")
    axes[1].set_ylim(50, 100)
    axes[1].set_ylabel("accuracy % (95% CI)")
    axes[1].set_title("5-way 1-shot (synthetic)")
    e = data["ewc"]
    lams = [max(r["lambda"], 1.0) for r in e]
    axes[2].semilogx(lams, [r["acc_A_after"] for r in e], "o-", label="task A after B")
    axes[2].semilogx(lams, [r["acc_B"] for r in e], "s--", label="task B")
    axes[2].set_xlabel("EWC λ (λ=0 drawn at 1)")
    axes[2].set_ylabel("accuracy %")
    axes[2].set_title("Forgetting vs EWC strength")
    axes[2].legend()
    fig.suptitle("metareason bench all — seed 0, CPU (measured)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/assets/bench.json")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/assets/bench.png")
    print(plot(json.loads(src.read_text()), dst))
