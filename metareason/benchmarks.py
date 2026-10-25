"""Reproducible benchmarks. Every number in the README comes from these functions.

Run ``metareason bench all`` (or ``python -m metareason bench all``).
"""

from __future__ import annotations

import time
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from metareason.continual.ewc import ElasticWeightConsolidation
from metareason.meta.maml import MAML, Task
from metareason.meta.proto_maml import ANIL, ANILModel, LinearEmbedder, PrototypicalNetwork, compute_confidence_interval
from metareason.tasks import GaussianClassTasks, SinusoidTasks, seed_everything


@dataclass
class BenchConfig:
    """Benchmark hyper-parameters (load overrides with :meth:`from_toml`)."""

    seed: int = 0
    meta_iters: int = 2000
    meta_batch: int = 8
    eval_tasks: int = 300
    inner_lr: float = 0.01
    inner_steps: int = 1
    eval_steps: int = 10
    outer_lr: float = 1e-3
    episodes: int = 1000
    eval_episodes: int = 600
    n_way: int = 5
    k_shot: int = 1

    @classmethod
    def from_toml(cls, path: str | Path) -> BenchConfig:
        """Load ``[bench]`` overrides from a TOML file."""
        data = tomllib.loads(Path(path).read_text()).get("bench", {})
        names = {f.name for f in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"unknown bench keys: {sorted(unknown)}")
        return cls(**data)

    def quick(self) -> BenchConfig:
        """Tiny settings for CI smoke runs."""
        return BenchConfig(**{**asdict(self), "meta_iters": 50, "eval_tasks": 20, "episodes": 50, "eval_episodes": 30})


def _mlp(i: int, h: int, o: int) -> nn.Module:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


def _eval_regression(model: nn.Module, tasks: list[Task], lr: float, steps: int) -> tuple[float, float]:
    learner = MAML(model, inner_lr=lr, inner_steps=steps, first_order=True)
    losses = []
    for t in tasks:
        p = learner.adapt(t.support_x, t.support_y, create_graph=False)
        with torch.no_grad():
            losses.append(float(F.mse_loss(learner.forward(t.query_x, p), t.query_y)))
    return compute_confidence_interval(losses)


def bench_sinusoid(cfg: BenchConfig) -> list[dict[str, Any]]:
    """MAML vs FOMAML vs joint pre-training on sinusoid regression (10-shot)."""
    test = SinusoidTasks(seed=cfg.seed + 10_000).batch(cfg.eval_tasks)
    rows = []
    for name in ("maml", "fomaml", "pretrain"):
        seed_everything(cfg.seed)
        model = _mlp(1, 40, 1)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.outer_lr)
        tasks = SinusoidTasks(seed=cfg.seed)
        learner = MAML(model, cfg.inner_lr, cfg.inner_steps, first_order=name == "fomaml")
        t0 = time.perf_counter()
        for _ in range(cfg.meta_iters):
            batch = tasks.batch(cfg.meta_batch)
            if name == "pretrain":
                opt.zero_grad()
                loss = torch.stack([F.mse_loss(model(t.support_x), t.support_y) for t in batch]).mean()
                loss.backward()  # type: ignore[no-untyped-call]
                opt.step()
            else:
                learner.meta_step(batch, opt)
        ms = 1e3 * (time.perf_counter() - t0) / cfg.meta_iters
        mean1, ci1 = _eval_regression(model, test, cfg.inner_lr, 1)
        meank, cik = _eval_regression(model, test, cfg.inner_lr, cfg.eval_steps)
        rows.append(
            {
                "method": name,
                "mse_1step": mean1,
                "ci_1step": ci1,
                f"mse_{cfg.eval_steps}step": meank,
                f"ci_{cfg.eval_steps}step": cik,
                "ms_per_iter": ms,
            }
        )
    return rows


def bench_fewshot(cfg: BenchConfig) -> list[dict[str, Any]]:
    """ProtoNet vs ANIL vs untrained-embedding nearest-prototype on Gaussian N-way K-shot."""
    rows = []
    for name in ("protonet", "anil", "untrained"):
        seed_everything(cfg.seed)
        emb = LinearEmbedder(16, 64, 64)
        tasks = GaussianClassTasks(cfg.n_way, cfg.k_shot, 15, seed=cfg.seed)
        proto = PrototypicalNetwork(emb)
        anil = ANIL(ANILModel(emb, 64, cfg.n_way), inner_lr=0.5, inner_steps=5)
        params = list(anil.model.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        t0 = time.perf_counter()
        n_train = 0 if name == "untrained" else cfg.episodes
        for _ in range(n_train):
            t = tasks.sample()
            opt.zero_grad()
            if name == "protonet":
                _, loss = proto(t.support_x, t.support_y, t.query_x, cfg.n_way, query_y=t.query_y)
                assert loss is not None
            else:
                loss = anil.meta_loss(t.support_x, t.support_y, t.query_x, t.query_y, nn.CrossEntropyLoss())
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        ms = 1e3 * (time.perf_counter() - t0) / max(n_train, 1)
        test = GaussianClassTasks(cfg.n_way, cfg.k_shot, 15, seed=cfg.seed + 10_000)
        accs = []
        for _ in range(cfg.eval_episodes):
            t = test.sample()
            if name == "anil":
                head = anil.adapt(t.support_x, t.support_y, nn.CrossEntropyLoss())
                with torch.no_grad():
                    pred = anil.model(t.query_x, head_params=head).argmax(1)
            else:
                with torch.no_grad():
                    pred = proto(t.support_x, t.support_y, t.query_x, cfg.n_way)[0].argmax(1)
            accs.append(100.0 * float((pred == t.query_y).float().mean()))
        mean, ci = compute_confidence_interval(accs)
        rows.append({"method": name, "acc": mean, "ci95": ci, "ms_per_episode": ms if n_train else 0.0})
    return rows


def bench_ewc(cfg: BenchConfig, lambdas: tuple[float, ...] = (0.0, 1e4, 1e5, 1e6, 1e7)) -> list[dict[str, Any]]:
    """Two sequential 5-way tasks; accuracy on task A after training on task B."""
    gen_a = GaussianClassTasks(5, 200, 200, dim=16, seed=cfg.seed + 1).sample()
    gen_b = GaussianClassTasks(5, 200, 200, dim=16, seed=cfg.seed + 2).sample()

    def train(model: nn.Module, t: Task, ewc: ElasticWeightConsolidation | None, steps: int = 300) -> None:
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(steps):
            opt.zero_grad()
            loss = F.cross_entropy(model(t.support_x), t.support_y)
            if ewc is not None:
                loss = loss + ewc.compute_penalty_loss()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()

    def acc(model: nn.Module, t: Task) -> float:
        with torch.no_grad():
            return 100.0 * float((model(t.query_x).argmax(1) == t.query_y).float().mean())

    rows = []
    for lam in lambdas:
        seed_everything(cfg.seed)
        model = _mlp(16, 64, 5)
        train(model, gen_a, None)
        a_before = acc(model, gen_a)
        ewc = ElasticWeightConsolidation(model, lam)
        ewc.register_previous_task([(gen_a.support_x, gen_a.support_y)], F.cross_entropy, mode="exact", batched=True)
        train(model, gen_b, ewc if lam > 0 else None)
        rows.append(
            {"lambda": lam, "acc_A_before": a_before, "acc_A_after": acc(model, gen_a), "acc_B": acc(model, gen_b)}
        )
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    """Markdown table."""
    if not rows:
        return ""
    keys = list(rows[0])
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines)
