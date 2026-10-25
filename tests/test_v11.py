"""Tests for the 1.1 features and the bug fixes."""

from __future__ import annotations

import itertools
import json
import random

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from metareason import (
    ANIL,
    MAML,
    ANILModel,
    ElasticWeightConsolidation,
    GridTask,
    LinearEmbedder,
    NeuralModuleNetwork,
    PrototypicalNetwork,
    RuleEngine,
    StepVerifier,
    Task,
    choice_variants,
    grid_variants,
)
from metareason.benchmarks import BenchConfig, bench_ewc, bench_fewshot, bench_sinusoid, format_table
from metareason.cli import main
from metareason.meta.amplify import DIHEDRAL, INVERSE
from metareason.meta.proto_maml import compute_confidence_interval
from metareason.symbolic.sat import evaluate, solve, unsat_core
from metareason.symbolic.verifier import parse_step
from metareason.tasks import GaussianClassTasks, SinusoidTasks, seed_everything


def brute(clauses: list[list[int]]) -> bool:
    vs = sorted({abs(lit) for c in clauses for lit in c})
    for bits in itertools.product([False, True], repeat=len(vs)):
        if evaluate(clauses, dict(zip(vs, bits, strict=True))):
            return True
    return False


# ------------------------------------------------------------------ MAML
class TestMAMLGeneric:
    def test_matches_manual_sgd(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(3, 1)
        x, y = torch.randn(5, 3), torch.randn(5, 1)
        p = MAML(model, inner_lr=0.1).adapt(x, y)
        loss = F.mse_loss(model(x), y)
        gw, gb = torch.autograd.grad(loss, [model.weight, model.bias])
        torch.testing.assert_close(p["weight"], model.weight - 0.1 * gw)
        torch.testing.assert_close(p["bias"], model.bias - 0.1 * gb)

    def test_second_order_differs_from_first_order(self) -> None:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
        t = SinusoidTasks(seed=1).sample()
        grads = []
        for fo in (False, True):
            model.zero_grad()
            MAML(model, 0.1, 2, first_order=fo).meta_loss([t]).backward()
            grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
        assert not torch.allclose(grads[0], grads[1])

    def test_second_order_gradient_matches_finite_difference(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(2, 1).double()
        x, y = torch.randn(4, 2, dtype=torch.double), torch.randn(4, 1, dtype=torch.double)
        task = Task(x, y, x + 1, y)
        learner = MAML(model, 0.05, 2)
        model.zero_grad()
        learner.meta_loss([task]).backward()
        g = model.weight.grad.clone()
        eps = 1e-6
        with torch.no_grad():
            model.weight[0, 0] += eps
            up = float(learner.meta_loss([task]))
            model.weight[0, 0] -= 2 * eps
            down = float(learner.meta_loss([task]))
            model.weight[0, 0] += eps
        assert abs((up - down) / (2 * eps) - float(g[0, 0])) < 1e-6

    def test_meta_training_reduces_post_adaptation_loss(self) -> None:
        seed_everything(0)
        model = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
        learner = MAML(model, 0.01, 1, first_order=True)
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        tasks = SinusoidTasks(seed=0)
        held = SinusoidTasks(seed=99).batch(30)
        before = float(learner.meta_loss(held))
        for _ in range(150):
            learner.meta_step(tasks.batch(4), opt)
        assert float(learner.meta_loss(held)) < before

    def test_zero_steps_is_identity_and_bad_args(self) -> None:
        model = nn.Linear(2, 1)
        p = MAML(model, inner_steps=0).adapt(torch.randn(3, 2), torch.randn(3, 1))
        assert torch.equal(p["weight"], model.weight)
        with pytest.raises(ValueError):
            MAML(model, inner_steps=-1)
        with pytest.raises(ValueError):
            MAML(model).meta_loss([])

    def test_model_not_mutated(self) -> None:
        model = nn.Linear(2, 1)
        w = model.weight.detach().clone()
        MAML(model, 0.5, 3).adapt(torch.randn(3, 2), torch.randn(3, 1))
        assert torch.equal(model.weight, w)


# ------------------------------------------------------------ ProtoNet / ANIL
class TestProtoANIL:
    def test_prototypes_any_order_and_unbalanced(self) -> None:
        z = torch.tensor([[1.0], [5.0], [3.0], [7.0], [9.0]])
        y = torch.tensor([1, 0, 1, 0, 0])
        p = PrototypicalNetwork.compute_prototypes(z, y, 2)
        torch.testing.assert_close(p, torch.tensor([[7.0], [2.0]]))

    def test_missing_class_or_bad_label_raises(self) -> None:
        with pytest.raises(ValueError):
            PrototypicalNetwork.compute_prototypes(torch.randn(2, 3), torch.tensor([0, 0]), 2)
        with pytest.raises(ValueError):
            PrototypicalNetwork.compute_prototypes(torch.randn(2, 3), torch.tensor([0, 2]), 2)

    def test_shuffled_queries_with_labels(self) -> None:
        net = PrototypicalNetwork(nn.Identity())
        sx = torch.tensor([[0.0, 0], [10, 10]])
        sy = torch.tensor([0, 1])
        qx = torch.tensor([[9.0, 9], [1, 1], [11, 10]])
        qy = torch.tensor([1, 0, 1])
        logits, loss = net(sx, sy, qx, 2, query_y=qy)
        assert logits.argmax(1).tolist() == [1, 0, 1]
        assert loss is not None and float(loss) < 0.01

    def test_anil_outer_loop_updates_embedder(self) -> None:
        """Bug fix: 1.0 ran the body under no_grad so it never trained."""
        torch.manual_seed(0)
        model = ANILModel(LinearEmbedder(4, 8, 8), 8, 3)
        t = GaussianClassTasks(3, 2, 4, dim=4, seed=0).sample()
        loss = ANIL(model).meta_loss(t.support_x, t.support_y, t.query_x, t.query_y, nn.CrossEntropyLoss())
        loss.backward()
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.embedder.parameters())

    def test_confidence_interval_uses_student_t(self) -> None:
        mean, half = compute_confidence_interval([1.0, 2.0, 3.0])
        assert mean == 2.0
        assert half == pytest.approx(4.302653 * 1.0 / 3**0.5, rel=1e-5)
        with pytest.raises(ValueError):
            compute_confidence_interval([])


# ----------------------------------------------------------------- EWC
class TestEWCv11:
    def _setup(self) -> tuple[nn.Module, list[tuple[torch.Tensor, torch.Tensor]]]:
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3)), [
            (torch.randn(4), torch.tensor(i % 3)) for i in range(12)
        ]

    def test_batched_equals_per_example(self) -> None:
        model, data = self._setup()
        ewc = ElasticWeightConsolidation(model)
        f1 = ewc.estimate_fisher(data, F.cross_entropy)
        xs, ys = torch.stack([d[0] for d in data]), torch.stack([d[1] for d in data])
        f2 = ewc.estimate_fisher([(xs, ys)], F.cross_entropy, batched=True)
        for k in f1:
            torch.testing.assert_close(f1[k], f2[k])

    def test_exact_fisher_and_mode_check(self) -> None:
        model, data = self._setup()
        ewc = ElasticWeightConsolidation(model)
        f = ewc.estimate_fisher(data, F.cross_entropy, mode="exact")
        assert all((v >= 0).all() for v in f.values())
        with pytest.raises(ValueError):
            ewc.estimate_fisher(data, F.cross_entropy, mode="bogus")
        with pytest.raises(ValueError):
            ewc.estimate_fisher([], F.cross_entropy)

    def test_penalty_zero_before_registration_is_tensor(self) -> None:
        model, _ = self._setup()
        p = ElasticWeightConsolidation(model).compute_penalty_loss()
        assert isinstance(p, torch.Tensor) and float(p) == 0.0

    def test_restores_training_mode(self) -> None:
        model, data = self._setup()
        model.train()
        ElasticWeightConsolidation(model).register_previous_task(data, F.cross_entropy)
        assert model.training

    def test_multi_task_and_online(self) -> None:
        model, data = self._setup()
        ewc = ElasticWeightConsolidation(model, 1.0)
        ewc.register_previous_task(data, F.cross_entropy)
        ewc.register_previous_task(data, F.cross_entropy)
        assert len(ewc.tasks) == 2
        online = ElasticWeightConsolidation(model, 1.0, online=True, gamma=0.5)
        online.register_previous_task(data, F.cross_entropy)
        f0 = {k: v.clone() for k, v in online.fisher_matrix.items()}
        online.register_previous_task(data, F.cross_entropy)
        assert len(online.tasks) == 1
        for k in f0:
            torch.testing.assert_close(online.fisher_matrix[k], 1.5 * f0[k])

    def test_penalty_value(self) -> None:
        model = nn.Linear(1, 1, bias=False)
        ewc = ElasticWeightConsolidation(model, 2.0)
        ewc.tasks = [({"weight": torch.tensor([[3.0]])}, {"weight": model.weight.detach() - 1.0})]
        assert float(ewc.compute_penalty_loss()) == pytest.approx(0.5 * 2.0 * 3.0 * 1.0)


# ---------------------------------------------------------------- rules
class TestRuleEngineSLD:
    def test_recursive_rule_without_forward_chaining(self) -> None:
        """Bug fix: 1.0 needed forward_chain() before recursive queries worked."""
        e = RuleEngine()
        for a, b in [("a", "b"), ("b", "c"), ("c", "d")]:
            e.add_fact("parent", a, b)
        e.add_rule(("anc", "X", "Y"), [("parent", "X", "Y")])
        e.add_rule(("anc", "X", "Z"), [("parent", "X", "Y"), ("anc", "Y", "Z")])
        assert e.query("anc", "a", "d")
        assert not e.query("anc", "d", "a")
        assert e.backward_chain(("anc", "a", "d")).depth() == 4

    def test_left_recursion_terminates_and_matches_forward(self) -> None:
        e = RuleEngine()
        for a, b in [("1", "2"), ("2", "3"), ("3", "1"), ("3", "4")]:
            e.add_fact("edge", a, b)
        e.add_rule(("path", "X", "Y"), [("edge", "X", "Y")])
        e.add_rule(("path", "X", "Z"), [("path", "X", "Y"), ("edge", "Y", "Z")])
        fwd = {f for f in e.forward_chain() if f[0] == "path"}
        e2 = RuleEngine()
        for a, b in [("1", "2"), ("2", "3"), ("3", "1"), ("3", "4")]:
            e2.add_fact("edge", a, b)
        e2.add_rule(("path", "X", "Y"), [("edge", "X", "Y")])
        e2.add_rule(("path", "X", "Z"), [("path", "X", "Y"), ("edge", "Y", "Z")])
        nodes = "1234"
        bwd = {("path", a, b) for a in nodes for b in nodes if e2.query("path", a, b)}
        assert fwd == bwd and len(fwd) == 12

    def test_non_ground_queries_enumerate_answers(self) -> None:
        e = RuleEngine()
        e.add_fact("likes", "ann", "tea")
        e.add_fact("likes", "bob", "tea")
        answers = {s["W"] for s, _ in e.solve(("likes", "W", "tea"))}
        assert answers == {"ann", "bob"}

    def test_safety_checks(self) -> None:
        e = RuleEngine()
        with pytest.raises(ValueError):
            e.add_rule(("p", "X"), [("q", "Y")])
        with pytest.raises(ValueError):
            e.add_fact("p", "X")

    def test_random_programs_forward_equals_backward(self) -> None:
        rng = random.Random(0)
        for _ in range(10):
            e = RuleEngine()
            nodes = [str(i) for i in range(5)]
            for _ in range(6):
                e.add_fact("e", rng.choice(nodes), rng.choice(nodes))
            e.add_rule(("r", "X", "Y"), [("e", "X", "Y")])
            e.add_rule(("r", "X", "Z"), [("e", "X", "Y"), ("r", "Y", "Z")])
            snapshot = RuleEngine()
            snapshot._original_facts = set(e._original_facts)
            snapshot._facts = set(e._facts)
            snapshot._rules = list(e._rules)
            fwd = {f for f in e.forward_chain() if f[0] == "r"}
            bwd = {("r", a, b) for a in nodes for b in nodes if snapshot.query("r", a, b)}
            assert fwd == bwd


# ------------------------------------------------------------------ NMN
class TestNMN:
    def test_string_leaf_and_custom_module(self) -> None:
        net = NeuralModuleNetwork(4)
        x = torch.randn(2, 4)
        out = net(x, ["count_objects", ["logical_and", "filter_color", "filter_shape"]])
        assert out.shape == (2, 1)
        net.register("ident", nn.Identity())
        torch.testing.assert_close(net(x, ["ident", "x"]), x)

    def test_module_to_dtype_works(self) -> None:
        """Regression: helper names must not shadow nn.Module internals such as _apply."""
        net = NeuralModuleNetwork(4).double()
        assert net(torch.randn(2, 4, dtype=torch.double), ["filter_color", "x"]).dtype == torch.double

    def test_errors(self) -> None:
        net = NeuralModuleNetwork(4, max_depth=3)
        x = torch.randn(1, 4)
        with pytest.raises(ValueError):
            net(x, ["nope"])
        with pytest.raises(ValueError):
            net(x, [])
        deep: list = ["filter_color"]
        for _ in range(5):
            deep = ["filter_color", deep]
        with pytest.raises(RecursionError):
            net(x, deep)


# --------------------------------------------------------- SAT & verifier
class TestSatVerifier:
    def test_solver_matches_brute_force(self) -> None:
        rng = random.Random(1)
        for _ in range(300):
            n = rng.randint(1, 7)
            cls = [
                [rng.choice([-1, 1]) * rng.randint(1, n) for _ in range(rng.randint(1, 3))]
                for _ in range(rng.randint(1, 20))
            ]
            r = solve(cls)
            assert r.sat == brute(cls)
            if r.sat:
                assert evaluate(cls, r.model)

    def test_unsat_core_is_minimal(self) -> None:
        rng = random.Random(2)
        checked = 0
        while checked < 30:
            n = rng.randint(2, 5)
            cls = [[rng.choice([-1, 1]) * rng.randint(1, n) for _ in range(2)] for _ in range(14)]
            if brute(cls):
                continue
            core = unsat_core(cls)
            sub = [cls[i] for i in core]
            assert not brute(sub)
            for i in range(len(sub)):
                assert brute(sub[:i] + sub[i + 1 :])
            checked += 1
        with pytest.raises(ValueError):
            unsat_core([[1]])

    def test_parse_step(self) -> None:
        assert parse_step("so x3 = true and x4 is false") == [[3], [-4]]
        assert parse_step("need (x1 OR ~x2) and (¬x3 ∨ x4)") == [[-2, 1], [-3, 4]]
        assert parse_step("consider the remaining cases") == []
        assert parse_step("(x1 OR banana)") == []  # malformed groups are not silently weakened

    def test_diagnosis_finds_first_failing_step_and_culprits(self) -> None:
        v = StepVerifier([[1, 2], [-1, 3]])
        res = v.verify(["Try x1 = true.", "Unrelated remark.", "Then x2 = false.", "So x3 = false."])
        assert [s.status for s in res.steps] == ["consistent", "unchecked", "consistent", "inconsistent"]
        d = res.diagnosis
        assert d is not None and d.failing_step == 3
        assert d.conflicting_steps == [0, 3] and d.conflicting_base_clauses == [1]
        assert d.restart_from == 0
        assert "step 4" in d.explain()

    def test_consistent_trace_returns_verified_model(self) -> None:
        v = StepVerifier([[1, 2], [-1, 3]])
        res = v.verify(["x1 = true", "x3 = true"])
        assert res.ok and res.model is not None and v.verify_answer(res.model)
        assert not v.verify_answer({1: True, 2: False, 3: False})

    def test_unsat_problem_diagnosed_up_front(self) -> None:
        res = StepVerifier([[1], [-1]]).verify(["x2 = true"])
        assert not res.ok and res.diagnosis.failing_step == -1


# ---------------------------------------------------------- amplification
class TestAmplify:
    def test_dihedral_inverses(self) -> None:
        g = [[1, 2, 3], [4, 5, 6]]
        for k, f in DIHEDRAL.items():
            assert DIHEDRAL[INVERSE[k]](f(g)) == g

    def test_grid_variants_are_consistent_and_distinct(self) -> None:
        a, b = [[0, 1], [2, 0]], [[1, 0], [0, 2]]
        task = GridTask([(a, b)], [[3, 0], [0, 4]], [[0, 3], [4, 0]])
        vs = grid_variants(task, n=20, seed=0)
        assert len(vs) == 20
        assert len({repr((v.train, v.test_input)) for v in vs}) == 20
        for v in vs:
            geo, perm = v.aug["geo"], v.aug["perm"]
            assert perm[0] == 0
            f = lambda g, geo=geo, perm=perm: [[perm[c] for c in r] for r in DIHEDRAL[geo](g)]
            assert v.train[0] == (f(a), f(b)) and v.test_output == f(task.test_output)
        assert grid_variants(task, n=20, seed=0) == vs

    def test_choice_variants(self) -> None:
        opts = ["red", "green", "blue", "cyan"]
        vs = choice_variants(opts, 2, n=8, seed=0)
        assert len(vs) == 8
        assert all(o[g] == "blue" and sorted(o) == sorted(opts) for o, g in vs)
        with pytest.raises(ValueError):
            choice_variants(opts, 9)


# ------------------------------------------------------- benches & CLI
class TestBenchCLI:
    def test_tasks_seeded(self) -> None:
        a, b = SinusoidTasks(seed=3).sample(), SinusoidTasks(seed=3).sample()
        assert torch.equal(a.support_x, b.support_x)
        t = GaussianClassTasks(4, 2, 3, seed=0).sample()
        assert sorted(t.support_y.tolist()) == [0, 0, 1, 1, 2, 2, 3, 3]

    def test_quick_benches_run(self) -> None:
        cfg = BenchConfig(meta_iters=3, eval_tasks=3, episodes=3, eval_episodes=3)
        assert len(bench_sinusoid(cfg)) == 3
        assert len(bench_fewshot(cfg)) == 3
        rows = bench_ewc(cfg, (0.0,))
        assert "|" in format_table(rows)

    def test_config_toml(self, tmp_path) -> None:
        p = tmp_path / "c.toml"
        p.write_text("[bench]\nseed = 7\nmeta_iters = 5\n")
        assert BenchConfig.from_toml(p).meta_iters == 5
        p.write_text("[bench]\nbogus = 1\n")
        with pytest.raises(ValueError):
            BenchConfig.from_toml(p)

    def test_cli_verify(self, tmp_path, capsys) -> None:
        f = tmp_path / "t.json"
        f.write_text(json.dumps({"clauses": [[1, 2]], "steps": ["x1 = false", "x2 = false"]}))
        assert main(["verify", str(f)]) == 1
        assert "DIAGNOSIS" in capsys.readouterr().out
        f.write_text(json.dumps({"clauses": [[1, 2]], "steps": ["x1 = true"]}))
        assert main(["verify", str(f)]) == 0
