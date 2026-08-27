"""
test_metareason.py
==================
Comprehensive test suite for MetaReason-Torch.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from metareason.meta.maml import MAML, MetaModel
from metareason.meta.proto_maml import (
    PrototypicalNetwork,
    LinearEmbedder,
    ANILModel,
    ANIL,
    compute_episode_accuracy,
    compute_confidence_interval,
)
from metareason.continual.ewc import ElasticWeightConsolidation
from metareason.symbolic.rule_engine import RuleEngine, unify, apply_substitution


@pytest.fixture
def simple_model():
    return MetaModel(input_dim=4, hidden_dim=8, output_dim=2)


@pytest.fixture
def embedder():
    return LinearEmbedder(input_dim=8, hidden_dim=16, embed_dim=16)


@pytest.fixture
def rule_engine():
    engine = RuleEngine()
    engine.add_fact("parent", "alice", "bob")
    engine.add_fact("parent", "bob", "charlie")
    engine.add_fact("parent", "charlie", "dave")
    engine.add_rule(head=("ancestor", "X", "Y"), body=[("parent", "X", "Y")])
    engine.add_rule(
        head=("ancestor", "X", "Z"),
        body=[("parent", "X", "Y"), ("ancestor", "Y", "Z")],
    )
    return engine


class TestMAML:
    def test_adapt_returns_dict(self, simple_model):
        maml = MAML(simple_model, inner_lr=0.01)
        sx = torch.randn(4, 4)
        sy = torch.randn(4, 2)
        adapted = maml.adapt_to_task(nn.MSELoss(), sx, sy)
        assert "weight1" in adapted and "weight2" in adapted

    def test_adapted_params_shape(self, simple_model):
        maml = MAML(simple_model, inner_lr=0.01)
        adapted = maml.adapt_to_task(nn.MSELoss(), torch.randn(4, 4), torch.randn(4, 2))
        assert adapted["weight1"].shape == (8, 4)
        assert adapted["weight2"].shape == (2, 8)

    def test_second_order_gradients(self, simple_model):
        maml = MAML(simple_model, inner_lr=0.01)
        adapted = maml.adapt_to_task(nn.MSELoss(), torch.randn(3, 4), torch.randn(3, 2))
        meta_loss = (simple_model(torch.randn(3, 4), params=adapted) ** 2).mean()
        meta_loss.backward()
        assert simple_model.weight1.grad is not None

    def test_adaptation_changes_params(self, simple_model):
        maml = MAML(simple_model, inner_lr=0.5)
        adapted = maml.adapt_to_task(nn.MSELoss(), torch.randn(8, 4), torch.randn(8, 2))
        diff = (adapted["weight1"] - simple_model.weight1).abs().max().item()
        assert diff > 1e-7

    def test_forward_with_adapted_params(self, simple_model):
        maml = MAML(simple_model, inner_lr=0.01)
        x = torch.randn(5, 4)
        adapted = maml.adapt_to_task(nn.MSELoss(), x, torch.randn(5, 2))
        assert simple_model(x, params=adapted).shape == (5, 2)


class TestPrototypicalNetworks:
    def test_prototype_shape(self, embedder):
        net = PrototypicalNetwork(embedder)
        z = torch.randn(15, 16)
        assert net.compute_prototypes(z, n_way=3, k_shot=5).shape == (3, 16)

    def test_prototype_is_mean(self, embedder):
        net = PrototypicalNetwork(embedder)
        z = torch.randn(6, 8)
        protos = net.compute_prototypes(z, n_way=3, k_shot=2)
        torch.testing.assert_close(protos[0], z[:2].mean(dim=0), atol=1e-5, rtol=1e-5)

    def test_forward_output_shapes(self, embedder):
        net = PrototypicalNetwork(embedder)
        support_x = torch.randn(15, 8)
        support_y = torch.tensor([0] * 5 + [1] * 5 + [2] * 5)
        logits, loss = net(support_x, support_y, torch.randn(9, 8), n_way=3, k_shot=5)
        assert logits.shape == (9, 3)
        assert loss.item() > 0

    def test_distance_ordering_correct(self):
        embedder = LinearEmbedder(input_dim=4, hidden_dim=8, embed_dim=4)
        net = PrototypicalNetwork(embedder)
        s0 = torch.tensor([[10.0, 0, 0, 0]] * 3)
        s1 = torch.tensor([[-10.0, 0, 0, 0]] * 3)
        support_x = torch.cat([s0, s1])
        support_y = torch.tensor([0, 0, 0, 1, 1, 1])
        q = torch.tensor([[9.9, 0, 0, 0]] * 2 + [[-9.9, 0, 0, 0]] * 2)
        logits, _ = net(support_x, support_y, q, n_way=2, k_shot=3)
        assert logits[0].argmax().item() == 0
        assert logits[2].argmax().item() == 1

    def test_well_separated_clusters_accuracy(self):
        net = PrototypicalNetwork(LinearEmbedder(8, 16, 8))
        torch.manual_seed(42)
        support_x = torch.cat([torch.randn(5, 8) + i * 10 for i in range(3)])
        support_y = torch.tensor([0] * 5 + [1] * 5 + [2] * 5)
        query_x = torch.cat([torch.randn(3, 8) + i * 10 for i in range(3)])
        protos = net.compute_prototypes(support_x[support_y.argsort()], n_way=3, k_shot=5)
        dists = torch.cdist(query_x, protos)
        preds = dists.argmin(dim=1)
        query_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        acc = (preds == query_y).float().mean().item()
        assert acc > 0.9


class TestANIL:
    def test_adapt_returns_dict(self, embedder):
        model = ANILModel(embedder, embed_dim=16, n_way=3)
        anil = ANIL(model, inner_lr=0.1)
        adapted = anil.adapt(torch.randn(9, 8), torch.tensor([0]*3+[1]*3+[2]*3), nn.CrossEntropyLoss())
        assert "weight" in adapted and "bias" in adapted

    def test_adapted_head_changes(self, embedder):
        model = ANILModel(embedder, embed_dim=16, n_way=3)
        anil = ANIL(model, inner_lr=0.5)
        adapted = anil.adapt(torch.randn(6, 8), torch.tensor([0,0,1,1,2,2]), nn.CrossEntropyLoss())
        assert (adapted["weight"] - model.head.weight).abs().max().item() > 1e-7

    def test_meta_loss_positive(self, embedder):
        model = ANILModel(embedder, embed_dim=16, n_way=2)
        anil = ANIL(model, inner_lr=0.1)
        x, y = torch.randn(4, 8), torch.tensor([0,0,1,1])
        assert anil.meta_loss(x, y, x, y, nn.CrossEntropyLoss()).item() > 0


class TestEWC:
    def _make(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        data = [(torch.randn(4), torch.tensor([0])) for _ in range(10)]
        return model, data

    def test_fisher_nonnegative(self):
        model, data = self._make()
        ewc = ElasticWeightConsolidation(model, 100.0)
        ewc.register_previous_task(data, nn.CrossEntropyLoss())
        for f in ewc.fisher_matrix.values():
            assert (f >= 0).all()

    def test_fisher_shape_matches(self):
        model, data = self._make()
        ewc = ElasticWeightConsolidation(model, 100.0)
        ewc.register_previous_task(data, nn.CrossEntropyLoss())
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert ewc.fisher_matrix[name].shape == param.shape

    def test_penalty_zero_at_optimal(self):
        model, data = self._make()
        ewc = ElasticWeightConsolidation(model, 500.0)
        ewc.register_previous_task(data, nn.CrossEntropyLoss())
        assert abs(ewc.compute_penalty_loss().item()) < 1e-5

    def test_penalty_increases_with_perturbation(self):
        model, data = self._make()
        ewc = ElasticWeightConsolidation(model, 500.0)
        ewc.register_previous_task(data, nn.CrossEntropyLoss())
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 2.0)
        assert ewc.compute_penalty_loss().item() > 1e-3

    def test_penalty_differentiable(self):
        model, data = self._make()
        ewc = ElasticWeightConsolidation(model, 100.0)
        ewc.register_previous_task(data, nn.CrossEntropyLoss())
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1)
        penalty = ewc.compute_penalty_loss()
        assert penalty.requires_grad
        penalty.backward()

    def test_optimal_weights_stored(self):
        model, data = self._make()
        ewc = ElasticWeightConsolidation(model, 100.0)
        ewc.register_previous_task(data, nn.CrossEntropyLoss())
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.testing.assert_close(ewc.optimal_weights[name], param.data, atol=1e-6, rtol=1e-5)


class TestRuleEngine:
    def test_forward_direct_facts(self, rule_engine):
        derived = rule_engine.forward_chain()
        assert ("parent", "alice", "bob") in derived

    def test_forward_one_hop(self, rule_engine):
        derived = rule_engine.forward_chain()
        assert ("ancestor", "alice", "bob") in derived

    def test_forward_transitive_closure(self, rule_engine):
        derived = rule_engine.forward_chain()
        assert ("ancestor", "alice", "charlie") in derived
        assert ("ancestor", "alice", "dave") in derived

    def test_backward_fact(self, rule_engine):
        rule_engine.forward_chain()
        proof = rule_engine.backward_chain(("parent", "alice", "bob"))
        assert proof is not None and proof.is_fact

    def test_backward_derived(self, rule_engine):
        rule_engine.forward_chain()
        proof = rule_engine.backward_chain(("ancestor", "alice", "charlie"))
        assert proof is not None and len(proof.children) > 0

    def test_backward_unknown_none(self, rule_engine):
        rule_engine.forward_chain()
        assert rule_engine.backward_chain(("parent", "dave", "alice")) is None

    def test_query_true(self, rule_engine):
        rule_engine.forward_chain()
        assert rule_engine.query("ancestor", "alice", "dave")

    def test_query_false(self, rule_engine):
        rule_engine.forward_chain()
        assert not rule_engine.query("ancestor", "dave", "alice")

    def test_unify_basic(self):
        subst = unify(("parent", "X", "Y"), ("parent", "alice", "bob"))
        assert subst == {"X": "alice", "Y": "bob"}

    def test_unify_constant_mismatch(self):
        assert unify(("parent", "alice", "Y"), ("parent", "bob", "charlie")) is None

    def test_unify_reflexive_conflict(self):
        assert unify(("sibling", "X", "X"), ("sibling", "alice", "bob")) is None

    def test_apply_substitution(self):
        result = apply_substitution(("ancestor", "X", "Z"), {"X": "alice", "Z": "charlie"})
        assert result == ("ancestor", "alice", "charlie")

    def test_forward_fixpoint_idempotent(self):
        engine = RuleEngine()
        engine.add_fact("a", "x")
        engine.add_rule(("b", "X"), [("a", "X")])
        d1 = engine.forward_chain()
        d2 = engine.forward_chain()
        assert d1 == d2

    def test_explain_returns_string(self, rule_engine):
        rule_engine.forward_chain()
        exp = rule_engine.explain("ancestor", "alice", "bob")
        assert isinstance(exp, str) and "ancestor" in exp


class TestAccuracyUtils:
    def test_perfect_accuracy(self):
        assert compute_episode_accuracy(torch.tensor([0,1,2]), torch.tensor([0,1,2])) == 100.0

    def test_zero_accuracy(self):
        assert compute_episode_accuracy(torch.tensor([1,2,0]), torch.tensor([0,1,2])) == 0.0

    def test_confidence_interval(self):
        accs = [85.0, 87.0, 83.0, 88.0, 86.0] * 6
        mean, ci = compute_confidence_interval(accs)
        assert 83.0 < mean < 89.0 and ci > 0
