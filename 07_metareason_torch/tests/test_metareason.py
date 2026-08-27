import pytest
import torch
import torch.nn as nn
from metareason.meta.maml import MAML, MetaModel
from metareason.continual.ewc import ElasticWeightConsolidation
from metareason.symbolic.nmn import NeuralModuleNetwork, NeuralModule

def test_maml_adaptation():
    model = MetaModel(input_dim=4, hidden_dim=8, output_dim=2)
    maml = MAML(model, inner_lr=0.1)
    loss_fn = nn.MSELoss()
    
    x = torch.randn(2, 4)
    y = torch.randn(2, 2)
    
    # Base forward
    out_base = model(x)
    loss_base = loss_fn(out_base, y)
    
    # Adapt
    adapted_params = maml.adapt_to_task(loss_fn, x, y)
    
    # Adapted forward
    out_adapted = model(x, adapted_params)
    loss_adapted = loss_fn(out_adapted, y)
    
    assert loss_adapted.item() < loss_base.item()

def test_ewc_regularization():
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )
    loss_fn = nn.MSELoss()
    ewc = ElasticWeightConsolidation(model, importance_weight=10.0)
    
    dataset = [(torch.randn(3), torch.randn(1)) for _ in range(5)]
    
    ewc.register_previous_task(dataset, loss_fn)
    assert len(ewc.optimal_weights) > 0
    assert len(ewc.fisher_matrix) > 0
    
    # No weight change -> zero penalty
    penalty_zero = ewc.compute_penalty_loss()
    assert abs(penalty_zero.item()) < 1e-6
    
    # Shift weights
    with torch.no_grad():
        model[0].weight.add_(torch.ones_like(model[0].weight) * 0.1)
        
    penalty_positive = ewc.compute_penalty_loss()
    assert penalty_positive.item() > 0.0

def test_nmn_routing():
    dim = 8
    nmn = NeuralModuleNetwork(dim=dim)
    x = torch.randn(1, dim)
    
    # Simple query
    query = ["logical_and", ["filter_color", x], ["filter_shape", x]]
    out = nmn(x, query)
    
    assert out.shape == (1, dim)
    
    # Complex count query
    query_count = ["count_objects", ["logical_and", ["filter_color", x], ["filter_shape", x]]]
    out_count = nmn(x, query_count)
    
    assert out_count.shape == (1, 1)
    assert 0.0 <= out_count.item() <= 1.0
