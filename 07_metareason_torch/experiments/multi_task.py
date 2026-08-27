"""
multi_task.py
=============
Runs multi-task adaptation and continual learning experiments using MAML, EWC, 
and Neural Module Networks (NMN) to evaluate model reasoning capacity.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from metareason.meta.maml import MAML, MetaModel
from metareason.continual.ewc import ElasticWeightConsolidation
from metareason.symbolic.nmn import NeuralModuleNetwork

def run_maml_experiment():
    print("--------------------------------------------------")
    print(" Running MAML Task Adaptation Experiment")
    print("--------------------------------------------------")
    
    # 1D regression meta-learning task (sine wave amplitude fitting)
    model = MetaModel(input_dim=1, hidden_dim=40, output_dim=1)
    maml = MAML(model, inner_lr=0.01)
    
    loss_fn = nn.MSELoss()
    
    # Generate synthetic support set (amplitude=1.5, phase=0)
    support_x = torch.randn(10, 1)
    support_y = 1.5 * torch.sin(support_x)
    
    # Measure baseline loss before inner adaptation
    baseline_pred = model(support_x)
    baseline_loss = loss_fn(baseline_pred, support_y)
    print(f"Baseline Support Loss (Unadapted): {baseline_loss.item():.6f}")
    
    # Perform inner loop adaptation step
    adapted_params = maml.adapt_to_task(loss_fn, support_x, support_y)
    
    # Measure adapted loss
    adapted_pred = model(support_x, adapted_params)
    adapted_loss = loss_fn(adapted_pred, support_y)
    print(f"Adapted Support Loss (1-step SGD): {adapted_loss.item():.6f}")
    assert adapted_loss.item() < baseline_loss.item()
    print("  [PASS] MAML successfully minimized support task loss.")

def run_ewc_experiment():
    print("\n--------------------------------------------------")
    print(" Running EWC Continual Learning Regularization")
    print("--------------------------------------------------")
    
    # Model configuration
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    
    loss_fn = nn.MSELoss()
    ewc = ElasticWeightConsolidation(model, importance_weight=50.0)
    
    # Generate Task A dataset
    task_a_dataset = [
        (torch.randn(5), torch.randn(1)) for _ in range(20)
    ]
    
    # Register optimal weights for Task A
    print("Registering Task A Fisher Information matrices...")
    ewc.register_previous_task(task_a_dataset, loss_fn)
    
    # Perturb weights slightly to simulate training on Task B
    print("Simulating parameter drifts during Task B training...")
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.05)
            
    # Calculate penalty loss
    penalty = ewc.compute_penalty_loss()
    print(f"Calculated EWC Elastic Penalty: {penalty.item():.6f}")
    assert penalty.item() > 0.0
    print("  [PASS] EWC detected parameter shift and generated positive penalty.")

def run_nmn_experiment():
    print("\n--------------------------------------------------")
    print(" Running Neural Module Network Graph Routing")
    print("--------------------------------------------------")
    
    # Input feature representation (dim=16)
    x = torch.randn(1, 16)
    
    nmn = NeuralModuleNetwork(dim=16)
    
    # Parse symbolic query tree: "logical_and( filter_color(x), filter_shape(x) )"
    query = ["count_objects", ["logical_and", ["filter_color", x], ["filter_shape", x]]]
    
    output = nmn(x, query)
    print(f"Query: count_objects( logical_and( filter_color, filter_shape ) )")
    print(f"NMN Output Probability: {output.item():.6f}")
    assert output.shape == (1, 1)
    print("  [PASS] NMN successfully parsed symbolic query and routed forward pass.")

def main():
    print("==================================================")
    print("      METAREASON-TORCH EXPERIMENT PIPELINES      ")
    print("==================================================")
    run_maml_experiment()
    run_ewc_experiment()
    run_nmn_experiment()
    print("==================================================")

if __name__ == "__main__":
    main()
