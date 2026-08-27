"""
ewc.py
======
Elastic Weight Consolidation (EWC) module in PyTorch.
Computes Fisher Information matrices to penalize changes to important weights 
during sequential multi-task training.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any

class ElasticWeightConsolidation:
    """Computes Fisher Information constraints for EWC regularization."""
    
    def __init__(self, model: nn.Module, importance_weight: float = 400.0):
        """
        Args:
            model: PyTorch model.
            importance_weight: Penalty scaling factor (lambda).
        """
        self.model = model
        self.importance = importance_weight
        self.fisher_matrix: Dict[str, torch.Tensor] = {}
        self.optimal_weights: Dict[str, torch.Tensor] = {}

    def register_previous_task(self, dataset: List[Tuple[torch.Tensor, torch.Tensor]], loss_fn: Any):
        """
        Computes the Fisher Information matrix and stores the optimal weight parameters
        for the current task before switching.
        """
        # Save current optimal parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_weights[name] = param.clone().detach()
                self.fisher_matrix[name] = torch.zeros_like(param)
                
        # Estimate diagonal Fisher matrix values
        self.model.eval()
        for x, y in dataset:
            self.model.zero_grad()
            pred = self.model(x.unsqueeze(0))
            loss = loss_fn(pred, y)
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_matrix[name] += (param.grad.data ** 2) / len(dataset)
                    
        # Detach Fisher values from graph
        for name in self.fisher_matrix:
            self.fisher_matrix[name] = self.fisher_matrix[name].detach()

    def compute_penalty_loss(self) -> torch.Tensor:
        """
        Computes the quadratic penalty constraint:
        L = sum( 0.5 * lambda * Fisher * (theta - theta_optimal)^2 )
        """
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_matrix:
                fisher = self.fisher_matrix[name]
                opt_w = self.optimal_weights[name]
                penalty += (fisher * (param - opt_w) ** 2).sum()
                
        return 0.5 * self.importance * penalty
