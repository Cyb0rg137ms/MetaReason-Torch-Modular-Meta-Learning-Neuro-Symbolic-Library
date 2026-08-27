"""
maml.py
=======
Model-Agnostic Meta-Learning (MAML) module in pure PyTorch.
Enables functional parameter overrides to compute gradients-of-gradients 
for task adaptation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List

class MetaModel(nn.Module):
    """
    A simple multilayer perceptron supporting weight overrides 
    for functional backpropagation in MAML inner loops.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # Keep track of layer dimensions for copy reconstructions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Meta-parameters
        self.weight1 = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
        self.weight2 = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)
        self.bias2 = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x: torch.Tensor, params: Dict[str, nn.Parameter] = None) -> torch.Tensor:
        """Forward pass with optional parameter overrides."""
        w1 = params["weight1"] if params else self.weight1
        b1 = params["bias1"] if params else self.bias1
        w2 = params["weight2"] if params else self.weight2
        b2 = params["bias2"] if params else self.bias2
        
        h = torch.relu(nn.functional.linear(x, w1, b1))
        return nn.functional.linear(h, w2, b2)

class MAML:
    """Manages MAML training loop, updating meta-weights across task batches."""
    
    def __init__(self, model: MetaModel, inner_lr: float = 0.01):
        self.model = model
        self.inner_lr = inner_lr

    def adapt_to_task(self, loss_fn: Any, support_x: torch.Tensor, support_y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Runs a single functional gradient step on support data.
        Returns the adapted parameter dictionary.
        """
        # Forward pass on support set with meta-weights
        pred = self.model(support_x)
        loss = loss_fn(pred, support_y)
        
        # Compute gradients of support loss with respect to meta-weights
        grads = torch.autograd.grad(
            loss, 
            [self.model.weight1, self.model.bias1, self.model.weight2, self.model.bias2],
            create_graph=True
        )
        
        # Manually compute SGD update: theta' = theta - lr * grad
        adapted_params = {
            "weight1": self.model.weight1 - self.inner_lr * grads[0],
            "bias1": self.model.bias1 - self.inner_lr * grads[1],
            "weight2": self.model.weight2 - self.inner_lr * grads[2],
            "bias2": self.model.bias2 - self.inner_lr * grads[3]
        }
        return adapted_params
