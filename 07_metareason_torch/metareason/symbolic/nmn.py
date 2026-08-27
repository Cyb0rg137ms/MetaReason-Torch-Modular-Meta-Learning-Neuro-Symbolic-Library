"""
nmn.py
======
Neural Module Network (NMN) module in PyTorch.
Parses symbolic query trees and dynamically routes execution paths through 
reusable neural modules.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Union

class NeuralModule(nn.Module):
    """A standard module designed for dynamic routing."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class NeuralModuleNetwork(nn.Module):
    """
    Parses symbolic query trees and dynamically strings together 
    NeuralModule blocks in forward passes.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.modules_dict = nn.ModuleDict({
            "filter_color": NeuralModule(dim),
            "filter_shape": NeuralModule(dim),
            "logical_and": NeuralModule(dim),
            "count_objects": nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        })

    def forward(self, x: torch.Tensor, query_tree: Union[str, List[Any]]) -> torch.Tensor:
        """
        Recursively parses the query tree and pipes execution.
        
        Args:
            x: Input feature representation.
            query_tree: Nested list query structure, e.g. ["count_objects", ["logical_and", ["filter_color", x]]]
            
        Returns:
            The output classification/regression tensor.
        """
        if isinstance(query_tree, torch.Tensor):
            return query_tree
            
        op = query_tree[0]
        args = query_tree[1:]
        
        # Evaluate child branches recursively
        evaluated_args = [self.forward(x, arg) for arg in args]
        
        # Compute combined input
        if len(evaluated_args) == 0:
            combined_input = x
        elif len(evaluated_args) == 1:
            combined_input = evaluated_args[0]
        else:
            # Multi-branch combines features (elementwise addition for simplicity)
            combined_input = sum(evaluated_args)
            
        # Dispatch to registered neural module
        if op in self.modules_dict:
            module = self.modules_dict[op]
            return module(combined_input)
        else:
            raise ValueError(f"Unknown symbolic operation node: {op}")
