"""
MetaReason-Torch package
"""

from metareason.meta.maml import MAML, MetaModel
from metareason.continual.ewc import ElasticWeightConsolidation
from metareason.symbolic.nmn import NeuralModuleNetwork, NeuralModule

__all__ = [
    "MAML",
    "MetaModel",
    "ElasticWeightConsolidation",
    "NeuralModuleNetwork",
    "NeuralModule"
]
