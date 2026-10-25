"""MetaReason-Torch: meta-learning, continual learning and verified neurosymbolic reasoning."""

from metareason.continual.ewc import ElasticWeightConsolidation
from metareason.meta.amplify import GridTask, choice_variants, grid_variants
from metareason.meta.maml import MAML, MetaModel, Task
from metareason.meta.proto_maml import ANIL, ANILModel, LinearEmbedder, PrototypicalNetwork
from metareason.symbolic.nmn import NeuralModule, NeuralModuleNetwork
from metareason.symbolic.rule_engine import RuleEngine
from metareason.symbolic.verifier import Diagnosis, StepVerifier, VerificationResult

__version__ = "1.1.0"

__all__ = [
    "ANIL",
    "MAML",
    "ANILModel",
    "Diagnosis",
    "ElasticWeightConsolidation",
    "GridTask",
    "LinearEmbedder",
    "MetaModel",
    "NeuralModule",
    "NeuralModuleNetwork",
    "PrototypicalNetwork",
    "RuleEngine",
    "StepVerifier",
    "Task",
    "VerificationResult",
    "choice_variants",
    "grid_variants",
]
