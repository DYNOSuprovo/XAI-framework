"""Built-in explainers. Importing this package registers them all."""

from .base import BaseExplainer
from .coalition import CoalitionExplainer
from .consensus import ConsensusExplainer, combine_explanations
from .permutation import PermutationExplainer
from .surrogate import LocalSurrogateExplainer

__all__ = [
    "BaseExplainer",
    "CoalitionExplainer",
    "ConsensusExplainer",
    "LocalSurrogateExplainer",
    "PermutationExplainer",
    "combine_explanations",
]
