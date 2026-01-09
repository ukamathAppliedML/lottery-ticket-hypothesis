"""
Lottery Ticket Hypothesis Implementation
========================================

Production-ready implementation of neural network pruning techniques
based on "The Lottery Ticket Hypothesis" (Frankle & Carlin, 2018).

Key modules:
- lottery_ticket: Core LTH algorithm (IMP, one-shot pruning)
- structured_sparsity: 2:4 sparsity for GPU acceleration
- models: Reference architectures
"""

from .lottery_ticket import (
    LotteryTicketFinder,
    PruningMask,
    MagnitudePruner,
    LotteryTicketConfig,
    find_lottery_ticket,
)

from .structured_sparsity import (
    Structured24Pruner,
    SparseConv2d,
    SparseLinear,
    convert_to_sparse_model,
)

from .models import (
    LeNet300,
    LeNet5,
    Conv2,
    Conv4,
    Conv6,
    VGG,
    ResNet,
    get_model,
    count_parameters,
)

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = [
    # Core
    "LotteryTicketFinder",
    "PruningMask", 
    "MagnitudePruner",
    "LotteryTicketConfig",
    "find_lottery_ticket",
    # Structured sparsity
    "Structured24Pruner",
    "SparseConv2d",
    "SparseLinear",
    "convert_to_sparse_model",
    # Models
    "LeNet300",
    "LeNet5",
    "Conv2",
    "Conv4", 
    "Conv6",
    "VGG",
    "ResNet",
    "get_model",
    "count_parameters",
]
