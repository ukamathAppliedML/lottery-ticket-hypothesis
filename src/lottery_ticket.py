"""
Lottery Ticket Hypothesis - Core Implementation
===============================================

Implementation of "The Lottery Ticket Hypothesis: Finding Sparse, Trainable 
Neural Networks" (Frankle & Carlin, 2018) - https://arxiv.org/abs/1803.03635

This module provides the core functionality for finding winning lottery tickets
in neural networks through iterative magnitude pruning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Callable, List
from dataclasses import dataclass
import copy
import json
from pathlib import Path


@dataclass
class LotteryTicketConfig:
    """Configuration for Lottery Ticket experiments."""
    target_sparsity: float = 0.9      # Final sparsity (0.9 = 90% pruned)
    pruning_rounds: int = 10          # Number of IMP rounds
    global_pruning: bool = True       # Global vs layer-wise


class PruningMask:
    """
    Manages binary pruning masks for neural network weights.
    
    The mask tracks which weights are active (1) vs pruned (0).
    Provides utilities for applying, saving, and analyzing masks.
    """
    
    def __init__(self, model: nn.Module):
        """Initialize masks to all ones (no pruning) for prunable layers."""
        self.masks: Dict[str, torch.Tensor] = {}
        self._initialize_masks(model)
    
    def _initialize_masks(self, model: nn.Module):
        """Create initial all-ones masks for weight tensors."""
        for name, param in model.named_parameters():
            if self._is_prunable(name, param):
                self.masks[name] = torch.ones_like(param, dtype=torch.bool)
    
    @staticmethod
    def _is_prunable(name: str, param: torch.Tensor) -> bool:
        """Determine if a parameter should be pruned."""
        # Prune weight matrices, not biases or 1D params
        return 'weight' in name and param.dim() >= 2
    
    def apply(self, model: nn.Module) -> None:
        """Apply masks to zero out pruned weights."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.masks:
                    param.mul_(self.masks[name].to(param.device).float())
    
    def update(self, name: str, new_mask: torch.Tensor) -> None:
        """Update mask for a specific layer (AND with existing)."""
        if name in self.masks:
            self.masks[name] = self.masks[name] & new_mask.bool()
    
    @property
    def sparsity(self) -> float:
        """Overall sparsity (fraction of zeros)."""
        total = sum(m.numel() for m in self.masks.values())
        zeros = sum((~m).sum().item() for m in self.masks.values())
        return zeros / total if total > 0 else 0.0
    
    @property
    def layer_sparsities(self) -> Dict[str, float]:
        """Per-layer sparsity breakdown."""
        return {
            name: (~mask).sum().item() / mask.numel()
            for name, mask in self.masks.items()
        }
    
    def active_params(self) -> int:
        """Count of non-pruned parameters."""
        return sum(m.sum().item() for m in self.masks.values())
    
    def save(self, path: str) -> None:
        """Save masks to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in self.masks.items()}, path)
    
    def load(self, path: str) -> None:
        """Load masks from file."""
        loaded = torch.load(path, map_location='cpu')
        self.masks = {k: v.bool() for k, v in loaded.items()}
    
    def to(self, device: str) -> 'PruningMask':
        """Move masks to device."""
        self.masks = {k: v.to(device) for k, v in self.masks.items()}
        return self


class MagnitudePruner:
    """
    Implements magnitude-based weight pruning.
    
    Weights with smallest absolute values are pruned first, based on the
    hypothesis that small weights contribute less to network function.
    """
    
    def __init__(self, global_pruning: bool = True):
        """
        Args:
            global_pruning: If True, prune globally across all layers.
                           If False, prune each layer independently.
        """
        self.global_pruning = global_pruning
    
    def compute_threshold(
        self,
        model: nn.Module,
        mask: PruningMask,
        target_sparsity: float
    ) -> float:
        """Compute magnitude threshold for target sparsity."""
        # Collect all currently active weights
        all_weights = []
        for name, param in model.named_parameters():
            if name in mask.masks:
                active = param[mask.masks[name].to(param.device)]
                all_weights.append(active.abs().flatten())
        
        if not all_weights:
            return 0.0
        
        all_weights = torch.cat(all_weights)
        n = len(all_weights)
        
        if n == 0:
            return 0.0
        
        # Use quantile for more accurate threshold computation
        # This avoids off-by-one issues with kthvalue
        threshold = torch.quantile(all_weights, target_sparsity).item()
        return threshold
    
    def prune(
        self,
        model: nn.Module,
        mask: PruningMask,
        target_sparsity: float
    ) -> PruningMask:
        """
        Prune model to target sparsity using magnitude criterion.
        
        Args:
            model: Neural network to prune
            mask: Current pruning mask
            target_sparsity: Desired sparsity level
            
        Returns:
            Updated pruning mask
        """
        if self.global_pruning:
            threshold = self.compute_threshold(model, mask, target_sparsity)
            
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in mask.masks:
                        # Keep weights above threshold
                        keep_mask = param.abs() >= threshold
                        mask.masks[name] = mask.masks[name].to(param.device) & keep_mask
        else:
            # Layer-wise pruning
            for name, param in model.named_parameters():
                if name in mask.masks:
                    active = param[mask.masks[name].to(param.device)]
                    k = int(active.numel() * target_sparsity)
                    if k > 0 and k < active.numel():
                        threshold = torch.kthvalue(active.abs().flatten(), k).values.item()
                        keep_mask = param.abs() >= threshold
                        mask.masks[name] = mask.masks[name].to(param.device) & keep_mask
        
        return mask


class LotteryTicketFinder:
    """
    Complete implementation of the Lottery Ticket Hypothesis algorithm.
    
    Orchestrates the process of:
    1. Storing original initialization (potential winning ticket)
    2. Training dense network
    3. Finding important weights via magnitude pruning  
    4. Resetting to original init with discovered mask
    5. Verifying sparse network matches dense performance
    
    Example:
        >>> finder = LotteryTicketFinder(model_class=LeNet300)
        >>> model = finder.initialize_network()
        >>> winner, mask, results = finder.find_winning_ticket(
        ...     model, train_fn, eval_fn, target_sparsity=0.9
        ... )
    """
    
    def __init__(
        self,
        model_class: type,
        model_kwargs: Optional[Dict] = None,
        device: str = None
    ):
        """
        Args:
            model_class: Neural network class to instantiate
            model_kwargs: Arguments to pass to model constructor
            device: Device to use (auto-detected if None)
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Experiment state
        self.original_init: Optional[Dict] = None
        self.winning_ticket_mask: Optional[PruningMask] = None
        self.experiment_log: List[Dict] = []
    
    def initialize_network(self) -> nn.Module:
        """Create new network and store its initialization."""
        model = self.model_class(**self.model_kwargs).to(self.device)
        self.original_init = copy.deepcopy(model.state_dict())
        return model
    
    def find_winning_ticket(
        self,
        model: nn.Module,
        train_fn: Callable[[nn.Module, Optional['PruningMask']], None],
        eval_fn: Callable[[nn.Module], float],
        target_sparsity: float = 0.9,
        pruning_rounds: int = 10,
        verbose: bool = True
    ) -> Tuple[nn.Module, PruningMask, Dict]:
        """
        Find winning lottery ticket via Iterative Magnitude Pruning.
        
        Algorithm (from paper):
        1. Randomly initialize network f(x; θ₀)
        2. Train for j iterations to get θⱼ
        3. Prune p% of parameters, creating mask m
        4. Reset remaining parameters to θ₀, creating f(x; m ⊙ θ₀)
        5. Repeat steps 2-4 until desired sparsity
        
        Args:
            model: Network to find winning ticket for
            train_fn: Function(model, mask) that trains model while maintaining sparsity.
                      Should call mask.apply(model) after each optimizer step.
            eval_fn: Function that returns accuracy/metric
            target_sparsity: Final sparsity level (0.9 = 90% pruned)
            pruning_rounds: Number of iterative pruning rounds
            verbose: Print progress
            
        Returns:
            Tuple of (winning_ticket_model, mask, results_dict)
        """
        if self.original_init is None:
            self.original_init = copy.deepcopy(model.state_dict())
        
        results = {
            'dense_accuracy': None,
            'sparse_accuracy': None,
            'sparsity': None,
            'rounds': [],
            'config': {
                'target_sparsity': target_sparsity,
                'pruning_rounds': pruning_rounds
            }
        }
        
        # Phase 1: Train dense network
        if verbose:
            print("=" * 60)
            print("Phase 1: Training Dense Network")
            print("=" * 60)
        
        train_fn(model, None)  # No mask for dense training
        dense_acc = eval_fn(model)
        results['dense_accuracy'] = dense_acc
        
        if verbose:
            print(f"Dense accuracy: {dense_acc:.4f}")
        
        # Store trained weights for magnitude analysis
        trained_state = copy.deepcopy(model.state_dict())
        
        # Phase 2: Iterative Magnitude Pruning
        if verbose:
            print("\n" + "=" * 60)
            print("Phase 2: Iterative Magnitude Pruning")
            print("=" * 60)
        
        mask = PruningMask(model)
        pruner = MagnitudePruner(global_pruning=True)
        
        # Calculate per-round keep ratio
        # After n rounds: remaining = keep_ratio^n = (1 - target_sparsity)
        # So: keep_ratio = (1 - target_sparsity)^(1/n)
        final_keep_ratio = 1 - target_sparsity
        per_round_keep = final_keep_ratio ** (1 / pruning_rounds)
        per_round_prune = 1 - per_round_keep
        
        for round_idx in range(pruning_rounds):
            # Load trained weights for magnitude computation
            model.load_state_dict(trained_state)
            
            # Prune this round's fraction of remaining weights
            # (NOT cumulative - each round prunes per_round_prune of what remains)
            mask = pruner.prune(model, mask, per_round_prune)
            
            # Reset to original init with mask
            model.load_state_dict(self.original_init)
            mask.apply(model)
            
            # Train sparse network - pass mask to maintain sparsity
            # (train_fn should call mask.apply(model) after each optimizer step)
            train_fn(model, mask)
            
            # Ensure mask is applied after training (numerical stability)
            mask.apply(model)
            round_acc = eval_fn(model)
            
            # Update trained state for next round's magnitude computation
            trained_state = copy.deepcopy(model.state_dict())
            
            round_result = {
                'round': round_idx + 1,
                'sparsity': mask.sparsity,
                'accuracy': round_acc
            }
            results['rounds'].append(round_result)
            
            if verbose:
                print(f"Round {round_idx + 1}/{pruning_rounds}: "
                      f"{mask.sparsity:.1%} sparse, "
                      f"accuracy: {round_acc:.4f}")
        
        # Final results
        results['sparse_accuracy'] = round_acc
        results['sparsity'] = mask.sparsity
        self.winning_ticket_mask = mask
        
        if verbose:
            print("\n" + "=" * 60)
            print("Results Summary")
            print("=" * 60)
            print(f"Dense accuracy:  {dense_acc:.4f}")
            print(f"Sparse accuracy: {round_acc:.4f} "
                  f"({round_acc/dense_acc:.1%} retained)")
            print(f"Sparsity: {mask.sparsity:.1%}")
        
        return model, mask, results
    
    def verify_random_ticket(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        verbose: bool = True
    ) -> float:
        """
        Control experiment: verify random init with same mask performs worse.
        
        This proves the importance of the SPECIFIC initialization,
        not just the architecture/mask.
        """
        if self.winning_ticket_mask is None:
            raise ValueError("Must find winning ticket first")
        
        if verbose:
            print("\n" + "=" * 60)
            print("Control: Random Ticket (same mask, random init)")
            print("=" * 60)
        
        # New random initialization
        random_model = self.model_class(**self.model_kwargs).to(self.device)
        
        # Apply winning ticket mask to random weights
        self.winning_ticket_mask.apply(random_model)
        
        # Train with mask enforced
        train_fn(random_model, self.winning_ticket_mask)
        
        # Ensure mask applied after training
        self.winning_ticket_mask.apply(random_model)
        random_acc = eval_fn(random_model)
        
        if verbose:
            print(f"Random ticket accuracy: {random_acc:.4f}")
        
        return random_acc
    
    def one_shot_pruning(
        self,
        model: nn.Module,
        train_fn: Callable,
        eval_fn: Callable,
        target_sparsity: float = 0.9,
        verbose: bool = True
    ) -> Tuple[nn.Module, PruningMask, Dict]:
        """
        Simplified one-shot pruning (train once, prune once, retrain).
        
        Faster than IMP but may not find optimal tickets at high sparsity.
        """
        if self.original_init is None:
            self.original_init = copy.deepcopy(model.state_dict())
        
        if verbose:
            print("One-shot pruning pipeline...")
        
        # Train dense
        train_fn(model, None)
        dense_acc = eval_fn(model)
        
        # Prune
        mask = PruningMask(model)
        pruner = MagnitudePruner()
        mask = pruner.prune(model, mask, target_sparsity)
        
        # Reset and retrain with mask enforced
        model.load_state_dict(self.original_init)
        mask.apply(model)
        
        # Train with mask
        train_fn(model, mask)
        
        # Ensure mask applied after training
        mask.apply(model)
        sparse_acc = eval_fn(model)
        
        results = {
            'dense_accuracy': dense_acc,
            'sparse_accuracy': sparse_acc,
            'sparsity': mask.sparsity
        }
        
        if verbose:
            print(f"Dense: {dense_acc:.4f}, Sparse: {sparse_acc:.4f}, "
                  f"Sparsity: {mask.sparsity:.1%}")
        
        return model, mask, results
    
    def save_experiment(self, path: str, results: Dict) -> None:
        """Save experiment results and mask."""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(Path(path) / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save mask
        if self.winning_ticket_mask:
            self.winning_ticket_mask.save(str(Path(path) / 'mask.pt'))
        
        # Save original init
        if self.original_init:
            torch.save(self.original_init, Path(path) / 'original_init.pt')
    
    def load_experiment(self, path: str) -> Dict:
        """Load previous experiment."""
        with open(Path(path) / 'results.json') as f:
            results = json.load(f)
        
        mask_path = Path(path) / 'mask.pt'
        if mask_path.exists():
            model = self.model_class(**self.model_kwargs)
            self.winning_ticket_mask = PruningMask(model)
            self.winning_ticket_mask.load(str(mask_path))
        
        init_path = Path(path) / 'original_init.pt'
        if init_path.exists():
            self.original_init = torch.load(init_path, map_location='cpu')
        
        return results


def _register_mask_hooks(model: nn.Module, mask: PruningMask) -> list:
    """
    Register gradient hooks to zero gradients for pruned weights.
    
    This ensures sparsity is maintained during training by preventing
    gradient updates to pruned weights.
    """
    hooks = []
    for name, param in model.named_parameters():
        if name in mask.masks:
            m = mask.masks[name].float().to(param.device)
            # Use default argument to capture current value of m
            hook = param.register_hook(lambda grad, m=m: grad * m)
            hooks.append(hook)
    return hooks


def _remove_hooks(hooks: list) -> None:
    """Remove all registered hooks."""
    for hook in hooks:
        hook.remove()


def create_sparse_forward_hook(mask: PruningMask):
    """Create forward hook that maintains sparsity during inference."""
    def hook(module, input, output):
        for name, param in module.named_parameters():
            if name in mask.masks:
                with torch.no_grad():
                    param.mul_(mask.masks[name].to(param.device).float())
        return output
    return hook


# Convenience function for quick experiments
def find_lottery_ticket(
    model_class: type,
    train_fn: Callable,
    eval_fn: Callable,
    target_sparsity: float = 0.9,
    pruning_rounds: int = 10,
    **model_kwargs
) -> Tuple[nn.Module, PruningMask, Dict]:
    """
    Convenience function to find a lottery ticket in one call.
    
    Example:
        >>> model, mask, results = find_lottery_ticket(
        ...     MyModel, train_fn, eval_fn, target_sparsity=0.9
        ... )
    """
    finder = LotteryTicketFinder(model_class, model_kwargs)
    model = finder.initialize_network()
    return finder.find_winning_ticket(
        model, train_fn, eval_fn, target_sparsity, pruning_rounds
    )
