"""
Tests for Lottery Ticket Hypothesis Implementation
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lottery_ticket import (
    LotteryTicketFinder,
    PruningMask,
    MagnitudePruner,
)
from src.structured_sparsity import Structured24Pruner
from src.models import LeNet300, Conv6


class TestPruningMask:
    """Tests for PruningMask class."""
    
    def test_initialization(self):
        """Test mask initialization."""
        model = LeNet300()
        mask = PruningMask(model)
        
        # All masks should be True initially
        assert mask.sparsity == 0.0
        assert all(m.all() for m in mask.masks.values())
    
    def test_apply_mask(self):
        """Test applying mask to model."""
        model = LeNet300()
        mask = PruningMask(model)
        
        # Set ~50% of weights to zero in mask (flatten, modify, reshape)
        for name in mask.masks:
            flat = mask.masks[name].view(-1)
            flat[:flat.numel()//2] = False
            mask.masks[name] = flat.view(mask.masks[name].shape)
        
        mask.apply(model)
        
        # Check sparsity
        assert mask.sparsity == pytest.approx(0.5, rel=0.1)
    
    def test_save_load(self, tmp_path):
        """Test saving and loading masks."""
        model = LeNet300()
        mask = PruningMask(model)
        
        # Modify mask
        for name in mask.masks:
            mask.masks[name][0] = False
        
        # Save and load
        path = str(tmp_path / "mask.pt")
        mask.save(path)
        
        new_mask = PruningMask(model)
        new_mask.load(path)
        
        # Verify
        for name in mask.masks:
            assert torch.equal(mask.masks[name], new_mask.masks[name])


class TestMagnitudePruner:
    """Tests for magnitude-based pruning."""
    
    def test_prune_to_sparsity(self):
        """Test pruning to target sparsity."""
        model = LeNet300()
        mask = PruningMask(model)
        pruner = MagnitudePruner(global_pruning=True)
        
        # Prune to 50% sparsity
        mask = pruner.prune(model, mask, target_sparsity=0.5)
        
        assert mask.sparsity == pytest.approx(0.5, rel=0.05)
    
    def test_prune_preserves_large_weights(self):
        """Test that pruning keeps large magnitude weights."""
        model = LeNet300()
        
        # Set first weight to be very large
        with torch.no_grad():
            model.fc1.weight[0, 0] = 1000.0
        
        mask = PruningMask(model)
        pruner = MagnitudePruner()
        mask = pruner.prune(model, mask, target_sparsity=0.9)
        
        # Large weight should still be active
        assert mask.masks['fc1.weight'][0, 0].item() == True


class TestStructured24Pruner:
    """Tests for 2:4 structured sparsity."""
    
    def test_24_pattern(self):
        """Test that 2:4 pattern is correctly applied."""
        weight = torch.randn(16, 16)
        pruner = Structured24Pruner()
        
        sparse_weight, mask = pruner.apply_24_mask(weight)
        
        # Check pattern: exactly 2 non-zeros per 4 elements
        flat = sparse_weight.view(-1)
        for i in range(0, len(flat) - 3, 4):
            group = flat[i:i+4]
            nonzeros = (group != 0).sum().item()
            assert nonzeros == 2, f"Group {i//4} has {nonzeros} non-zeros, expected 2"
    
    def test_verify_pattern(self):
        """Test pattern verification."""
        pruner = Structured24Pruner()
        
        # Valid pattern
        valid = torch.tensor([1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0])
        assert pruner.verify_24_pattern(valid)
        
        # Invalid pattern
        invalid = torch.tensor([1.0, 2.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0])
        assert not pruner.verify_24_pattern(invalid)
    
    def test_prune_model(self):
        """Test pruning full model."""
        model = LeNet300()
        pruner = Structured24Pruner()
        
        masks = pruner.prune_model(model)
        
        assert len(masks) > 0
        
        # Verify each pruned layer follows 2:4 pattern
        for name, param in model.named_parameters():
            if name in masks:
                assert pruner.verify_24_pattern(param)


class TestLotteryTicketFinder:
    """Tests for LotteryTicketFinder."""
    
    def test_initialize_network(self):
        """Test network initialization."""
        finder = LotteryTicketFinder(
            model_class=LeNet300,
            model_kwargs={'num_classes': 10}
        )
        
        model = finder.initialize_network()
        
        assert finder.original_init is not None
        assert isinstance(model, LeNet300)
    
    def test_one_shot_pruning(self):
        """Test one-shot pruning pipeline."""
        finder = LotteryTicketFinder(model_class=LeNet300)
        model = finder.initialize_network()
        
        # Simple training/eval functions (train_fn must accept mask)
        def train_fn(m, mask=None):
            optimizer = torch.optim.SGD(m.parameters(), lr=0.01)
            for _ in range(2):
                x = torch.randn(8, 784)
                y = torch.randint(0, 10, (8,))
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(m(x), y)
                loss.backward()
                optimizer.step()
                # Enforce mask after each step
                if mask is not None:
                    mask.apply(m)
        
        def eval_fn(m):
            m.eval()
            with torch.no_grad():
                x = torch.randn(32, 784)
                y = torch.randint(0, 10, (32,))
                pred = m(x).argmax(dim=1)
                return (pred == y).float().mean().item()
        
        model, mask, results = finder.one_shot_pruning(
            model, train_fn, eval_fn, target_sparsity=0.5
        )
        
        assert 'dense_accuracy' in results
        assert 'sparse_accuracy' in results
        assert mask.sparsity == pytest.approx(0.5, rel=0.1)


class TestModels:
    """Tests for model architectures."""
    
    @pytest.mark.parametrize("model_class,input_shape", [
        (LeNet300, (1, 784)),
        (Conv6, (1, 3, 32, 32)),
    ])
    def test_forward_pass(self, model_class, input_shape):
        """Test forward pass works."""
        model = model_class(num_classes=10)
        x = torch.randn(*input_shape)
        
        output = model(x)
        
        assert output.shape == (1, 10)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        from src.models import count_parameters
        
        model = LeNet300()
        counts = count_parameters(model)
        
        assert counts['total'] > 0
        assert counts['trainable'] == counts['total']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
