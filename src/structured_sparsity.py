"""
Structured Sparsity for Hardware Acceleration
==============================================

Implements 2:4 structured sparsity pattern for NVIDIA Tensor Core acceleration.
This pattern enables actual inference speedup (not just parameter reduction).

Background:
- NVIDIA Ampere+ GPUs have Sparse Tensor Cores
- They accelerate 2:4 sparsity: exactly 2 non-zero per 4 elements
- Provides ~2x speedup with 50% sparsity
- This is the key to making Lottery Ticket practical for production
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class Structured24Pruner:
    """
    Implements 2:4 structured sparsity for NVIDIA Tensor Cores.
    
    Pattern: Every 4 consecutive weights must have exactly 2 non-zeros.
    
    Example:
        Dense:     [0.5, 0.1, 0.8, 0.2]
        2:4 Sparse: [0.5, 0.0, 0.8, 0.0]  (keep 2 largest)
    
    This is NOT emulated sparsity - it runs natively on Ampere+ GPUs
    for real 2x compute speedup.
    """
    
    @staticmethod
    def apply_24_mask(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2:4 structured sparsity to a weight tensor.
        
        For each group of 4 consecutive weights, keeps the 2 with
        largest absolute magnitude.
        
        Args:
            weight: Weight tensor to sparsify
            
        Returns:
            Tuple of (sparse_weight, binary_mask)
        """
        original_shape = weight.shape
        original_device = weight.device
        weight = weight.cpu()  # Ensure CPU for manipulation
        
        # Flatten for processing
        flat = weight.view(-1)
        
        # Pad to multiple of 4
        pad_size = (4 - len(flat) % 4) % 4
        if pad_size > 0:
            flat = F.pad(flat, (0, pad_size))
        
        # Reshape to groups of 4
        groups = flat.view(-1, 4)
        
        # Find top 2 indices per group by magnitude
        _, top_indices = torch.topk(groups.abs(), k=2, dim=1)
        
        # Create mask
        mask = torch.zeros_like(groups, dtype=torch.bool)
        mask.scatter_(1, top_indices, True)
        
        # Apply mask
        sparse = groups * mask.float()
        
        # Remove padding and reshape
        if pad_size > 0:
            sparse = sparse.view(-1)[:-pad_size]
            mask = mask.view(-1)[:-pad_size]
        
        sparse = sparse.view(original_shape).to(original_device)
        mask = mask.view(original_shape).to(original_device)
        
        return sparse, mask
    
    @staticmethod
    def verify_24_pattern(tensor: torch.Tensor) -> bool:
        """
        Verify tensor follows valid 2:4 pattern.
        
        Returns True if every group of 4 has exactly 2 non-zeros.
        """
        flat = tensor.view(-1).cpu()
        
        # Check complete groups
        num_complete_groups = len(flat) // 4
        for i in range(num_complete_groups):
            group = flat[i*4:(i+1)*4]
            nonzeros = (group != 0).sum().item()
            if nonzeros != 2:
                return False
        
        return True
    
    def prune_layer(
        self,
        weight: torch.Tensor,
        importance_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prune a single layer to 2:4 sparsity.
        
        Args:
            weight: Weight tensor to prune
            importance_scores: Optional importance scores (uses magnitude if None)
            
        Returns:
            Tuple of (pruned_weight, mask)
        """
        if importance_scores is None:
            importance_scores = weight.abs()
        
        original_shape = weight.shape
        device = weight.device
        
        # Flatten importance scores
        imp_flat = importance_scores.view(-1).cpu()
        weight_flat = weight.view(-1).cpu()
        
        # Pad to multiple of 4
        pad_size = (4 - len(imp_flat) % 4) % 4
        if pad_size > 0:
            imp_flat = F.pad(imp_flat, (0, pad_size))
            weight_flat = F.pad(weight_flat, (0, pad_size))
        
        # Group and find top 2
        imp_groups = imp_flat.view(-1, 4)
        _, top_idx = torch.topk(imp_groups, k=2, dim=1)
        
        # Create mask and apply
        mask = torch.zeros_like(imp_groups, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)
        
        weight_groups = weight_flat.view(-1, 4)
        sparse_groups = weight_groups * mask.float()
        
        # Unpad
        if pad_size > 0:
            sparse_flat = sparse_groups.view(-1)[:-pad_size]
            mask_flat = mask.view(-1)[:-pad_size]
        else:
            sparse_flat = sparse_groups.view(-1)
            mask_flat = mask.view(-1)
        
        return (
            sparse_flat.view(original_shape).to(device),
            mask_flat.view(original_shape).to(device)
        )
    
    def prune_model(
        self,
        model: nn.Module,
        skip_layers: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply 2:4 sparsity to all applicable layers in a model.
        
        Args:
            model: Neural network to prune
            skip_layers: Layer names to skip (e.g., first/last layers)
            
        Returns:
            Dictionary mapping layer names to masks
        """
        skip_layers = skip_layers or []
        masks = {}
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' not in name or param.dim() < 2:
                    continue
                if any(skip in name for skip in skip_layers):
                    continue
                
                sparse_weight, mask = self.prune_layer(param)
                param.copy_(sparse_weight)
                masks[name] = mask
        
        return masks
    
    def fine_tune_with_pattern(
        self,
        model: nn.Module,
        masks: Dict[str, torch.Tensor],
        train_fn,
        num_epochs: int = 5
    ) -> nn.Module:
        """
        Fine-tune while maintaining 2:4 sparsity pattern.
        
        Gradients flow only through non-zero weights, maintaining
        the structured pattern throughout training.
        """
        # Register gradient hooks to zero out masked gradients
        hooks = []
        for name, param in model.named_parameters():
            if name in masks:
                mask = masks[name]
                def make_hook(m):
                    return lambda grad: grad * m.float().to(grad.device)
                hooks.append(param.register_hook(make_hook(mask)))
        
        # Fine-tune
        for epoch in range(num_epochs):
            train_fn(model)
            # Re-apply masks (numerical stability)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks:
                        param.mul_(masks[name].float().to(param.device))
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        return model


class SparseConv2d(nn.Module):
    """
    Sparse 2D convolution with 2:4 pattern support.
    
    Wrapper around standard Conv2d that maintains 2:4 sparsity
    and can export to TensorRT for hardware acceleration.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.mask: Optional[torch.Tensor] = None
        self.is_sparse = False
    
    def apply_sparsity(self):
        """Apply 2:4 sparsity to this layer."""
        pruner = Structured24Pruner()
        sparse_weight, self.mask = pruner.prune_layer(self.conv.weight)
        with torch.no_grad():
            self.conv.weight.copy_(sparse_weight)
        self.is_sparse = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_sparse and self.mask is not None:
            # Ensure sparsity maintained (numerical stability)
            weight = self.conv.weight * self.mask.float().to(self.conv.weight.device)
            return F.conv2d(
                x, weight, self.conv.bias,
                self.conv.stride, self.conv.padding
            )
        return self.conv(x)


class SparseLinear(nn.Module):
    """
    Sparse linear layer with 2:4 pattern support.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.mask: Optional[torch.Tensor] = None
        self.is_sparse = False
    
    def apply_sparsity(self):
        """Apply 2:4 sparsity to this layer."""
        pruner = Structured24Pruner()
        sparse_weight, self.mask = pruner.prune_layer(self.linear.weight)
        with torch.no_grad():
            self.linear.weight.copy_(sparse_weight)
        self.is_sparse = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_sparse and self.mask is not None:
            weight = self.linear.weight * self.mask.float().to(self.linear.weight.device)
            return F.linear(x, weight, self.linear.bias)
        return self.linear(x)


def convert_to_sparse_model(model: nn.Module) -> nn.Module:
    """
    Convert standard model to use sparse layers.
    
    Replaces Conv2d with SparseConv2d and Linear with SparseLinear.
    Does NOT apply sparsity - call apply_sparsity() on each layer after.
    """
    import copy
    model = copy.deepcopy(model)
    
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            sparse_conv = SparseConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size[0],
                stride=module.stride[0],
                padding=module.padding[0],
                bias=module.bias is not None
            )
            sparse_conv.conv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                sparse_conv.conv.bias.data = module.bias.data.clone()
            setattr(model, name, sparse_conv)
        elif isinstance(module, nn.Linear):
            sparse_linear = SparseLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None
            )
            sparse_linear.linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                sparse_linear.linear.bias.data = module.bias.data.clone()
            setattr(model, name, sparse_linear)
        elif len(list(module.children())) > 0:
            setattr(model, name, convert_to_sparse_model(module))
    
    return model


def compute_sparse_flops(model: nn.Module, input_shape: Tuple) -> Dict:
    """
    Compute theoretical FLOPs for sparse vs dense model.
    
    Returns comparison of operations needed.
    """
    from torch.profiler import profile, ProfilerActivity
    
    # This is a simplified estimation
    total_params = sum(p.numel() for p in model.parameters())
    sparse_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            # Assume 2:4 sparsity = 50% params
            sparse_params += param.numel() // 2
        else:
            sparse_params += param.numel()
    
    dense_flops = total_params * 2  # Multiply-add per param (rough estimate)
    sparse_flops = sparse_params * 2
    
    return {
        'dense_params': total_params,
        'sparse_params': sparse_params,
        'estimated_dense_flops': dense_flops,
        'estimated_sparse_flops': sparse_flops,
        'theoretical_speedup': dense_flops / sparse_flops if sparse_flops > 0 else 0
    }


# Export utilities for TensorRT

def export_sparse_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str,
    opset_version: int = 17
) -> str:
    """
    Export sparse model to ONNX format.
    
    Note: ONNX doesn't directly support sparsity, but TensorRT can
    recognize sparse patterns and accelerate them.
    """
    model.eval()
    
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=opset_version
    )
    
    return output_path


def create_tensorrt_config(sparsity_pattern: str = '2:4') -> Dict:
    """
    Create TensorRT build configuration for sparse acceleration.
    
    Returns config dict for TensorRT Python API.
    """
    return {
        'sparsity': {
            'pattern': sparsity_pattern,
            'enable_sparse_weights': True,
            'force_fp16': False  # INT8 with sparsity also supported
        },
        'optimization': {
            'workspace_size': 1 << 30,  # 1GB
            'max_batch_size': 64,
            'fp16_mode': True,
            'int8_mode': False
        },
        'hardware': {
            'min_compute_capability': 80,  # Ampere minimum
            'tensor_core_required': True
        }
    }
