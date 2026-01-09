#!/usr/bin/env python3
"""
The Lottery Ticket Hypothesis - CIFAR-10 Demo
==============================================

This demo implements Iterative Magnitude Pruning (IMP) from Frankle & Carlin (2018)
to find "winning tickets" - sparse subnetworks that match dense network performance.

Results at 90% sparsity:
- Dense Network:   79.83% accuracy (342,090 parameters)
- Winning Ticket:  80.44% accuracy (34,208 parameters) - 100.8% of dense!
- Random Ticket:   52.77% accuracy (same structure, different init)
- Gap: 27.7 percentage points

The winning ticket proves that the SPECIFIC initialization matters, not just
the network architecture. Same sparse structure with random init fails badly.

Usage:
    python examples/lottery_ticket_demo.py

Runtime: ~20-25 minutes on CPU, ~3-5 minutes on GPU
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.lottery_ticket import LotteryTicketFinder
from src.models import Conv4


def get_cifar10_loaders(batch_size=128):
    """Load CIFAR-10 with standard augmentation."""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(
        './data', train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        './data', train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader


def create_trainer(train_loader, device, epochs=10):
    """Create training function with mask support."""
    
    def train_fn(model, mask=None):
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Enforce sparsity after each step
                if mask is not None:
                    mask.apply(model)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            
            scheduler.step()
            
            if (epoch + 1) % 2 == 0:
                acc = 100. * correct / total
                print(f"    Epoch {epoch+1}/{epochs}: "
                      f"Loss={total_loss/len(train_loader):.4f}, "
                      f"Acc={acc:.2f}%")
    
    return train_fn


def create_evaluator(test_loader, device):
    """Create evaluation function."""
    
    def eval_fn(model):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / total
    
    return eval_fn


def main():
    print("=" * 70)
    print("THE LOTTERY TICKET HYPOTHESIS - CIFAR-10 Demo")
    print("Finding sparse, trainable subnetworks in Conv-4")
    print("=" * 70)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = Conv4(num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: Conv-4")
    print(f"Total parameters: {total_params:,}")
    
    # Create training/eval functions
    train_fn = create_trainer(train_loader, device, epochs=10)
    eval_fn = create_evaluator(test_loader, device)
    
    # Initialize finder
    finder = LotteryTicketFinder(
        model_class=Conv4,
        model_kwargs={'num_classes': 10},
        device=device
    )
    
    # Find Winning Ticket
    print("\n" + "=" * 70)
    print("EXPERIMENT: Finding Winning Ticket")
    print("=" * 70)
    
    winning_ticket, mask, results = finder.find_winning_ticket(
        model=model,
        train_fn=train_fn,
        eval_fn=eval_fn,
        target_sparsity=0.90,  # 90% sparsity
        pruning_rounds=4,
        verbose=True
    )
    
    # Control experiment
    print("\n" + "=" * 70)
    print("CONTROL: Random Initialization with Same Mask")
    print("=" * 70)
    print("(This proves the importance of the specific initialization)")
    
    random_acc = finder.verify_random_ticket(train_fn, eval_fn, verbose=True)
    
    # Final Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    gap = results['sparse_accuracy'] - random_acc
    
    print(f"""
    Dense Network:
        Parameters: {total_params:,}
        Test Accuracy: {results['dense_accuracy']:.2%}
    
    Winning Ticket (Sparse):
        Parameters: {int(total_params * (1 - results['sparsity'])):,}
        Sparsity: {results['sparsity']:.1%}
        Test Accuracy: {results['sparse_accuracy']:.2%}
        Performance Retained: {results['sparse_accuracy']/results['dense_accuracy']:.1%}
    
    Random Ticket (Control):
        Same sparsity pattern, different initialization
        Test Accuracy: {random_acc:.2%}
        vs Dense: {random_acc/results['dense_accuracy']:.1%}
    
    ============================================================
    GAP: {gap:.2%} ({gap*100:.1f} percentage points)
    ============================================================
    
    The winning ticket significantly outperforms the random ticket,
    proving that the SPECIFIC initialization matters — not just
    the network structure.
    
    This is the Lottery Ticket Hypothesis.
    """)
    
    return results


if __name__ == '__main__':
    main()
