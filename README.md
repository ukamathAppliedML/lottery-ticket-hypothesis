# The Lottery Ticket Hypothesis
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready implementation of [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) (Frankle & Carbin, ICLR 2019).

## The Core Insight

When you initialize a neural network with random weights, most of those weights are useless. Hidden inside is a "winning ticket"—a small subnetwork that, trained in isolation, matches the full network's performance.

The proof is striking: prune 90% of weights, reset to original initialization, retrain. The sparse network works. A random sparse network with the same structure fails completely.

**It's the specific initialization that matters, not just the architecture.**

## Demo Results (CIFAR-10)

```
======================================================================
EXPERIMENT SUMMARY
======================================================================
    Dense Network:
        Parameters: 342,090
        Test Accuracy: 79.83%
    
    Winning Ticket (Sparse):
        Parameters: 34,208
        Sparsity: 90.0%
        Test Accuracy: 80.44%
        Performance Retained: 100.8%
    
    Random Ticket (Control):
        Same sparsity pattern, different initialization
        Test Accuracy: 52.77%
        vs Dense: 66.1%
    
    ============================================================
    GAP: 27.67% (27.7 percentage points)
    ============================================================
```

Same structure. Same sparsity. Different initialization. **27.7 percentage point gap.**

The winning ticket actually *beats* dense accuracy while the random ticket collapses to barely better than random guessing.

## Installation

```bash
git clone https://github.com/yourusername/lottery-ticket-hypothesis.git
cd lottery-ticket-hypothesis
pip install -e .
```

## Quick Start

```bash
python examples/lottery_ticket_demo.py
```

**Runtime:** ~10-15 minutes on CPU, ~3-5 minutes on GPU

## How It Works

### The Iterative Magnitude Pruning (IMP) Algorithm

1. **Initialize** a network with random weights θ₀
2. **Train** the dense network to completion
3. **Prune** the smallest weights by magnitude (globally across all layers)
4. **Reset** surviving weights to their original values θ₀ (not trained values)
5. **Retrain** the sparse network while enforcing the mask
6. **Repeat** steps 3-5 until target sparsity is reached

The key insight is step 4: you don't keep the trained weights. You keep the original random values that happened to be "lucky."

### A Toy Example

Let's trace the algorithm with a tiny 6-weight network.

**Step 1: Random Initialization (θ₀)**
```
θ₀ = [0.5, -0.1, 0.8, -0.3, 0.2, 0.7]
      w1    w2   w3    w4   w5   w6

We save this. This is the potential winning ticket.
```

**Step 2: Train Dense Network**
```
θ_trained = [0.9, 0.02, 1.2, -0.05, 0.4, 1.1]
             w1   w2    w3    w4    w5   w6

Notice: w2 and w4 ended up near zero — the network learned they don't matter.
```

**Step 3: Prune by Magnitude (50% sparsity)**
```
Rank by |value|: w4(0.05) < w2(0.02) < w5(0.4) < w1(0.9) < w6(1.1) < w3(1.2)
Prune smallest 50%: w2, w4, w5

mask = [1, 0, 1, 0, 0, 1]
        ✓  ✗  ✓  ✗  ✗  ✓
```

**Step 4: Reset to Original Init + Apply Mask**
```
θ₀        = [0.5, -0.1, 0.8, -0.3, 0.2, 0.7]
mask      = [  1,    0,   1,    0,   0,   1]
                  
θ₀ ⊙ mask = [0.5,  0.0, 0.8,  0.0, 0.0, 0.7]  ← Winning Ticket

We keep ORIGINAL random values of w1, w3, w6 — not their trained values.
```

**Step 5: Train Sparse Network → Same accuracy, 50% fewer parameters**

**Step 6: Control — Same mask, NEW random init**
```
θ₀_new ⊙ mask = [0.3, 0.0, 0.1, 0.0, 0.0, 0.2]  ← Random Ticket

Train this → Much worse accuracy!
```

**The Point:**

| | Winning Ticket | Random Ticket |
|---|---|---|
| Mask | [1,0,1,0,0,1] | [1,0,1,0,0,1] |
| Init | [0.5, -, 0.8, -, -, 0.7] | [0.3, -, 0.1, -, -, 0.2] |
| Result | 80% accuracy | 53% accuracy |

Same structure. Different numbers. The specific initialization matters.

### Implementation Details

**Mask Enforcement During Training**

The mask must be enforced after every optimizer step, not just at the beginning and end:

```python
def train_fn(model, mask=None):
    for data, target in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        
        # Critical: enforce sparsity after each step
        if mask is not None:
            mask.apply(model)
```

Without per-step enforcement, pruned weights "come back to life" during gradient updates, breaking the sparse training requirement.

**Global Magnitude Pruning**

Weights are pruned globally across all layers based on magnitude:

```python
# Collect all weights, find threshold at target percentile
all_weights = torch.cat([p.abs().flatten() for p in model.parameters()])
threshold = torch.quantile(all_weights, target_sparsity)

# Prune weights below threshold
mask = all_weights.abs() >= threshold
```

This allows different layers to have different sparsity levels based on weight importance.

**Iterative vs One-Shot Pruning**

Iterative pruning (multiple rounds) finds better tickets than one-shot pruning at high sparsity:

| Approach | 90% Sparsity | 95% Sparsity | 99% Sparsity |
|----------|-------------|--------------|--------------|
| One-shot | Good | Degraded | Poor |
| Iterative (5 rounds) | Excellent | Good | Reasonable |

Each round prunes a fraction of remaining weights, allowing the network to adapt gradually.

## Supported Models

### MNIST Architectures

| Model | Parameters | Description |
|-------|-----------|-------------|
| `LeNet300` | 266,610 | Fully-connected 784→300→100→10 |
| `LeNet5` | 61,706 | Classic CNN from LeCun et al. |

### CIFAR-10 Architectures

| Model | Parameters | Description |
|-------|-----------|-------------|
| `Conv2` | 88,986 | 2 conv + 2 FC layers |
| `Conv4` | 342,090 | 4 conv + 2 FC layers |
| `Conv6` | 1,122,858 | 6 conv + 2 FC layers |

### ImageNet-Scale Architectures

| Model | Parameters | Description |
|-------|-----------|-------------|
| `VGG11` | 132.9M | VGG with 11 layers |
| `VGG13` | 133.0M | VGG with 13 layers |
| `VGG16` | 138.4M | VGG with 16 layers |
| `VGG19` | 143.7M | VGG with 19 layers |
| `ResNet18` | 11.7M | ResNet with 18 layers |
| `ResNet34` | 21.8M | ResNet with 34 layers |
| `ResNet50` | 25.6M | ResNet with 50 layers |
| `ResNet101` | 44.5M | ResNet with 101 layers |

## API Reference

### LotteryTicketFinder

```python
from src.lottery_ticket import LotteryTicketFinder

finder = LotteryTicketFinder(
    model_class=Conv4,
    model_kwargs={'num_classes': 10},
    device='cuda'
)

# Find winning ticket via IMP
model, mask, results = finder.find_winning_ticket(
    model=model,
    train_fn=train_fn,      # Function(model, mask) -> None
    eval_fn=eval_fn,        # Function(model) -> float
    target_sparsity=0.9,    # 90% weights pruned
    pruning_rounds=4,       # Iterative rounds
    verbose=True
)

# Control experiment: random init with same mask
random_acc = finder.verify_random_ticket(train_fn, eval_fn)

# Save/load experiments
finder.save_experiment('./experiments/my_exp', results)
results = finder.load_experiment('./experiments/my_exp')
```

### PruningMask

```python
from src.lottery_ticket import PruningMask

mask = PruningMask(model)

# Apply mask to zero pruned weights
mask.apply(model)

# Check sparsity
print(f"Overall: {mask.sparsity:.1%}")
print(f"Per-layer: {mask.layer_sparsities}")

# Save/load masks
mask.save('mask.pt')
mask.load('mask.pt')
```

### MagnitudePruner

```python
from src.lottery_ticket import MagnitudePruner

pruner = MagnitudePruner(global_pruning=True)

# Prune to target sparsity
mask = pruner.prune(model, mask, target_sparsity=0.5)
```

## Structured 2:4 Sparsity

For production deployment on NVIDIA Ampere+ GPUs, the library includes 2:4 structured sparsity support:

```python
from src.structured_sparsity import Structured24Pruner

pruner = Structured24Pruner()
masks = pruner.prune_model(model)

# Verify pattern compliance
is_valid = pruner.verify_24_pattern(tensor)
```

The 2:4 pattern (exactly 2 non-zeros per 4 consecutive weights) enables hardware acceleration via Sparse Tensor Cores, providing real 2x speedup.

## Repository Structure

```
lottery-ticket-hypothesis/
├── src/
│   ├── lottery_ticket.py      # Core IMP implementation
│   │   ├── LotteryTicketFinder
│   │   ├── PruningMask
│   │   └── MagnitudePruner
│   ├── structured_sparsity.py # 2:4 GPU sparsity
│   │   └── Structured24Pruner
│   └── models.py              # Reference architectures
│       ├── LeNet300, LeNet5
│       ├── Conv2, Conv4, Conv6
│       ├── VGG11/13/16/19
│       └── ResNet18/34/50/101
├── examples/
│   └── lottery_ticket_demo.py # CIFAR-10 demonstration
├── tests/
│   └── test_pruning.py        # Unit tests
└── docs/
    └── medium_article.md      # Technical writeup
```

## Why This Matters Now

The 2018 paper was theoretical. Today it's practical:

- **Hardware support**: NVIDIA Ampere+ GPUs accelerate structured sparsity natively (2x speedup)
- **Framework support**: PyTorch 2.0 has native sparse tensor operations
- **Economic pressure**: LLM inference costs make efficiency essential

Production deployments stack optimizations: magnitude pruning → structured 2:4 sparsity → INT8 quantization, achieving 20-50x efficiency gains.

## Citation

```bibtex
@inproceedings{frankle2019lottery,
  title={The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks},
  author={Frankle, Jonathan and Carbin, Michael},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

## License

MIT
