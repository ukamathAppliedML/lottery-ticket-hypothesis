"""
Reference Model Architectures
=============================

Standard architectures used in Lottery Ticket research.
These match the configurations from the original paper and follow-up works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


# =============================================================================
# LeNet Variants (MNIST)
# =============================================================================

class LeNet300(nn.Module):
    """
    LeNet-300-100: Fully-connected network from original LTH paper.
    
    Architecture: 784 -> 300 -> 100 -> 10
    Used for MNIST experiments in Frankle & Carlin (2018).
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5(nn.Module):
    """
    LeNet-5: Classic convolutional network for MNIST.
    
    Architecture from LeCun et al. (1998).
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# =============================================================================
# Conv Variants (CIFAR-10)
# =============================================================================

class Conv2(nn.Module):
    """Conv-2: Simple 2-layer CNN for CIFAR-10."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Conv4(nn.Module):
    """Conv-4: 4-layer CNN from LTH paper."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.fc = nn.Linear(128 * 8 * 8, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Conv6(nn.Module):
    """Conv-6: 6-layer CNN from LTH paper."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Linear(256 * 4 * 4, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# =============================================================================
# VGG Variants
# =============================================================================

class VGG(nn.Module):
    """
    VGG-style network with configurable depth.
    
    From "Very Deep Convolutional Networks" (Simonyan & Zisserman, 2014).
    """
    
    CONFIGS = {
        'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
    
    def __init__(
        self,
        config: str = 'vgg16',
        num_classes: int = 10,
        batch_norm: bool = True
    ):
        super().__init__()
        self.features = self._make_layers(self.CONFIGS[config], batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def _make_layers(self, config: List, batch_norm: bool) -> nn.Sequential:
        layers = []
        in_channels = 3
        
        for v in config:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                conv = nn.Conv2d(in_channels, v, 3, padding=1)
                if batch_norm:
                    layers.extend([conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                else:
                    layers.extend([conv, nn.ReLU(inplace=True)])
                in_channels = v
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def vgg11(num_classes: int = 10, **kwargs) -> VGG:
    return VGG('vgg11', num_classes, **kwargs)

def vgg16(num_classes: int = 10, **kwargs) -> VGG:
    return VGG('vgg16', num_classes, **kwargs)

def vgg19(num_classes: int = 10, **kwargs) -> VGG:
    return VGG('vgg19', num_classes, **kwargs)


# =============================================================================
# ResNet Variants
# =============================================================================

class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
    
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50/101/152."""
    
    expansion = 4
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """ResNet architecture."""
    
    def __init__(
        self,
        block,
        num_blocks: List[int],
        num_classes: int = 10
    ):
        super().__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(num_classes: int = 10) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes: int = 10) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes: int = 10) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes: int = 10) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


# =============================================================================
# Utility Functions
# =============================================================================

def get_model(name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """
    Factory function to get model by name.
    
    Args:
        name: Model name (e.g., 'lenet300', 'vgg16', 'resnet18')
        num_classes: Number of output classes
        **kwargs: Additional model arguments
        
    Returns:
        Instantiated model
    """
    models = {
        'lenet300': LeNet300,
        'lenet5': LeNet5,
        'conv2': Conv2,
        'conv4': Conv4,
        'conv6': Conv6,
        'vgg11': lambda nc: VGG('vgg11', nc, **kwargs),
        'vgg13': lambda nc: VGG('vgg13', nc, **kwargs),
        'vgg16': lambda nc: VGG('vgg16', nc, **kwargs),
        'vgg19': lambda nc: VGG('vgg19', nc, **kwargs),
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
    }
    
    if name.lower() not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    
    return models[name.lower()](num_classes)


def count_parameters(model: nn.Module) -> dict:
    """Count parameters by type."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    conv_params = sum(
        p.numel() for n, p in model.named_parameters() 
        if 'conv' in n.lower()
    )
    fc_params = sum(
        p.numel() for n, p in model.named_parameters() 
        if 'fc' in n.lower() or 'linear' in n.lower() or 'classifier' in n.lower()
    )
    
    return {
        'total': total,
        'trainable': trainable,
        'conv': conv_params,
        'fc': fc_params
    }
