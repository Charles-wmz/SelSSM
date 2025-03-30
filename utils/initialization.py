"""
Weight initialization utilities for different network components.
"""

import torch
import torch.nn as nn
import torch.nn.init as init

def weights_init_normal(m):
    """
    Generic weight initialization function for standard layers.
    Uses Kaiming initialization for convolutional layers and Xavier for linear layers.
    """
    if hasattr(m, 'weight') and m.weight is not None:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

def weights_init_ssm(m):
    """
    Weight initialization function specifically for SelectiveSSM module.
    Uses Kaiming initialization for convolutional layers and Xavier for linear layers.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    
def weights_init_discriminator(m):
    """
    Weight initialization function specifically for Discriminator module.
    Note: Spectral normalization layers should not be reinitialized as it overwrites their weights.
    """
    if isinstance(m, nn.Conv2d):
        if not hasattr(m, 'weight') or m.weight.requires_grad:
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0) 