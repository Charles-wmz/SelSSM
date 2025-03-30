"""
Convolution modules for the model.
"""

import torch
import torch.nn as nn
from utils.initialization import weights_init_normal

class DynamicConv(nn.Module):
    """
    Dynamic convolution module that generates adaptive convolution kernels.
    
    Args:
        input_channels (int): Number of input channels
        kernel_size (int): Size of the dynamic convolution kernel
    """
    def __init__(self, input_channels, kernel_size=3):
        super(DynamicConv, self).__init__()
        self.input_channels = input_channels
        self.kernel_size = kernel_size

        # Generate dynamic convolution kernels
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, input_channels * kernel_size * kernel_size, kernel_size=1),
            nn.ReLU()
        )
        
        # Initialize weights
        self.apply(weights_init_normal)

    def forward(self, x):
        """
        Forward pass to apply dynamic convolution.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Dynamically convolved output
        """
        batch_size, channels, height, width = x.size()

        # Generate dynamic kernels
        dynamic_kernels = self.kernel_generator(x)  # [batch_size, channels * kernel_size^2, 1, 1]
        dynamic_kernels = dynamic_kernels.view(batch_size * channels, 1, self.kernel_size, self.kernel_size)

        # Apply dynamic convolution
        x = x.view(1, batch_size * channels, height, width)  # Merge batch and channels
        output = nn.functional.conv2d(x, dynamic_kernels, groups=batch_size * channels, padding=self.kernel_size // 2)
        output = output.view(batch_size, channels, height, width)  # Restore original dimensions
        return output 