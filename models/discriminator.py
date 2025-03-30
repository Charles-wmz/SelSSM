"""
PatchGAN discriminator with spectral normalization.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from utils.initialization import weights_init_discriminator

class Discriminator(nn.Module):
    """
    PatchGAN discriminator with spectral normalization.
    
    The discriminator uses a convolutional architecture with spectral normalization
    to stabilize training. It outputs patch-wise predictions for real/fake classification.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Encoder path: downsampling
        self.conv1 = spectral_norm(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1))
        self.conv3 = spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1))

        # Output layer
        self.conv_out = spectral_norm(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1))

        # Activation functions and regularization
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        
        # Initialize weights
        self.apply(weights_init_discriminator)

    def forward(self, x):
        """
        Forward pass of the discriminator.
        
        Args:
            x: Input tensor [batch_size, 1, height, width]
            
        Returns:
            Patch-wise predictions [batch_size, 1, height/8, width/8]
        """
        # First convolutional layer
        x = self.relu(self.conv1(x))  # [batch_size, 32, height/2, width/2]
        x = self.dropout(x)

        # Second convolutional layer
        x = self.relu(self.conv2(x))  # [batch_size, 64, height/4, width/4]
        x = self.dropout(x)

        # Third convolutional layer
        x = self.relu(self.conv3(x))  # [batch_size, 128, height/8, width/8]
        x = self.dropout(x)

        # Output layer
        x = self.conv_out(x)  # [batch_size, 1, height/8, width/8]

        return x 