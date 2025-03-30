"""
Generator network with U-Net architecture and SelectiveSSM integration.
"""

import torch
import torch.nn as nn
from utils.initialization import weights_init_normal
from models.attention import SpatialAttention
from models.ssm import SelectiveSSM

class Generator(nn.Module):
    """
    Generator network with U-Net architecture and SelectiveSSM integration.
    
    The generator uses a U-Net style architecture with skip connections and incorporates
    the SelectiveSSM module for feature enhancement. It includes both encoder and decoder
    paths with appropriate normalization and activation functions.
    """
    def __init__(self):
        super(Generator, self).__init__()
        
        self.ssm = SelectiveSSM()  # Integrate SelectiveSSM module
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable fusion weight
        
        # Encoder path: downsampling
        self.enc_conv_input = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.enc_conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.enc_conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1)
        
        # Decoder path: upsampling
        self.dec_conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(in_channels=512*2, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(in_channels=256*2, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(in_channels=128*2, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.dec_conv5 = nn.ConvTranspose2d(in_channels=64*2, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.dec_conv6 = nn.ConvTranspose2d(in_channels=32*2, out_channels=16, kernel_size=4, stride=2, padding=1)
        
        # Output layer
        self.conv_out = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        
        # Spatial attention layer
        self.spatial_attention = SpatialAttention(kernel_size=5)
        
        # Instance normalization for decoder
        self.ibn16 = nn.InstanceNorm2d(16)
        self.ibn32 = nn.InstanceNorm2d(32)
        self.ibn64 = nn.InstanceNorm2d(64)
        self.ibn128 = nn.InstanceNorm2d(128)
        self.ibn256 = nn.InstanceNorm2d(256)
        self.ibn512 = nn.InstanceNorm2d(512)

        # Batch normalization for encoder
        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)
        self.bn1024 = nn.BatchNorm2d(1024)

        # Activation functions and regularization
        self.relu = nn.LeakyReLU(0.3)
        self.dropout = nn.Dropout(0.1)
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self.apply(weights_init_normal)

    def forward(self, x, target_modality):
        """
        Forward pass of the generator.
        
        Args:
            x: Input tensor [batch_size, 3, height, width]
            target_modality: Target modality index [batch_size]
            
        Returns:
            Tuple of (generated image, contrastive loss)
        """
        # SelectiveSSM feature enhancement
        ssm_output, contrastive_loss = self.ssm(x, target_modality)

        # Weighted fusion of original input and SSM output
        x = self.alpha * x + (1 - self.alpha) * ssm_output
        
        # Encoder path: feature extraction
        x_enc_1 = self.relu(self.bn32(self.enc_conv_input(x)))  # (B, 3, 256, 256) -> (B, 32, 128, 128)
        x_enc_1_d = self.dropout(x_enc_1)

        x_enc_2 = self.relu(self.bn64(self.enc_conv1(x_enc_1_d)))  # (B, 32, 128, 128) -> (B, 64, 64, 64)
        x_enc_2_d = self.dropout(x_enc_2)

        x_enc_3 = self.relu(self.bn128(self.enc_conv2(x_enc_2_d)))  # (B, 64, 64, 64) -> (B, 128, 32, 32)
        x_enc_3_d = self.dropout(x_enc_3)

        x_enc_4 = self.relu(self.bn256(self.enc_conv3(x_enc_3_d)))  # (B, 128, 32, 32) -> (B, 256, 16, 16)
        x_enc_4_d = self.dropout(x_enc_4)
        
        x_enc_5 = self.relu(self.bn512(self.enc_conv4(x_enc_4_d)))  # (B, 256, 16, 16) -> (B, 512, 8, 8)
        x_enc_5_d = self.dropout(x_enc_5)

        x_enc_6 = self.relu(self.bn1024(self.enc_conv5(x_enc_5_d)))  # (B, 512, 8, 8) -> (B, 1024, 4, 4)
        x_enc_6_d = self.dropout(x_enc_6)

        # Decoder path: image reconstruction
        x_dec_1 = self.relu(self.ibn512(self.dec_conv1(x_enc_6_d)))  # (B, 1024, 4, 4) -> (B, 512, 8, 8)
        
        jump_1 = torch.cat([x_dec_1, x_enc_5], 1)
        x_dec_2 = self.relu(self.ibn256(self.dec_conv2(jump_1)))  # (B, 512*2, 8, 8) -> (B, 256, 16, 16)
        
        jump_2 = torch.cat([x_dec_2, x_enc_4], 1)
        x_dec_3 = self.relu(self.ibn128(self.dec_conv3(jump_2)))  # (B, 256*2, 16, 16) -> (B, 128, 32, 32)
        
        jump_3 = torch.cat([x_dec_3, x_enc_3], 1)
        x_dec_4 = self.relu(self.ibn64(self.dec_conv4(jump_3)))  # (B, 128*2, 32, 32) -> (B, 64, 64, 64)
        self.spatial_attention(x_dec_4)
        
        jump_4 = torch.cat([x_dec_4, x_enc_2], 1)
        x_dec_5 = self.relu(self.ibn32(self.dec_conv5(jump_4)))  # (B, 64*2, 64, 64) -> (B, 32, 128, 128)
        self.spatial_attention(x_dec_5)
        
        jump_5 = torch.cat([x_dec_5, x_enc_1], 1)
        x_dec_6 = self.relu(self.ibn16(self.dec_conv6(jump_5)))  # (B, 32*2, 128, 128) -> (B, 16, 256, 256)
        self.spatial_attention(x_dec_6)
        
        # Output layer
        x_out = self.conv_out(x_dec_6)  # (B, 16, 256, 256) -> (B, 1, 256, 256)
        x_out = self.tanh(x_out)  # Limit output to [-1, 1]
        
        return x_out, contrastive_loss 