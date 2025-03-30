"""
Selective State Space Model for medical image synthesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.initialization import weights_init_ssm
from models.attention import SpatialAttention
from models.conv import DynamicConv
from utils.losses import ContrastiveLoss

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model for medical image synthesis.
    
    Args:
        input_channels (int): Number of input modalities (channels)
        iterations (int): Number of state iterations
        local_window (int): Size of local window for gradient similarity
        sigma (float): Standard deviation for Gaussian similarity
    """
    def __init__(self, input_channels=3, iterations=3, local_window=5, sigma=0.5):
        super(SelectiveSSM, self).__init__()
        self.input_channels = input_channels
        self.iterations = iterations
        self.local_window = local_window
        self.sigma = sigma

        # State update components
        self.self_update = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, input_channels, kernel_size=3, padding=1)
        )
        
        self.neighbor_update = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(128, input_channels, kernel_size=3, padding=2, dilation=2)
        )
        
        # Input control matrix
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(input_channels, input_channels)
        self.sigmoid = nn.Sigmoid()
        
        # Learnable modality weights
        self.modality_weights = nn.Parameter(torch.ones(input_channels, dtype=torch.float32))
        
        # Sobel filters for gradient computation
        self.sobel_x = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False, groups=input_channels)
        self.sobel_y = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False, groups=input_channels)
        sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x.repeat(input_channels, 1, 1, 1), requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y.repeat(input_channels, 1, 1, 1), requires_grad=False)

        # Dynamic high-frequency enhancement module
        self.dynamic_conv = DynamicConv(input_channels, kernel_size=3)
        
        # Spatial attention mechanism
        self.spatial_attention = SpatialAttention(kernel_size=5)

        # Initialize all layers
        self.apply(weights_init_ssm)

    def gradient_similarity(self, state):
        """
        Compute gradient-based similarity between local neighborhoods.
        
        Args:
            state: Current state tensor [batch_size, channels, height, width]
            
        Returns:
            Similarity matrix
        """
        batch_size, channels, height, width = state.size()

        # Compute gradients
        grad_x = self.sobel_x(state)
        grad_y = self.sobel_y(state)
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Unfold local neighborhoods
        unfolded = F.unfold(gradient_magnitude, kernel_size=self.local_window, padding=self.local_window // 2)
        unfolded = unfolded.view(batch_size, channels, self.local_window ** 2, height * width).permute(0, 1, 3, 2)

        # Flatten state
        state_flat = gradient_magnitude.view(batch_size, channels, -1).unsqueeze(-1)

        # Compute gradient similarity
        diff = unfolded - state_flat
        dist = torch.sum(diff ** 2, dim=-1)
        similarity = torch.exp(-dist / (2 * self.sigma ** 2))
        return similarity
    
    def generate_positive_negative_samples(self, state, target_modality):
        """
        Dynamically generate positive and negative sample weights based on target modality.
        
        Args:
            state: Current state matrix [batch_size, channels, height, width]
            target_modality: Target modality index [batch_size]
            
        Returns:
            Tuple of (positive_weights, negative_weights)
        """
        batch_size, channels, height, width = state.size()
    
        # Compute Sobel gradient magnitude
        grad_x = self.sobel_x(state)
        grad_y = self.sobel_y(state)
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    
        # Dynamic high-frequency enhancement
        high_freq = self.dynamic_conv(state)
        high_freq = torch.sigmoid(high_freq)  # Limit to [0, 1]
    
        # Initialize weight tensors
        positive_weights = torch.zeros_like(state)
        negative_weights = torch.zeros_like(state)
    
        # Define weights for each modality
        for modality_idx in range(channels):
            mask = (target_modality == modality_idx).view(batch_size, 1, 1, 1)
            if modality_idx == 0:  # T1 modality
                positive = high_freq
                negative = 1 - torch.sigmoid(gradient_magnitude)
            elif modality_idx == 1:  # T1ce modality
                positive = torch.sigmoid(gradient_magnitude)
                negative = 1 - high_freq
            elif modality_idx == 2:  # T2 modality
                positive = high_freq * torch.sigmoid(gradient_magnitude)
                negative = 1 - positive
            elif modality_idx == 3:  # FLAIR modality
                positive = torch.sigmoid(gradient_magnitude * high_freq)
                negative = 1 - positive
            else:
                raise ValueError(f"Unknown target modality: {modality_idx}")
            
            # Ensure weights are on the correct device
            positive = positive.to(state.device)
            negative = negative.to(state.device)
            mask = mask.to(state.device)

            # Apply mask and update weights
            positive_weights += mask * positive
            negative_weights += mask * negative
    
        # Ensure weights are in [0, 1] range
        positive_weights = positive_weights.clamp(0, 1)
        negative_weights = negative_weights.clamp(0, 1)
    
        return positive_weights, negative_weights
    
    def forward(self, x, target_modality):
        """
        Forward pass of the SelectiveSSM module.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            target_modality: Target modality index [batch_size]
            
        Returns:
            Tuple of (output, contrastive_loss)
        """
        batch_size, channels, height, width = x.size()
    
        if channels != self.input_channels:
            raise ValueError(f"Input channel mismatch: expected {self.input_channels}, got {channels}")
    
        # Input control matrix
        global_features = self.global_avg_pool(x).view(batch_size, channels)
        input_weights = self.sigmoid(self.fc(global_features)).view(batch_size, channels, 1, 1)
        controlled_input = x * input_weights
        
        # Explicit modality weight control
        modality_weights = self.sigmoid(self.modality_weights).view(1, -1, 1, 1)
        controlled_input = controlled_input * modality_weights
    
        # Initialize state matrix
        state = controlled_input.clone()
        
        for _ in range(self.iterations):
            # Self state update
            updated_state = self.self_update(state)
    
            # Neighbor information update
            neighbor_info = self.neighbor_update(state)
    
            # Compute gradient similarity
            similarity = self.gradient_similarity(state)
            similarity = similarity / (similarity.sum(dim=-1, keepdim=True) + 1e-8)
            similarity = similarity.unsqueeze(-1)
    
            # Weighted state update
            state_unfolded = F.unfold(state, kernel_size=self.local_window, padding=self.local_window // 2)
            state_unfolded = state_unfolded.view(batch_size, channels, self.local_window ** 2, height * width).permute(0, 1, 3, 2)
            weighted_state = (state_unfolded * similarity).sum(dim=-1)
            weighted_state = weighted_state.view(batch_size, channels, height, width)
    
            # Dynamic convolution for high-frequency enhancement
            high_freq = self.dynamic_conv(state)
    
            # Update state
            state = weighted_state + updated_state + neighbor_info + high_freq
        
        # Generate positive and negative samples
        positive_weights, negative_weights = self.generate_positive_negative_samples(state, target_modality)
        
        # Selection matrix
        selection_weights = torch.exp(positive_weights) / (torch.exp(positive_weights) + torch.exp(negative_weights) + 1e-8)

        # Compute contrastive loss
        contrastive_loss_fn = ContrastiveLoss()
        contrastive_loss = contrastive_loss_fn(selection_weights, positive_weights, negative_weights)
    
        # Final output
        output = state * selection_weights
        
        return output, contrastive_loss 