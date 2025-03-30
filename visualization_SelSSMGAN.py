"""
Visualization script for the Pix2Pix model with Selective State Space Model (SSM).

This script generates visualizations of the model's outputs, including:
- Generated images vs ground truth
- Attention maps
- Feature maps
- Loss curves

Author: Cai Yize
Date: Nov 6, 2024
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils.data import BraTSLoader
from utils.seed import set_seed
from models.generator import Generator
from models.discriminator import Discriminator

def plot_image_comparison(generated, target, save_path, title=None):
    """
    Plot and save a comparison between generated and target images.
    
    Args:
        generated: Generated image tensor [1, height, width]
        target: Target image tensor [1, height, width]
        save_path: Path to save the plot
        title: Optional title for the plot
    """
    plt.figure(figsize=(10, 4))
    
    # Plot generated image
    plt.subplot(1, 2, 1)
    plt.imshow(generated.squeeze(), cmap='gray')
    plt.title('Generated Image')
    plt.axis('off')
    
    # Plot target image
    plt.subplot(1, 2, 2)
    plt.imshow(target.squeeze(), cmap='gray')
    plt.title('Target Image')
    plt.axis('off')
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_attention_maps(attention_maps, save_path):
    """
    Plot and save attention maps.
    
    Args:
        attention_maps: Dictionary of attention maps
        save_path: Path to save the plot
    """
    n_maps = len(attention_maps)
    fig, axes = plt.subplots(1, n_maps, figsize=(5*n_maps, 5))
    
    if n_maps == 1:
        axes = [axes]
    
    for (name, map_data), ax in zip(attention_maps.items(), axes):
        ax.imshow(map_data.squeeze(), cmap='hot')
        ax.set_title(name)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_maps(feature_maps, save_path):
    """
    Plot and save feature maps.
    
    Args:
        feature_maps: Dictionary of feature maps
        save_path: Path to save the plot
    """
    n_maps = len(feature_maps)
    n_cols = min(4, n_maps)
    n_rows = (n_maps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for (name, map_data), ax in zip(feature_maps.items(), axes.flat):
        if map_data is not None:
            ax.imshow(map_data.squeeze(), cmap='viridis')
            ax.set_title(name)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_loss_curves(losses_path, save_path):
    """
    Plot and save training loss curves.
    
    Args:
        losses_path: Path to the losses.json file
        save_path: Path to save the plot
    """
    with open(losses_path, 'r') as f:
        losses = json.load(f)
    
    plt.figure(figsize=(10, 6))
    for name, values in losses.items():
        plt.plot(values, label=name)
    
    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def visualize_model(generator, test_loader, device, args):
    """
    Generate visualizations for the model's outputs and internal states.
    
    Args:
        generator: Trained generator model
        test_loader: DataLoader for test data
        device: Device to run visualization on (CPU/GPU)
        args: Visualization arguments
    """
    generator.eval()
    
    # Create visualization directory
    os.makedirs(args.vis_path, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, subject_paths, target_modality) in enumerate(tqdm(test_loader, desc="Generating visualizations")):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Generate images and get attention maps
            generated_images, _ = generator(inputs, target_modality)
            
            # Process each image in the batch
            for i, (gen_img, target_img, subject_path) in enumerate(zip(generated_images, targets, subject_paths)):
                # Create subject directory
                subject_dir = os.path.join(args.vis_path, os.path.basename(subject_path))
                os.makedirs(subject_dir, exist_ok=True)
                
                # Plot image comparison
                comparison_path = os.path.join(subject_dir, f'comparison_{i}.png')
                plot_image_comparison(
                    gen_img.cpu().numpy(),
                    target_img.cpu().numpy(),
                    comparison_path,
                    f'Subject {os.path.basename(subject_path)} - Slice {i}'
                )
                
                # Calculate and save metrics
                metrics = {
                    'psnr': psnr(target_img.squeeze().cpu().numpy(), gen_img.squeeze().cpu().numpy()),
                    'ssim': ssim(target_img.squeeze().cpu().numpy(), gen_img.squeeze().cpu().numpy())
                }
                
                metrics_path = os.path.join(subject_dir, f'metrics_{i}.json')
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)

def main():
    def parse_args():
        """Parse command line arguments for visualization configuration."""
        parser = argparse.ArgumentParser(description='Visualize MambaGAN for BraTS Dataset')
        parser.add_argument('--test_path', type=str, default="../dataset/BraTS_2020_entire_DP150_80_SMt1_256/test", help='Path to test dataset')
        parser.add_argument('--model_path', type=str, default='./task4_pth_1_1000_t1/best_SSM_gan.pth', help='Path to trained model')
        parser.add_argument('--vis_path', type=str, default='./task4_visualization_1_1000_t1', help='Path to save visualizations')
        parser.add_argument('--batch_size', type=int, default=1, help='Batch size for visualization')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
        parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
        return parser.parse_args()
    
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Print configuration
    print("Visualization configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define input modalities for different tasks
    desired_inputs_task1 = ['t1.npy', 't1ce.npy', 't2.npy']  # t1, t1ce, t2 -> flair [0001]
    desired_inputs_task2 = ['t1.npy', 't1ce.npy', 'flair.npy']  # t1, t1ce, flair -> t2 [0010]
    desired_inputs_task3 = ['t1.npy', 't2.npy', 'flair.npy']  # t1, t2, flair -> t1ce [0100]
    desired_inputs_task4 = ['t1ce.npy', 't2.npy', 'flair.npy']  # t1ce, t2, flair -> t1 [1000]
    
    # Load test dataset
    test_dataset = BraTSLoader(args.test_path, input_modalities=desired_inputs_task4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Initialize generator
    generator = Generator().to(device)
    
    # Load trained model
    checkpoint = torch.load(args.model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Generate visualizations
    visualize_model(generator, test_loader, device, args)
    print(f"\nVisualizations saved to {args.vis_path}")

if __name__ == '__main__':
    main()
