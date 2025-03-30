"""
Test script for the Pix2Pix model with Selective State Space Model (SSM).

This script evaluates the trained model on test data and computes various metrics
including PSNR, SSIM, and MAE for image quality assessment.

Author: Cai Yize
Date: Nov 6, 2024
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import json

from utils.data import BraTSLoader
from utils.seed import set_seed
from models.generator import Generator
from models.discriminator import Discriminator

def calculate_metrics(generated, target):
    """
    Calculate image quality metrics between generated and target images.
    
    Args:
        generated: Generated image tensor [batch_size, 1, height, width]
        target: Target image tensor [batch_size, 1, height, width]
        
    Returns:
        Dictionary containing PSNR, SSIM, and MAE values
    """
    # Convert to numpy arrays
    generated_np = generated.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    
    # Calculate metrics
    psnr_value = psnr(target_np, generated_np)
    ssim_value = ssim(target_np, generated_np)
    mae_value = np.mean(np.abs(target_np - generated_np))
    
    return {
        'psnr': psnr_value,
        'ssim': ssim_value,
        'mae': mae_value
    }

def test_model(generator, test_loader, device, args):
    """
    Test the trained model on test data.
    
    Args:
        generator: Trained generator model
        test_loader: DataLoader for test data
        device: Device to run testing on (CPU/GPU)
        args: Testing arguments
        
    Returns:
        Dictionary containing test results and metrics
    """
    generator.eval()
    results = {
        'metrics': {
            'psnr': [],
            'ssim': [],
            'mae': []
        },
        'generated_images': [],
        'target_images': [],
        'subject_paths': []
    }
    
    with torch.no_grad():
        for inputs, targets, subject_paths, target_modality in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Generate images
            generated_images, _ = generator(inputs, target_modality)
            
            # Calculate metrics
            batch_metrics = calculate_metrics(generated_images, targets)
            
            # Store results
            results['metrics']['psnr'].append(batch_metrics['psnr'])
            results['metrics']['ssim'].append(batch_metrics['ssim'])
            results['metrics']['mae'].append(batch_metrics['mae'])
            results['generated_images'].append(generated_images.cpu())
            results['target_images'].append(targets.cpu())
            results['subject_paths'].extend(subject_paths)
    
    # Calculate average metrics
    results['metrics']['psnr'] = np.mean(results['metrics']['psnr'])
    results['metrics']['ssim'] = np.mean(results['metrics']['ssim'])
    results['metrics']['mae'] = np.mean(results['metrics']['mae'])
    
    return results

def save_results(results, args):
    """
    Save test results and metrics to files.
    
    Args:
        results: Dictionary containing test results
        args: Testing arguments
    """
    # Create results directory if it doesn't exist
    os.makedirs(args.results_path, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(args.results_path, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results['metrics'], f, indent=4)
    
    # Save generated images
    for i, (gen_img, target_img, subject_path) in enumerate(zip(
        results['generated_images'], 
        results['target_images'], 
        results['subject_paths']
    )):
        # Create subject directory
        subject_dir = os.path.join(args.results_path, os.path.basename(subject_path))
        os.makedirs(subject_dir, exist_ok=True)
        
        # Save images
        np.save(os.path.join(subject_dir, f'generated_{i}.npy'), gen_img.squeeze().numpy())
        np.save(os.path.join(subject_dir, f'target_{i}.npy'), target_img.squeeze().numpy())

def main():
    def parse_args():
        """Parse command line arguments for testing configuration."""
        parser = argparse.ArgumentParser(description='Test MambaGAN for BraTS Dataset')
        parser.add_argument('--test_path', type=str, default="../dataset/BraTS_2020_entire_DP150_80_SMt1_256/test", help='Path to test dataset')
        parser.add_argument('--model_path', type=str, default='./task4_pth_1_1000_t1/best_SSM_gan.pth', help='Path to trained model')
        parser.add_argument('--results_path', type=str, default='./task4_results_1_1000_t1', help='Path to save test results')
        parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
        parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
        return parser.parse_args()
    
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Print configuration
    print("Testing configuration:")
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
    
    # Test model
    start_time = time.time()
    results = test_model(generator, test_loader, device, args)
    end_time = time.time()
    
    # Print results
    print("\nTest Results:")
    print(f"PSNR: {results['metrics']['psnr']:.2f}")
    print(f"SSIM: {results['metrics']['ssim']:.4f}")
    print(f"MAE: {results['metrics']['mae']:.4f}")
    print(f"Testing time: {end_time - start_time:.2f} seconds")
    
    # Save results
    save_results(results, args)
    print(f"\nResults saved to {args.results_path}")

if __name__ == '__main__':
    main()