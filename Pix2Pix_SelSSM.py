"""
Pix2Pix with Selective State Space Model (SSM) for Medical Image Synthesis

This module implements a GAN-based medical image synthesis model that uses a Selective State Space Model
for feature extraction and generation. The model is specifically designed for multi-modal medical image
synthesis tasks, particularly for the BraTS dataset.

Key components:
- SelectiveSSM: A state space model with selective attention mechanism
- Generator: U-Net style architecture with SSM integration
- Discriminator: PatchGAN discriminator with spectral normalization
- Training utilities with gradient clipping and learning rate scheduling

Author: Cai Yize
Date: Nov 6, 2024
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import time
import torch.nn.utils as nn_utils
import json

from utils.data import BraTSLoader
from utils.seed import set_seed
from utils.losses import edge_loss
from models.generator import Generator
from models.discriminator import Discriminator

def train_gan(generator, discriminator, train_loader, g_optimizer, d_optimizer, device, num_epochs, args, checkpoint_path=None):
    """
    Train the GAN model with the specified configuration.
    
    Args:
        generator: Generator model instance
        discriminator: Discriminator model instance
        train_loader: DataLoader for training data
        g_optimizer: Optimizer for generator
        d_optimizer: Optimizer for discriminator
        device: Device to run training on (CPU/GPU)
        num_epochs: Number of training epochs
        args: Training arguments
        checkpoint_path: Path to load checkpoint from (optional)
    """
    best_g_loss = float("inf")  # Track best generator loss
    start_epoch = 0
    adversarial_loss = nn.BCEWithLogitsLoss()
    
    generator.train()
    discriminator.train()
    
    # Track losses
    epoch_losses = {
        "generator_loss": [],
        "discriminator_loss": []
    }
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_g_loss = checkpoint.get('best_g_loss', float("inf"))
        print(f"Resuming training from epoch {start_epoch}")
        
        # Update learning rates
        for param_group in g_optimizer.param_groups:
            param_group['lr'] = args.g_lr
    
        for param_group in d_optimizer.param_groups:
            param_group['lr'] = args.d_lr
    
    else:
        # Set initial learning rates
        for param_group in g_optimizer.param_groups:
            param_group['lr'] = args.g_lr

        for param_group in d_optimizer.param_groups:
            param_group['lr'] = args.d_lr
        
    # Initialize learning rate schedulers
    g_lr_scheduler = lr_scheduler.MultiStepLR(g_optimizer, milestones=[80], gamma=0.1)
    d_lr_scheduler = lr_scheduler.MultiStepLR(d_optimizer, milestones=[80], gamma=0.1)

    # Load scheduler states if available
    if checkpoint_path is not None and 'g_scheduler_state_dict' in checkpoint and 'd_scheduler_state_dict' in checkpoint:
        g_lr_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
        d_lr_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
        print("Loaded scheduler states from checkpoint.")
        
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Get current learning rates
        current_g_lr = g_optimizer.param_groups[0]['lr']
        current_d_lr = d_optimizer.param_groups[0]['lr']
        print(f"Learning Rates -> Generator: {current_g_lr:.7f}, Discriminator: {current_d_lr:.7f}")
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        total_d_loss = 0.0
        total_g_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets, _, target_modality) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
                        
            # ----------------------
            # Train Discriminator
            # ----------------------
            d_optimizer.zero_grad()
            generated_images, contrastive_loss = generator(inputs, target_modality)

            # Discriminator predictions for real data
            real_preds = discriminator(targets)
            real_labels = torch.ones_like(real_preds).to(device) * (1 - args.smoothing_factor)

            # Discriminator predictions for fake data
            fake_preds = discriminator(generated_images.detach())
            fake_labels = torch.zeros_like(fake_preds).to(device) + args.smoothing_factor

            # Discriminator loss
            real_loss = adversarial_loss(real_preds, real_labels)
            fake_loss = adversarial_loss(fake_preds, fake_labels)
            d_loss = (real_loss + fake_loss) / 2

            # Optimize discriminator
            d_loss.backward()
            nn_utils.clip_grad_norm_(d_optimizer.param_groups[0]['params'], args.max_norm)
            d_optimizer.step()

            # Accumulate discriminator loss
            total_d_loss += d_loss.item()

            # ----------------------
            # Train Generator
            # ----------------------
            g_optimizer.zero_grad()

            # Content loss (L1 loss)
            l1_content_loss = nn.L1Loss()(generated_images, targets)
            
            # Discriminator predictions for generated data
            g_preds = discriminator(generated_images)
            target_labels = torch.ones_like(g_preds).to(device)
            
            # Edge detail loss based on intensity gradients
            detail_loss = edge_loss(generated_images, targets)

            # Learning rate phase settings and loss function weights
            if epoch < 50:  # Early phase
                lambda_contrastive = 0.5
            elif epoch < 70:  # Middle phase
                lambda_contrastive = 1.0
            else:  # Late phase
                lambda_contrastive = 1.0

            # Generator loss
            g_loss = adversarial_loss(g_preds, target_labels) + \
                     lambda_contrastive * contrastive_loss
                     
            g_loss.backward()

            # Gradient clipping
            nn_utils.clip_grad_norm_(g_optimizer.param_groups[0]['params'], args.max_norm)
            g_optimizer.step()

            # Accumulate losses
            total_g_loss += g_loss.item()
            num_batches += 1

        # Calculate average losses for the epoch
        avg_d_loss = total_d_loss / num_batches
        avg_g_loss = total_g_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}")
                
        # Save losses
        epoch_losses["generator_loss"].append(avg_g_loss)
        epoch_losses["discriminator_loss"].append(avg_d_loss)

        # Save losses to file
        losses_path = os.path.join(args.save_path, "losses.json")
        with open(losses_path, "w") as f:
            json.dump(epoch_losses, f)

        # Save model checkpoint
        epoch_save_path = os.path.join(args.save_path, f"SSM_gan_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'g_scheduler_state_dict': g_lr_scheduler.state_dict(),
            'd_scheduler_state_dict': d_lr_scheduler.state_dict(),
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss,
            'best_g_loss': best_g_loss,
            'epoch_losses': epoch_losses,
        }, epoch_save_path)
        print(f"Model saved at {epoch_save_path}")

        # Save best model if generator loss improves
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            best_model_path = os.path.join(args.save_path, "best_SSM_gan.pth")
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_scheduler_state_dict': g_lr_scheduler.state_dict(),
                'd_scheduler_state_dict': d_lr_scheduler.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss,
                'best_g_loss': best_g_loss,
                'epoch_losses': epoch_losses,
            }, best_model_path)
            print(f"Best model updated and saved at {best_model_path}")
                
        # Step learning rate schedulers
        g_lr_scheduler.step()
        d_lr_scheduler.step()
        
if __name__ == '__main__':
    def parse_args():
        """
        Parse command line arguments for training configuration.
        
        Returns:
            Parsed arguments
        """
        parser = argparse.ArgumentParser(description='MambaGAN for BraTS Dataset')
        parser.add_argument('--train_path', type=str, default="../dataset/BraTS_2020_entire_DP150_80_SMt1_256/train", help='Path to training dataset')
        parser.add_argument('--train_model', type=str, default='gan', help='Setting up the trained model')
        parser.add_argument('--num_work', type=int, default=16, help='Number of processes for dataloader')
        parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
        parser.add_argument('--num_slices', type=int, default=80, help='Number of slices')
        parser.add_argument('--height', type=int, default=256, help='Height of the input sample')
        parser.add_argument('--width', type=int, default=256, help='Width of the input sample')
        parser.add_argument('--g_lr', type=float, default=1e-4, help='Learning rate for generator')
        parser.add_argument('--d_lr', type=float, default=5e-5, help='Learning rate for discriminator')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='L2 regularization')
        parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
        parser.add_argument('--smoothing_factor', type=float, default=0.1, help='Label Smoothness')
        parser.add_argument('--max_norm', type=float, default=10, help='Degree of gradient clipping')
        parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
        parser.add_argument('--patience', type=int, default=20, help='Number of epochs to wait for improvement before early stopping')
        parser.add_argument('--save_path', type=str, default='./task4_pth_1_1000_t1', help='Path to save trained models')
        parser.add_argument('--checkpoint_path', type=str, default='./task4_pth_1_1000_t1/best_SSM_gan.pth', help='Path to load checkpoint models')
        parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
        return parser.parse_args()
    
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Print hyperparameters
    print("Training hyperparameters:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
       
    # Save hyperparameters to file
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, 'hyperparameters.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
    
    torch.cuda.empty_cache()
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define input modalities for different tasks
    desired_inputs_task1 = ['t1.npy', 't1ce.npy', 't2.npy']  # t1, t1ce, t2 -> flair [0001]
    desired_inputs_task2 = ['t1.npy', 't1ce.npy', 'flair.npy']  # t1, t1ce, flair -> t2 [0010]
    desired_inputs_task3 = ['t1.npy', 't2.npy', 'flair.npy']  # t1, t2, flair -> t1ce [0100]
    desired_inputs_task4 = ['t1ce.npy', 't2.npy', 'flair.npy']  # t1ce, t2, flair -> t1 [1000]

    # Load dataset
    train_dataset_task4 = BraTSLoader(args.train_path, input_modalities=desired_inputs_task4)
    train_loader_task4 = DataLoader(train_dataset_task4, batch_size=args.batch_size, shuffle=True, num_workers=args.num_work, pin_memory=True)
    
    # Initialize models
    generator_task4 = Generator().to(device)
    discriminator_task4 = Discriminator().to(device)
    
    if args.train_model == 'gan':
        start_time_gan = time.time()
        
        # Initialize optimizers
        g_optimizer_task4 = optim.Adam(generator_task4.parameters(), lr=args.g_lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
        d_optimizer_task4 = optim.Adam(discriminator_task4.parameters(), lr=args.d_lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
        
        # Train GAN
        train_gan(generator_task4, 
                  discriminator_task4, 
                  train_loader_task4, 
                  g_optimizer_task4, 
                  d_optimizer_task4, 
                  device,
                  num_epochs=args.num_epochs,
                  args=args,
                  checkpoint_path=args.checkpoint_path)
        
        # Calculate and print training time
        end_time_gan = time.time()
        total_time_gan = end_time_gan - start_time_gan
        hours_gan = int(total_time_gan // 3600)
        minutes_gan = int((total_time_gan % 3600) // 60)
        seconds_gan = total_time_gan % 60
        print(f"Total training time for Mamba GAN: {hours_gan} hours, {minutes_gan} minutes, {seconds_gan:.2f} seconds")
