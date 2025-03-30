"""
Data loading utilities for the BraTS dataset.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class BraTSLoader(Dataset):
    """
    Dataset loader for BraTS medical imaging data.
    
    Args:
        folder_path (str): Path to the folder containing subject subfolders
        input_modalities (list of str, optional): List of modality filenames to include as input.
            If None, all available modalities except the missing one will be used.
            Example: ['t1.npy', 'flair.npy']
    """
    def __init__(self, folder_path, input_modalities=None):
        self.folder_path = folder_path
        self.slice_paths = []
        
        # Define all possible modalities
        self.all_modalities = ['t1.npy', 't1ce.npy', 't2.npy', 'flair.npy']
        
        # If input_modalities is not provided, use all except one
        if input_modalities is None:
            self.input_modalities = self.all_modalities.copy()
        else:
            # Validate input_modalities
            for modality in input_modalities:
                if modality not in self.all_modalities:
                    raise ValueError(f"Invalid modality '{modality}'. Valid options are: {self.all_modalities}")
            self.input_modalities = input_modalities.copy()
        
        # Traverse all subjects
        for subject in os.listdir(folder_path):
            subject_path = os.path.join(folder_path, subject)
            
            if not os.path.isdir(subject_path):
                continue
            
            # Determine the number of slices based on available modalities
            num_slices = None
            for modality in self.all_modalities:
                modality_path = os.path.join(subject_path, modality)
                if os.path.exists(modality_path):
                    num_slices = len(np.load(modality_path))
                    break
            
            if num_slices is None:
                print(f"No valid modalities found for subject {subject}. Skipping.")
                continue
            
            # Add slice paths
            for slice_idx in range(num_slices):
                self.slice_paths.append((subject_path, slice_idx))
    
    def __len__(self):
        return len(self.slice_paths)
    
    def __getitem__(self, idx):
        subject_path, slice_idx = self.slice_paths[idx]
    
        # Determine the missing modality
        missing_modality_file = os.path.join(subject_path, "missing_modality.txt")
        if os.path.exists(missing_modality_file):
            with open(missing_modality_file, "r") as f:
                missing_modality = f.readline().strip()
        else:
            raise FileNotFoundError(f"Missing modality file not found for {subject_path}")
    
        # Ensure the missing modality is not in the input modalities
        if missing_modality in self.input_modalities:
            raise ValueError(f"Missing modality '{missing_modality}' cannot be part of input modalities.")
    
        # Load specified input modalities
        input_modalities = []
        for modality in self.input_modalities:
            modality_path = os.path.join(subject_path, modality)
            if modality == missing_modality:
                continue
            if os.path.exists(modality_path):
                slice_data = np.load(modality_path)[slice_idx]  # [height, width]
                input_modalities.append(slice_data)
            else:
                raise FileNotFoundError(f"Expected modality '{modality}' not found at '{modality_path}'")
    
        if len(input_modalities) == 0:
            raise ValueError(f"No valid input modalities found for slice {slice_idx} in {subject_path}.")
                
        # Determine target modality index
        target_modality = self.all_modalities.index(missing_modality)
    
        # Stack input modalities to create multi-channel input tensor
        input_tensor = torch.tensor(np.stack(input_modalities, axis=0), dtype=torch.float)
    
        # Load the missing modality's real data as the target
        original_modalities_folder = os.path.join(subject_path, "original_modalities")
        missing_modality_path = os.path.join(original_modalities_folder, f"original_{missing_modality}")
        if os.path.exists(missing_modality_path):
            missing_data = np.load(missing_modality_path)[slice_idx]  # [height, width]
            target_tensor = torch.tensor(missing_data, dtype=torch.float)  # [height, width]
            target_tensor = target_tensor.unsqueeze(0)  # [1, height, width]
        else:
            raise FileNotFoundError(f"Real missing modality not found at '{missing_modality_path}'")
        
        return input_tensor, target_tensor, subject_path, target_modality 