# -*- coding: utf-8 -*-
"""
@author: D-03

"""
import os
import numpy as np

# Original data path and output data path
input_data_path = '../dataset/BraTS_2020_entire_DP155'
output_data_path = '../dataset/BraTS_2020_entire_DP155_80'

# Create output directory (if it doesn't exist)
os.makedirs(output_data_path, exist_ok=True)

# Index range for the middle 80 slices
start_slice = 37
end_slice = 117

# Traverse train, val and test folders
for subset in ['train', 'val', 'test']:
    print(f"Processing {subset} folder")
    subset_input_path = os.path.join(input_data_path, subset)
    subset_output_path = os.path.join(output_data_path, subset)
    os.makedirs(subset_output_path, exist_ok=True)

    for subject in os.listdir(subset_input_path):
        subject_input_path = os.path.join(subset_input_path, subject)
        subject_output_path = os.path.join(subset_output_path, subject)
        os.makedirs(subject_output_path, exist_ok=True)
        
        # Process segmentation label files
        seg_path = os.path.join(subject_input_path, 'segmentation_labels.npy')
        if os.path.exists(seg_path):
            seg_data = np.load(seg_path)
            seg_data = seg_data.transpose(2, 0, 1)  # Convert to [155, 240, 240]
            seg_data = seg_data[start_slice:end_slice]  # Get the middle 80 slices
            np.save(os.path.join(subject_output_path, 'segmentation_labels.npy'), seg_data)  # Save to new path
        
        # Process modality files
        modalities = ['t1.npy', 't1ce.npy', 't2.npy', 'flair.npy']
        for modality in modalities:
            modality_path = os.path.join(subject_input_path, modality)
            if os.path.exists(modality_path):
                modality_data = np.load(modality_path)
                modality_data = modality_data[start_slice:end_slice]  # Get the middle 80 slices
                np.save(os.path.join(subject_output_path, modality), modality_data)  # Save to new path

print("Processing completed, the middle 80 slices of each modality and segmentation label file have been saved to the new path.")


