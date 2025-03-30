# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 17:49:00 2024

@author: D-03
"""

import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from Spliting_dataset import split_dataset

# Dataset path
data_path = "../dataset/BraTS_2020_entire"

# Output path (for preprocessed data)
output_path = "../dataset/BraTS_2020_entire_DP155"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# N4 bias field correction
def n4_bias_correction(input_image):
    sitk_image = sitk.ReadImage(input_image)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)  # Convert image to 32-bit float type
    mask_image = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(sitk_image, mask_image)
    return sitk.GetArrayFromImage(corrected_image)

# Z-score normalization
def z_score_normalization(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

# Intensity scaling to [-1, 1]
def intensity_scaling(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return 2 * (data - data_min) / (data_max - data_min) - 1

# Process data in each folder
for folder_name in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder_name)
    if os.path.isdir(folder_path):
        output_folder = os.path.join(output_path, folder_name)
        # Skip this folder if output directory exists and all modality files are already saved
        if os.path.exists(output_folder) and all(os.path.exists(os.path.join(output_folder, f"{modality}.npy")) for modality in ["t1", "t1ce", "t2", "flair"]):
            print(f"Skipping {folder_name} as it is already processed.")
            continue

        print(f"Processing {folder_name}...")

        # Load modality images (T1, T1ce, T2, FLAIR)
        modalities = ["t1", "t1ce", "t2", "flair"]
        images = {}
        for modality in modalities:
            image_path = os.path.join(folder_path, f"{folder_name}_{modality}.nii")
            # N4 bias correction
            n4_corrected = n4_bias_correction(image_path)
            # Z-score normalization
            normalized = z_score_normalization(n4_corrected)
            # Intensity scaling to [-1, 1]
            scaled = intensity_scaling(normalized)
            images[modality] = scaled

        # Save processed data in .npy format
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for modality, image_data in images.items():
            np.save(os.path.join(output_folder, f"{modality}.npy"), image_data)

        # Load labels
        seg_path = os.path.join(folder_path, f"{folder_name}_seg.nii")
        seg_image = nib.load(seg_path).get_fdata()

        # Keep original labels (0: Background, 1: Non-Enhancing Tumor, 2: Peritumoral Edema, 4: Enhancing Tumor)
        combined_label = seg_image.astype(np.uint8)

        # Save labels in .npy format
        np.save(os.path.join(output_folder, "segmentation_labels.npy"), combined_label)

print("Preprocessing complete.")

# Use the preprocessed output path for dataset splitting
split_dataset(output_path)
print("Splitting complete.")
