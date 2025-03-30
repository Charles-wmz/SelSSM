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

# 数据集路径
data_path = r"D:\Project\dataset\BraTS_2019_entire"

# 输出路径（预处理数据的输出路径）
output_path = r"D:\Project\dataset\BraTS_2019_entire_DP155"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# N4 偏置场校正
def n4_bias_correction(input_image):
    sitk_image = sitk.ReadImage(input_image)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)  # 将图像转换为 32 位浮点类型
    mask_image = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(sitk_image, mask_image)
    return sitk.GetArrayFromImage(corrected_image)

# Z-score 归一化
def z_score_normalization(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

# 强度缩放到 [-1, 1]
def intensity_scaling(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return 2 * (data - data_min) / (data_max - data_min) - 1

# 处理每个文件夹中的数据
for folder_name in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder_name)
    if os.path.isdir(folder_path):
        output_folder = os.path.join(output_path, folder_name)
        # 如果输出目录存在且所有模态文件都已经保存，则跳过该文件夹
        if os.path.exists(output_folder) and all(os.path.exists(os.path.join(output_folder, f"{modality}.npy")) for modality in ["t1", "t1ce", "t2", "flair"]):
            print(f"Skipping {folder_name} as it is already processed.")
            continue

        print(f"Processing {folder_name}...")

        # 加载模态影像 (T1, T1ce, T2, FLAIR)
        modalities = ["t1", "t1ce", "t2", "flair"]
        images = {}
        for modality in modalities:
            image_path = os.path.join(folder_path, f"{folder_name}_{modality}.nii")
            # N4 偏置校正
            n4_corrected = n4_bias_correction(image_path)
            # Z-score 标准化
            normalized = z_score_normalization(n4_corrected)
            # 强度缩放到 [-1, 1]
            scaled = intensity_scaling(normalized)
            images[modality] = scaled

        # 保存处理后的数据为 .npy 格式
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for modality, image_data in images.items():
            np.save(os.path.join(output_folder, f"{modality}.npy"), image_data)

        # 加载标签
        seg_path = os.path.join(folder_path, f"{folder_name}_seg.nii")
        seg_image = nib.load(seg_path).get_fdata()

        # 保留原始标签（0：背景，1：Non-Enhancing Tumor，2：Peritumoral Edema，4：Enhancing Tumor）
        combined_label = seg_image.astype(np.uint8)

        # 保存标签为 .npy 格式
        np.save(os.path.join(output_folder, "segmentation_labels.npy"), combined_label)

print("Preprocessing complete.")

# 使用预处理后的输出路径进行数据集划分
split_dataset(output_path)
print("Splitting complete.")
