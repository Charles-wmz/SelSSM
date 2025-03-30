# -*- coding: utf-8 -*-
"""
Spyder 编辑器

这是一个临时脚本文件。
"""
import os
import numpy as np

# 原始数据路径和输出数据路径
input_data_path = r'D:\Project\dataset\BraTS_2019_entire_DP155'
output_data_path = r'D:\Project\dataset\BraTS_2019_entire_DP155_80'

# 创建输出目录（如果不存在）
os.makedirs(output_data_path, exist_ok=True)

# 中间 80 个切片的索引范围
start_slice = 37
end_slice = 117

# 遍历 train、val 和 test 文件夹
for subset in ['train', 'val', 'test']:
    print(f"正在处理{subset}文件夹")
    subset_input_path = os.path.join(input_data_path, subset)
    subset_output_path = os.path.join(output_data_path, subset)
    os.makedirs(subset_output_path, exist_ok=True)

    for subject in os.listdir(subset_input_path):
        subject_input_path = os.path.join(subset_input_path, subject)
        subject_output_path = os.path.join(subset_output_path, subject)
        os.makedirs(subject_output_path, exist_ok=True)
        
        # 处理分割标签文件
        seg_path = os.path.join(subject_input_path, 'segmentation_labels.npy')
        if os.path.exists(seg_path):
            seg_data = np.load(seg_path)
            seg_data = seg_data.transpose(2, 0, 1)  # 变为 [155, 240, 240]
            seg_data = seg_data[start_slice:end_slice]  # 获取中间 80 个切片
            np.save(os.path.join(subject_output_path, 'segmentation_labels.npy'), seg_data)  # 保存到新路径
        
        # 处理模态文件
        modalities = ['t1.npy', 't1ce.npy', 't2.npy', 'flair.npy']
        for modality in modalities:
            modality_path = os.path.join(subject_input_path, modality)
            if os.path.exists(modality_path):
                modality_data = np.load(modality_path)
                modality_data = modality_data[start_slice:end_slice]  # 获取中间 80 个切片
                np.save(os.path.join(subject_output_path, modality), modality_data)  # 保存到新路径

print("处理完成，已将每个模态和分割标签文件的中间 80 个切片保存到新路径。")


