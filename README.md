# SelSSM: Medical Image Synthesis with Selective State Space Model

## Project Introduction
SelSSM is a medical image synthesis framework based on Selective State Space Model (SelSSM), specifically designed for multi-modal medical image synthesis tasks, particularly for the BraTS dataset. This project implements a combination of Pix2Pix architecture with Selective State Space Model for high-quality medical image generation.

## Dataset
Before training, you need to get the data on BraTS2020 and BraTS2019 first. Please change the path in the code to your path.

### BraTS2020 Dataset
- Download from: [BraTS2020 Dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
- Contains training and validation data
- Multi-modal MRI scans (T1, T1ce, T2, FLAIR)

### BraTS2019 Dataset
- Download from: [BraTS2019 Dataset](https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019)
- Contains training and validation data
- Multi-modal MRI scans (T1, T1ce, T2, FLAIR)

### Dataset Structure
The preprocessed dataset structure used in this project is organized as follows:
```
BraTS_2019_entire_DP155_80_SMflair_256/
├── train/                         # Training set
│   ├── BraTS19_XXXX_XX_X/         # Patient cases
│   │   ├── t1.npy                 # T1 modality data
│   │   ├── t1ce.npy               # T1ce modality data
│   │   ├── t2.npy                 # T2 modality data
│   │   ├── segmentation_labels.npy # Segmentation masks
│   │   ├── missing_modality.txt   # Information about missing modalities
│   │   └── original_modalities/   # Original modality data
│   │       └── original_flair.npy # Original FLAIR modality
│   └── ...
├── val/                           # Validation set
│   └── ...
└── test/                          # Test set
    └── ...
```

The dataset has been preprocessed into NumPy arrays (.npy files) with a standard size of 256×256 for easier loading and processing.

## Key Features
- Plug-and-play Selective State Space Model (SSM) for feature extraction and generation
- U-Net based generator architecture with SSM integration
- Enhanced edge and detail preservation in generated images
- Support for gradient clipping and learning rate scheduling
- Multi-modal medical image synthesis capability

## Requirement
- Python 3.9
- PyTorch 2.2.0
- CUDA 12.2
- Numpy 1.26.4
- Scikit-learn 1.5.2
- SimpleITK 2.4.0

## Project Structure
```
SelSSM/
├── models/                 # Model definitions
├── utils/                  # Utility functions
├── Pix2Pix_SelSSM.py      # Main training script
├── test_SelSSMGAN.py      # Testing script
└── visualization_SelSSMGAN.py  # Visualization tools
```

## Usage
1. Train the model:
```bash
python Pix2Pix_SelSSM.py --data_path /path/to/data --save_path /path/to/save
```

2. Test the model:
```bash
python test_SelSSMGAN.py --model_path /path/to/model --test_data /path/to/test_data
```

3. Visualize results:
```bash
python visualization_SelSSMGAN.py --results_path /path/to/results
```

## Key Parameters
- `--data_path`: Path to training data
- `--save_path`: Path to save model checkpoints
- `--g_lr`: Generator learning rate
- `--d_lr`: Discriminator learning rate
- `--num_epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--max_norm`: Gradient clipping threshold

## Experimental Results
Model performance on BraTS dataset:
- High-quality image generation
- Superior edge and detail preservation
- Accurate multi-modal conversion

## Citation
If you use this project in your research, please cite:
```bibtex
@misc{selssm2024,
  author = {Mingzhi Wang},
  title = {SelSSM: Selective State Space Module Empowered Generative Adversarial Networks for Multi-modal MR Image Synthesis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Charles-wmz/SelSSM}
}
```

## License
MIT License

## Author
- Author: Cai Yize
- Email: [your-email@example.com]
- Institution: [your-institution]

## Acknowledgments
Thanks to all contributors who provided help and suggestions for this project. 