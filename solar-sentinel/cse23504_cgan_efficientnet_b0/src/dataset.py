"""
SolarSentinel AI — PyTorch Dataset Implementations
Provides dataset loaders for:
1. PVELADDataset: Real multi-label EL image dataset for EfficientNet-B0.
2. GANDataset: Class-conditioned dataset for cDCGAN training.
3. AugmentedPVELADDataset: Combined real + quality-filtered GAN synthetic images for Experiment B.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import config
from src.preprocessing import (
    create_reproducible_splits,
    get_classifier_transforms,
    get_gan_transforms
)

def get_primary_class_label(row):
    """
    Determines single primary class index for conditional GAN training.
    Prioritizes minority defects over common defects and background.
    """
    priority_order = [
        'star_crack',
        'other_defect',
        'thick_line',
        'crack',
        'finger',
        'black_core',
        'material',
        'good'
    ]
    for cname in priority_order:
        if row.get(cname, 0) == 1:
            return config.CLASS_TO_IDX[cname]
    return config.CLASS_TO_IDX['good']

class PVELADDataset(Dataset):
    """
    Multi-label PyTorch Dataset for real Solar Panel EL images.
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_paths = self.df['image_path'].values
        self.labels = self.df[config.CLASS_NAMES].values.astype(np.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Handle Windows/Unix path formatting
        if not os.path.exists(img_path):
            norm_path = os.path.normpath(img_path)
            if os.path.exists(norm_path):
                img_path = norm_path

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback to neutral black image if corrupted file
            image = Image.new('RGB', (config.CLASSIFIER_IMG_SIZE, config.CLASSIFIER_IMG_SIZE))

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

class GANDataset(Dataset):
    """
    Dataset specifically tailored for Conditional DCGAN (cDCGAN) training.
    Returns (normalized_image_tensor, class_condition_idx).
    """
    def __init__(self, df, transform=None, focus_defects_only=False):
        self.df = df.reset_index(drop=True)
        
        # Optionally filter to focus more heavily on defective solar cells
        if focus_defects_only:
            is_defect = self.df[config.DEFECT_CLASSES].sum(axis=1) > 0
            self.df = self.df[is_defect].reset_index(drop=True)

        self.transform = transform or get_gan_transforms(config.GAN_IMG_SIZE)
        self.image_paths = self.df['image_path'].values
        
        # Compute condition label for each sample
        self.conditions = np.array([get_primary_class_label(row) for _, row in self.df.iterrows()], dtype=np.int64)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if not os.path.exists(img_path):
            img_path = os.path.normpath(img_path)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (config.GAN_IMG_SIZE, config.GAN_IMG_SIZE))

        if self.transform is not None:
            image = self.transform(image)

        condition = torch.tensor(self.conditions[idx], dtype=torch.long)
        return image, condition

class AugmentedPVELADDataset(Dataset):
    """
    Experiment B Dataset: Combines real training split images with quality-filtered
    GAN synthetic images. Tracks provenance to prevent synthetic images from ever leaking into test sets.
    """
    def __init__(self, real_train_df, synthetic_manifest_csv=None, transform=None):
        self.transform = transform or get_classifier_transforms(is_train=True)
        self.samples = []

        # 1. Add real training images
        for _, row in real_train_df.iterrows():
            lbl = row[config.CLASS_NAMES].values.astype(np.float32)
            self.samples.append({
                'path': row['image_path'],
                'label': lbl,
                'is_synthetic': False
            })
        self.num_real = len(self.samples)

        # 2. Add quality-checked synthetic images if manifest exists
        self.num_synthetic = 0
        if synthetic_manifest_csv and os.path.exists(synthetic_manifest_csv):
            syn_df = pd.read_csv(synthetic_manifest_csv)
            for _, row in syn_df.iterrows():
                # Synthetic sample has one-hot label corresponding to its generated class condition
                lbl = np.zeros(config.NUM_CLASSES, dtype=np.float32)
                cls_idx = int(row['class_idx'])
                lbl[cls_idx] = 1.0
                
                self.samples.append({
                    'path': row['image_path'],
                    'label': lbl,
                    'is_synthetic': True
                })
                self.num_synthetic += 1

        print(f"AugmentedPVELADDataset initialized: {self.num_real:,} real + {self.num_synthetic:,} synthetic = {len(self.samples):,} total")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['path']
        if not os.path.exists(img_path):
            img_path = os.path.normpath(img_path)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (config.CLASSIFIER_IMG_SIZE, config.CLASSIFIER_IMG_SIZE))

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(sample['label'], dtype=torch.float32)
        return image, label

def load_split_datasets(splits_csv=config.SPLITS_INDEX_CSV):
    """
    Convenience function to load train, val, test datasets using existing or generated splits.
    """
    df_splits = create_reproducible_splits(splits_csv=splits_csv)
    
    train_df = df_splits[df_splits['split'] == 'train'].reset_index(drop=True)
    val_df = df_splits[df_splits['split'] == 'val'].reset_index(drop=True)
    test_df = df_splits[df_splits['split'] == 'test'].reset_index(drop=True)

    train_ds = PVELADDataset(train_df, transform=get_classifier_transforms(is_train=True))
    val_ds = PVELADDataset(val_df, transform=get_classifier_transforms(is_train=False))
    test_ds = PVELADDataset(test_df, transform=get_classifier_transforms(is_train=False))

    return train_ds, val_ds, test_ds, train_df, val_df, test_df
