"""
SolarSentinel AI — Preprocessing and Transformation Pipelines
Handles dataset splitting, reproducible transforms for EfficientNet-B0 and cDCGAN,
and image denormalization for visualization.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

import config

def create_reproducible_splits(
    index_csv=config.DATASET_INDEX_CSV,
    splits_csv=config.SPLITS_INDEX_CSV,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=config.RANDOM_SEED
):
    """
    Creates a deterministic 70:15:15 split across the 13,414 PVELAD dataset images.
    Guarantees zero overlap between train, val, and test partitions.
    """
    if os.path.exists(splits_csv):
        df_splits = pd.read_csv(splits_csv)
        print(f"Loaded existing splits from {splits_csv}:")
        print(df_splits['split'].value_counts())
        return df_splits

    if not os.path.exists(index_csv):
        raise FileNotFoundError(f"Source index CSV not found: {index_csv}")

    df = pd.read_csv(index_csv)
    total_samples = len(df)
    print(f"Creating reproducible splits from {total_samples:,} images...")

    # First split off test set (15%)
    val_plus_test_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=val_plus_test_ratio,
        random_state=random_seed,
        shuffle=True
    )

    # Split remaining 30% equally into val (15%) and test (15%)
    relative_test_ratio = test_ratio / val_plus_test_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=random_seed,
        shuffle=True
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df.to_csv(splits_csv, index=False)

    print(f"Splits saved to {splits_csv}:")
    print(f"  Train: {len(train_df):,} ({len(train_df)/total_samples:.1%})")
    print(f"  Val:   {len(val_df):,} ({len(val_df)/total_samples:.1%})")
    print(f"  Test:  {len(test_df):,} ({len(test_df)/total_samples:.1%})")

    return combined_df

def get_classifier_transforms(is_train=True, img_size=config.CLASSIFIER_IMG_SIZE):
    """
    Returns image transformation pipelines for EfficientNet-B0.
    Standard ImageNet normalization is used for transfer learning.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])

def get_gan_transforms(img_size=config.GAN_IMG_SIZE):
    """
    Returns transformation pipeline for cDCGAN training.
    Normalizes images to [-1, 1] range to match Generator Tanh output.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.GAN_MEAN, std=config.GAN_STD)
    ])

def denormalize_gan_tensor(tensor):
    """
    Inverts GAN [-1, 1] normalization back to [0, 1] RGB tensor.
    """
    img = tensor.clone().detach().cpu()
    img = img * 0.5 + 0.5
    img = torch.clamp(img, 0.0, 1.0)
    return img

def denormalize_classifier_tensor(tensor):
    """
    Inverts ImageNet normalization back to [0, 1] RGB tensor.
    """
    img = tensor.clone().detach().cpu()
    for c, (m, s) in enumerate(zip(config.IMAGENET_MEAN, config.IMAGENET_STD)):
        img[c] = img[c] * s + m
    img = torch.clamp(img, 0.0, 1.0)
    return img

def validate_synthetic_tensor(img_tensor, min_var=0.005):
    """
    Checks if a generated synthetic image tensor is valid:
    - Finite (no NaNs or Infs)
    - Has sufficient pixel variance (rejects collapsed solid-color images)
    """
    if not torch.isfinite(img_tensor).all():
        return False, "Non-finite values detected (NaN or Inf)"
    
    var = torch.var(img_tensor).item()
    if var < min_var:
        return False, f"Mode collapse detected: pixel variance too low ({var:.6f} < {min_var})"
        
    return True, "Valid synthetic image"
