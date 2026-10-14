"""
SolarSentinel AI — EfficientNet-B0 Training Pipeline
Trains either:
- Experiment A (Baseline): Real training split images only.
- Experiment B (GAN-Augmented): Real images + quality-filtered synthetic images from cDCGAN.
Evaluates both on the identical real validation split.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

import config
from src.preprocessing import create_reproducible_splits, get_classifier_transforms
from src.dataset import PVELADDataset, AugmentedPVELADDataset
from src.classifier import create_classifier

def plot_training_curves(history, mode_name, save_path):
    """
    Plots training and validation loss and F1-score across epochs.
    """
    df = pd.DataFrame(history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='#2563eb', lw=2)
    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#dc2626', lw=2)
    ax1.set_title(f'{mode_name} — Loss Curves', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCEWithLogits Loss')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # F1 curves
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='#2563eb', lw=2)
    ax2.plot(df['epoch'], df['train_f1'], label='Train Macro F1', color='#059669', lw=2)
    ax2.plot(df['epoch'], df['val_f1'], label='Val Macro F1', color='#7c3aed', lw=2)
    ax2.set_title(f'{mode_name} — Macro F1-Score', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved training curves to {save_path}")

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Train]", leave=True)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val * images.size(0)
        probs = torch.sigmoid(logits)

        all_targets.append(targets.cpu().numpy())
        all_preds.append(probs.detach().cpu().numpy())
        pbar.set_postfix({'loss': f"{loss_val:.4f}"})

    epoch_loss = running_loss / len(dataloader.dataset)
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    binary_preds = (all_preds > config.DECISION_THRESHOLD).astype(np.float32)

    f1 = f1_score(all_targets, binary_preds, average='macro', zero_division=0)
    prec = precision_score(all_targets, binary_preds, average='macro', zero_division=0)
    rec = recall_score(all_targets, binary_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_targets, binary_preds)

    return epoch_loss, acc, f1, prec, rec

def validate_epoch(model, dataloader, criterion, device, epoch, total_epochs):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Val]  ", leave=False)
    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    binary_preds = (all_preds > config.DECISION_THRESHOLD).astype(np.float32)

    f1 = f1_score(all_targets, binary_preds, average='macro', zero_division=0)
    prec = precision_score(all_targets, binary_preds, average='macro', zero_division=0)
    rec = recall_score(all_targets, binary_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_targets, binary_preds)

    return epoch_loss, acc, f1, prec, rec

def train_classifier(
    mode='baseline',
    epochs=config.CLASSIFIER_EPOCHS,
    batch_size=config.CLASSIFIER_BATCH_SIZE,
    lr=config.CLASSIFIER_LR,
    smoke_test=False,
    subset_size=None,
    save_path=None,
    init_from=None,
    device=config.DEVICE
):
    """
    Main training execution function for Experiment A (Baseline) or Experiment B (Augmented).
    """
    mode_title = "EXPERIMENT A — BASELINE" if mode == 'baseline' else "EXPERIMENT B — GAN-AUGMENTED"
    default_save = config.BASELINE_CKPT if mode == 'baseline' else config.AUGMENTED_CKPT
    save_path = save_path or default_save

    print("=" * 70)
    print(f"SOLARSENTINEL AI — {mode_title}")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    if smoke_test:
        print(">>> RUNNING IN SMOKE-TEST MODE <<<")
    print("=" * 70)

    # 1. Load Splits
    splits_df = create_reproducible_splits()
    train_df = splits_df[splits_df['split'] == 'train'].reset_index(drop=True)
    val_df = splits_df[splits_df['split'] == 'val'].reset_index(drop=True)

    if smoke_test:
        train_df = train_df.sample(n=min(128, len(train_df)), random_state=config.RANDOM_SEED).reset_index(drop=True)
        val_df = val_df.sample(n=min(64, len(val_df)), random_state=config.RANDOM_SEED).reset_index(drop=True)
        epochs = min(epochs, 2)
    elif subset_size and subset_size < len(train_df):
        train_df = train_df.sample(n=subset_size, random_state=config.RANDOM_SEED).reset_index(drop=True)

    # 2. Setup Dataset & Loader
    val_dataset = PVELADDataset(val_df, transform=get_classifier_transforms(is_train=False))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS)

    if mode == 'augmented':
        manifest_path = os.path.join(config.GENERATED_SAMPLES_DIR, "synthetic_manifest.csv")
        train_dataset = AugmentedPVELADDataset(
            real_train_df=train_df,
            synthetic_manifest_csv=manifest_path,
            transform=get_classifier_transforms(is_train=True)
        )
    else:
        train_dataset = PVELADDataset(train_df, transform=get_classifier_transforms(is_train=True))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS)
    print(f"Training dataset size: {len(train_dataset):,} | Validation dataset size: {len(val_dataset):,}")

    # 3. Model, Pos-Weight Loss, Optimizer
    model = create_classifier(num_classes=config.NUM_CLASSES, pretrained=True, device=device)
    if init_from and os.path.exists(init_from):
        print(f"Initializing model weights from: {init_from}")
        init_ckpt = torch.load(init_from, map_location=device)
        sd = init_ckpt['model_state_dict'] if 'model_state_dict' in init_ckpt else init_ckpt
        model.load_state_dict(sd)

    # Calculate pos_weight from real training split
    labels_matrix = train_df[config.CLASS_NAMES].values
    pos_counts = labels_matrix.sum(axis=0)
    neg_counts = len(train_df) - pos_counts
    pos_weight_vals = np.clip(neg_counts / np.maximum(pos_counts, 1), 1.0, 20.0)
    pos_weight_tensor = torch.tensor(pos_weight_vals, dtype=torch.float32, device=device)
    print(f"Class pos_weight: {dict(zip(config.CLASS_NAMES, [round(w, 2) for w in pos_weight_vals]))}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.CLASSIFIER_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = 0.0
    best_val_loss = float('inf')
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()

        train_loss, train_acc, train_f1, train_p, train_r = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs
        )
        val_loss, val_acc, val_f1, val_p, val_r = validate_epoch(
            model, val_loader, criterion, device, epoch, epochs
        )

        scheduler.step()
        ep_time = time.time() - ep_start

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f}")

        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 4), 'train_acc': round(train_acc, 4), 'train_f1': round(train_f1, 4),
            'val_loss': round(val_loss, 4), 'val_acc': round(val_acc, 4), 'val_f1': round(val_f1, 4),
            'duration_sec': round(ep_time, 1)
        })

        if val_f1 > best_val_f1 or val_loss < best_val_loss:
            best_val_f1 = max(best_val_f1, val_f1)
            best_val_loss = min(best_val_loss, val_loss)
            checkpoint = {
                'epoch': epoch,
                'mode': mode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_f1': best_val_f1,
                'class_names': config.CLASS_NAMES
            }
            torch.save(checkpoint, save_path)
            print(f"  >>> Checkpoint saved to {save_path}! (Best Val F1: {best_val_f1:.4f})")

    total_time = time.time() - start_time
    print(f"\n{mode_title} Complete in {total_time:.1f}s. Final Best Val F1: {best_val_f1:.4f}")

    # Save metrics and plot
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(config.METRICS_DIR, f"{mode}_training_history.csv")
    history_df.to_csv(history_csv, index=False)

    curve_path = os.path.join(config.PLOTS_DIR, f"{mode}_training_curves.png")
    plot_training_curves(history, mode_title, curve_path)

    return model, history_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SolarSentinel EfficientNet-B0")
    parser.add_argument('--mode', choices=['baseline', 'augmented'], default='baseline', help="Experiment mode")
    parser.add_argument('--epochs', type=int, default=config.CLASSIFIER_EPOCHS, help="Epoch count")
    parser.add_argument('--batch-size', type=int, default=config.CLASSIFIER_BATCH_SIZE, help="Batch size")
    parser.add_argument('--lr', type=float, default=config.CLASSIFIER_LR, help="Learning rate")
    parser.add_argument('--smoke-test', action='store_true', help="Fast smoke test on small subset")
    parser.add_argument('--subset-size', type=int, default=None, help="Custom subset size")
    parser.add_argument('--init-from', type=str, default=None, help="Initialize weights from existing checkpoint")
    args = parser.parse_args()

    train_classifier(
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        smoke_test=args.smoke_test,
        subset_size=args.subset_size,
        init_from=args.init_from
    )
