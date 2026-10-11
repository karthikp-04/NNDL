import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import PVELADDataset, build_dataset_index, get_transforms, CLASS_NAMES, NUM_CLASSES
from model import create_model

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
    binary_preds = (all_preds > 0.5).astype(np.float32)
    
    f1_macro = f1_score(all_targets, binary_preds, average='macro', zero_division=0)
    precision_macro = precision_score(all_targets, binary_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_targets, binary_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_targets, binary_preds)
    
    return epoch_loss, acc, f1_macro, precision_macro, recall_macro

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
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    epoch_loss = running_loss / len(dataloader.dataset)
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    binary_preds = (all_preds > 0.5).astype(np.float32)
    
    f1_macro = f1_score(all_targets, binary_preds, average='macro', zero_division=0)
    precision_macro = precision_score(all_targets, binary_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_targets, binary_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_targets, binary_preds)
    
    return epoch_loss, acc, f1_macro, precision_macro, recall_macro, all_targets, all_preds

def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on PVELAD dataset")
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--img-size', type=int, default=224, help='Image resolution')
    parser.add_argument('--subset-size', type=int, default=None, help='Optionally train on a subset (e.g., 3000)')
    parser.add_argument('--quick-run', action='store_true', help='Subsample 1,000 images for fast test')
    parser.add_argument('--save-path', type=str, default=None, help='Path to save best checkpoint')
    parser.add_argument('--use-pos-weight', action='store_true', help='Calculate and apply pos_weight in BCEWithLogitsLoss for class imbalance')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers')
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = 'quick_run_model.pth' if args.quick_run else 'best_efficientnet_b0.pth'

    # Set PyTorch thread optimization for CPU
    threads = max(1, os.cpu_count() or 4)
    torch.set_num_threads(threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} ({threads} CPU threads)")
    print(f"Checkpoint output path: {args.save_path}")

    # 1. Load Dataset Index
    csv_path = 'dataset_index.csv'
    if not os.path.exists(csv_path):
        df = build_dataset_index(csv_path=csv_path)
    else:
        df = pd.read_csv(csv_path)

    if args.quick_run:
        print("Quick Run Mode: Subsampling 1,000 images...")
        df = df.sample(n=min(1000, len(df)), random_state=42).reset_index(drop=True)
    elif args.subset_size is not None and args.subset_size < len(df):
        print(f"Subset Training Mode: Subsampling {args.subset_size:,} images...")
        df = df.sample(n=args.subset_size, random_state=42).reset_index(drop=True)

    print(f"Total images for training/validation: {len(df):,}")

    # 2. Train/Validation Split (80/20)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train split: {len(train_df):,} images | Val split: {len(val_df):,} images")

    train_dataset = PVELADDataset(train_df, transform=get_transforms(img_size=args.img_size, is_train=True))
    val_dataset = PVELADDataset(val_df, transform=get_transforms(img_size=args.img_size, is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 3. Initialize Model, Loss, Optimizer, Scheduler
    model = create_model(num_classes=NUM_CLASSES, pretrained=True).to(device)
    
    if args.use_pos_weight:
        # Calculate positive weights per class: (N_neg / N_pos)
        labels_matrix = train_df[CLASS_NAMES].values
        pos_counts = labels_matrix.sum(axis=0)
        neg_counts = len(train_df) - pos_counts
        pos_weight_vals = np.clip(neg_counts / np.maximum(pos_counts, 1), 1.0, 20.0)
        pos_weight_tensor = torch.tensor(pos_weight_vals, dtype=torch.float32).to(device)
        print(f"Using class pos_weight: {dict(zip(CLASS_NAMES, [round(w, 2) for w in pos_weight_vals]))}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    best_f1 = 0.0
    history = []

    print("\nStarting Training Pipeline...")
    print("=" * 70)
    
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        
        train_loss, train_acc, train_f1, train_p, train_r = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        val_loss, val_acc, val_f1, val_p, val_r, _, _ = validate_epoch(
            model, val_loader, criterion, device, epoch, args.epochs
        )
        
        scheduler.step()
        ep_time = time.time() - ep_start
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({ep_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1
        })
        
        # Save Best Model Checkpoint
        if val_f1 > best_f1 or val_loss < best_val_loss:
            best_val_loss = min(best_val_loss, val_loss)
            best_f1 = max(best_f1, val_f1)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_f1': best_f1,
                'class_names': CLASS_NAMES
            }
            torch.save(checkpoint, args.save_path)
            print(f"   >>> Checkpoint saved to {args.save_path}! Best Val F1: {best_f1:.4f}, Best Val Loss: {best_val_loss:.4f}")

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Training completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation F1-Score: {best_f1:.4f}")

    # Save History & Plot
    hist_df = pd.DataFrame(history)
    hist_df.to_csv('training_history.csv', index=False)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist_df['epoch'], hist_df['train_loss'], label='Train Loss')
    plt.plot(hist_df['epoch'], hist_df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(hist_df['epoch'], hist_df['train_f1'], label='Train F1')
    plt.plot(hist_df['epoch'], hist_df['val_f1'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training & Validation F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=200)
    print("Saved training curves to training_curves.png")

if __name__ == '__main__':
    main()
