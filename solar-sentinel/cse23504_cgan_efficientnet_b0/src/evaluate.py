"""
SolarSentinel AI — Evaluation Engine
Evaluates EfficientNet-B0 classifiers on the untouched, held-out real test set.
Computes multi-label metrics, ROC-AUC, per-class confusion matrices, and latency.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, roc_curve, confusion_matrix, hamming_loss
)
from tqdm import tqdm

import config
from src.preprocessing import create_reproducible_splits, get_classifier_transforms
from src.dataset import PVELADDataset
from src.classifier import create_classifier

def plot_confusion_matrices(all_targets, all_binary_preds, model_name, save_path):
    """
    Plots an 8-panel grid of 2x2 confusion matrices for all multi-label classes.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, cname in enumerate(config.CLASS_NAMES):
        cm = confusion_matrix(all_targets[:, idx], all_binary_preds[:, idx], labels=[0, 1])
        ax = axes[idx]
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
            xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos']
        )
        ax.set_title(f"{cname}", fontsize=11, fontweight='bold')
        ax.set_ylabel('Ground Truth' if idx % 4 == 0 else '')
        ax.set_xlabel('Predicted')

    plt.suptitle(f"SolarSentinel AI — Confusion Matrices ({model_name})", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrices to {save_path}")

def plot_roc_curves(all_targets, all_probs, model_name, save_path):
    """
    Plots multi-label ROC curves with individual class AUC values.
    """
    plt.figure(figsize=(10, 7))
    colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899']

    for idx, cname in enumerate(config.CLASS_NAMES):
        y_true = all_targets[:, idx]
        y_prob = all_probs[:, idx]

        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
            plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2, label=f"{cname} (AUC = {auc_val:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Chance Line (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    plt.title(f'SolarSentinel AI — Multi-Label ROC Curves ({model_name})', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved ROC curves to {save_path}")

def evaluate_model(
    checkpoint_path,
    model_name="Baseline",
    batch_size=32,
    threshold=config.DECISION_THRESHOLD,
    device=config.DEVICE
):
    """
    Evaluates checkpoint on held-out test split.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print("=" * 70)
    print(f"SOLARSENTINEL AI — MODEL EVALUATION: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device} | Decision Threshold: {threshold}")
    print("=" * 70)

    # 1. Load Test Split
    splits_df = create_reproducible_splits()
    test_df = splits_df[splits_df['split'] == 'test'].reset_index(drop=True)
    print(f"Test samples: {len(test_df):,} held-out images")

    test_ds = PVELADDataset(test_df, transform=get_classifier_transforms(is_train=False))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS)

    # 2. Load Model
    model = create_classifier(num_classes=config.NUM_CLASSES, pretrained=False, device=device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    total_params, trainable_params = model.count_parameters()
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    # 3. Inference Pass with Latency Profiling
    all_targets = []
    all_probs = []
    latencies = []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            images = images.to(device)

            t0 = time.perf_counter()
            probs = model.predict_proba(images)
            t1 = time.perf_counter()

            batch_latency_ms = ((t1 - t0) / images.size(0)) * 1000.0
            latencies.append(batch_latency_ms)

            all_targets.append(targets.numpy())
            all_probs.append(probs.cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    binary_preds = (all_probs > threshold).astype(int)

    avg_latency_ms = float(np.mean(latencies))
    throughput = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    # 4. Compute Aggregate Metrics
    macro_f1 = f1_score(all_targets, binary_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_targets, binary_preds, average='micro', zero_division=0)
    weighted_f1 = f1_score(all_targets, binary_preds, average='weighted', zero_division=0)

    macro_precision = precision_score(all_targets, binary_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_targets, binary_preds, average='macro', zero_division=0)

    exact_acc = accuracy_score(all_targets, binary_preds)
    h_loss = hamming_loss(all_targets, binary_preds)
    hamming_acc = 1.0 - h_loss

    try:
        macro_auc = roc_auc_score(all_targets, all_probs, average='macro')
    except Exception:
        macro_auc = 0.0

    # 5. Per-Class Metrics
    per_class_rows = []
    print("\n" + "-" * 75)
    print(f"{'Class Name':<14} | {'Pos':<5} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'ROC-AUC':<7}")
    print("-" * 75)

    for i, cname in enumerate(config.CLASS_NAMES):
        y_t = all_targets[:, i]
        y_p = binary_preds[:, i]
        y_prob = all_probs[:, i]

        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))
        pos = int(np.sum(y_t))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        try:
            auc = roc_auc_score(y_t, y_prob) if len(np.unique(y_t)) > 1 else 0.0
        except Exception:
            auc = 0.0

        per_class_rows.append({
            'class': cname,
            'positives': pos,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'roc_auc': round(auc, 4)
        })
        print(f"{cname:<14} | {pos:<5d} | {prec:<7.4f} | {rec:<7.4f} | {f1:<7.4f} | {auc:<7.4f}")

    print("-" * 75)
    print(f"Macro F1: {macro_f1:.4f} | Macro Precision: {macro_precision:.4f} | Macro Recall: {macro_recall:.4f}")
    print(f"Exact Match Accuracy: {exact_acc:.4f} | Hamming Accuracy: {hamming_acc:.4f} | Macro AUC: {macro_auc:.4f}")
    print(f"Inference Latency: {avg_latency_ms:.2f} ms/sample ({throughput:.1f} FPS)")
    print(f"Model Parameters: {trainable_params:,} | Checkpoint Size: {file_size_mb:.2f} MB\n")

    # 6. Save Results
    clean_name = model_name.lower().replace(" ", "_")
    per_class_df = pd.DataFrame(per_class_rows)
    eval_csv_path = os.path.join(config.METRICS_DIR, f"{clean_name}_evaluation.csv")
    per_class_df.to_csv(eval_csv_path, index=False)

    summary_dict = {
        'model_name': model_name,
        'checkpoint_path': checkpoint_path,
        'total_params': total_params,
        'checkpoint_size_mb': round(file_size_mb, 2),
        'avg_latency_ms': round(avg_latency_ms, 2),
        'throughput_fps': round(throughput, 1),
        'exact_match_accuracy': round(exact_acc, 4),
        'hamming_accuracy': round(hamming_acc, 4),
        'macro_f1': round(macro_f1, 4),
        'micro_f1': round(micro_f1, 4),
        'weighted_f1': round(weighted_f1, 4),
        'macro_precision': round(macro_precision, 4),
        'macro_recall': round(macro_recall, 4),
        'macro_roc_auc': round(macro_auc, 4),
        'per_class': per_class_rows
    }
    eval_json_path = os.path.join(config.METRICS_DIR, f"{clean_name}_evaluation.json")
    with open(eval_json_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)

    # 7. Generate Plots
    cm_plot_path = os.path.join(config.PLOTS_DIR, f"{clean_name}_confusion_matrices.png")
    plot_confusion_matrices(all_targets, binary_preds, model_name, cm_plot_path)

    roc_plot_path = os.path.join(config.PLOTS_DIR, f"{clean_name}_roc_curves.png")
    plot_roc_curves(all_targets, all_probs, model_name, roc_plot_path)

    return summary_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate SolarSentinel Model")
    parser.add_argument('--checkpoint', type=str, default=config.BASELINE_CKPT, help="Path to checkpoint")
    parser.add_argument('--name', type=str, default="Baseline", help="Model identification name")
    args = parser.parse_args()

    evaluate_model(checkpoint_path=args.checkpoint, model_name=args.name)
