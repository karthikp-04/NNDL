import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score
from PIL import Image
from tqdm import tqdm

from dataset import PVELADDataset, get_transforms, CLASS_NAMES, NUM_CLASSES
from model import create_model

def evaluate_checkpoint(checkpoint_path='best_efficientnet_b0.pth', csv_path='dataset_index.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating checkpoint: {checkpoint_path} on device {device}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    df = pd.read_csv(csv_path)
    val_dataset = PVELADDataset(df, transform=get_transforms(img_size=224, is_train=False))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            probs = model.predict_proba(images)
            all_targets.append(targets.numpy())
            all_probs.append(probs.cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    binary_preds = (all_probs > 0.5).astype(int)

    print("\n" + "=" * 60)
    print("PER-CLASS EVALUATION REPORT")
    print("=" * 60)
    
    metrics = []
    for i, cname in enumerate(CLASS_NAMES):
        y_true = all_targets[:, i]
        y_pred = binary_preds[:, i]
        y_prob = all_probs[:, i]
        
        try:
            auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        except Exception:
            auc = 0.0

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics.append({
            'Class': cname,
            'Total_Positives': int(np.sum(y_true)),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(auc, 4)
        })
        print(f"{cname:15s} | Positives: {int(np.sum(y_true)):5d} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('evaluation_report.csv', index=False)
    print("Saved evaluation report to evaluation_report.csv")
    return metrics_df

def predict_test_set(checkpoint_path='best_efficientnet_b0.pth', test_dir='test_PVELAD/test/JPEGImages', limit=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nRunning test set prediction on images in {test_dir}...")
    
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = get_transforms(img_size=224, is_train=False)
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if limit is not None and limit < len(test_files):
        print(f"Predicting on sample subset of {limit} test images out of {len(test_files)}...")
        test_files = test_files[:limit]

    results = []
    with torch.no_grad():
        for fname in tqdm(test_files, desc="Inferring Test Images"):
            fpath = os.path.join(test_dir, fname)
            img = Image.open(fpath).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            probs = model.predict_proba(tensor).cpu().numpy()[0]
            
            pred_classes = [CLASS_NAMES[i] for i, p in enumerate(probs) if p > 0.5]
            if not pred_classes:
                top_idx = np.argmax(probs)
                pred_classes = [f"{CLASS_NAMES[top_idx]} ({probs[top_idx]:.2f})"]
                
            rec = {'filename': fname, 'predicted_labels': ', '.join(pred_classes)}
            for i, cname in enumerate(CLASS_NAMES):
                rec[cname + '_prob'] = round(float(probs[i]), 4)
            results.append(rec)

    res_df = pd.DataFrame(results)
    res_df.to_csv('test_predictions_sample.csv', index=False)
    print("Saved test set predictions to test_predictions_sample.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate model checkpoint on PVELAD dataset")
    parser.add_argument('--checkpoint', type=str, default='best_efficientnet_b0.pth', help='Path to checkpoint file')
    parser.add_argument('--limit', type=int, default=500, help='Limit test set prediction sample size')
    args = parser.parse_args()
    
    evaluate_checkpoint(checkpoint_path=args.checkpoint)
    predict_test_set(checkpoint_path=args.checkpoint, limit=args.limit)
