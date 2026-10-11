"""
evaluate.py
-----------
Evaluation script for the ResNet-50 (384x384) PVELAD solar-panel defect
classifier trained by train.py.

It loads the saved checkpoint (resnet50_solar_model.pth), runs inference on
the held-out test set (Dataset/test_PVELAD/test/), and reports:

    - Accuracy
    - Precision / Recall / F1 (macro, micro, weighted)
    - Balanced Accuracy
    - Cohen's Kappa
    - Per-class accuracy
    - Full classification report
    - Confusion matrix (raw counts + normalized)
    - ROC-AUC (One-vs-Rest, macro) + ROC curves for all 9 classes
    - Per-image predictions CSV
    - Average inference time per image

Outputs are written to a `results/` folder next to this script:

    results/
        roc_curve.png
        confusion_matrix.png
        classification_report.txt
        predictions.csv
        metrics.txt

USAGE
-----
    pip install scikit-learn seaborn pandas matplotlib
    python evaluate.py

By default it expects this layout (same as your training script):

    <this folder>/
        resnet50_solar_model.pth
        Dataset/
            test_PVELAD/
                test/
                    JPEGImages/
                    Annotations/
            othertypes_PVELAD/            (optional, for 'normal' class)
                othertypes/
                    good/

If your paths differ, edit CHECKPOINT_PATH and TEST_ROOT below.
"""

import os
import glob
import time
import random
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless / no-display environments
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    auc,
)
from sklearn.preprocessing import label_binarize

# ===========================================================
# CONFIG — edit these paths if your folder layout differs
# ===========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "resnet50_solar_model.pth")

# NOTE: test_PVELAD has no annotations (unlabeled benchmark test split).
# The only labeled data available is train_PVELAD. Since train.py's
# random_split was not seeded/saved, we cannot recover a clean held-out
# set — evaluating here will include images the model already trained on,
# so treat these numbers as an OPTIMISTIC ESTIMATE, not a true test result.
TEST_ROOT = os.path.join(BASE_DIR, "Dataset", "train_PVELAD", "train")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BATCH_SIZE = 16
IMG_SIZE = 384  # must match training resolution

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------
# 1. Test Dataset
#    Mirrors PVELADDatasetAllClasses from train.py, but WITHOUT
#    synthetic oversampling (evaluation must use real, unrepeated
#    samples) and WITHOUT train-time augmentation.
# ---------------------------------------------------------
class PVELADTestDataset(Dataset):
    def __init__(self, root_dir, transform=None, include_normal=True, max_normal=300, seed=42):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "JPEGImages")
        self.ann_dir = os.path.join(root_dir, "Annotations")
        self.transform = transform

        self.classes = [
            "crack", "finger", "black_core", "thick_line",
            "star_crack", "corner", "fragment", "scratch", "normal",
        ]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []

        # Defect images from Annotations/JPEGImages (same logic as training)
        print(f"🔎 Looking for annotations in: {self.ann_dir}")
        print(f"🔎 Looking for images in     : {self.img_dir}")

        if os.path.exists(self.ann_dir):
            xml_files = sorted(glob.glob(os.path.join(self.ann_dir, "*.xml")))
            print(f"   Found {len(xml_files)} XML annotation files.")
            matched = 0
            for xml_file in xml_files:
                base_name = os.path.splitext(os.path.basename(xml_file))[0]
                img_path = os.path.join(self.img_dir, base_name + ".jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.img_dir, base_name + ".png")
                if os.path.exists(img_path):
                    label = self._parse_annotation(xml_file)
                    self.samples.append((img_path, label))
                    matched += 1
            print(f"   Matched {matched} XML files to an existing image.")
            if len(xml_files) > 0 and matched == 0:
                print(f"   ⚠️  XML files exist but none matched an image in {self.img_dir}. "
                      f"Check that image filenames (minus extension) match XML filenames.")
        else:
            print(f"⚠️  No Annotations folder found at {self.ann_dir}. "
                  f"Defect classes (crack, finger, black_core, thick_line, star_crack, "
                  f"corner, fragment, scratch) will be EMPTY — only 'normal' images will "
                  f"be loaded. Double check this path exists and contains .xml files.")

        # Optional healthy ("normal") images, same source used in training
        if include_normal:
            dataset_base_dir = os.path.dirname(os.path.dirname(self.root_dir))
            good_img_dir = os.path.join(dataset_base_dir, "othertypes_PVELAD", "othertypes", "good")
            if os.path.exists(good_img_dir):
                good_imgs = sorted(
                    glob.glob(os.path.join(good_img_dir, "*.jpg"))
                    + glob.glob(os.path.join(good_img_dir, "*.png"))
                )
                random.seed(seed)
                random.shuffle(good_imgs)
                good_imgs = good_imgs[:max_normal]
                for img_path in good_imgs:
                    self.samples.append((img_path, self.class_to_idx["normal"]))

        counts = Counter([lbl for _, lbl in self.samples])
        print("\n📊 Test Set Counts (no oversampling):")
        for name, idx in self.class_to_idx.items():
            print(f"   Class '{name}': {counts.get(idx, 0)} samples")
        print(f"✅ Total test samples: {len(self.samples)}\n")

    def _parse_annotation(self, xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            obj = root.find("object")
            if obj is not None:
                name = obj.find("name").text.strip()
                if name in self.class_to_idx:
                    return self.class_to_idx[name]
        except Exception:
            pass
        return self.class_to_idx["normal"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


# ---------------------------------------------------------
# 2. Model architecture — MUST exactly match train.py so that
#    load_state_dict succeeds.
# ---------------------------------------------------------
def build_high_res_resnet50(num_classes=9):
    try:
        model = models.resnet50(weights=None)  # we're loading our own weights, no need for ImageNet ones
    except Exception:
        model = models.resnet50(pretrained=False)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )
    return model


# ---------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    print("\n" + "=" * 70)
    print("⚠️  CAVEAT: test_PVELAD has no annotations (unlabeled benchmark split).")
    print("   Evaluating on train_PVELAD instead. Since train.py's random_split")
    print("   was unseeded and not saved, some/most of these images were likely")
    print("   used during training. These metrics are an OPTIMISTIC ESTIMATE —")
    print("   NOT a clean test-set result. See the fixed-split retraining option")
    print("   for a trustworthy evaluation.")
    print("=" * 70 + "\n")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
    if not os.path.exists(TEST_ROOT):
        raise FileNotFoundError(f"Test dataset folder not found at {TEST_ROOT}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    classes = checkpoint["classes"]
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(classes)

    print(f"Loaded checkpoint with classes: {classes}")

    model = build_high_res_resnet50(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_dataset = PVELADTestDataset(TEST_ROOT, transform=eval_transform)
    if len(test_dataset) == 0:
        raise RuntimeError("Test dataset is empty — check TEST_ROOT and folder structure.")

    # Sanity check that dataset class order matches checkpoint class order
    if test_dataset.classes != classes:
        print("⚠️  Warning: test dataset class order differs from checkpoint. "
              "Using checkpoint's class order/indices for evaluation.")

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_labels = []
    all_preds = []
    all_probs = []
    all_paths = []

    total_images = 0
    start_time = time.time()

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_paths.extend(paths)

            total_images += images.size(0)

    total_time = time.time() - start_time
    avg_time_ms = (total_time / total_images) * 1000 if total_images else 0

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)  # shape: (N, num_classes)

    # -------------------------------------------------------
    # Metrics
    # -------------------------------------------------------
    acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="micro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    # ROC-AUC (One-vs-Rest, macro). Requires all classes present in y_true
    # to be well-defined; guard against classes missing from the test set.
    present_classes = sorted(set(all_labels.tolist()))
    roc_auc_macro = None
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}

    try:
        y_true_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        roc_auc_macro = roc_auc_score(
            y_true_bin, all_probs, average="macro", multi_class="ovr"
        )
        for i in range(num_classes):
            if i in present_classes:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
                fpr_dict[i] = fpr
                tpr_dict[i] = tpr
                auc_dict[i] = auc(fpr, tpr)
    except Exception as e:
        print(f"⚠️  Could not compute ROC-AUC: {e}")

    # Per-class accuracy
    per_class_acc = {}
    for idx, cls_name in enumerate(classes):
        mask = all_labels == idx
        if mask.sum() > 0:
            per_class_acc[cls_name] = (all_preds[mask] == idx).mean()
        else:
            per_class_acc[cls_name] = float("nan")

    report_str = classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),   # force all 9 classes to appear even if absent from this test run
        target_names=classes,
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    # -------------------------------------------------------
    # Print summary
    # -------------------------------------------------------
    print("\n===============================")
    print("Evaluation Results")
    print("===============================\n")
    print(f"Test Images           : {total_images}")
    print(f"Accuracy              : {acc*100:.2f} %")
    print(f"Balanced Accuracy     : {balanced_acc*100:.2f} %")
    print(f"Cohen's Kappa         : {kappa:.4f}")
    print()
    print(f"Precision (Macro)     : {precision_macro*100:.2f} %")
    print(f"Recall (Macro)        : {recall_macro*100:.2f} %")
    print(f"F1 Score (Macro)      : {f1_macro*100:.2f} %")
    print()
    print(f"Precision (Weighted)  : {precision_weighted*100:.2f} %")
    print(f"Recall (Weighted)     : {recall_weighted*100:.2f} %")
    print(f"F1 Score (Weighted)   : {f1_weighted*100:.2f} %")
    if roc_auc_macro is not None:
        print()
        print(f"ROC-AUC (Macro, OvR)  : {roc_auc_macro:.4f}")
    print()
    print(f"Avg inference time    : {avg_time_ms:.2f} ms/image")
    print("\nPer-class accuracy:")
    for cls_name, a in per_class_acc.items():
        print(f"   {cls_name:12s}: {a*100:.2f} %" if not np.isnan(a) else f"   {cls_name:12s}: N/A (no samples)")
    print("\nClassification report:\n")
    print(report_str)

    # -------------------------------------------------------
    # Save results
    # -------------------------------------------------------

    # metrics.txt
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write("Evaluation Results\n")
        f.write("===============================\n\n")
        f.write(f"Test Images           : {total_images}\n")
        f.write(f"Accuracy              : {acc*100:.2f} %\n")
        f.write(f"Balanced Accuracy     : {balanced_acc*100:.2f} %\n")
        f.write(f"Cohen's Kappa         : {kappa:.4f}\n\n")
        f.write(f"Precision (Macro)     : {precision_macro*100:.2f} %\n")
        f.write(f"Recall (Macro)        : {recall_macro*100:.2f} %\n")
        f.write(f"F1 Score (Macro)      : {f1_macro*100:.2f} %\n\n")
        f.write(f"Precision (Micro)     : {precision_micro*100:.2f} %\n")
        f.write(f"Recall (Micro)        : {recall_micro*100:.2f} %\n")
        f.write(f"F1 Score (Micro)      : {f1_micro*100:.2f} %\n\n")
        f.write(f"Precision (Weighted)  : {precision_weighted*100:.2f} %\n")
        f.write(f"Recall (Weighted)     : {recall_weighted*100:.2f} %\n")
        f.write(f"F1 Score (Weighted)   : {f1_weighted*100:.2f} %\n\n")
        if roc_auc_macro is not None:
            f.write(f"ROC-AUC (Macro, OvR)  : {roc_auc_macro:.4f}\n\n")
        f.write(f"Avg inference time    : {avg_time_ms:.2f} ms/image\n\n")
        f.write("Per-class accuracy:\n")
        for cls_name, a in per_class_acc.items():
            f.write(f"   {cls_name:12s}: {a*100:.2f} %\n" if not np.isnan(a) else f"   {cls_name:12s}: N/A\n")

    # classification_report.txt
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report_str)

    # predictions.csv
    pred_class_names = [classes[p] for p in all_preds]
    true_class_names = [classes[t] for t in all_labels]
    df = pd.DataFrame({
        "image_path": all_paths,
        "true_label": true_class_names,
        "predicted_label": pred_class_names,
        "correct": all_preds == all_labels,
    })
    for idx, cls_name in enumerate(classes):
        df[f"prob_{cls_name}"] = all_probs[:, idx]
    df.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)

    # confusion_matrix.png
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title("Confusion Matrix (row-normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    # roc_curve.png
    if fpr_dict:
        plt.figure(figsize=(9, 8))
        for idx in fpr_dict:
            plt.plot(fpr_dict[idx], tpr_dict[idx],
                     label=f"{classes[idx]} (AUC = {auc_dict[idx]:.3f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        title = "ROC Curves (One-vs-Rest)"
        if roc_auc_macro is not None:
            title += f" — Macro AUC = {roc_auc_macro:.3f}"
        plt.title(title)
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=200)
        plt.close()
    else:
        print("⚠️  Skipped ROC curve plot — no valid per-class ROC data.")

    print(f"\n✅ All results saved to: {RESULTS_DIR}")
    print("   - metrics.txt")
    print("   - classification_report.txt")
    print("   - predictions.csv")
    print("   - confusion_matrix.png")
    if fpr_dict:
        print("   - roc_curve.png")


if __name__ == "__main__":
    evaluate()