import os
import random
import time
import numpy as np
from collections import Counter
from PIL import Image
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Set seed for strict reproducibility
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Load dataset paths and resolve priority classes
def load_all_dataset_paths():
    paths = []
    labels = []
    
    class_to_id = {
        'good': 0, 'crack': 1, 'finger': 2, 'black_core': 3, 'thick_line': 4,
        'star_crack': 5, 'corner': 6, 'fragment': 7, 'scratch': 8,
        'dislocation': 9, 'meterial': 10
    }
    
    priority_classes = ['scratch', 'fragment', 'corner', 'star_crack', 'thick_line', 'black_core', 'crack', 'finger']
    priority_map = {name: idx for idx, name in enumerate(priority_classes)}
    
    train_xml_dir = os.path.join(".", "train_PVELAD", "train", "Annotations")
    train_img_dir = os.path.join(".", "train_PVELAD", "train", "JPEGImages")
    
    if os.path.exists(train_xml_dir):
        for f in sorted(os.listdir(train_xml_dir)):
            if not f.endswith('.xml'):
                continue
            xml_path = os.path.join(train_xml_dir, f)
            img_name = f.replace('.xml', '.jpg')
            img_path = os.path.join(train_img_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
                
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objs = root.findall('object')
            if not objs:
                continue
                
            img_classes = [obj.find('name').text for obj in objs]
            valid_classes = [c for c in img_classes if c in priority_map]
            if not valid_classes:
                continue
                
            selected_class = min(valid_classes, key=lambda c: priority_map[c])
            paths.append(img_path)
            labels.append(class_to_id[selected_class])
            
    othertypes_dir = os.path.join(".", "othertypes_PVELAD", "othertypes")
    subfolders = ["good", "dislocation", "meterial"]
    for folder in subfolders:
        folder_path = os.path.join(othertypes_dir, folder)
        if os.path.exists(folder_path):
            for file_name in sorted(os.listdir(folder_path)):
                if file_name.lower().endswith(('.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, file_name)
                    paths.append(img_path)
                    labels.append(class_to_id[folder])
                    
    return np.array(paths), np.array(labels), class_to_id

class SolarPanelDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label

def robust_stratified_split(paths, labels, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    train_idx, val_idx, test_idx = [], [], []
    class_to_indices = {}
    for idx, label in enumerate(labels):
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
        
    for label, idxs in class_to_indices.items():
        idxs = np.array(idxs)
        np.random.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            train_idx.extend(idxs)
        elif n == 2:
            train_idx.append(idxs[0])
            val_idx.append(idxs[1])
        elif n == 3:
            train_idx.append(idxs[0])
            val_idx.append(idxs[1])
            test_idx.append(idxs[2])
        else:
            n_train = max(1, int(round(train_ratio * n)))
            n_val = max(1, int(round(val_ratio * n)))
            n_test = n - n_train - n_val
            if n_test == 0 and n_val > 1:
                n_val -= 1
                n_test = 1
            elif n_test == 0 and n_train > 1:
                n_train -= 1
                n_test = 1
            train_idx.extend(idxs[:n_train])
            val_idx.extend(idxs[n_train:n_train+n_val])
            test_idx.extend(idxs[n_train+n_val:])
            
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_paths, labels, class_to_id = load_all_dataset_paths()
    id_to_class = {v: k for k, v in class_to_id.items()}
    
    train_idx, val_idx, test_idx = robust_stratified_split(file_paths, labels)
    print(f"Dataset splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Define augmentations and normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = SolarPanelDataset(file_paths[train_idx], labels[train_idx], transform=train_transform)
    val_dataset = SolarPanelDataset(file_paths[val_idx], labels[val_idx], transform=val_test_transform)
    test_dataset = SolarPanelDataset(file_paths[test_idx], labels[test_idx], transform=val_test_transform)
    
    # We use shuffle=True for training instead of WeightedRandomSampler to prevent extreme overfitting.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)
    
    # Calculate stabilized class weights (square-root scaling of inverse frequency)
    # This prevents the rarest classes from having weight multipliers that span 4 orders of magnitude,
    # ensuring training stability while still penalizing their misclassification appropriately.
    inv_freq = []
    for i in range(11):
        count = np.sum(labels[train_idx] == i)
        inv_freq.append(1.0 / np.sqrt(count) if count > 0 else 1.0)
    
    # Normalize weights relative to 'good' class (label 0)
    class_weights = np.array(inv_freq) / inv_freq[0]
    print("Stabilized Class Loss Weights:")
    for idx, w in enumerate(class_weights):
        print(f"  {id_to_class[idx]}: {w:.4f}")
        
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Load model
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 11)
    )
    model = model.to(device)
    
    # --- PHASE 1: Head-Only training ---
    print("\n=== Phase 1: Classification Head Only Training ===")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_val_f1 = 0.0
    best_model_weights = None
    
    for epoch in range(3):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        print(f"P1 Epoch {epoch+1}/3 | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1-Macro: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_weights = model.state_dict().copy()
            os.makedirs("results", exist_ok=True)
            torch.save(best_model_weights, os.path.join("results", "solarguard_best.pth"))
            
    # --- PHASE 2: Fine-Tuning Block 4 + Classifier ---
    print("\n=== Phase 2: Fine-Tuning Block 4 + Classifier ===")
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    for name, param in model.named_parameters():
        if "features.denseblock4" in name or "features.norm5" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    params_group = [
        {"params": [p for n, p in model.named_parameters() if "classifier" in n], "lr": 1e-4},
        {"params": [p for n, p in model.named_parameters() if ("denseblock4" in n or "norm5" in n)], "lr": 1e-5}
    ]
    optimizer = optim.AdamW(params_group, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    patience = 5
    epochs_no_improve = 0
    
    for epoch in range(12):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        scheduler.step(val_f1)
        
        print(f"P2 Epoch {epoch+1}/12 | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1-Macro: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            print(f"  F1 Improved ({best_val_f1:.4f} -> {val_f1:.4f}). Saving model state.")
            best_val_f1 = val_f1
            best_model_weights = model.state_dict().copy()
            torch.save(best_model_weights, os.path.join("results", "solarguard_best.pth"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping triggered after {patience} epochs without F1-macro improvement.")
                break
                
    # Restore best weights and run test set evaluation
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print("Restored best model weights.")
        
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
            
    print("\n--- TEST SET EVALUATION ---")
    cm = confusion_matrix(test_targets, test_preds)
    print("\nConfusion Matrix:")
    header = "True \\ Pred | " + " | ".join([f"{id_to_class[i]:11}" for i in range(11)])
    print(header)
    print("-" * len(header))
    for i in range(11):
        row_str = f"{id_to_class[i]:11} | " + " | ".join([f"{cm[i, j]:11d}" for j in range(11)])
        print(row_str)
        
    report = classification_report(test_targets, test_preds, target_names=[id_to_class[i] for i in range(11)], zero_division=0)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    main()
