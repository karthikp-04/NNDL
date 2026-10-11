import os
import torch
import numpy as np
from collections import Counter
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import confusion_matrix, classification_report

# 1. Dataset loading functions from the notebook
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

def main():
    file_paths, labels, class_to_id = load_all_dataset_paths()
    id_to_class = {v: k for k, v in class_to_id.items()}
    
    # Stratified split using the same logic
    np.random.seed(42)
    train_ratio, val_ratio, test_ratio = 0.70, 0.15, 0.15
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
            
    test_paths = file_paths[test_idx]
    test_labels = labels[test_idx]
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = SolarPanelDataset(test_paths, test_labels, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)
    
    # Set up model architecture
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(num_features, 11)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    best_model_path = os.path.join("results", "solarguard_best.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Model weights successfully loaded.")
    else:
        print(f"Error: checkpoint {best_model_path} not found.")
        return
        
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = outputs.argmax(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    cm = confusion_matrix(all_targets, all_preds)
    print("\nConfusion Matrix:")
    print("Columns are predictions, rows are true labels.")
    
    header = "True \\ Pred | " + " | ".join([f"{id_to_class[i]:11}" for i in range(11)])
    print(header)
    print("-" * len(header))
    for i in range(11):
        row_str = f"{id_to_class[i]:11} | " + " | ".join([f"{cm[i, j]:11d}" for j in range(11)])
        print(row_str)
        
    report = classification_report(all_targets, all_preds, target_names=[id_to_class[i] for i in range(11)], zero_division=0)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    main()
