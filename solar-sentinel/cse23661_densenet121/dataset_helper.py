import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset

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
