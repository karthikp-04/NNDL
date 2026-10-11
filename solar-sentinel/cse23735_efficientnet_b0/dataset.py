import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# 8 Target Classification Categories
CLASS_NAMES = [
    'crack',
    'finger',
    'black_core',
    'thick_line',
    'star_crack',
    'other_defect',
    'good',
    'material'
]

NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def build_dataset_index(root_dir=".", csv_path="dataset_index.csv"):
    """
    Builds a cached CSV index of all 13,414 images in PVELAD dataset with multi-label targets.
    """
    records = []
    
    train_dir = os.path.join(root_dir, 'train_PVELAD', 'train')
    annotations_dir = os.path.join(train_dir, 'Annotations')
    images_dir = os.path.join(train_dir, 'JPEGImages')
    
    # 1. Parse Defective Images (train_PVELAD)
    if os.path.exists(annotations_dir):
        xml_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.xml')])
        for xml_file in xml_files:
            base_name = os.path.splitext(xml_file)[0]
            # Find matching image
            img_path = None
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                possible_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(possible_path):
                    img_path = possible_path
                    break
            
            if img_path is None:
                continue
                
            labels = [0] * NUM_CLASSES
            tree = ET.parse(os.path.join(annotations_dir, xml_file))
            for obj in tree.getroot().findall('object'):
                name = obj.find('name').text
                if name in CLASS_TO_IDX:
                    labels[CLASS_TO_IDX[name]] = 1
                elif name in ['corner', 'fragment', 'scratch']:
                    labels[CLASS_TO_IDX['other_defect']] = 1
            
            record = {'image_path': img_path, 'source': 'train_PVELAD'}
            for idx, cname in enumerate(CLASS_NAMES):
                record[cname] = labels[idx]
            records.append(record)
            
    # 2. Add Good Images (othertypes_PVELAD/good)
    good_dir = os.path.join(root_dir, 'othertypes_PVELAD', 'othertypes', 'good')
    if os.path.exists(good_dir):
        good_files = sorted([f for f in os.listdir(good_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG'))])
        for f in good_files:
            img_path = os.path.join(good_dir, f)
            labels = [0] * NUM_CLASSES
            labels[CLASS_TO_IDX['good']] = 1
            record = {'image_path': img_path, 'source': 'good'}
            for idx, cname in enumerate(CLASS_NAMES):
                record[cname] = labels[idx]
            records.append(record)

    # 3. Add Material Images (othertypes_PVELAD/meterial)
    mat_dir = os.path.join(root_dir, 'othertypes_PVELAD', 'othertypes', 'meterial')
    if os.path.exists(mat_dir):
        mat_files = sorted([f for f in os.listdir(mat_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG'))])
        for f in mat_files:
            img_path = os.path.join(mat_dir, f)
            labels = [0] * NUM_CLASSES
            labels[CLASS_TO_IDX['material']] = 1
            record = {'image_path': img_path, 'source': 'material'}
            for idx, cname in enumerate(CLASS_NAMES):
                record[cname] = labels[idx]
            records.append(record)
            
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Dataset index created with {len(df)} total images saved to {csv_path}")
    return df

def get_transforms(img_size=224, is_train=True):
    """
    Returns image transformation pipelines for training and evaluation.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

class PVELADDataset(Dataset):
    """
    PyTorch Dataset for PVELAD Solar Cell EL Classification.
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_paths = self.df['image_path'].values
        self.labels = self.df[CLASS_NAMES].values.astype(np.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

if __name__ == '__main__':
    df = build_dataset_index()
    print("Class Distributions:")
    for c in CLASS_NAMES:
        print(f"  - {c}: {df[c].sum()} images")
