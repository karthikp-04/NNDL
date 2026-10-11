import os
import glob
import xml.etree.ElementTree as ET
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import time

# ---------------------------------------------------------
# 1. Dataset Definition with Synthetic Oversampling for Rare Classes
# ---------------------------------------------------------
class PVELADDatasetAllClasses(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.ann_dir = os.path.join(root_dir, 'Annotations')
        self.transform = transform
        
        self.classes = ['crack', 'finger', 'black_core', 'thick_line', 'star_crack', 'corner', 'fragment', 'scratch', 'normal']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        raw_samples = []
        if os.path.exists(self.ann_dir):
            xml_files = glob.glob(os.path.join(self.ann_dir, "*.xml"))
            for xml_file in xml_files:
                base_name = os.path.splitext(os.path.basename(xml_file))[0]
                img_path = os.path.join(self.img_dir, base_name + ".jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.img_dir, base_name + ".png")
                
                if os.path.exists(img_path):
                    label = self._parse_annotation(xml_file)
                    raw_samples.append((img_path, label))

        # Load healthy (normal) images from othertypes_PVELAD\othertypes\good
        dataset_base_dir = os.path.dirname(os.path.dirname(self.root_dir))
        good_img_dir = os.path.join(dataset_base_dir, 'othertypes_PVELAD', 'othertypes', 'good')
        if os.path.exists(good_img_dir):
            good_imgs = glob.glob(os.path.join(good_img_dir, "*.jpg")) + glob.glob(os.path.join(good_img_dir, "*.png"))
            
            # Shuffle and cap to 300 to prevent normal class from overpowering defects
            import random
            random.seed(42)
            random.shuffle(good_imgs)
            good_imgs = good_imgs[:300]
            
            for img_path in good_imgs:
                raw_samples.append((img_path, self.class_to_idx['normal']))

        # Synthetic Oversampling for Minority Classes (<50 samples)
        self.samples = []
        counts = Counter([lbl for _, lbl in raw_samples])
        
        print("\n📊 Raw Dataset Counts:")
        for name, idx in self.class_to_idx.items():
            print(f"   Class '{name}': {counts.get(idx, 0)} original samples")

        for img_path, label in raw_samples:
            self.samples.append((img_path, label))
            c_count = counts.get(label, 0)
            if c_count > 0 and c_count < 10:
                for _ in range(25):  # Oversample rare classes (scratch, corner, fragment)
                    self.samples.append((img_path, label))
            elif c_count >= 10 and c_count < 100:
                for _ in range(2):   # Oversample star_crack
                    self.samples.append((img_path, label))

        print(f"\n✅ Synthetic Oversampling Complete: Dataset expanded to {len(self.samples)} samples across ALL 9 classes!")

    def _parse_annotation(self, xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            obj = root.find('object')
            if obj is not None:
                name = obj.find('name').text.strip()
                if name in self.class_to_idx:
                    return self.class_to_idx[name]
        except Exception:
            pass
        return self.class_to_idx['normal']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ---------------------------------------------------------
# 2. Focal Loss Function
# ---------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()

# ---------------------------------------------------------
# 3. High-Resolution (384x384) ResNet-50 Architecture
# ---------------------------------------------------------
def build_high_res_resnet50(num_classes=9):
    print("Loading ResNet-50 architecture...")
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except Exception:
        model = models.resnet50(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False

    print("🔓 Unfreezing Layer 3 & Layer 4 for High-Resolution Micro-Crack Extraction...")
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model

# ---------------------------------------------------------
# 4. Ultra High-Resolution Training Pipeline (384x384)
# ---------------------------------------------------------
def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "Dataset", "train_PVELAD", "train")
    
    if not os.path.exists(train_path):
        print(f"Error: Dataset not found at {train_path}")
        return

    # Ultra High Resolution Transforms (384x384 pixels)
    data_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")

    dataset = PVELADDatasetAllClasses(train_path, transform=data_transforms)
    if len(dataset) == 0:
        return

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    num_classes = len(dataset.classes)
    model = build_high_res_resnet50(num_classes=num_classes).to(device)

    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.Adam([
        {'params': model.layer3.parameters(), 'lr': 1e-4},
        {'params': model.layer4.parameters(), 'lr': 2e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ])

    epochs = 10
    total_batches = len(train_loader)
    
    print(f"\n=======================================================")
    print(f"🔬 ResNet-50 ULTRA HIGH-RESOLUTION (384x384) TRAINING")
    print(f"Focal Loss + Synthetic Oversampling | {epochs} Epochs")
    print(f"=======================================================\n")
    
    start_time = time.time()
    train_losses, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        print(f"⏳ Epoch [{epoch+1}/{epochs}] training at 384x384 resolution...")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            if (batch_idx + 1) % 30 == 0 or (batch_idx + 1) == total_batches:
                print(f"   ↳ Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{total_batches}] processed...")

        epoch_loss = running_loss / len(train_ds)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = (correct / total) * 100 if total > 0 else 0
        val_accuracies.append(val_acc)
        print(f"🔥 Epoch [{epoch+1}/{epochs}] Complete ➔ Focal Loss: {epoch_loss:.4f} | Val Accuracy: {val_acc:.2f}%\n")

    total_time = time.time() - start_time
    print(f"\n🎉 Ultra High-Res Training Complete in {total_time/60:.2f} minutes!")

    # Save Model Weights
    save_path = os.path.join(base_dir, "resnet50_solar_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': dataset.classes,
        'class_to_idx': dataset.class_to_idx
    }, save_path)
    print(f"Saved upgraded 384x384 weights to: {save_path}")

    # Plot & Save Metrics Graph
    plt.figure(figsize=(9, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, 'b-o', linewidth=2, label='Focal Loss')
    plt.title('ResNet-50 (384x384) Focal Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_accuracies, 'g-o', linewidth=2, label='Val Accuracy')
    plt.title('Validation Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    chart_path = os.path.join(base_dir, "training_metrics.png")
    plt.savefig(chart_path, dpi=300)
    print(f"High-resolution metrics graph saved to: {chart_path}")

if __name__ == "__main__":
    train_model()
