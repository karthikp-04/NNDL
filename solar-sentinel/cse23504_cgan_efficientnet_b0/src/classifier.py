"""
SolarSentinel AI — EfficientNet-B0 Classifier Architecture
Transfer-learning convolutional neural network tailored for multi-label solar panel defect classification.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torchvision.models as models

import config

class EfficientNetB0Solar(nn.Module):
    """
    EfficientNet-B0 fine-tuned for Solar Panel Electroluminescence (EL) Defect Classification.
    Features:
    - Pretrained ImageNet convolutional backbone
    - Dropout regularization (0.3)
    - Custom Linear projection head for 8 defect categories
    - Sigmoid probability activation for multi-label classification
    """
    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=True, dropout=config.CLASSIFIER_DROPOUT):
        super(EfficientNetB0Solar, self).__init__()
        self.num_classes = num_classes

        if pretrained:
            try:
                weights = models.EfficientNet_B0_Weights.DEFAULT
                self.backbone = models.efficientnet_b0(weights=weights)
            except Exception as e:
                print(f"Warning: Pretrained weights download failed ({e}). Initializing random backbone.")
                self.backbone = models.efficientnet_b0(weights=None)
        else:
            self.backbone = models.efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features  # 1280
        
        # Multi-Label Classification Head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        Computes forward pass returning raw unnormalized logits.
        """
        return self.backbone(x)

    def predict_proba(self, x):
        """
        Computes calibrated sigmoid multi-label probabilities in [0.0, 1.0].
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def count_parameters(self):
        """
        Returns (total_params, trainable_params).
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

def create_classifier(num_classes=config.NUM_CLASSES, pretrained=True, dropout=config.CLASSIFIER_DROPOUT, device=config.DEVICE):
    """
    Factory function to instantiate and place model onto device.
    """
    model = EfficientNetB0Solar(num_classes=num_classes, pretrained=pretrained, dropout=dropout).to(device)
    return model
