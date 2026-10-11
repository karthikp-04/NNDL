import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetB0Solar(nn.Module):
    """
    EfficientNet-B0 fine-tuned for Solar Panel Electroluminescence (EL) Defect Classification.
    """
    def __init__(self, num_classes=8, pretrained=True, dropout=0.3):
        super(EfficientNetB0Solar, self).__init__()
        
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
            
        # EfficientNet-B0 feature dimension is 1280
        in_features = self.backbone.classifier[1].in_features
        
        # Custom Multi-Label Classification Head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass returning raw logits (unnormalized log-odds).
        """
        return self.backbone(x)

    def predict_proba(self, x):
        """
        Returns sigmoid multi-label probabilities for inference.
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

def create_model(num_classes=8, pretrained=True):
    model = EfficientNetB0Solar(num_classes=num_classes, pretrained=pretrained)
    return model

if __name__ == '__main__':
    print("Testing EfficientNetB0Solar Model...")
    net = create_model(num_classes=8, pretrained=True)
    net.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = net(dummy_input)
        probs = net.predict_proba(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits output shape: {logits.shape}")
    print(f"Probabilities output shape: {probs.shape}")
    print(f"Sample probabilities: \n{probs[0].numpy()}")
    print("EfficientNet-B0 Model test passed!")
