"""
SolarSentinel AI — Defect Inference Engine
Loads trained EfficientNet-B0 checkpoint, runs inference on real EL images,
computes sigmoid defect probabilities, and formats diagnostic interpretations.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import argparse
import numpy as np
import torch
from PIL import Image

import config
from src.preprocessing import get_classifier_transforms
from src.classifier import create_classifier

class SolarSentinelPredictor:
    """
    Inference service for Solar Panel Defect Detection.
    """
    def __init__(self, checkpoint_path=config.BASELINE_CKPT, device=config.DEVICE):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.transform = get_classifier_transforms(is_train=False)

        if os.path.exists(checkpoint_path):
            self.load_model(checkpoint_path)
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Predictor initialized in unweighted state.")

    def load_model(self, checkpoint_path):
        """
        Loads weights from PyTorch checkpoint.
        """
        self.model = create_classifier(num_classes=config.NUM_CLASSES, pretrained=False, device=self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.checkpoint_path = checkpoint_path
        print(f"Loaded classifier weights from {checkpoint_path}")

    def predict(self, image_input, threshold=config.DECISION_THRESHOLD):
        """
        Predicts multi-label defects on a single image.
        image_input: file path (str) or PIL.Image.Image instance.
        """
        if self.model is None:
            raise RuntimeError("No model weights loaded. Cannot perform prediction.")

        # Handle file path vs PIL Image
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Input image not found: {image_input}")
            img_pil = Image.open(image_input).convert('RGB')
            input_name = os.path.basename(image_input)
        elif isinstance(image_input, Image.Image):
            img_pil = image_input.convert('RGB')
            input_name = "uploaded_image.png"
        else:
            raise ValueError("Input must be a valid file path string or PIL Image.")

        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            probs_tensor = self.model.predict_proba(img_tensor).squeeze(0)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        probs_np = probs_tensor.cpu().numpy()
        probs_dict = {config.CLASS_NAMES[i]: float(probs_np[i]) for i in range(config.NUM_CLASSES)}

        # Active predictions above threshold
        active_classes = [cname for cname, p in probs_dict.items() if p >= threshold]

        # Determine primary defect
        defect_probs = {c: probs_dict[c] for c in config.DEFECT_CLASSES}
        primary_defect = max(defect_probs, key=defect_probs.get) if defect_probs else 'none'
        max_defect_prob = defect_probs[primary_defect]

        is_defective = max_defect_prob >= threshold or any(probs_dict[d] >= threshold for d in config.DEFECT_CLASSES)

        # Generate diagnostic interpretation
        if not is_defective:
            if probs_dict['good'] >= 0.5:
                diagnosis = "Healthy Solar Cell — Homogeneous electroluminescence emission. No structural or electrical defects detected."
            elif probs_dict['material'] >= 0.5:
                diagnosis = "Wafer Inhomogeneity — Material pattern observed; no active electrical defect detected."
            else:
                diagnosis = "Nominal Operation — Low confidence of defect markers."
        else:
            defects_detected = [f"{c.replace('_', ' ').title()} ({probs_dict[c]:.1%})" for c in config.DEFECT_CLASSES if probs_dict[c] >= threshold]
            if not defects_detected:
                defects_detected = [f"{primary_defect.replace('_', ' ').title()} ({max_defect_prob:.1%})"]
            diagnosis = f"Defect Alert — Detected {', '.join(defects_detected)}. Requires maintenance inspection."

        return {
            'image_name': input_name,
            'probabilities': probs_dict,
            'active_classes': active_classes,
            'primary_defect': primary_defect,
            'is_defective': is_defective,
            'max_defect_prob': round(float(max_defect_prob), 4),
            'latency_ms': round(latency_ms, 2),
            'interpretation': diagnosis
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run defect inference on a solar panel image")
    parser.add_argument('--image', type=str, required=True, help="Path to input image")
    parser.add_argument('--checkpoint', type=str, default=config.BASELINE_CKPT, help="Path to model checkpoint")
    args = parser.parse_args()

    predictor = SolarSentinelPredictor(checkpoint_path=args.checkpoint)
    res = predictor.predict(args.image)
    print("\nPrediction Result:")
    print(f"Image: {res['image_name']} | Latency: {res['latency_ms']} ms")
    print(f"Diagnosis: {res['interpretation']}")
    print("Probabilities:")
    for c, p in res['probabilities'].items():
        print(f"  {c:15s}: {p:.4f} {'<-- ACTIVE' if p >= config.DECISION_THRESHOLD else ''}")
