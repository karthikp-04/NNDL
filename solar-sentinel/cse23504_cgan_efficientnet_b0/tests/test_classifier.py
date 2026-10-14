"""
Unit Tests for SolarSentinel AI EfficientNet-B0 Classifier.
Validates forward pass, output logit dimensions, parameter counts, and dimension resizing.
"""

import unittest
import torch
from PIL import Image

import config
from src.classifier import create_classifier
from src.preprocessing import get_classifier_transforms

class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = create_classifier(num_classes=config.NUM_CLASSES, pretrained=False, device=self.device)
        self.model.eval()

    def test_classifier_output_dimensions(self):
        """Scenario 7: Model produces 8 multi-label logits corresponding to dataset classes."""
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, config.CLASSIFIER_IMG_SIZE, config.CLASSIFIER_IMG_SIZE, device=self.device)
        with torch.no_grad():
            logits = self.model(dummy_input)
            probs = self.model.predict_proba(dummy_input)

        self.assertEqual(logits.shape, (batch_size, config.NUM_CLASSES))
        self.assertEqual(probs.shape, (batch_size, config.NUM_CLASSES))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all(), "Sigmoid probabilities outside [0, 1] range!")

    def test_incorrect_dimension_handling(self):
        """Scenario 3: Non-standard image dimensions (e.g. 1024x1024 or 300x500) handled gracefully by transforms."""
        non_standard_img = Image.new('RGB', (1024, 768), color=(128, 128, 128))
        transform = get_classifier_transforms(is_train=False)
        tensor = transform(non_standard_img)

        self.assertEqual(tensor.shape, (3, config.CLASSIFIER_IMG_SIZE, config.CLASSIFIER_IMG_SIZE))
        with torch.no_grad():
            logits = self.model(tensor.unsqueeze(0))
        self.assertEqual(logits.shape, (1, config.NUM_CLASSES))

    def test_parameter_count(self):
        """Validates parameter counts align with EfficientNet-B0 backbone (~4M parameters)."""
        total, trainable = self.model.count_parameters()
        self.assertGreater(total, 3_500_000)
        self.assertLess(total, 6_000_000)
        self.assertEqual(total, trainable)

if __name__ == '__main__':
    unittest.main()
