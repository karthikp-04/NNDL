"""
Unit Tests for SolarSentinel AI Defect Prediction Engine.
Validates inference execution, error handling for corrupted inputs, and missing checkpoint handling.
"""

import os
import unittest
import torch
from PIL import Image

import config
from src.predict import SolarSentinelPredictor

class TestPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid_img_path = "test_valid_solar.png"
        cls.invalid_txt_path = "test_invalid_file.txt"

        # Create valid image
        img = Image.new('RGB', (224, 224), color=(60, 60, 60))
        img.save(cls.valid_img_path)

        # Create invalid non-image file
        with open(cls.invalid_txt_path, 'w') as f:
            f.write("This is not an image file.")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.valid_img_path):
            os.remove(cls.valid_img_path)
        if os.path.exists(cls.invalid_txt_path):
            os.remove(cls.invalid_txt_path)

    def test_missing_checkpoint_handling(self):
        """Scenario 4: Missing model checkpoints produce a clear message and raise on predict."""
        predictor = SolarSentinelPredictor(checkpoint_path="non_existent_model_checkpoint.pth")
        self.assertIsNone(predictor.model)
        with self.assertRaises(RuntimeError):
            predictor.predict(self.valid_img_path)

    def test_invalid_file_format_handling(self):
        """Scenario 2: Invalid file formats or unreadable files are rejected gracefully."""
        # Using existing checkpoint if available, or create dummy checkpoint
        ckpt_path = config.BASELINE_CKPT if os.path.exists(config.BASELINE_CKPT) else "dummy_test_ckpt.pth"
        if not os.path.exists(ckpt_path):
            from src.classifier import create_classifier
            m = create_classifier(pretrained=False)
            torch.save({'model_state_dict': m.state_dict()}, ckpt_path)

        predictor = SolarSentinelPredictor(checkpoint_path=ckpt_path)
        with self.assertRaises(Exception):
            predictor.predict(self.invalid_txt_path)

    def test_valid_image_prediction(self):
        """Scenario 1: Valid EL image produces prediction dictionary when weights are available."""
        ckpt_path = config.BASELINE_CKPT if os.path.exists(config.BASELINE_CKPT) else "dummy_test_ckpt.pth"
        if not os.path.exists(ckpt_path):
            from src.classifier import create_classifier
            m = create_classifier(pretrained=False)
            torch.save({'model_state_dict': m.state_dict()}, ckpt_path)

        predictor = SolarSentinelPredictor(checkpoint_path=ckpt_path)
        res = predictor.predict(self.valid_img_path)

        self.assertIn('probabilities', res)
        self.assertEqual(len(res['probabilities']), config.NUM_CLASSES)
        self.assertIn('latency_ms', res)
        self.assertGreater(res['latency_ms'], 0.0)
        self.assertIn('interpretation', res)
        self.assertIsInstance(res['is_defective'], bool)

if __name__ == '__main__':
    unittest.main()
