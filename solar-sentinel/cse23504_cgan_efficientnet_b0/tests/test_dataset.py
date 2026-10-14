"""
Unit Tests for SolarSentinel AI Dataset and Preprocessing Pipelines.
Validates label dimensions, transform outputs, and data split integrity.
"""

import os
import unittest
import pandas as pd
import numpy as np
import torch
from PIL import Image

import config
from src.preprocessing import create_reproducible_splits, get_classifier_transforms, get_gan_transforms
from src.dataset import PVELADDataset, GANDataset

class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a small synthetic dataframe for deterministic testing
        cls.test_csv_path = "test_mock_index.csv"
        cls.dummy_img_dir = "test_mock_images"
        os.makedirs(cls.dummy_img_dir, exist_ok=True)

        rows = []
        for i in range(10):
            img_path = os.path.join(cls.dummy_img_dir, f"mock_{i}.png")
            # Save small random image
            arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            Image.fromarray(arr).save(img_path)

            record = {'image_path': img_path, 'source': 'mock'}
            for c in config.CLASS_NAMES:
                record[c] = 1 if c == 'crack' and i < 5 else (1 if c == 'good' and i >= 5 else 0)
            rows.append(record)

        cls.df = pd.DataFrame(rows)
        cls.df.to_csv(cls.test_csv_path, index=False)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_csv_path):
            os.remove(cls.test_csv_path)
        if os.path.exists(cls.dummy_img_dir):
            import shutil
            shutil.rmtree(cls.dummy_img_dir)

    def test_dataset_label_dimensions(self):
        """Scenario 7: Dataset labels match model output dimensions (8 classes)."""
        dataset = PVELADDataset(self.df, transform=get_classifier_transforms(is_train=False))
        self.assertEqual(len(dataset), 10)
        img, label = dataset[0]
        self.assertEqual(label.shape[0], config.NUM_CLASSES)
        self.assertEqual(img.shape, (3, config.CLASSIFIER_IMG_SIZE, config.CLASSIFIER_IMG_SIZE))

    def test_gan_dataset_dimensions(self):
        """Validates GANDataset yields correct condition label and image resolution."""
        dataset = GANDataset(self.df, transform=get_gan_transforms(config.GAN_IMG_SIZE))
        img, cond = dataset[0]
        self.assertEqual(img.shape, (3, config.GAN_IMG_SIZE, config.GAN_IMG_SIZE))
        self.assertIsInstance(cond.item(), int)
        self.assertTrue(0 <= cond.item() < config.NUM_CLASSES)

    def test_zero_leakage_in_splits(self):
        """Scenario 9: Verify reproducible splits have zero overlap between train, val, test."""
        splits_df = create_reproducible_splits(index_csv=self.test_csv_path, splits_csv="test_splits_temp.csv")
        try:
            train_paths = set(splits_df[splits_df['split'] == 'train']['image_path'])
            val_paths = set(splits_df[splits_df['split'] == 'val']['image_path'])
            test_paths = set(splits_df[splits_df['split'] == 'test']['image_path'])

            # Verify mutual exclusivity
            self.assertEqual(len(train_paths.intersection(val_paths)), 0, "Leakage between train and val!")
            self.assertEqual(len(train_paths.intersection(test_paths)), 0, "Leakage between train and test!")
            self.assertEqual(len(val_paths.intersection(test_paths)), 0, "Leakage between val and test!")
        finally:
            if os.path.exists("test_splits_temp.csv"):
                os.remove("test_splits_temp.csv")

if __name__ == '__main__':
    unittest.main()
