"""
Integration & End-to-End Pipeline Tests for SolarSentinel AI.
Validates held-out evaluation protocol, absence of synthetic leakage, and application imports.
"""

import os
import unittest
import pandas as pd

import config
from src.preprocessing import create_reproducible_splits

class TestPipeline(unittest.TestCase):
    def test_held_out_test_set_split_sizes(self):
        """Scenario 8: Evaluation runs on held-out test set with expected ratio."""
        if os.path.exists(config.DATASET_INDEX_CSV):
            df_splits = create_reproducible_splits()
            test_df = df_splits[df_splits['split'] == 'test']
            self.assertGreater(len(test_df), 1000, "Held-out test set too small!")
            self.assertEqual(len(df_splits), 13414, "Unexpected total dataset size!")

    def test_no_synthetic_leakage_in_test_set(self):
        """Scenario 9: GAN augmentation does not introduce test-set leakage."""
        manifest_path = os.path.join(config.GENERATED_SAMPLES_DIR, "synthetic_manifest.csv")
        if os.path.exists(manifest_path) and os.path.exists(config.SPLITS_INDEX_CSV):
            syn_df = pd.read_csv(manifest_path)
            splits_df = pd.read_csv(config.SPLITS_INDEX_CSV)
            test_paths = set(splits_df[splits_df['split'] == 'test']['image_path'])
            syn_paths = set(syn_df['image_path'])

            # Ensure zero intersection
            overlap = test_paths.intersection(syn_paths)
            self.assertEqual(len(overlap), 0, f"Critical Leakage: {len(overlap)} synthetic images found in test split!")

    def test_streamlit_app_import_cleanliness(self):
        """Scenario 10: The web application modules import cleanly without runtime syntax or import errors."""
        import streamlit
        import PIL.Image
        import matplotlib.pyplot
        import seaborn
        import src.predict
        import src.gan_model
        import src.classifier
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
