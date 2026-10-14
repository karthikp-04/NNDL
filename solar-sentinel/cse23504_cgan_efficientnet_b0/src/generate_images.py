"""
SolarSentinel AI — Synthetic Image Generator & Quality Filtering
Loads the trained cDCGAN Generator and produces quality-validated synthetic EL defect images
for data augmentation in Experiment B.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import pandas as pd
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
from torchvision.transforms.functional import resize
from PIL import Image
from tqdm import tqdm

import config
from src.preprocessing import denormalize_gan_tensor, validate_synthetic_tensor
from src.gan_model import ConditionalGenerator

def load_generator(ckpt_path=config.GENERATOR_CKPT, device=config.DEVICE):
    """
    Loads trained Conditional Generator from checkpoint.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Generator checkpoint not found at {ckpt_path}. Train the GAN first.")

    checkpoint = torch.load(ckpt_path, map_location=device)
    netG = ConditionalGenerator(
        latent_dim=checkpoint.get('latent_dim', config.GAN_LATENT_DIM),
        num_classes=checkpoint.get('num_classes', config.NUM_CLASSES),
        embed_dim=checkpoint.get('embed_dim', config.GAN_EMBED_DIM)
    ).to(device)
    netG.load_state_dict(checkpoint['generator_state_dict'])
    netG.eval()
    return netG

def generate_synthetic_samples(
    num_per_defect=50,
    classes=config.MINORITY_CLASSES,
    output_dir=config.SYNTHETIC_DATASET_DIR,
    manifest_csv=os.path.join(config.GENERATED_SAMPLES_DIR, "synthetic_manifest.csv"),
    device=config.DEVICE,
    random_seed=config.RANDOM_SEED
):
    """
    Generates, quality-filters, and saves synthetic EL defect images.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print("=" * 70)
    print("SOLARSENTINEL AI — SYNTHETIC DEFECT IMAGE GENERATION")
    print(f"Target classes: {classes} | Images per class: {num_per_defect}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    netG = load_generator(device=device)
    os.makedirs(output_dir, exist_ok=True)

    manifest_records = []
    generated_tensors = []

    for cname in classes:
        cls_idx = config.CLASS_TO_IDX[cname]
        cls_dir = os.path.join(output_dir, cname)
        os.makedirs(cls_dir, exist_ok=True)

        accepted_count = 0
        attempts = 0
        max_attempts = num_per_defect * 3

        pbar = tqdm(total=num_per_defect, desc=f"Generating '{cname}'", leave=True)
        while accepted_count < num_per_defect and attempts < max_attempts:
            attempts += 1
            z = torch.randn(1, config.GAN_LATENT_DIM, device=device)
            c = torch.tensor([cls_idx], dtype=torch.long, device=device)

            with torch.no_grad():
                raw_tensor = netG(z, c)
                denorm_tensor = denormalize_gan_tensor(raw_tensor).squeeze(0)  # (3, 64, 64)

            # Quality validation
            is_valid, reason = validate_synthetic_tensor(denorm_tensor)
            if not is_valid:
                continue

            # Resize to 224x224 RGB for classifier pipeline
            denorm_tensor_224 = resize(denorm_tensor, [config.CLASSIFIER_IMG_SIZE, config.CLASSIFIER_IMG_SIZE], antialias=True)
            
            # Save as PNG
            img_filename = f"syn_{cname}_{accepted_count:04d}.png"
            img_path = os.path.join(cls_dir, img_filename)
            save_image(denorm_tensor_224, img_path)

            variance_val = torch.var(denorm_tensor).item()

            manifest_records.append({
                'image_path': img_path,
                'class_name': cname,
                'class_idx': cls_idx,
                'is_synthetic': True,
                'variance': round(variance_val, 5),
                'status': 'accepted'
            })

            if accepted_count < 4:
                generated_tensors.append(denorm_tensor)

            accepted_count += 1
            pbar.update(1)

        pbar.close()
        print(f"Class '{cname}': Generated {accepted_count} images (Yield rate: {accepted_count/attempts:.1%})")

    # Save Manifest CSV
    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"\nSynthetic dataset manifest saved to {manifest_csv} ({len(manifest_df)} total images)")

    # Save composite preview grid
    if generated_tensors:
        preview_grid = make_grid(torch.stack(generated_tensors), nrow=4, padding=4, normalize=False)
        preview_path = os.path.join(config.GENERATED_SAMPLES_DIR, "synthetic_augmentation_grid.png")
        save_image(preview_grid, preview_path)
        print(f"Augmentation preview grid saved to {preview_path}")

    return manifest_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic solar panel images")
    parser.add_argument('--num-per-class', type=int, default=30, help="Number of synthetic images per minority defect")
    parser.add_argument('--classes', nargs='+', default=config.MINORITY_CLASSES, help="Defect classes to generate")
    args = parser.parse_args()

    generate_synthetic_samples(num_per_defect=args.num_per_class, classes=args.classes)
