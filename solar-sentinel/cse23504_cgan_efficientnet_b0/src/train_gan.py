"""
SolarSentinel AI — Conditional DCGAN (cDCGAN) Training Pipeline
Trains the Generator and Discriminator on real solar panel EL images conditioned on defect classes.
Saves sample image grids, training history, loss plots, and model checkpoints.
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
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

import config
from src.preprocessing import (
    create_reproducible_splits,
    get_gan_transforms,
    denormalize_gan_tensor
)
from src.dataset import GANDataset
from src.gan_model import (
    ConditionalGenerator,
    ConditionalDiscriminator,
    build_gan_models
)

def plot_gan_losses(history, save_path=os.path.join(config.PLOTS_DIR, "gan_loss_curve.png")):
    """
    Plots Generator and Discriminator adversarial losses over training epochs.
    """
    df = pd.DataFrame(history)
    plt.figure(figsize=(9, 5))
    plt.plot(df['epoch'], df['loss_d'], label='Discriminator Loss (D)', color='#ef4444', lw=2)
    plt.plot(df['epoch'], df['loss_g'], label='Generator Loss (G)', color='#3b82f6', lw=2)
    plt.title('SolarSentinel AI — cDCGAN Adversarial Training Dynamics', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('BCE Loss', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved GAN loss plot to {save_path}")

def save_fixed_samples(netG, fixed_noise, fixed_labels, epoch, save_dir=config.GENERATED_SAMPLES_DIR):
    """
    Generates and saves a grid of synthetic images for fixed noise & labels across all 8 classes.
    """
    netG.eval()
    with torch.no_grad():
        fake_imgs = netG(fixed_noise, fixed_labels)
        fake_imgs = denormalize_gan_tensor(fake_imgs)
        grid = make_grid(fake_imgs, nrow=config.NUM_CLASSES, padding=2, normalize=False)
        out_path = os.path.join(save_dir, f"progress_grid_epoch_{epoch:03d}.png")
        save_image(grid, out_path)
    netG.train()
    return out_path

def train_gan(
    epochs=config.GAN_EPOCHS,
    batch_size=config.GAN_BATCH_SIZE,
    smoke_test=False,
    subset_size=None,
    focus_defects=True,
    device=config.DEVICE
):
    """
    Main cDCGAN training routine.
    """
    print("=" * 70)
    print("SOLARSENTINEL AI — CONDITIONAL DCGAN TRAINING")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    if smoke_test:
        print(">>> RUNNING IN SMOKE-TEST MODE (Fast verification subset) <<<")
    print("=" * 70)

    # 1. Load Dataset
    splits_df = create_reproducible_splits()
    train_df = splits_df[splits_df['split'] == 'train'].reset_index(drop=True)

    if smoke_test:
        train_df = train_df.sample(n=min(128, len(train_df)), random_state=config.RANDOM_SEED).reset_index(drop=True)
        epochs = min(epochs, 2)
    elif subset_size and subset_size < len(train_df):
        train_df = train_df.sample(n=subset_size, random_state=config.RANDOM_SEED).reset_index(drop=True)

    gan_dataset = GANDataset(
        train_df,
        transform=get_gan_transforms(config.GAN_IMG_SIZE),
        focus_defects_only=focus_defects
    )
    dataloader = DataLoader(
        gan_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True
    )
    print(f"cDCGAN Training Samples: {len(gan_dataset):,} | Batches per epoch: {len(dataloader)}")

    # 2. Instantiate Models
    netG, netD = build_gan_models(device=device)

    # 3. Setup Optimizers & Loss Function
    optimizerD = torch.optim.Adam(netD.parameters(), lr=config.GAN_LR, betas=(config.GAN_BETA1, config.GAN_BETA2))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=config.GAN_LR, betas=(config.GAN_BETA1, config.GAN_BETA2))
    criterion = nn.BCEWithLogitsLoss()

    # Fixed evaluation noise: 2 samples per class (16 total images)
    num_samples = config.NUM_CLASSES * 2
    fixed_noise = torch.randn(num_samples, config.GAN_LATENT_DIM, device=device)
    fixed_labels = torch.tensor([cls for cls in range(config.NUM_CLASSES) for _ in range(2)], dtype=torch.long, device=device)

    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        running_loss_d = 0.0
        running_loss_g = 0.0

        pbar = tqdm(dataloader, desc=f"cGAN Epoch {epoch:02d}/{epochs:02d}", leave=True)
        for real_imgs, real_labels in pbar:
            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)

            # Targets: Label smoothing for real images (0.9), fake images (0.0)
            real_target = torch.full((b_size, 1), config.GAN_LABEL_SMOOTHING, dtype=torch.float32, device=device)
            fake_target = torch.zeros((b_size, 1), dtype=torch.float32, device=device)

            # ----------------------------------------------------
            # 1. Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # ----------------------------------------------------
            optimizerD.zero_grad()

            # Real batch forward
            d_real_out = netD(real_imgs, real_labels)
            loss_d_real = criterion(d_real_out, real_target)

            # Fake batch forward
            noise = torch.randn(b_size, config.GAN_LATENT_DIM, device=device)
            gen_labels = torch.randint(0, config.NUM_CLASSES, (b_size,), device=device)
            fake_imgs = netG(noise, gen_labels)

            d_fake_out = netD(fake_imgs.detach(), gen_labels)
            loss_d_fake = criterion(d_fake_out, fake_target)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizerD.step()

            # ----------------------------------------------------
            # 2. Update Generator: maximize log(D(G(z)))
            # ----------------------------------------------------
            optimizerG.zero_grad()

            d_fake_for_g = netD(fake_imgs, gen_labels)
            loss_g = criterion(d_fake_for_g, real_target)  # Generator tricks discriminator into thinking images are real

            loss_g.backward()
            optimizerG.step()

            running_loss_d += loss_d.item()
            running_loss_g += loss_g.item()

            pbar.set_postfix({'Loss_D': f"{loss_d.item():.4f}", 'Loss_G': f"{loss_g.item():.4f}"})

        epoch_loss_d = running_loss_d / max(1, len(dataloader))
        epoch_loss_g = running_loss_g / max(1, len(dataloader))
        ep_duration = time.time() - ep_start

        history.append({
            'epoch': epoch,
            'loss_d': round(epoch_loss_d, 4),
            'loss_g': round(epoch_loss_g, 4),
            'duration_sec': round(ep_duration, 1)
        })

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_duration:.1f}s) | Loss_D: {epoch_loss_d:.4f} | Loss_G: {epoch_loss_g:.4f}")

        # Save progress visual samples
        sample_path = save_fixed_samples(netG, fixed_noise, fixed_labels, epoch)
        print(f"  --> Saved sample grid: {sample_path}")

    total_training_time = time.time() - start_time
    print(f"\ncDCGAN Training Complete! Total Duration: {total_training_time:.1f}s")

    # 4. Save Checkpoints
    torch.save({
        'epoch': epochs,
        'generator_state_dict': netG.state_dict(),
        'latent_dim': config.GAN_LATENT_DIM,
        'num_classes': config.NUM_CLASSES,
        'embed_dim': config.GAN_EMBED_DIM
    }, config.GENERATOR_CKPT)
    print(f"Saved Generator checkpoint: {config.GENERATOR_CKPT}")

    torch.save({
        'epoch': epochs,
        'discriminator_state_dict': netD.state_dict(),
        'num_classes': config.NUM_CLASSES,
        'img_size': config.GAN_IMG_SIZE
    }, config.DISCRIMINATOR_CKPT)
    print(f"Saved Discriminator checkpoint: {config.DISCRIMINATOR_CKPT}")

    # 5. Save History & Plot
    history_df = pd.DataFrame(history)
    history_df.to_csv(config.GAN_HISTORY_CSV, index=False)
    plot_gan_losses(history)

    return netG, netD, history_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SolarSentinel cDCGAN")
    parser.add_argument('--epochs', type=int, default=config.GAN_EPOCHS, help="Number of epochs")
    parser.add_argument('--batch-size', type=int, default=config.GAN_BATCH_SIZE, help="Batch size")
    parser.add_argument('--smoke-test', action='store_true', help="Fast smoke test on small subset")
    parser.add_argument('--subset-size', type=int, default=None, help="Train on custom subset size")
    parser.add_argument('--all-classes', action='store_true', help="Train on all images including non-defective")
    args = parser.parse_args()

    train_gan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        smoke_test=args.smoke_test,
        subset_size=args.subset_size,
        focus_defects=not args.all_classes
    )
