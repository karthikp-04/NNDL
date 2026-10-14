"""
SolarSentinel AI — Conditional DCGAN (cDCGAN) Architecture
Generates class-conditioned electroluminescence (EL) images of solar panel cells.
Includes Generator, Discriminator, custom weight initialization, and utility methods.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

import config

def weights_init(m):
    """
    Standard DCGAN weight initialization:
    Convolutional weights ~ N(0.0, 0.02), BatchNorm weights ~ N(1.0, 0.02), bias = 0.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class ConditionalGenerator(nn.Module):
    """
    Conditional Generator for Solar Cell EL Images.
    Maps noise vector z in R^{latent_dim} and class condition c into 64x64x3 synthetic EL images.
    """
    def __init__(
        self,
        latent_dim=config.GAN_LATENT_DIM,
        num_classes=config.NUM_CLASSES,
        embed_dim=config.GAN_EMBED_DIM,
        ngf=config.GAN_FEATURE_MAP_G,
        out_channels=config.IMAGE_CHANNELS
    ):
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Condition Embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Input dimension: latent_dim + embed_dim
        in_dim = latent_dim + embed_dim

        # Projection to 4x4 spatial feature map
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 8 * 4 * 4),
            nn.BatchNorm1d(ngf * 8 * 4 * 4),
            nn.ReLU(True)
        )

        # Transposed Convolution Layers (4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64)
        self.deconv_blocks = nn.Sequential(
            # State: (ngf*8) x 4 x 4 -> (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # State: (ngf*4) x 8 x 8 -> (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # State: (ngf*2) x 16 x 16 -> ngf x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # State: ngf x 32 x 32 -> out_channels x 64 x 64
            nn.ConvTranspose2d(ngf, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output range [-1, 1]
        )

        self.apply(weights_init)

    def forward(self, noise, labels):
        """
        noise: (B, latent_dim)
        labels: (B,) long tensor
        """
        c_emb = self.label_embedding(labels)  # (B, embed_dim)
        gen_input = torch.cat([noise, c_emb], dim=1)  # (B, latent_dim + embed_dim)

        feat = self.fc(gen_input)
        feat = feat.view(-1, config.GAN_FEATURE_MAP_G * 8, 4, 4)
        img = self.deconv_blocks(feat)
        return img

class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator for Solar Cell EL Images.
    Evaluates whether an image x conditioned on class c is real or synthetically generated.
    """
    def __init__(
        self,
        num_classes=config.NUM_CLASSES,
        img_size=config.GAN_IMG_SIZE,
        ndf=config.GAN_FEATURE_MAP_D,
        in_channels=config.IMAGE_CHANNELS
    ):
        super(ConditionalDiscriminator, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes

        # Spatial condition channel embedding (1 x img_size x img_size)
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)

        # Input channels: 3 (RGB) + 1 (Condition plane) = 4
        total_in_channels = in_channels + 1

        self.conv_blocks = nn.Sequential(
            # (total_in_channels) x 64 x 64 -> ndf x 32 x 32
            nn.Conv2d(total_in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf x 32 x 32 -> (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2) x 16 x 16 -> (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4) x 8 x 8 -> (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*8) x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )

        self.apply(weights_init)

    def forward(self, img, labels):
        """
        img: (B, 3, 64, 64)
        labels: (B,) long tensor
        Returns raw logits (B, 1)
        """
        c_emb = self.label_embedding(labels)  # (B, 64*64)
        c_emb = c_emb.view(-1, 1, self.img_size, self.img_size)  # (B, 1, 64, 64)
        d_input = torch.cat([img, c_emb], dim=1)  # (B, 4, 64, 64)

        out = self.conv_blocks(d_input)
        return out.view(-1, 1)

def build_gan_models(device=config.DEVICE):
    """
    Instantiates both Conditional Generator and Discriminator on the specified device.
    """
    netG = ConditionalGenerator().to(device)
    netD = ConditionalDiscriminator().to(device)
    return netG, netD
