"""
Unit Tests for SolarSentinel AI cDCGAN Architecture.
Validates Generator and Discriminator forward passes, conditional embeddings, and checkpoint saving.
"""

import os
import unittest
import torch

import config
from src.gan_model import ConditionalGenerator, ConditionalDiscriminator, build_gan_models

class TestGAN(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.netG, self.netD = build_gan_models(device=self.device)

    def test_generator_output_dimensions(self):
        """Scenario 5: GAN generator produces an image with expected dimensions [B, 3, 64, 64]."""
        batch_size = 4
        noise = torch.randn(batch_size, config.GAN_LATENT_DIM, device=self.device)
        labels = torch.randint(0, config.NUM_CLASSES, (batch_size,), device=self.device)

        fake_imgs = self.netG(noise, labels)
        self.assertEqual(fake_imgs.shape, (batch_size, config.IMAGE_CHANNELS, config.GAN_IMG_SIZE, config.GAN_IMG_SIZE))
        self.assertTrue(torch.isfinite(fake_imgs).all(), "Generator produced non-finite values!")
        self.assertTrue(fake_imgs.min() >= -1.0 and fake_imgs.max() <= 1.0, "Generator output outside [-1, 1] range!")

    def test_discriminator_output_dimensions(self):
        """Discriminator evaluates conditional image and produces scalar logit per sample [B, 1]."""
        batch_size = 4
        dummy_imgs = torch.randn(batch_size, config.IMAGE_CHANNELS, config.GAN_IMG_SIZE, config.GAN_IMG_SIZE, device=self.device)
        labels = torch.randint(0, config.NUM_CLASSES, (batch_size,), device=self.device)

        logits = self.netD(dummy_imgs, labels)
        self.assertEqual(logits.shape, (batch_size, 1))
        self.assertTrue(torch.isfinite(logits).all(), "Discriminator produced non-finite values!")

    def test_checkpoint_saving_and_loading(self):
        """Scenario 6: Checkpoints save and load correctly."""
        temp_ckpt = "temp_gan_test.pth"
        try:
            torch.save({
                'generator_state_dict': self.netG.state_dict(),
                'latent_dim': config.GAN_LATENT_DIM,
                'num_classes': config.NUM_CLASSES
            }, temp_ckpt)
            self.assertTrue(os.path.exists(temp_ckpt))

            loaded = torch.load(temp_ckpt, map_location='cpu')
            new_G = ConditionalGenerator().to(self.device)
            new_G.load_state_dict(loaded['generator_state_dict'])
            self.assertEqual(loaded['latent_dim'], config.GAN_LATENT_DIM)
        finally:
            if os.path.exists(temp_ckpt):
                os.remove(temp_ckpt)

if __name__ == '__main__':
    unittest.main()
