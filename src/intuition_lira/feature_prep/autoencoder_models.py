'''
Created on November 24th 2025

@author: Bella Chung

'''

import objax.nn as nn
import objax.functional as F
from objax.module import Module

# --- Convolutional Autoencoder (CAE) for Images ---

class CAE_Encoder(Module):
    """Encodes an image into a latent vector."""
    def __init__(self, latent_dim: int = 256, in_channels: int = 3):
        self.conv1 = nn.Sequential([
            nn.Conv2D(in_channels, 32, k=3, strides=1, padding='SAME'),
            nn.BatchNorm2D(32),
            F.relu,
            nn.MaxPool2D(k=2, strides=2) # 32x32 -> 16x16
        ])
        self.conv2 = nn.Sequential([
            nn.Conv2D(32, 64, k=3, strides=1, padding='SAME'),
            nn.BatchNorm2D(64),
            F.relu,
            nn.MaxPool2D(k=2, strides=2) # 16x16 -> 8x8
        ])
        self.conv3 = nn.Sequential([
            nn.Conv2D(64, 128, k=3, strides=1, padding='SAME'),
            nn.BatchNorm2D(128),
            F.relu,
            nn.MaxPool2D(k=2, strides=2) # 8x8 -> 4x4
        ])
        # Latent layer: 4*4*128 = 2048 features flattened to latent_dim
        self.fc_mu = nn.Linear(4 * 4 * 128, latent_dim)

    def __call__(self, x, training: bool):
        h = self.conv1(x, training=training)
        h = self.conv2(h, training=training)
        h = self.conv3(h, training=training)
        
        # Flatten and produce latent vector (feature)
        h = h.reshape((h.shape[0], -1))
        latent_features = self.fc_mu(h)
        return latent_features, h.shape # Return latent vector and the shape before flattening

class CAE_Decoder(Module):
    """Decodes a latent vector back into an image."""
    def __init__(self, out_channels: int = 3, latent_dim: int = 256):
        # We need the pre-flattening size from the encoder: 4x4x128
        self.fc_decode = nn.Linear(latent_dim, 4 * 4 * 128)
        
        self.deconv1 = nn.Sequential([
            nn.Conv2DTranspose(128, 64, k=3, strides=2, padding='SAME'), # 4x4 -> 8x8
            nn.BatchNorm2D(64),
            F.relu
        ])
        self.deconv2 = nn.Sequential([
            nn.Conv2DTranspose(64, 32, k=3, strides=2, padding='SAME'), # 8x8 -> 16x16
            nn.BatchNorm2D(32),
            F.relu
        ])
        # Final layer uses Tanh to output reconstructed pixel values
        # We cannot use Sigmoid because the inputs were normalized to [-1, 1]
        self.deconv3 = nn.Sequential([
            nn.Conv2DTranspose(32, out_channels, k=3, strides=2, padding='SAME'), # 16x16 -> 32x32
            F.tanh
        ])

    def __call__(self, z, pre_flatten_shape, training: bool):
        h = self.fc_decode(z)
        # Reshape to 4x4x128 volume before transposed convolutions
        h = h.reshape((h.shape[0], pre_flatten_shape[1], pre_flatten_shape[2], pre_flatten_shape[3]))
        
        h = self.deconv1(h, training=training)
        h = self.deconv2(h, training=training)
        reconstruction = self.deconv3(h, training=training)
        return reconstruction

class ConvolutionalAutoencoder(Module):
    def __init__(self, latent_dim: int = 256, in_channels: int = 3):
        self.encoder = CAE_Encoder(latent_dim, in_channels)
        self.decoder = CAE_Decoder(in_channels, latent_dim)

    def __call__(self, x, training: bool):
        latent, pre_flatten_shape = self.encoder(x, training=training)
        reconstruction = self.decoder(latent, pre_flatten_shape, training=training)
        return latent, reconstruction
    

# --- Densely Connected Autoencoder (DAE) for Tabular Data ---

class DenselyConnectedAutoencoder(Module):
    """
    Autoencoder for non-image, tabular data (like Purchase-100).
    Uses a symmetric architecture of Dense layers.
    """
    def __init__(self, input_dim: int, latent_dim: int = 128):
        # Encoder: Reduce dimensions to latent_dim
        self.encoder_layers = nn.Sequential([
            nn.Linear(input_dim, input_dim // 2),
            F.relu,
            nn.Linear(input_dim // 2, latent_dim),
        ])
        
        # Decoder: Expand dimensions back to input_dim
        self.decoder_layers = nn.Sequential([
            nn.Linear(latent_dim, input_dim // 2),
            F.relu,
            nn.Linear(input_dim // 2, input_dim),
            # Final layer uses Sigmoid to reconstruct binary hot-coded data
            F.sigmoid
        ])

    def __call__(self, x, training: bool):
        # Flatten is often unnecessary if input is already 1D vector
        latent = self.encoder_layers(x)
        reconstruction = self.decoder_layers(latent)
        return latent, reconstruction