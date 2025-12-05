'''
Created on December 2nd 2025

@author: Bella Chung

'''

import torch
import numpy as np
from torchvision.transforms import Compose, Resize, Normalize, InterpolationMode

# The original DINOv2 models are not available directly in Objax/JAX.
# We will use PyTorch, which is standard for DINOv2 models.

# --- ImageNet Standard Mean and Std Dev ---
# Required for DINOv2 and ViT models
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def load_dinov2_model(arch_size: str = 'vit_base'):
    # Load the pre-trained model
    # The official PyTorch hub is the easiest way to load DINOv2.
    repo_id = 'facebookresearch/dinov2'

    # Use a dictionary to map common names to the official hubcallables
    model_map = {
        'vit_base': 'dinov2_vitb14',
        'vit_large': 'dinov2_vitl14',
        'vit_small': 'dinov2_vits14',
        # Add other architectures if needed
    }

    # Get the official callable name, default to the input if not found
    callable_name = model_map.get(arch_size, arch_size)

    # Load the model
    model = torch.hub.load(repo_id, callable_name)
    model.eval() # Set to evaluation mode
    return model

def extract_dinov2_features(images_nhwc: np.ndarray, model):
    # 0. Get the device from the model
    device = next(model.parameters()).device

    # 1. Convert NHWC (NumPy) to NCHW (PyTorch Tensor)
    images_tensor = torch.from_numpy(images_nhwc).permute(0, 3, 1, 2).float()

    # 2. Preprocess Images
    # Apply Resize and Normalize sequentially
    preprocess = Compose([
        # Resize: Upsample 32x32 to 224x224
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        # Normalize: Apply ImageNet Mean/Std Dev
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Apply the transforms across the batch
    # We apply the transforms to each image in the batch using torch.stack
    processed_images = torch.stack([preprocess(img) for img in images_tensor])

    # Move the input batch to the GPU device
    processed_images = processed_images.to(device)

    # 3. Extract Features
    with torch.no_grad():
        # The 'forward_features' method gives the global CLS token feature
        # or the features before the final layer.
        features = model.forward_features(processed_images)['x_norm_clstoken']
        
    return features.cpu().numpy() # Convert back to NumPy