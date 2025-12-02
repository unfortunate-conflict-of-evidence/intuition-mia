'''
Created on November 29th 2025

@author: Bella Chung

'''

import numpy as np
import os
import argparse
import objax
import jax.numpy as jn
import matplotlib.pyplot as plt

# We need the model definitions
from autoencoder_models import ConvolutionalAutoencoder 
# We need the utility for loading data
import data_utils 
import objax.functional as F
from objax.util import EasyDict
from objax.module import Module

# --- AETrainLoop Class Definition (COPIED from extract_features.py) ---
# NOTE: This class is needed because the saved checkpoint includes its optimizer variables.

class AETrainLoop(objax.Module):
    def __init__(self, model, lr: float = 1e-3, is_image_data: bool = True):
        self.model = model
        # Initialize the optimizer, even though we won't use it for reconstruction
        self.opt = objax.optimizer.Adam(self.model.vars())
        self.is_image_data = is_image_data
        
        # Define a placeholder loss_fn/train_op, primarily to define the variables
        @objax.Function.with_vars(self.model.vars())
        def loss_fn(x_in):
            latent, reconstruction = self.model(x_in, training=True)
            loss = jn.mean(jn.square(x_in - reconstruction)) if self.is_image_data else jn.mean(F.sigmoid_logistic_loss(reconstruction, x_in))
            return loss, {}

        gv = objax.GradValues(loss_fn, self.model.vars())
        
        # Placeholder train_op:
        @objax.Function.with_vars(self.vars())
        def train_op(x_in):
            g, v = gv(x_in)
            self.opt(lr, g)
            return v[1]

        self.train_op = objax.Jit(train_op)
        
# --- END AETrainLoop ---


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Converts image from [-1, 1] range back to [0, 255] and converts to uint8."""
    # Formula: I_original = ((I_reconstructed + 1) / 2) * 255
    image = (image + 1.0) / 2.0  # Scale to [0, 1]
    image = image * 255.0        # Scale to [0, 255]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def visualize_reconstruction(dataset_name: str, 
                             checkpoint_path: str, # e.g., 'feature_prep/results/ckpt/0000000100.npz'
                             num_samples: int = 8):
    
    # 1. Load Data
    x_all = data_utils.load_and_normalize_data(dataset_name, 'train')
    
    indices = np.random.choice(len(x_all), num_samples, replace=False)
    sample_batch_nhwc = x_all[indices]
    
    # 2. Setup Model and Restore Weights
    C = x_all.shape[-1]
    
    # Instantiate the base model
    model = ConvolutionalAutoencoder(in_channels=C)

    # Instantiate the wrapper module (AETrainLoop) which holds model + optimizer variables
    # We must set is_image_data=True for image AEs (CIFAR10)
    trainer = AETrainLoop(model, is_image_data=True) 

    # --- FIX 1: Corrected Path Calculation ---
    # Given path: 'feature_prep/results/ckpt/0000000100.npz'
    # We need the logdir for Checkpoint to be 'feature_prep/results' 
    # to avoid the doubled 'ckpt/ckpt' error.

    # 2a. Calculate the base directory ('feature_prep/results')
    # os.path.dirname(os.path.dirname(checkpoint_path)) goes up two levels.
    ckpt_dir = os.path.dirname(os.path.dirname(checkpoint_path)) 
    
    # 2b. Extract the integer step/epoch
    filename = os.path.basename(checkpoint_path)
    epoch_index = int(filename.split('.')[0])
    
    # 2c. Instantiate Checkpoint with the corrected base directory
    checkpoint = objax.io.Checkpoint(ckpt_dir, keep_ckpts=1)
    
    # --- FIX 2: Variable Restoration Fix ---
    # Restore ALL saved variables (model + optimizer) into the 'trainer' module.
    print(f"Attempting to restore trainer variables from index {epoch_index} in directory {ckpt_dir}...")
    checkpoint.restore(trainer.vars(), epoch_index)
    # The weights for 'model' are now loaded inside 'trainer.model'

    # 3. Create JIT-compiled Reconstruction Function
    # The model we use is now trainer.model
    reconstruct_fn = objax.Jit(lambda x: trainer.model(x, training=False)[1], trainer.model.vars()) 
    
    # Transpose input for the model: NHWC -> NCHW
    sample_batch_nchw = np.transpose(sample_batch_nhwc, (0, 3, 1, 2))

    # Perform reconstruction
    reconstruction_nchw = np.asarray(reconstruct_fn(sample_batch_nchw))
    
    # Transpose back for visualization: NCHW -> NHWC
    reconstruction_nhwc = np.transpose(reconstruction_nchw, (0, 2, 3, 1))

    # 4. Plotting
    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))
    
    for i in range(num_samples):
        original_img = denormalize_image(sample_batch_nhwc[i])
        reconstructed_img = denormalize_image(reconstruction_nhwc[i])
        
        # Original Row
        axes[0, i].imshow(original_img)
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')

        # Reconstruction Row
        axes[1, i].imshow(reconstructed_img)
        axes[1, i].set_title(f"Recon {i+1}")
        axes[1, i].axis('off')

    plt.suptitle(f"Autoencoder Reconstruction Check: {dataset_name}")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to the final checkpoint file (e.g., feature_prep/results/ckpt/0000000100.npz)')
    args = parser.parse_args()
    
    visualize_reconstruction(args.dataset, args.ckpt_path)