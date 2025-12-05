'''
Created on November 24th 2025

@author: Bella Chung

'''

import os
import argparse
import jax.numpy as jn
import numpy as np
import objax
import objax.functional as F # Import F for sigmoid_logistic_loss
from objax.jaxboard import SummaryWriter
import objax.io.checkpoint as objax_ckpt

from autoencoder_models import ConvolutionalAutoencoder, DenselyConnectedAutoencoder
import data_utils

# --- Training Loop Setup ---
class AETrainLoop(objax.Module):
    def __init__(self, model, lr: float = 1e-3, is_image_data: bool = True):
        self.model = model
        self.opt = objax.optimizer.Adam(self.model.vars())
        self.is_image_data = is_image_data # Store the flag
        
        # Define the training step
        @objax.Function.with_vars(self.model.vars())
        def loss_fn(x_in):
            latent, reconstruction = self.model(x_in, training=True)
            
            if self.is_image_data:
                # 1. MSE Loss for Image AE (Tanh output for [-1, 1] range)
                loss = jn.mean(jn.square(x_in - reconstruction))
                metric_name = 'losses/mse'
            else:
                # 2. BCE Loss for Tabular AE (Sigmoid output for [0, 1] range)
                # F.sigmoid_logistic_loss is the stable Objax implementation of BCE
                loss = jn.mean(F.sigmoid_logistic_loss(reconstruction, x_in))
                metric_name = 'losses/bce'
                
            return loss, {metric_name: loss}

        gv = objax.GradValues(loss_fn, self.model.vars())

        @objax.Function.with_vars(self.vars())
        def train_op(x_in):
            g, v = gv(x_in)
            self.opt(lr, g)
            return v[1] # Return metrics

        self.train_op = objax.Jit(train_op)

# --- Main Execution ---
def train_and_extract(dataset_name: str, 
                      epochs: int = 100, 
                      batch_size: int = 128, 
                      latent_dim: int = 256,
                      output_dir: str = 'feature_prep/results'):

    x_all = data_utils.load_and_normalize_data(dataset_name, ['train', 'test'])

    is_image = dataset_name in ['cifar10', 'celeba']

    # 2. Setup Model
    if is_image:
        # Assuming x_all is (N, H, W, C), C is x_all.shape[-1]
        model = ConvolutionalAutoencoder(latent_dim=latent_dim, in_channels=x_all.shape[-1])
    else:
        # Tabular data, D is x_all.shape[-1]
        input_dim = x_all.shape[-1]
        model = DenselyConnectedAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

    # Pass the image flag to the trainer to select the correct loss
    trainer = AETrainLoop(model, is_image_data=is_image)

    # Checkpoint setup
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    # Checkpoint object tracks variables for the model and the optimizer
    checkpoint = objax_ckpt.Checkpoint(ckpt_dir, keep_ckpts=10)

    # Prepare Results Directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Training Loop
    print("Starting training...")
    num_batches = len(x_all) // batch_size
    writer = SummaryWriter(os.path.join(output_dir, f'runs/{dataset_name}_ae')) # Define writer

    for epoch in range(epochs):
        # Shuffle data indices each epoch
        indices = np.random.permutation(len(x_all))
        
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_indices = indices[start:end]
            batch = x_all[batch_indices]

            if is_image:
                # Apply [-1, 1] scaling for AE before training
                batch_scaled = (batch * 2.0) - 1.0
                # Apply the required NHWC->NCHW transpose for image data before training
                batch_input = np.transpose(batch_scaled, (0, 3, 1, 2))
            else:
                batch_input = batch
                
            metrics = trainer.train_op(batch_input)

            # Define the global step
            step = epoch * num_batches + i
            
            # Log metrics to the writer
            if i % 10 == 0: # Log every 10 batches
                loss_value = next(iter(metrics.values())).item()
                metric_name = next(iter(metrics.keys()))

                print(f"Epoch {epoch:03d} | Step {step:05d}/{num_batches * epochs:05d} | {metric_name}: {loss_value:.6f}")

    writer.close()

    # Checkpoint Save
    # Save the model state at the end of training (using epoch + 1 as the step name)
    # This will create the file feature_prep/results/ckpt/0000000100.npz
    checkpoint.save(trainer.vars(), epochs) 
    print(f"Training complete. Checkpoint saved to {ckpt_dir}/{epochs:010d}.npz")

    # 4. Feature Extraction
    print("Extracting features...")
    # Get the encoder function
    encoder_fn = objax.Jit(lambda x: model.encoder(x, training=False)[0], model.vars())
    
    # Process data in batches and collect latent vectors
    features = []
    for i in range(0, len(x_all), batch_size):
        batch = x_all[i:i + batch_size]

        if is_image:
            # Apply [-1, 1] scaling for AE before feature extraction
            batch_scaled = (batch * 2.0) - 1.0
            # Convert NHWC (loaded) -> NCHW (required by CAE model)
            batch_input = np.transpose(batch_scaled, (0, 3, 1, 2)) 
        else:
            # Tabular data is already (N, D)
            batch_input = batch
        
        # features.append(encoder_fn(batch_input).to_numpy())
        jax_array_output = encoder_fn(batch_input)
        features.append(np.asarray(jax_array_output))

    features_array = np.concatenate(features, axis=0)

    # 5. Save Results
    np.save(os.path.join(output_dir, f'{dataset_name}_features.npy'), features_array)
    print(f"Saved features for {dataset_name} to {dataset_name}_features.npy")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--output_dir', type=str, default='feature_prep/results',
                        help='Directory to save logs and extracted features.')
    args = parser.parse_args()
    
    train_and_extract(args.dataset, output_dir=args.output_dir)