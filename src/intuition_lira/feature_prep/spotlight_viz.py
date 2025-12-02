'''
Created on December 1st 2025

@author: Bella Chung

'''

import numpy as np
import os
import pandas as pd
import tensorflow_datasets as tfds
from renumics import spotlight

def prepare_for_spotlight(dataset_name: str, logdir: str):
    
    # 1. Load Data
    print("Loading data for visualization (Train + Test splits)...")
    
    splits = ['train', 'test']
    images_list = []
    labels_list = []
    
    for split_name in splits:
        # Load the split individually
        ds = tfds.load(dataset_name, split=split_name, shuffle_files=False, as_supervised=True)
        
        # We load as NumPy arrays, keeping them in the normalized float32 format
        for image, label in ds:
            images_list.append(image.numpy().astype(np.float32))
            labels_list.append(label.numpy())
            
    labels = np.array(labels_list)
    
    # Load metadata
    scores = np.load(os.path.join(logdir, f'{dataset_name}_density_scores.npy'))
    features = np.load(os.path.join(logdir, f'{dataset_name}_features.npy'))

    print(f"TFDS Images loaded: {len(images_list)}")
    print(f"NumPy Scores loaded: {len(scores)}")

    # 2. Create DataFrame
    # Spotlight is a tabular viewer, so we pass the data as a Pandas DataFrame
    # Pandas can handle the N-dimensional NumPy arrays when initialized this way, 
    # treating them as a single column of objects.
    df = pd.DataFrame({
        'ID': np.arange(len(images_list)),
        'Label': labels,
        'Density_Score': scores,
        # Image and Latent Vector columns (N-dimensional)
        'Image': list(images_list), # Convert to list of objects to avoid ValueError
        'Latent_Vector': list(features),  # Convert to list of objects to avoid ValueError
    })

    # 3. Launch Spotlight
    # Spotlight should automatically recognize the 'Image' and vector types
    print(f"Launching Renumics Spotlight for {dataset_name}...")
    spotlight.show(df, dtype={'Image': spotlight.Image, 'Latent_Vector': spotlight.Embedding})


if __name__ == '__main__':
    # You may need to run this command from the base directory
    prepare_for_spotlight('cifar10', 'feature_prep/results')