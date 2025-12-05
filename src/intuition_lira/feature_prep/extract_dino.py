'''
Created on December 4th 2025

@author: Bella Chung

'''

import os
import argparse
import numpy as np
import torch

import dinov2_extractor
import data_utils

DEFAULT_BATCH_SIZE = 32

def run_dino_extraction(dataset_name: str, 
                        output_dir: str, 
                        batch_size: int = DEFAULT_BATCH_SIZE):

    # 1. Prepare Results Directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output features will be saved to: {output_dir}")

    # 2. Load and Normalize Data
    print(f"Loading and normalizing {dataset_name} data...")
    # x_all contains both 'train' and 'test' splits, normalized to [0, 1]
    x_all = data_utils.load_and_normalize_data(dataset_name, ['train', 'test'])
    print(f"Data loaded successfully. Total shape: {x_all.shape}")

    # 3. Load DINOv2 Model (PyTorch)
    print("Using pre-trained DINOv2 (ViT-Base) for feature extraction.")
    # dinov2_extractor.load_dinov2_model handles caching/downloading if needed
    dinov2_model = dinov2_extractor.load_dinov2_model('vit_base') 
    
    # Move model to GPU if available and set to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov2_model.to(device)
    dinov2_model.eval()
    
    # 4. Feature Extraction Loop
    print("Extracting features in batches...")
    features = []
    
    num_batches = int(np.ceil(len(x_all) / batch_size))
    
    for i in range(0, len(x_all), batch_size):
        batch = x_all[i:i + batch_size]
        
        # Convert NumPy batch to PyTorch tensor on the correct device
        # DINOv2 extractor handles the rest of the preprocessing (Resize, Normalize, NCHW)
        dinov2_output = dinov2_extractor.extract_dinov2_features(batch, dinov2_model)
        features.append(dinov2_output)
        
        # Simple progress tracking
        if (i // batch_size) % 100 == 0:
            print(f"  Processed batch {i // batch_size + 1}/{num_batches}")

    features_array = np.concatenate(features, axis=0)
    print(f"Extraction complete. Final feature array shape: {features_array.shape}")
    
    # 5. Save Results
    feature_file_name = f'{dataset_name}_dino_features.npy'
    output_path = os.path.join(output_dir, feature_file_name)
    
    # Save the feature array
    np.save(output_path, features_array)
    
    print(f"Successfully saved features to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DINOv2 Feature Extraction")
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Dataset to process (e.g., cifar10).')
    parser.add_argument('--output_dir', type=str, default='feature_prep/results',
                        help='Directory to save extracted features.')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size for feature extraction.')
    args = parser.parse_args()
    
    run_dino_extraction(args.dataset, args.output_dir, args.batch_size)