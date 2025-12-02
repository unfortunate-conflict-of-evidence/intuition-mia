'''
Created on November 24th 2025

@author: Bella Chung

'''

import sys
import numpy as np
import os
import argparse
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import json
import warnings
# Suppress NumPy warnings that can occur during normalization of zero vectors (though unlikely here)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def calculate_knn_density(features: np.ndarray, k: int = 20):
    """
    1. L2-normalizes the features.
    2. Calculates the average Euclidean distance to the k-nearest neighbors.
    """
    print("Performing L2-Normalization...")
    # L2-normalize each feature vector (row) to a unit norm.
    # This transforms Euclidean distance into Cosine distance.
    features_norm = features / norm(features, axis=1, keepdims=True)
    
    print(f"Finding {k} nearest neighbors for {len(features)} points...")
    
    # Use brute-force for simplicity and to guarantee the exact k nearest neighbors
    # n_neighbors is k + 1 because the 1st neighbor is always the point itself (distance 0).
    knn = NearestNeighbors(n_neighbors=k + 1, algorithm='brute', metric='euclidean')
    knn.fit(features_norm)
    
    # distances will be shape (N, k+1), the first column (index 0) is distance to self (0).
    distances, _ = knn.kneighbors(features_norm)
    
    # Calculate the average distance to the *true* k-nearest neighbors (excluding self)
    # The higher the average distance, the lower the density (i.e., it's an outlier).
    avg_distance = np.mean(distances[:, 1:], axis=1)
    
    return avg_distance

def select_targets(avg_distance: np.ndarray, M: int, output_dir: str, dataset_name: str):
    """
    Selects M outliers, M inliers, and M midliers based on density scores 
    and saves the indices to a JSON file.
    """
    N = len(avg_distance)
    
    # Sort indices by distance (Outliers have high distance, Inliers have low distance)
    sorted_indices = np.argsort(avg_distance)
    
    # Inliers (Lowest distances - start of the sorted list)
    inlier_indices = sorted_indices[:M].tolist()
    
    # Outliers (Highest distances - end of the sorted list)
    outlier_indices = sorted_indices[N - M:].tolist()
    
    # Midliers (Indices around the median)
    mid_start = (N // 2) - (M // 2)
    mid_end = (N // 2) + (M // 2)
    midlier_indices = sorted_indices[mid_start:mid_end].tolist()

    target_data = {
        'outliers': outlier_indices,
        'inliers': inlier_indices,
        'midliers': midlier_indices,
        'N_total': N,
    }
    
    output_path = os.path.join(output_dir, f'{dataset_name}_target_indices.json')
    with open(output_path, 'w') as f:
        json.dump(target_data, f, indent=4)
        
    print(f"Selected {M} targets per category. Indices saved to {output_path}")
    
    # Select the single most extreme outlier for the Causal Test (Phase 2)
    extreme_outlier_id = outlier_indices[-1]
    
    return extreme_outlier_id

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--features_path', type=str, required=True,
                        help='Path to the numpy file containing extracted features.')
    parser.add_argument('--num_targets', type=int, default=100,
                        help='Number of targets (M) to select for each category.')
    parser.add_argument('--k_neighbors', type=int, default=20,
                        help='Number of neighbors (k) for k-NN distance calculation.')
    parser.add_argument('--output_dir', type=str, default='feature_prep/results')
    args = parser.parse_args()

    features = np.load(args.features_path)
    
    density_scores = calculate_knn_density(features, k=args.k_neighbors)
    extreme_outlier = select_targets(density_scores, args.num_targets, args.output_dir, args.dataset)
    
    # Save the density scores for visualization/further analysis
    np.save(os.path.join(args.output_dir, f'{args.dataset}_density_scores.npy'), density_scores)
    
    print(f"\nPhase 1 Complete.")
    print(f"Extreme Outlier ID selected for Causal Test: {extreme_outlier}")