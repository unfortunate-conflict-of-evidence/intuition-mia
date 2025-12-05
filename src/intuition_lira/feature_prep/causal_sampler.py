'''
Created on December 1st 2025

@author: Bella Chung

'''

import numpy as np
import os

def generate_causal_subsets(target_id: int, N_total: int, N_keep: int, K: int, output_dir: str):
    """
    Generates K subsets for each of the four causal piles (total 4*K files).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a consistent seed for reproducibility
    np.random.seed(42) 

    # --- Counters to track files saved for each pile ---
    counts = {
        'Observed_In': 0,
        'Interventional_Out': 0,
        'Observed_Out': 0,
        'Interventional_In': 0,
    }
    
    # We loop until all four counts reach K
    current_iter = 0
    while any(c < K for c in counts.values()):
        
        # 1. Sample a base set S_base
        S_base = np.zeros(N_total, dtype=bool)
        S_base[np.random.choice(N_total, size=N_keep, replace=False)] = True
        
        # --- Case 1: Target X was naturally sampled (Observed In) ---
        if S_base[target_id]: 
            
            # Save Observed_In files only if we haven't reached K yet
            if counts['Observed_In'] < K:
                np.save(os.path.join(output_dir, f'Observed_In_{counts["Observed_In"]}.npy'), S_base)
                counts['Observed_In'] += 1
            
            # Save Interventional_Out files only if we haven't reached K yet
            if counts['Interventional_Out'] < K:
                S_interventional_out = S_base.copy()
                S_interventional_out[target_id] = False 
                np.save(os.path.join(output_dir, f'Interventional_Out_{counts["Interventional_Out"]}.npy'), S_interventional_out)
                counts['Interventional_Out'] += 1

        # --- Case 2: Target X was naturally excluded (Observed Out) ---
        else: 

            # Save Observed_Out files only if we haven't reached K yet
            if counts['Observed_Out'] < K:
                np.save(os.path.join(output_dir, f'Observed_Out_{counts["Observed_Out"]}.npy'), S_base)
                counts['Observed_Out'] += 1
            
            # Save Interventional_In files only if we haven't reached K yet
            if counts['Interventional_In'] < K:
                S_interventional_in = S_base.copy()
                S_interventional_in[target_id] = True
                np.save(os.path.join(output_dir, f'Interventional_In_{counts["Interventional_In"]}.npy'), S_interventional_in)
                counts['Interventional_In'] += 1
                
        current_iter += 1

    total_files = sum(counts.values())
    print(f"Generated {total_files} files (K={K} per pile) for Causal Test.")

TARGET_ID = 36290
N_TOTAL = 50000
N_KEEP = 25000
K_MODELS = 20
OUTPUT_DIR = 'feature_prep/causal_subsets/cifar10'

generate_causal_subsets(TARGET_ID, N_TOTAL, N_KEEP, K_MODELS, OUTPUT_DIR)