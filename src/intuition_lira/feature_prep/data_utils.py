'''
Created on November 24th 2025

@author: Bella Chung

'''

import numpy as np
import os
import tensorflow_datasets as tfds

def load_and_normalize_data(dataset_name: str, split: str = 'train'):
    """
    Loads the full, normalized dataset as a NumPy array (NHWC or flat).

    The 'split' argument can be a single string (e.g., 'train') or a list
    of strings (e.g., ['train', 'test']) to load combined splits.
    """

    splits_to_load = [split] if isinstance(split, str) else split

    if dataset_name in ['cifar10', 'cifar100']:
        tfds_name = 'cifar10' if dataset_name == 'cifar10' else 'cifar100'
        
        all_images = []
        for s in splits_to_load:
            # TFDS allows loading multiple splits in one call using a list, 
            # but loading separately is cleaner for concatenation here.
            data = tfds.load(tfds_name, split=s, shuffle_files=False, as_supervised=True)
            
            images = []
            for image, _ in data: # Discard the label
                images.append(image.numpy())
            all_images.extend(images)
            
        inputs = np.array(all_images)
        
        # Apply the consistent image normalization: [0, 255] -> [-1, 1]
        # Images are uint8, so convert to float first
        inputs = inputs.astype(np.float32)
        inputs = (inputs / 127.5) - 1.0 # NHWC format (N, 32, 32, 3)
        
    elif dataset_name == 'celeba':
        all_images = []
        for s in splits_to_load:
            data = tfds.load('celeba', split=s, shuffle_files=False, as_supervised=False, 
                             data_dir=os.environ.get('TFDS_DIR', '~/TFDS'))

            images = []
            for example in data:
                images.append(example['image'].numpy())
            all_images.extend(images)
            
        inputs = np.array(all_images)

        # NOTE: If CelebA images are larger (e.g., 218x178), you MUST resize/crop
        # them to 32x32 to match your ConvolutionalAutoencoder architecture.
        # Example: inputs = resize_and_crop(inputs, target_size=(32, 32))
        
        # Apply the consistent image normalization: [0, 255] -> [-1, 1]
        inputs = inputs.astype(np.float32)
        inputs = (inputs / 127.5) - 1.0
        
    elif dataset_name == 'purchase-100':
        # This is a tabular dataset, typically loaded from an NPZ file
        # You will need to determine the specific path where Purchase-100 is stored
        data = np.load('path/to/purchase100.npz') # Path needs adjustment
        inputs = data['features'].astype(np.float32)
        
        # Tabular data is typically standardized or min-max scaled, 
        # NOT by 127.5. Apply the standard preprocessing used in the MIA literature 
        # for this dataset (e.g., standard scaling).
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return inputs