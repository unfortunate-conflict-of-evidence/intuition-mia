'''
Created on June 3rd 2025

@author: Bella Chung

Based on https://github.com/AhmedSalem2/ML-Leaks/blob/master/mlLeaks.py

'''

import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from tensorflow import keras

labelsCIFAR10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def readCIFAR10(data_path, normalize=True, train_ratio=0.8, random_state=None):
	'''
	Reads and formats downloaded CIFAR-10 data from a specified filepath.
	Returns image data in (num_images, height, width, channels) format.
	Also returns lists of unique IDs for train and test images.
	Allows for custom train:test split.
	
	Args:
	    data_path: path to the CIFAR-10 data folder
		normalize: If True, returns float data in [0, 1] range.
                   If False, returns uint8 data in [0, 255] range.
		train_ratio: ratio of training to test images [0, 1]
		random_state: used to fix shuffling and splitting
		
	Returns:
        tuple: (train_images, train_labels, train_ids,
		        test_images, test_labels, test_ids)
	'''
	X_all_train = []
	y_all_train = []
	
	for i in range(5):
		f = open(data_path + '/data_batch_' + str(i + 1), 'rb')
		train_data_dict = pickle.load(f, encoding='bytes')
		f.close()

		X_all_train.append(train_data_dict[b'data'])
		y_all_train.append(train_data_dict[b'labels'])
		
	X_all_train = np.concatenate(X_all_train, axis=0)
	y_all_train = np.concatenate(y_all_train, axis=0)
		
	f = open(data_path + '/test_batch', 'rb')
	test_data_dict = pickle.load(f, encoding='bytes')
	f.close()
	
	XTest = np.array(test_data_dict[b'data'])
	yTest = np.array(test_data_dict[b'labels'])
	
    # Combine all data
	X_combined = np.concatenate((X_all_train, XTest), axis=0)
	y_combined = np.concatenate((y_all_train, yTest), axis=0)

	if normalize:
		X_combined = X_combined / 255.0
		
    # Generate unique IDs for all images before shuffling
	image_ids = np.arange(len(X_combined))

    # Split data and IDs
	X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_combined, y_combined, image_ids, train_size=train_ratio, random_state=random_state, stratify=y_combined
    )
	return (X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), y_train, ids_train,
            X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), y_test, ids_test)

def loadCIFAR(dataset='cifar10', normalize=True, train_ratio=0.8, random_state=None):
    '''
    Formats CIFAR data loaded from keras. Slower than readCIFAR10.
    Returns image data in (num_images, height, width, channels) format.
    Also returns lists of unique IDs for train and test images.
    Allows for custom train:test split.

    Args:
        dataset: 'cifar10' or 'cifar100'
        normalize: If True, returns float data in [0, 1] range.
                    If False, returns uint8 data in [0, 255] range.
        train_ratio: ratio of training to test images [0, 1]
        random_state: used to fix shuffling and splitting
        
    Returns:
        tuple: (train_images, train_labels, train_ids,
                test_images, test_labels, test_ids)
    '''
    if dataset == 'cifar10':
        dataset_loader = keras.datasets.cifar10
    elif dataset == 'cifar100':
        dataset_loader = keras.datasets.cifar100
    else:
        raise ValueError("Unsupported dataset. Choose 'cifar10' or 'cifar100'.")

    (x_train, y_train), (x_test, y_test) = dataset_loader.load_data()

    # Combine all data
    X_combined = np.concatenate((x_train, x_test), axis=0)
    y_combined = np.concatenate((y_train, y_test), axis=0).flatten()

    if normalize:
        X_combined = X_combined / 255.0

    # Generate unique IDs for all images before shuffling
    image_ids = np.arange(len(X_combined))
        
    # Split data and IDs
    x_train, x_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_combined, y_combined, image_ids, train_size=train_ratio, random_state=random_state, stratify=y_combined
    )
    return x_train, y_train, ids_train, x_test, y_test, ids_test

def loadFullCIFAR(dataset='cifar10', normalize=True):
    '''
    Loads the full CIFAR dataset without splitting it into train/test.
    Returns the concatenated images, labels, and unique IDs.
	Image data is in (num_images, height, width, channels) format.
	
	Args:
        dataset: 'cifar10' or 'cifar100'
		normalize: If True, returns float data in [0, 1] range.
		
	Returns:
        tuple: (images, labels, ids)
    '''
    if dataset == 'cifar10':
        dataset_loader = keras.datasets.cifar10
    elif dataset == 'cifar100':
        dataset_loader = keras.datasets.cifar100
    else:
        raise ValueError("Unsupported dataset. Choose 'cifar10' or 'cifar100'.")

    (x_train, y_train), (x_test, y_test) = dataset_loader.load_data()
    
    full_x = np.concatenate((x_train, x_test), axis=0)
    full_y = np.concatenate((y_train, y_test), axis=0).flatten()
    
    num_samples = len(full_y)
    full_ids = np.arange(num_samples)
	
    if normalize:
        full_x = full_x / 255.0
		
    return full_x, full_y, full_ids

def standardCIFAR(train_images: np.ndarray, test_images: np.ndarray):
    '''
    Performs standardization (mean subtraction and division by standard deviation).

    This function expects input images to be in (N, H, W, C) format (channels-last),
	which is the default for Keras/TensorFlow. If normalize=True was used in loading,
	they should also already be scaled to the [0, 1] range (float32).

    Mean and standard deviation are calculated per channel from the training data only.

    Args:
        train_images (np.ndarray): Training images in (N, H, W, C) format, float.
        test_images (np.ndarray): Test images in (M, H, W, C) format, float.

    Returns:
        tuple: (train_images_processed, test_images_processed)
               Both are np.ndarray in (N, H, W, C) format, float32, standardized.
    '''
    # Ensure inputs are float32 (they should be if normalize=True was used in loading)
    train_images_float32 = train_images.astype(np.float32)
    test_images_float32 = test_images.astype(np.float32)

    # Calculate mean and standard deviation per channel from the training data
    # For (N, H, W, C) format, axis=(0, 1, 2) means average across all images (axis 0),
    # and all height/width pixels (axes 1, 2), keeping dimensions for broadcasting.
    # This calculates a mean/std for each of the 3 channels.
    mean = train_images_float32.mean(axis=(0, 1, 2), keepdims=True)
    std = train_images_float32.std(axis=(0, 1, 2), keepdims=True)

    # Add a small epsilon to std to prevent division by zero for channels with zero variance
    std = np.maximum(std, 1e-7)

    # Apply standardization
    train_images_standardized = (train_images_float32 - mean) / std
    test_images_standardized = (test_images_float32 - mean) / std

    return train_images_standardized, test_images_standardized

def getLabelCIFAR10(index):
	'''
	Returns the string name for a given CIFAR-10 label.
      
    Args:
        index (int): The integer label.
	'''
	return labelsCIFAR10[index]