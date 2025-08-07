'''
Created on June 10th 2025

@author: Bella Chung

Based on https://github.com/AhmedSalem2/ML-Leaks/blob/master/mlLeaks.py

'''

import os
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import classifier
import dataloader
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import date
import shutil
from multiprocessing import Pool, cpu_count
import warnings

# Suppress common warnings that can be noisy
warnings.filterwarnings('ignore', category=UserWarning)

def check_gpu():
    '''
    Configures TensorFlow to use GPUs with dynamic memory allocation.

    This enables memory growth on all detected GPUs, preventing the
    framework from pre-allocating all VRAM. This is ideal for running 
    multiple processes concurrently.

    If no GPU is found, default to using the CPU.
    '''
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {gpus}")
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")
    else:
        print("No GPU found. Running on CPU.")

def imshow(img):
    '''
    View single image.
    '''
    plt.imshow(img)
    plt.show()

def multi_imshow(img_arr, start=None, stop=None, labels_arr=None):
    '''
    Plots img_arr[start:stop] (inclusive) on a square grid.
    If start and/or stop is not given, assume beginning and end of img_arr.
    Optionally, display image labels (currently hardcoded for CIFAR-10).
    '''
    if start is None:
        start = 0
    if stop is None:
        stop = len(img_arr) - 1

    dim = math.ceil(math.sqrt(stop + 1 - start))
    _, ax = plt.subplots(dim, dim)
    k = start

    for i in range(dim):
        for j in range(dim):
            ax[i][j].imshow(img_arr[k], aspect='auto')
            if labels_arr is not None:
                ax[i][j].set_title(dataloader.getLabelCIFAR10(labels_arr[k]), fontsize = 10)
            ax[i][j].axis('off')
            k += 1

            if k > stop:
                plt.subplots_adjust(hspace=0.4)
                plt.show()
                return

def get_image_features(model_to_extract_features, images, batch_size=100):
    '''
    Extracts features for a given set of images using a pre-trained truncated model.
    '''
    features = model_to_extract_features.predict(images, batch_size=batch_size)
    return features

def norm_angular_distance(feature1, feature2):
    '''
    Calculates the normalized angular distance between two feature vectors.

    Args:
        feature1 (np.ndarray): The first feature vector.
        feature2 (np.ndarray): The second feature vector.

    Returns:
        float: The normalized angular distance, ranging from 0 to 1.
    '''
    # Ensure features are unit vectors (L2 normalized) for cosine similarity to directly represent cosine of angle
    feature1_norm = feature1 / np.linalg.norm(feature1, axis=-1, keepdims=True)
    feature2_norm = feature2 / np.linalg.norm(feature2, axis=-1, keepdims=True)

    # cosine_similarity expects 2D arrays, so reshape if necessary
    if feature1_norm.ndim == 1:
        feature1_norm = feature1_norm.reshape(1, -1)
    if feature2_norm.ndim == 1:
        feature2_norm = feature2_norm.reshape(1, -1)

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(feature1_norm, feature2_norm)
    
    # Clip values to ensure they are within the valid range [-1, 1] for arccos
    # Floating point inaccuracies can sometimes cause values slightly outside this range.
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    
    # Calculate the angle in radians and normalize by pi
    angular_dist = np.arccos(cosine_sim) / np.pi
    return angular_dist.flatten()

def nearest_train(query_img_feature, query_img_label, train_features, train_labels, metric='angular_distance', same_class=False):
    '''
    Finds the single nearest neighbor in the training set to a query image feature.

    Args:
        query_img_feature (np.ndarray): The feature of the single query image.
                                        Expected shape: (feature_dim,)
        query_img_label (any): The label of the query image.
        train_features (np.ndarray): Features of the training images.
                                     Expected shape: (num_train_images, feature_dim)
        train_labels (np.ndarray): Labels corresponding to the train_features.
                                   Expected shape: (num_train_images,)
        metric (str): The similarity/distance metric to use. Supports 'angular_distance'.
        same_class (bool): If True, only search for nearest neighbors within the same class
                            as the query image.

    Returns:
        tuple: A tuple containing:
                - index (int): Index of the nearest neighbor in train_features.
                - score (float): Normalized angular distance for the nearest neighbor.
                                 Returns None, None if no training images are found for the specified class.
    '''
    if query_img_feature.ndim == 1:
        # Reshape to (1, feature_dim) for similarity/distance functions
        query_img_feature = query_img_feature.reshape(1, -1)

    if same_class:
        # Filter train_features and train_labels for the same class as the query image
        same_class_indices = np.where(train_labels == query_img_label)[0]

        if len(same_class_indices) == 0:
            print(f"Warning: No training images found for class '{query_img_label}'. Returning empty results.")
            return None, None # Return None, None when no images are found for the class
        
        subset_train_features = train_features[same_class_indices]
        
        if metric == 'angular_distance':
            # Calculate distances only with the same-class subset
            scores = norm_angular_distance(query_img_feature, subset_train_features).flatten()
            # Get the index that corresponds to the lowest distance
            best_score_index_subset = np.argmin(scores)
            
        else:
            raise ValueError(f"Metric '{metric}' not supported. Only 'angular_distance' are supported.")

        # Get the single best index within the subset and its corresponding score
        best_score = scores[best_score_index_subset]
        
        # Map the subset index back to the original train_features indices
        best_original_index = same_class_indices[best_score_index_subset]
        
        return best_original_index, best_score
    else:
        if metric == 'angular_distance':
            # Calculate distances with all train_features
            scores = norm_angular_distance(query_img_feature, train_features).flatten()
            # Get the index that corresponds to the lowest distance
            best_score_index = np.argmin(scores)
            
        else:
            raise ValueError(f"Metric '{metric}' not supported. Only 'angular_distance' are supported.")
        
        # Get the single best index and its corresponding score
        best_score = scores[best_score_index]
        best_original_index = best_score_index
        
        return best_original_index, best_score
    
def norm_angular_distance_batch(features1: np.ndarray, features2: np.ndarray) -> np.ndarray:
    '''
    Calculates the normalized angular distance between two sets of feature vectors.

    Args:
        features1 (np.ndarray): The first set of feature vectors.
                                Expected shape: (N, feature_dim) or (feature_dim,) if N=1.
        features2 (np.ndarray): The second set of feature vectors.
                                Expected shape: (M, feature_dim) or (feature_dim,) if M=1.

    Returns: (N, M) array of distances.
    '''
    # Ensure features are unit vectors (L2 normalized)
    # Handle cases where input might be 1D (e.g., if one of the inputs is a single feature vector)
    if features1.ndim == 1:
        features1 = features1.reshape(1, -1)
    if features2.ndim == 1:
        features2 = features2.reshape(1, -1)

    features1_norm = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
    features2_norm = features2 / np.linalg.norm(features2, axis=1, keepdims=True)

    cosine_sim = cosine_similarity(features1_norm, features2_norm)

    # Clip values to ensure they are within the valid range [-1, 1] for arccos
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

    angular_dist = np.arccos(cosine_sim) / np.pi
    return angular_dist

def nearest_train_batch(
    query_features: np.ndarray,
    query_labels: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    metric: str = 'angular_distance',
    same_class: bool = False
) -> np.ndarray:
    '''
    Finds the single nearest neighbor in the training set for each query image feature,
    using batch processing for efficiency.

    Args:
        query_features (np.ndarray): Features of the query images.
                                     Expected shape: (num_query_images, feature_dim)
        query_labels (np.ndarray): Labels corresponding to the query_features.
                                   Expected shape: (num_query_images,)
        train_features (np.ndarray): Features of the training images.
                                     Expected shape: (num_train_images, feature_dim)
        train_labels (np.ndarray): Labels corresponding to the train_features.
                                   Expected shape: (num_train_images,)
        metric (str): The similarity/distance metric to use. Supports 'angular_distance'.
        same_class (bool): If True, only search for nearest neighbors within the same class
                            as the query image.

    Returns:
        np.ndarray: A 1D array of scores (distances or similarities) for the nearest neighbor
                    of each query image. Shape: (num_query_images,).
                    Values will be np.nan if 'same_class=True' and no training images
                    are found for a particular query image's class.
    '''
    num_query_images = len(query_features)
    distances = np.full(num_query_images, np.nan) # Initialize with NaN

    if same_class:
        unique_classes = np.unique(query_labels) # Iterate through classes present in query set

        for cls in unique_classes:
            # Identify query images for the current class
            query_indices_in_class = np.where(query_labels == cls)[0]
            if len(query_indices_in_class) == 0:
                continue # Should not happen if unique_classes came from query_labels, but good for safety

            current_query_features = query_features[query_indices_in_class]

            # Identify training images for the current class
            train_indices_in_class = np.where(train_labels == cls)[0]
            if len(train_indices_in_class) == 0:
                print(f"Warning: No training images found for class '{cls}'. Distances for related queries will be NaN.")
                # The 'distances' array for these indices will remain NaN
                continue

            subset_train_features = train_features[train_indices_in_class]

            if metric == 'angular_distance':
                class_distances = norm_angular_distance_batch(current_query_features, subset_train_features)
                min_class_distances = np.min(class_distances, axis=1)
                distances[query_indices_in_class] = min_class_distances
            else:
                raise ValueError(f"Metric '{metric}' not supported. Only 'angular_distance' are supported.")
    else: # same_class is False
        if metric == 'angular_distance':
            all_pairwise_scores = norm_angular_distance_batch(query_features, train_features)
            distances = np.min(all_pairwise_scores, axis=1)
        else:
            raise ValueError(f"Metric '{metric}' not supported. Only 'angular_distance' are supported.")

    return distances


def top_bottom_percent_conf_subsets(norm_confidences, norm_train_distances, ids, percent):
    '''
    Extracts subsets of data corresponding to the top and bottom percent of
    normalized confidence values.

    This function identifies the specified percentage of data points
    with the lowest and highest confidence scores and returns their
    corresponding confidences, training distances, and ids.

    Args:
        norm_confidences (np.ndarray):
            A 1D array of normalized confidence values. These values
            are used to determine the 'top' and 'bottom' subsets.
        norm_train_distances (np.ndarray):
            A 1D array of normalized training distances, where each
            element corresponds positionally to a confidence value in
            `norm_confidences`.
        ids (np.ndarray):
            A 1D array of unique identifiers, where each element
            corresponds positionally to a confidence value in
            `norm_confidences`.
        percent (float):
            The percentage (e.g., 10.0 for 10%) of the total data points
            to select for both the top and bottom subsets. The number of
            elements selected will be `round(len(norm_confidences) * percent / 100)`.

    Returns:
        tuple: A tuple containing six elements:
            - bottom_conf (np.ndarray): Normalized confidence values for the
              bottom 'percent' of data points.
            - top_conf (np.ndarray): Normalized confidence values for the
              top 'percent' of data points.
            - bottom_train_dist (np.ndarray): Normalized training distances
              corresponding to the bottom 'percent' of data points.
            - top_train_dist (np.ndarray): Normalized training distances
              corresponding to the top 'percent' of data points.
            - bottom_ids (np.ndarray): Identifiers corresponding to the
              bottom 'percent' of data points.
            - top_ids (np.ndarray): Identifiers corresponding to the
              top 'percent' of data points.
    '''
    length = len(norm_confidences)
    percent_amount = min(length, round(length * percent * 0.01))

    # --- Bottom Percent ---
    # Get indices of the bottom percent
    bottom_indices = np.argpartition(norm_confidences, percent_amount - 1)[:percent_amount]

    # Get bottom confidences
    bottom_conf = norm_confidences[bottom_indices]

    # Get corresponding training point distances
    bottom_train_dist = norm_train_distances[bottom_indices]

    # --- Top Percent ---
    # Get indices of the top percent
    top_indices = np.argpartition(norm_confidences, -percent_amount)[-percent_amount:]

    # Get top confidences
    top_conf = norm_confidences[top_indices]

    # Get corresponding training point distances
    top_train_dist = norm_train_distances[top_indices]

    return bottom_conf, top_conf, bottom_train_dist, top_train_dist, ids[bottom_indices], ids[top_indices]

def raw_experiment(dataset, train_ratio=0.5, percent=5, model_type='cnn', num_trials=100, same_class=False, folder_name='experiment_data'):
    '''
    Conducts a series of machine learning experiments to analyze model confidence and
    nearest training point distances for top and bottom confidence predictions.

    For each trial, the function trains a specified model on a dataset, queries the
    model for prediction confidences on the dataset, and calculates the distances
    to the nearest training point for each query sample. It then identifies the
    top and bottom percent of samples based on prediction confidence and saves
    the relevant data (IDs, training distances, confidences, and trial number)
    for these subsets.

    The results from all trials are aggregated and saved into two CSV files:
    'bottom_percent.csv' and 'top_percent.csv', located in the specified
    'folder_name'.

    Args:
        dataset (str): The name of the dataset to use (e.g., 'cifar10').
        train_ratio (float, optional): The proportion of the dataset to use for
            training. Defaults to 0.5.
        percent (int, optional): The percentage of top and bottom confident
            predictions to save. For example, if percent=5, the top 5% and
            bottom 5% will be saved. Defaults to 5.
        model_type (str, optional): The type of model to train (e.g., 'cnn').
            Defaults to 'cnn'.
        num_trials (int, optional): The total number of experiment trials to run.
            Defaults to 100.
        same_class (bool, optional): If True, when calculating nearest training
            point distances, only consider training points from the same class
            as the test point. Defaults to False.
        folder_name (str, optional): The name of the folder where the experiment
            results will be saved. Defaults to 'experiment_data'.

    Returns:
        None: The function saves the results to CSV files and does not return
        any value.
    '''
    all_trials_bottom_conf = []
    all_trials_top_conf = []
    all_trials_bottom_train_dist = []
    all_trials_top_train_dist = []
    all_trials_bottom_ids = []
    all_trials_top_ids = []
    all_trials = []

    for trial in range(num_trials):
        # Load and preprocess data
        if dataset == 'cifar10' or 'cifar100':
            train_x, train_y, train_ids, test_x, test_y, test_ids = dataloader.loadCIFAR(dataset, normalize=True, train_ratio=train_ratio, random_state=None)
            train_x, test_x = dataloader.standardCIFAR(train_x, test_x)
        
        # Train model
        model = classifier.train_model(train_x, train_y, test_x, test_y, model_type=model_type)

        # Combine train and test sets to process the full dataset
        full_x = np.concatenate((train_x, test_x), axis=0)
        full_y = np.concatenate((train_y, test_y), axis=0)
        full_ids = np.concatenate((train_ids, test_ids), axis=0)

        # Query model for confidences on the full dataset
        confidences = np.max(model.predict(full_x, batch_size=100), axis=1)

        # Get nearest training point distances
        if dataset == 'cifar10' or dataset == 'cifar100':
            # Get image features before the softmax layer
            features_model = classifier.get_cnn_features(model, train_x.shape[1:])
            train_features = get_image_features(features_model, train_x)
            full_features = get_image_features(features_model, full_x)

            distances = nearest_train_batch(full_features, full_y, train_features, train_y, 'angular_distance', same_class)

        # Get top/bottom percent confidence, distance, and id subsets
        bottom_conf, top_conf, bottom_train_dist, top_train_dist, bottom_ids, top_ids = top_bottom_percent_conf_subsets(confidences, distances, full_ids, percent)

        #Save trial data
        all_trials_bottom_conf.extend(bottom_conf)
        all_trials_top_conf.extend(top_conf)
        all_trials_bottom_train_dist.extend(bottom_train_dist)
        all_trials_top_train_dist.extend(top_train_dist)
        all_trials_bottom_ids.extend(bottom_ids)
        all_trials_top_ids.extend(top_ids)
        all_trials.extend(np.full(bottom_conf.shape, trial+1))

    # Save all trials post experiment data
    df_bottom = pd.DataFrame(
        {"ID": all_trials_bottom_ids,
        "Train distance": all_trials_bottom_train_dist,
        "Confidence": all_trials_bottom_conf,
        "Trial": all_trials}
    )

    df_top = pd.DataFrame(
        {"ID": all_trials_top_ids,
        "Train distance": all_trials_top_train_dist,
        "Confidence": all_trials_top_conf,
        "Trial": all_trials}
    )

    os.makedirs(folder_name, exist_ok=True)
    df_bottom.to_csv(os.path.join(folder_name, "bottom_percent.csv"), index=False)
    df_top.to_csv(os.path.join(folder_name, "top_percent.csv"), index=False)

def run_single_trial(trial, dataset, train_ratio, percent, model_type, same_class, full_x, full_y, full_ids):
    '''
    A worker function to run a single experiment trial. This function
    is designed to be used with multiprocessing.Pool.
    
    Returns:
        dict: A dictionary containing the aggregated data for the single trial.
    '''
    print(f"Starting trial {trial}...")

    # Perform a randomized train/test split inside the worker function
    train_x, test_x, train_y, test_y, train_ids, test_ids = train_test_split(
        full_x, full_y, full_ids, train_size=train_ratio, random_state=trial, stratify=full_y
    )

    # Preprocess the data after splitting
    if dataset == 'cifar10' or dataset == 'cifar100':
        train_x, test_x = dataloader.standardCIFAR(train_x, test_x)
    
    # Train model
    model = classifier.train_model(train_x, train_y, test_x, test_y, model_type=model_type)

    # Combine train and test sets to process the full dataset
    full_x = np.concatenate((train_x, test_x), axis=0)
    full_y = np.concatenate((train_y, test_y), axis=0)
    full_ids = np.concatenate((train_ids, test_ids), axis=0)

    # Query model for confidences on the full dataset
    confidences = np.max(model.predict(full_x, batch_size=100), axis=1)

    # Get nearest training point distances
    if dataset == 'cifar10' or dataset == 'cifar100':
        # Get image features before the softmax layer
        features_model = classifier.get_cnn_features(model, train_x.shape[1:])
        train_features = get_image_features(features_model, train_x)
        full_features = get_image_features(features_model, full_x)

        distances = nearest_train_batch(full_features, full_y, train_features, train_y, 'angular_distance', same_class)

    # Get top/bottom percent confidence, distance, and id subsets
    bottom_conf, top_conf, bottom_train_dist, top_train_dist, bottom_ids, top_ids = top_bottom_percent_conf_subsets(confidences, distances, full_ids, percent)

    print(f"Trial {trial} completed.")
    
    # Return a dictionary with all the collected data
    return {
        "bottom_conf": bottom_conf,
        "top_conf": top_conf,
        "bottom_train_dist": bottom_train_dist,
        "top_train_dist": top_train_dist,
        "bottom_ids": bottom_ids,
        "top_ids": top_ids,
        "trial_num": np.full(bottom_conf.shape, trial)
    }

def raw_experiment_parallel(dataset, train_ratio=0.5, percent=5, model_type='cnn', num_trials=100, same_class=False, folder_name='experiment_data'):
    '''
    Conducts a series of machine learning experiments in parallel using multiprocessing
    to analyze model confidence and nearest training point distances for top and bottom 
    confidence predictions.

    The results from all trials are aggregated and saved into two CSV files:
    'bottom_percent.csv' and 'top_percent.csv', located in the specified
    'folder_name'.

    Args:
        dataset (str): The name of the dataset to use (e.g., 'cifar10').
        train_ratio (float, optional): The proportion of the dataset to use for
            training. Defaults to 0.5.
        percent (int, optional): The percentage of top and bottom confident
            predictions to save. For example, if percent=5, the top 5% and
            bottom 5% will be saved. Defaults to 5.
        model_type (str, optional): The type of model to train (e.g., 'cnn').
            Defaults to 'cnn'.
        num_trials (int, optional): The total number of experiment trials to run.
            Defaults to 100.
        same_class (bool, optional): If True, when calculating nearest training
            point distances, only consider training points from the same class
            as the test point. Defaults to False.
        folder_name (str, optional): The name of the folder where the experiment
            results will be saved. Defaults to 'experiment_data'.

    Returns:
        None: The function saves the results to CSV files and does not return
        any value.
    '''
    #num_processes = min(num_trials, cpu_count())
    num_processes = int(os.environ['SLURM_CPUS_PER_TASK'])
    print(f"Running {num_trials} trials in parallel using {num_processes} processes...")

    # Load the full dataset once in the main process to reduce memory overhead
    print("Loading dataset once in main process...")
    if dataset == 'cifar10' or dataset == 'cifar100':
        full_x, full_y, full_ids = dataloader.loadFullCIFAR(dataset, normalize=True)

    # Create a list of tuples, where each tuple contains all parameters for a single trial
    trial_params_list = [
        (trial + 1, dataset, train_ratio, percent, model_type, same_class, full_x, full_y, full_ids)
        for trial in range(num_trials)
    ]

    # Use a process pool to run trials in parallel
    with Pool(processes=num_processes) as pool:
        all_results = pool.starmap(run_single_trial, trial_params_list)

    # Aggregate results from all trials
    all_trials_bottom_conf = []
    all_trials_top_conf = []
    all_trials_bottom_train_dist = []
    all_trials_top_train_dist = []
    all_trials_bottom_ids = []
    all_trials_top_ids = []
    all_trials_num = []
    
    for result in all_results:
        all_trials_bottom_conf.extend(result["bottom_conf"])
        all_trials_top_conf.extend(result["top_conf"])
        all_trials_bottom_train_dist.extend(result["bottom_train_dist"])
        all_trials_top_train_dist.extend(result["top_train_dist"])
        all_trials_bottom_ids.extend(result["bottom_ids"])
        all_trials_top_ids.extend(result["top_ids"])
        all_trials_num.extend(result["trial_num"])

    # Save all trials post experiment data
    df_bottom = pd.DataFrame(
        {"ID": all_trials_bottom_ids,
         "Train distance": all_trials_bottom_train_dist,
         "Confidence": all_trials_bottom_conf,
         "Trial": all_trials_num}
    )

    df_top = pd.DataFrame(
        {"ID": all_trials_top_ids,
         "Train distance": all_trials_top_train_dist,
         "Confidence": all_trials_top_conf,
         "Trial": all_trials_num}
    )

    os.makedirs(folder_name, exist_ok=True)
    df_bottom.to_csv(os.path.join(folder_name, "bottom_percent.csv"), index=False)
    df_top.to_csv(os.path.join(folder_name, "top_percent.csv"), index=False)
    print(f"All {num_trials} trials completed and data saved to '{folder_name}'.")

def save_model_weights(model, file_name):
    '''
    Saves the given model weights only to the file_name location.
    '''
    model.save_weights(f"./{file_name}.weights.h5")

def save_model(model, file_name):
    '''
    Saves the given model architecture to the file_name location.
    '''
    model.save(f"./{file_name}.h5")

def load_model(file_name):
    '''
    Loads the model architecture from the given file_name location.
    '''
    return tf.keras.models.load_model(file_name)

def zip_folder(folder_path, output_path, format='zip'):
    '''
    Zips a specified folder.

    Args:
        folder_path (str): The path to the folder to be zipped.
        output_path (str): The path and name for the output zip file (without the extension).
        format (str): The archive format. Common options include 'zip', 'tar', 'gztar', 'bztar', 'xztar'.
                      Defaults to 'zip'.
    '''
    try:
        # Ensure the folder path is valid
        if not os.path.isdir(folder_path):
            print(f"Error: Folder '{folder_path}' does not exist or is not a directory.")
            return

        # Create the archive
        shutil.make_archive(output_path, format, folder_path)
        print(f"Successfully zipped '{folder_path}' to '{output_path}.{format}'")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    # Get the current date
    today = date.today()
    month_number = today.month
    day_number = today.day

    # 1:5, 1:1, 5:1
    train_ratio_string = ['1-to-5', '1-to-1', '5-to-1']
    for i, train_ratio in enumerate([0.1667, 0.5, 0.8333]):
        folder_name = f"{month_number}-{day_number}-cnn-cifar10-same-class-train-ratio-{train_ratio_string[i]}"
        raw_experiment_parallel('cifar10', train_ratio, percent=5, model_type='cnn', num_trials=100, same_class=True, folder_name=folder_name)
        zip_folder(folder_name, folder_name)

        folder_name = f"{month_number}-{day_number}-cnn-cifar10-diff-class-train-ratio-{train_ratio_string[i]}"
        raw_experiment_parallel('cifar10', train_ratio, percent=5, model_type='cnn', num_trials=100, same_class=False, folder_name=folder_name)
        zip_folder(folder_name, folder_name)

if __name__ == '__main__':
    main()