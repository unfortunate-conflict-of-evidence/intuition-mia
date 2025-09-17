'''
Created on August 9th 2025

@author: Bella Chung

'''

import os
import numpy as np
import pandas as pd
import scipy.special as sp
from scipy.spatial import cKDTree
from linprog_classifier import LinearBinaryClassifier
from experimenter import top_bottom_percent_conf_subsets
from multiprocessing import Pool, cpu_count

def rand_train_data(train_size, full_data, full_ids, ideal_weights, ideal_bias, random_seed=None):
    '''
    Generates synthetic training data by sampling from a full set of grid points.

    Args:
        train_size (int): The number of training data points to generate.
        full_data (np.ndarray): The complete array of all possible grid points.
        full_ids (np.ndarray): The corresponding unique IDs for each point in full_data.
        ideal_weights (np.ndarray): The weight vector of the ideal separating hyperplane.
        ideal_bias (float): The bias of the ideal separating hyperplane.
        random_seed (int): An optional integer to set the random seed for reproducibility.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - X_train (np.ndarray): The training data points.
            - y_train (np.ndarray): The corresponding class labels (1 or -1) for each data point.
            - train_ids (np.ndarray): The unique IDs corresponding to the training data.
    '''
    if random_seed is not None:
        np.random.seed(random_seed)

    # Randomly sample unique indices from the full set of points
    sampled_indices = np.random.choice(len(full_ids), size=train_size, replace=False)
    
    # Use these indices to select the training data and their IDs
    X_train = full_data[sampled_indices]
    train_ids = full_ids[sampled_indices]
    
    # Generate labels based on the ideal separating hyperplane
    y_train = np.where(np.dot(X_train, ideal_weights) + ideal_bias > 0, 1, -1)

    return X_train, y_train, train_ids

def rand_test_data(test_size, grid_size, granularity, dimensions):
    '''
    Generates a uniform random sample of test data points without replacement
    from a discrete grid in n-dimensions. If the requested test_size is greater
    than or equal to the total number of points on the grid, it performs an
    exhaustive test and returns all points.

    Args:
        test_size (int): The number of test data points to generate.
        grid_size (float): The size of the grid.
        granularity (float): The step size between points on the grid.
        dimensions (int): The number of dimensions for the grid.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X_test (np.ndarray): The randomly generated test data points.
            - test_point_ids (np.ndarray): Unique IDs corresponding to the test data.
    '''
    # The number of unique values on each dimension of the grid
    num_points_per_dim = int(grid_size / granularity) + 1
    
    # Calculate the total number of points in the exhaustive grid
    total_points = num_points_per_dim ** dimensions
    
    # If test_size is too large, use all points instead of sampling
    if test_size >= total_points:
        print(f"Warning: test_size ({test_size}) is greater than or equal to the total number of points on the grid ({total_points}). Performing an exhaustive test instead.")
        sampled_indices = np.arange(total_points)
    else:
        # Use a memory-efficient method to generate unique random indices
        sampled_indices = set()
        while len(sampled_indices) < test_size:
            sampled_indices.add(np.random.randint(0, total_points))
        sampled_indices = np.array(list(sampled_indices))

    # Calculate coordinates from the 1D indices without generating the full grid
    X_test = np.zeros((len(sampled_indices), dimensions))
    temp_indices = sampled_indices.copy()
    for d in range(dimensions - 1, -1, -1):
        coord_val = temp_indices % num_points_per_dim
        X_test[:, d] = coord_val * granularity
        temp_indices = temp_indices // num_points_per_dim
    
    test_point_ids = sampled_indices
    return X_test, test_point_ids

def exhaust_test_data(grid_size, granularity, dimensions):
    '''
    Generates all possible test data points on a discrete grid in n-dimensions.

    This function creates a complete grid of points within a specified range and granularity,
    which can be used to thoroughly test the classifier's performance across the entire domain.

    Args:
        grid_size (float): The size of the square grid (e.g., a grid from 0 to grid_size).
        granularity (float): The step size between points on the grid.
        dimensions (int): The number of dimensions for the grid.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X_test (np.ndarray): All possible data points on the grid.
            - test_point_ids (np.ndarray): A unique identifier for each test point.
    '''
    values = np.arange(0, grid_size + granularity, granularity)

    # Use a list of value arrays for meshgrid to handle n-dimensions
    mesh_args = [values] * dimensions
    X_test = np.array(np.meshgrid(*mesh_args)).T.reshape(-1, dimensions)

    # generate unique IDs for each test point
    test_point_ids = np.arange(len(X_test))
    return X_test, test_point_ids

def norm_bound_dist(bound_distances, predict_weights, predict_bias, grid_size):
    '''
    Normalizes the distance from a point to the decision boundary by dividing by the
    maximum possible distance within the grid.
    
    Args:
        bound_distances (np.ndarray): The distances to the decision boundary.
        predict_weights (np.ndarray): The weight vector of the predicted separating hyperplane.
        predict_bias (float): The bias of the predicted separating hyperplane.
        grid_size (float): The size of the grid.
        
    Returns:
        np.ndarray: The normalized distances.
    '''
    dimensions = len(predict_weights)
    # Generate corners of the n-dimensional hypercube
    corners_coords = np.array(np.meshgrid(*[[0, grid_size]] * dimensions)).T.reshape(-1, dimensions)

    corner_dist = np.abs(np.dot(corners_coords, predict_weights) + predict_bias)
    
    max_corner_dist = np.max(corner_dist)
    if max_corner_dist == 0:
      norm_bound_distances = np.zeros_like(bound_distances)
    else:
      norm_bound_distances = bound_distances / max_corner_dist

    return norm_bound_distances

def norm_train_dist(train_distances):
    '''
    Normalizes the distance from a point to its nearest training point by dividing
    by the maximum nearest-training-point distance found in the dataset.
    
    Args:
        train_distances (np.ndarray): The distances to the nearest training point.

    Returns:
        np.ndarray: The normalized distances.
    '''
    max_near = np.max(train_distances)
    norm_train_distances = train_distances / max_near
    return norm_train_distances

def get_train_distances(X_train, y_train, X_test, ideal_weights, ideal_bias, same_class):
    '''
    Calculates normalized distances from each test point to its nearest training point.
    
    Args:
        X_train (np.ndarray): The training data.
        y_train (np.ndarray): The labels for the training data.
        X_test (np.ndarray): The test data.
        ideal_weights (np.ndarray): The weight vector of the ideal separating hyperplane.
        ideal_bias (float): The bias of the ideal separating hyperplane.
        same_class (bool): If True, finds the nearest neighbor from the same class. If False,
                          finds the nearest neighbor from all classes.

    Returns:
        train_distances (np.ndarray): Normalized distances to the nearest training point.
    '''
    # 1. Determine the ideal labels (ground truth) for the test set
    ideal_labels = np.where(np.dot(X_test, ideal_weights) + ideal_bias > 0, 1, -1)

    # 2. Use KD-Trees for efficient nearest neighbor search
    train_distances = np.zeros(len(X_test))
    if same_class:
        # Build KD-Trees once for each class
        tree_class_1 = cKDTree(X_train[y_train == 1])
        tree_class_neg_1 = cKDTree(X_train[y_train == -1])

        # Query the appropriate tree based on the test point's class
        for i, test_point in enumerate(X_test):
            if ideal_labels[i] == 1:
                distance, _ = tree_class_1.query(test_point, k=1)
            else:
                distance, _ = tree_class_neg_1.query(test_point, k=1)
            train_distances[i] = distance
    else:
        # Build a single KD-Tree on the entire training set
        tree = cKDTree(X_train)
        
        # Query the tree for all test points at once
        train_distances, _ = tree.query(X_test, k=1)
    
    return norm_train_dist(train_distances)

def get_sig_confidence(signed_bound_distances):
    '''
    Calculates confidence using the sigmoid function on the decision distances.
    The confidence is a value between 0 (low confidence) and 1 (high confidence).
    
    Args:
        signed_bound_distances (np.ndarray): The signed distances to the decision boundary.
        
    Returns:
        np.ndarray: The confidence scores.
    '''
    sigmoid_values = sp.expit(signed_bound_distances.astype(np.float64))
    confidence_values = np.abs(sigmoid_values - 0.5) * 2

    return confidence_values

def run_single_trial(trial, grid_size, train_size, X_test, test_ids, percent, same_class, ideal_weights, ideal_bias, use_sigmoid, base_folder, exp_name):
    '''
    Runs a single trial of the linear programming experiment.

    Returns:
        dict: A dictionary containing the aggregated data for the single trial.
    '''
    print(f"Running trial {trial+1}...")

    if use_sigmoid:
            # Load pre-trained models from the 'distance_conf' folder
            model_file = os.path.join(base_folder, 'distance_conf', exp_name, 'experiment_models.npz')
            model_data = np.load(model_file, allow_pickle=True)
            X_train = model_data['X_train'][trial]
            y_train = model_data['y_train'][trial]
            train_ids = model_data['train_ids'][trial]
            weights = model_data['weights'][trial]
            bias = model_data['bias'][trial]
            model = LinearBinaryClassifier(weights=weights, bias=bias)
    else:
        # Train a new model for each trial
        X_train, y_train, train_ids = rand_train_data(train_size, X_test, test_ids, ideal_weights, ideal_bias, random_seed=trial+1)
        model = LinearBinaryClassifier()
        model.train(X_train, y_train)

    # Make predictions and get decision distances
    decision_distances = model.get_decision_distance(X_test)

    # Calculate distances to the nearest training point
    train_distances = get_train_distances(
        X_train, y_train, X_test, ideal_weights, ideal_bias, same_class
    )

    # Normalize distances to the decision boundary as confidences
    if use_sigmoid:
        confidences = get_sig_confidence(decision_distances)
    else:
        decision_distances = np.abs(decision_distances)
        predict_weights = model.weights
        predict_bias = model.bias
        confidences = norm_bound_dist(decision_distances, predict_weights, predict_bias, grid_size)

    # Boolean list to determine membership
    is_member = np.isin(test_ids, train_ids)

    # Get top and bottom percentile results
    bottom_conf, top_conf, bottom_train_dist, top_train_dist, bottom_ids, top_ids, bottom_is_member, top_is_member = top_bottom_percent_conf_subsets(confidences, train_distances, test_ids, is_member, percent)

    # Return a dictionary with all the collected data
    trial_result = {
        "bottom_conf": bottom_conf,
        "top_conf": top_conf,
        "bottom_train_dist": bottom_train_dist,
        "top_train_dist": top_train_dist,
        "bottom_ids": bottom_ids,
        "top_ids": top_ids,
        "bottom_is_member": bottom_is_member,
        "top_is_member": top_is_member,
        "trial_num": np.full(bottom_conf.shape, trial+1)
    }

    # Return the result and the model parameters if needed
    if not use_sigmoid:
        return trial_result, X_train, y_train, train_ids, model.weights, model.bias
    else:
        return trial_result    

def run_experiment(exp_name, num_trials, train_size, grid_size,
                   X_test, test_ids, percent, same_class,
                   ideal_weights, ideal_bias, base_folder, use_sigmoid=False, use_multiprocessing=True):
    '''
    Conducts a series of machine learning experiments to analyze model confidence and
    nearest training point distances for top and bottom confidence predictions.

    For each trial, the function trains a binary classifer using linear programming
    on linearly separable data, whose ground truth is determined by the given ideal
    weights and ideal bias describing an ideal decision boundary. It queries the
    model for prediction confidences on the dataset, and calculates the distances
    to the nearest training point for each query sample. It then identifies the
    top and bottom percent of samples based on prediction confidence and saves
    the relevant data (IDs, training distances, confidences, and trial number)
    for these subsets.

    The results from all trials are aggregated and saved into two CSV files:
    'bottom_percent.csv' and 'top_percent.csv', located in the specified
    'base_folder'.

    If use_sigmoid is True, the function loads previously saved model parameters
    and runs the experiment with sigmoid confidence scores.
    
    If use_sigmoid is False, the function trains new models for each trial,
    saves the model parameters, and runs the experiment using normalized
    boundary distance as confidence.

    Args:
        exp_name (str): The name for the experiment folder.
        num_trials (int): The number of trials to run.
        train_size (int): The number of training data points per trial.
        grid_size (float): The size of the grid.
        X_test (np.ndarray): All possible test data points on the grid.
        test_ids (np.ndarray): Unique IDs for each test point.
        percent (int): The top/bottom percentage to analyze.
        same_class (bool): Whether to find the nearest neighbor from the same class.
        ideal_weights (np.ndarray): The weight vector of the ideal separating hyperplane.
        ideal_bias (float): The bias of the ideal separating hyperplane.
        base_folder (str): The base directory to save the results.
        use_sigmoid (bool): If True, use sigmoid confidence and load models.
                            If False, use normalized distance and train new models.
        use_multiprocessing (bool): If True, use multiprocessing to run trials in parallel.
    '''
    # Set the sub-folder and confidence metric based on the use_sigmoid flag
    if use_sigmoid:
        sub_folder = 'sigmoid_conf'
        confidence_metric = "sigmoid"
    else:
        sub_folder = 'distance_conf'
        confidence_metric = "normalized distance from the decision boundary"
    
    print(f"Starting experiment '{exp_name}' using {confidence_metric} as confidence...")
    
    # Construct the full experiment directory path
    exp_dir = os.path.join(base_folder, sub_folder, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    all_trials_bottom_conf, all_trials_top_conf = [], []
    all_trials_bottom_train_dist, all_trials_top_train_dist = [], []
    all_trials_bottom_ids, all_trials_top_ids = [], []
    all_trials_bottom_is_member, all_trials_top_is_member = [], []
    all_trials_num = []

    if not use_sigmoid:
        # Train new models and save their parameters for future use
        all_X_train = []
        all_y_train = []
        all_train_ids = []
        all_weights = []
        all_bias = []

    # Prepare arguments for multiprocessing or sequential execution
    shared_args = (grid_size, train_size, X_test, test_ids, percent, same_class, ideal_weights, ideal_bias, use_sigmoid, base_folder, exp_name)
    trial_numbers = range(0, num_trials)
    
    if use_multiprocessing:
        print(f"Running {num_trials} trials in parallel...")

        # Get number of cpus from environment or default to a safe value if not a SLURM environment
        try:
            num_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
        except KeyError:
            num_cpus = cpu_count()

        with Pool(processes=num_cpus) as pool:
            # Create an iterator for the arguments to pass to the parallel function
            args_iterator = ((trial,) + shared_args for trial in trial_numbers)
            results = pool.starmap(run_single_trial, args_iterator)
    else:
        print(f"Running {num_trials} trials sequentially...")
        results = [run_single_trial(trial, *shared_args) for trial in trial_numbers]

    # Aggregate the results from all trials
    for result in results:
        if not use_sigmoid:
            trial_result, X_train, y_train, train_ids, weights, bias = result
            all_X_train.append(X_train)
            all_y_train.append(y_train)
            all_train_ids.append(train_ids)
            all_weights.append(weights)
            all_bias.append(bias)
        else:
            trial_result = result

        all_trials_bottom_conf.extend(trial_result["bottom_conf"])
        all_trials_top_conf.extend(trial_result["top_conf"])
        all_trials_bottom_train_dist.extend(trial_result["bottom_train_dist"])
        all_trials_top_train_dist.extend(trial_result["top_train_dist"])
        all_trials_bottom_ids.extend(trial_result["bottom_ids"])
        all_trials_top_ids.extend(trial_result["top_ids"])
        all_trials_bottom_is_member.extend(trial_result["bottom_is_member"])
        all_trials_top_is_member.extend(trial_result["top_is_member"])
        all_trials_num.extend(trial_result["trial_num"])

    # If we trained new models, save their parameters
    if not use_sigmoid:
        model_data = {
            "X_train": np.array(all_X_train, dtype=object),
            "y_train": np.array(all_y_train, dtype=object),
            "train_ids": np.array(all_train_ids, dtype=object),
            "weights": np.array(all_weights, dtype=object),
            "bias": np.array(all_bias, dtype=object)
        }
        file_path = os.path.join(exp_dir, "experiment_models.npz")
        np.savez(file_path, **model_data)
        print(f"Model parameters saved to '{file_path}'.")

    # Create and save the dataframes
    df_bottom = pd.DataFrame({
        "ID": all_trials_bottom_ids,
        "Train distance": all_trials_bottom_train_dist,
        "Confidence": all_trials_bottom_conf,
        "Membership": all_trials_bottom_is_member,
        "Trial": all_trials_num
    })

    df_top = pd.DataFrame({
        "ID": all_trials_top_ids,
        "Train distance": all_trials_top_train_dist,
        "Confidence": all_trials_top_conf,
        "Membership": all_trials_top_is_member,
        "Trial": all_trials_num
    })

    bottom_filename = "bottom_percent.csv"
    top_filename = "top_percent.csv"
    
    df_bottom.to_csv(os.path.join(exp_dir, bottom_filename), index=False)
    df_top.to_csv(os.path.join(exp_dir, top_filename), index=False)

    print(f"Results saved to '{exp_dir}'.")

def execute_linprog_experiments(exp_name, ideal_weights, ideal_bias, train_size, test_size, grid_size, granularity, num_trials, percent, same_class, base_folder, num_dimensions, use_multiprocessing):
    '''
    Executes a pair of linear programming experiments on the same
    trained models using distance from the decision boundary first 
    as confidence, then sigmoid confidence.

    Args:
        exp_name (str): The name for the experiment folder.
        ideal_weights (np.ndarray): The weight vector of the ideal separating hyperplane.
        ideal_bias (float): The bias of the ideal separating hyperplane.
        train_size (int): The number of training data points per trial.
        test_size (int): The number of test data points to generate.
        grid_size (float): The size of the grid.
        granularity (float): The step size between points on the grid.
        num_trials (int): The number of trials to run.
        percent (int): The top/bottom percentage to analyze.
        same_class (bool): Whether to find the nearest neighbor from the same class.
        base_folder (str): The base directory to save the results.
        num_dimensions (int): The number of dimensions for the data.
        use_multiprocessing (bool): Whether to use multiprocessing to run trials in parallel.
    '''
    X_test, test_ids = rand_test_data(test_size, grid_size, granularity, num_dimensions)
    run_experiment(exp_name, num_trials, train_size, grid_size,
                   X_test, test_ids, percent, same_class,
                   ideal_weights, ideal_bias, base_folder, use_sigmoid=False, use_multiprocessing=use_multiprocessing)  # distance confidence
    run_experiment(exp_name, num_trials, train_size, grid_size,
                   X_test, test_ids, percent, same_class,
                   ideal_weights, ideal_bias, base_folder, use_sigmoid=True, use_multiprocessing=use_multiprocessing)  # sigmoid confidence