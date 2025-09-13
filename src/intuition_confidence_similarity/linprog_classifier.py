'''
Created on August 9th 2025

@author: Bella Chung

'''

import numpy as np
from scipy.optimize import linprog

class LinearBinaryClassifier:
    '''
    A binary classifier that uses linear programming to find the optimal separating hyperplane.
    
    This class solves a linear program to determine the weights and bias of a linear classifier,
    aiming to maximize the separation between two classes (labeled 1 and -1) in the feature space.
    The optimization problem is formulated to find a hyperplane `w . x + b = 0` that correctly
    classifies all data points while maximizing the margin.

    Attributes:
        w (numpy.ndarray): The weight vector of the hyperplane.
        b (float): The bias or intercept of the hyperplane.
    '''
    def __init__(self, weights=None, bias=None):
        self.weights = weights
        self.bias = bias

    def train(self, X, y):
        '''
        Solves the linear program to find the optimal weights and bias.

        Args:
            X (numpy.ndarray): The input data matrix, where each row is a data point.
            y (numpy.ndarray): The target labels for each data point (must be 1 or -1).
        '''
        num_samples, num_features = X.shape
        
        A_ub = np.concatenate([
        -y.reshape(-1, 1) * X, 
        -y.reshape(-1, 1),           # bias term scaled by -y
        -np.eye(num_samples)
        ], axis=1)
    
        b_ub = -np.ones(num_samples)
    
        bounds = [(None, None)] * (num_features + 1) + [(0, None)] * num_samples
        c = np.concatenate([np.zeros(num_features + 1), np.ones(num_samples)])

        # linear program
        soln = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

        if soln.success:
            self.weights = soln.x[:num_features]
            self.bias = soln.x[num_features]

            # normalize weights
            weight_norm = np.linalg.norm(self.weights)
            if weight_norm > 0:  # watch out for division by zero
                self.weights /= weight_norm
                self.bias /= weight_norm

        else:
            raise Exception("Linear programming optimization failed!")

    def predict(self, X):
        '''
        Predicts the class label for new data points.

        Args:
            X (numpy.ndarray): The input data matrix for classification.

        Returns:
            numpy.ndarray: The predicted labels (1 or -1) for each data point.
        '''
        decision_values = np.dot(X, self.weights) + self.bias
        return np.where(decision_values >= 0, 1, -1)

    def get_decision_distance(self, X):
        '''
        Calculates the signed distance of each data point to the decision boundary.

        The magnitude of the distance indicates how far a point is from the boundary,
        and the sign indicates which side of the boundary the point lies on.

        Args:
            X (np.ndarray): The input data matrix for which to calculate the distances.

        Returns:
            np.ndarray: The signed distances for each data point, with shape (n_samples,).
        '''
        return np.dot(X, self.weights) + self.bias # signed distance