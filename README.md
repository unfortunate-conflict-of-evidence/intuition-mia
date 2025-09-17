# Intuition-based Membership Inference Vulnerability Analysis

## Overview
This repository contains the source code for a research project that investigates membership inference vulnerabilities in machine learning models. The project analyzes the relationship between a model's prediction confidence and the similarity of a data point to its nearest training set neighbor.

The core of the research involves two distinct experimental pipelines: one using a custom linear classifier on synthetic data and another using a deep learning classifier (a CNN) on real-world image datasets. The project is designed to be highly scalable, using high-performance computing to run hundreds of experimental trials in parallel to generate and analyze large datasets.

## Key Features

* **Diverse Experimental Pipelines:** The project includes two main experiment pipelines for membership inference analysis: one using a linear classifier trained via linear programming on synthetic data, and another using deep learning models on datasets like CIFAR-10.
* **Scalable and High-Performance:** The code is engineered for efficiency, leveraging multiprocessing to run trials in parallel. It is also optimized for use on supercomputing clusters, with explicit support for Slurm and GPU acceleration.
* **Command-Line Interface:** Experiments are easily configured and executed using a robust command-line interface powered by `argparse`. This allows for dynamic control over key parameters such as data size, trial count, and model type.
* **Comprehensive Metrics:** The project quantifies the relationship between confidence and data similarity using metrics like normalized boundary distance for the linear classifier and normalized angular distance for deep learning models.
* **Custom Model Implementation:** The code includes custom-built classifiers, such as a `LinearBinaryClassifier` trained from scratch using `scipy.optimize.linprog` and a custom CNN with `keras`.

## Repository Structure

The core source code is located in `src/intuition_confidence_similarity`.

* `linprog_classifier.py`: Defines the custom linear binary classifier that uses linear programming to find an optimal separating hyperplane.
* `linprog_runner.py`: A command-line script to run the linear classifier experiments.
* `classifier.py`: Contains the deep learning model architectures (CNN, MLP) and training logic using `tensorflow` and `keras`.
* `dataloader.py`: Handles the loading, preprocessing, and splitting of image datasets like CIFAR-10.
* `experimenter.py`: Contains the core logic for running machine learning experiments, including functions for GPU configuration, nearest neighbor calculations, and data aggregation.
* `runner.py`: A command-line script to run the deep learning experiments.

## Setup and Dependencies

To run this project, you will need to install the required Python libraries.

```bash
pip install numpy pandas scipy scikit-learn tensorflow keras
