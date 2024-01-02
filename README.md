# AeroSurrogateTune

## Description
A repository for hyperparameter tuning using Keras tuner for the creation of a surrogate model in the context of aircraft conceptual design.

## Table of Contents
- [Import libraries](#import-libraries)
- [Load data](#load-data)
- [Visualize sampling space](#visualize-sampling-space)
- [Data preprocessing](#data-preprocessing)
- [Define the NN and hyperparameters](#define-the-nn-and-hyperparameters)
- [Callbacks](#callbacks)
- [Run the hyperparameter search](#run-the-hyperparameter-search)
- [Extract best hyperparameters and model](#extract-best-hyperparameters-and-model)
- [Postprocessing](#postprocessing)

## Dependencies
- Python 3
- TensorFlow
- Keras
- keras_tuner
- numpy
- matplotlib
- seaborn
- pandas
- scikit-learn

## Import libraries
This section includes the necessary Python libraries for the project, including TensorFlow, Keras, and data preprocessing tools.

## Load data
Load and preprocess the aircraft conceptual design data from CSV files, removing outliers and preparing features and targets.

## Visualize sampling space
Visualize the feature space with scatter plots to gain insights into the distribution of data.

## Data preprocessing
Standardize and scale the data, and split it into training and testing sets.

## Define the NN and hyperparameters
Define the neural network architecture and hyperparameters for the Keras tuner.

## Callbacks
Configure callbacks, including checkpoints, early stopping, TensorBoard logging, and learning rate adjustment.

## Run the hyperparameter search
Perform hyperparameter tuning using Keras tuner's Random Search.

## Extract best hyperparameters and model
Retrieve the best hyperparameters and the corresponding neural network model.

## Postprocessing
Evaluate the best model on the test set, calculate metrics like R2 and RMSE, and visualize regression plots.
