# AeroSurrogateTune

## Description
A repository for hyperparameter tuning using Keras tuner for the creation of a surrogate model in the context of aircraft conceptual design. This code is associated with the work submitted in ASME Turbo Expo 2024, GT2024-127421 titled "Navigating Technological Risks: An Uncertainty Analysis Of Powertrain Technology In
Hybrid-Electric Commuter Aircraft."

## Authors
- Jerol Soibam
- Dimitra-Eirini Diamantidou
  
## Dependencies
Python 3, TensorFlow, keras_tuner, numpy, matplotlib, pandas, scikit-learn, seaborn

## Notes 
To monitor the results during training, Tensorboard can be used with the following command:

- tensorboard --logdir=random_search/tb1_logs

Screenshots from training can be found in the "Plots" folder.

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
Configure callbacks to enhance the model's performance and efficiency during the training process. Additional care was taken by introducing key callbacks:

- **ModelCheckpoint:** Save the model's weights whenever there is an improvement in validation loss, ensuring progress is not lost.

- **EarlyStopping:** Halt the training if there is no improvement in validation loss for 50 epochs, preventing overfitting and saving computational resources.

- **ReduceLROnPlateau:** Dynamically adjust the learning rate when the validation loss plateaus, facilitating more precise weight adjustments and aiding in avoiding local minima. These callbacks collectively ensure a more robust and effective training process.

## Run the hyperparameter search
Perform hyperparameter tuning using Keras tuner's Random Search. For this repository, 5 trials were ran to showcase its capabilities. However, a larger number of trials was performed for the publication. 

## Extract best hyperparameters and model
Retrieve the best hyperparameters and the corresponding neural network model.

## Postprocessing
Evaluate the best model on the test set, calculate metrics like R2 and RMSE, and visualize regression plots.
