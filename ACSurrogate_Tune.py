#########################################
# Hyperparameter tuning using Keras tuner
# Authors: jsoibam, ddiama
#########################################

##############################
# Import libraries
##############################

# General libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Tensoflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import regularizers

# Kera tuner libraries
from keras import backend as K
from keras_tuner import RandomSearch
from keras_tuner import Objective
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

# Data preprocessing
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set random seed state
np.random.seed(42)

##############################
# Load data
##############################
path_train = "Data//"

file_names = ["data.csv"]
concatenated_data = pd.DataFrame()

# Loop through each file and concatenate the data
for file_name in file_names:
    data = pd.read_csv(path_train + file_name)
    concatenated_data = pd.concat([concatenated_data, data], ignore_index=True)

##############################
# Preprocess data
##############################
# Calculate IQR for each column
Q1 = concatenated_data.quantile(0.25)
Q3 = concatenated_data.quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the data to exclude outliers
concatenated_data_no_outliers = concatenated_data[~((concatenated_data < lower_bound) | (concatenated_data > upper_bound)).any(axis=1)]

# Split the dataset between inputs (X) and outputs (y)
X = concatenated_data_no_outliers.drop(['Unnamed: 0'], axis = 1).iloc[:,:9]
y = concatenated_data_no_outliers.drop(['Unnamed: 0'], axis = 1)[['MTOW', "Block_fuel", "El_energy", "Block_energy"]]

df_x = X.values
df_y = y.values

x_scaler = StandardScaler()
x_scaled = x_scaler.fit_transform(df_x)

y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(df_y)

##############################
# Visualize sampling space
##############################
# Input space
scatter_matrix = sns.pairplot(X, diag_kind='hist')
scatter_matrix.fig.suptitle('Scatter Plot Matrix', y=1.02)

plot_filename = f'plots//Xscatter_plot.png'
scatter_matrix.savefig(plot_filename)

# Output space
scatter_matrix = sns.pairplot(y, diag_kind='hist')
scatter_matrix.fig.suptitle('Scatter Plot Matrix', y=1.02)

plot_filename = f'plots//distribution//yscatter_plot.png'
scatter_matrix.savefig(plot_filename)

##############################
# Data split
##############################
X_train, X_test, y_train, y_test = train_test_split(x_scaled,
                                                    y_scaled, 
                                                    test_size = 0.20, 
                                                    random_state = 42)


# Save the data scalers in order to use them when employing the surrogate model 
joblib.dump(x_scaler, 'x_scaler.pkl')
joblib.dump(y_scaler, 'y_scaler.pkl')

####################################
# Define the NN and hyperparameters
####################################
def build_model(hyperparams):
    model = Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))

    num_layers = hyperparams.Int("num_layers", 2, 6)

    for i in range(num_layers):
        model.add(layers.Dense(
            units=hyperparams.Int(f"units_{i+1}", 64, 512, step=32),
            activation=hyperparams.Choice(f"act_{i+1}", ["relu", "tanh"]),
            kernel_regularizer=regularizers.l2(hyperparams.Float(f"l2_{i+1}", 1e-5, 1e-1, sampling="log"))
        ))

    model.add(layers.Dense(y_train.shape[1],))

    batch_size = hyperparams.Int("batch_size", 16, 128, step=16)
    hp_learning_rate = hyperparams.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])

    optimizers_dict = {
        "Adam": keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
        "RMSprop": keras.optimizers.legacy.RMSprop(learning_rate=hp_learning_rate)
    }

    hp_optimizers = hyperparams.Choice('optimizer', values=["Adam", "RMSprop"])

    model.compile(
        loss='mean_squared_error',
        optimizer=optimizers_dict[hp_optimizers],
        metrics=["mean_squared_error"]
    )

    return model


####################################
# Callbacks
####################################
# Callback for writing checkpoints and saving the weights
path_checkpoint = 'weights//hyperparameter_search'
callback_checpoint = ModelCheckpoint(filepath=path_checkpoint, 
                                    monitor ='val_loss',
                                    verbose=1,
                                    save_weights_only = True,
                                    save_best_only = True)

# Callback for early stopping the optimization when performance worsens on the validation-set
callback_early_stopping = EarlyStopping(monitor = 'val_loss',
                                       patience =60,
                                       verbose=1)


# Callback for TensorBoard log during training
tensorboard_log_dir = "./random_search/tb1_logs"
os.makedirs(tensorboard_log_dir, exist_ok=True)

callback_tensorboard = TensorBoard(tensorboard_log_dir)

# Callback for learning rate
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor =0.1,
                                      min_lr = 1e-5,
                                      patience=30,
                                      verbose=1)

callbacks = [callback_early_stopping,
            callback_checpoint,
            callback_tensorboard,
            callback_reduce_lr]

####################################
# Run the hyperparameter search
####################################
tuner1 = RandomSearch(
    hypermodel=build_model,
    objective=Objective(name="val_mean_squared_error", direction="min"),
    max_trials=40,
    project_name="Regression",
    overwrite=True,
)

tuner1.search(X_train, y_train, epochs=2000, validation_split = 0.1, callbacks=callbacks)

####################################
# Extract the best performing model
####################################
best_params = tuner1.get_best_hyperparameters()
best_params[0].values

best_model = tuner1.get_best_models()[0]
best_model.summary()

best_model.save('best_model.keras', save_format='tf', signatures=None)

####################################
# Postprocessing
####################################
y_pred = best_model.predict(X_test)

y_pred_inverse = y_scaler.inverse_transform(y_pred)
y_test_inverse = y_scaler.inverse_transform(y_test)

ynames = ['MTOW', "Block_fuel", "El_energy", "Block_energy"]
ylen = len(ynames)

# Regression plots
for element in range(ylen):

    plt.figure(figsize = [5,5])
    plt.scatter(y_test_inverse[:,element], y_pred_inverse[:,element], alpha=0.3)
    plt.plot([np.min(y_test_inverse[:,element]),np.max(y_test_inverse[:,element])],[np.min(y_test_inverse[:,element]),np.max(y_test_inverse[:,element])], "k-")
    plt.xlabel(f"{ynames[element]} True")
    plt.ylabel(f"{ynames[element]} Pred")
    plt.tight_layout()
    plt.savefig(f"plots//testing//{ynames[element]}.png")
    plt.close()

# R2 score
for element in range(ylen):
    r2 = (r2_score(y_test_inverse[:,element],y_pred_inverse[:,element]))
    print(f"R2 for {ynames[element]} is {r2}")

# RMSE
for element in range(ylen):
    rmse = np.sqrt(mean_squared_error(y_test_inverse[:,element],y_pred_inverse[:,element]))
    print(f"RMSE for {ynames[element]} is {rmse}")


