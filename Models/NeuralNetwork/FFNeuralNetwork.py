import warnings, os, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split

def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def DNN(n_components):
    model = Sequential()
    model.add(Dense(64, input_dim=n_components, activation='relu'))  # Adjust input features according to PCA components
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # For regression, the activation can be linear for single output.

    # Set a learning rate
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[r_squared])
    plot_model(model, to_file=f'c:/Users/dannt/Documents/GitHub/PSDI_4_priv/IUPAC_Dataset/Scripts/ML_Models/FFNeuralNetwork/ModelArchitectures/model_pca_{n_components}.png', show_shapes=True, show_layer_names=True)

    return model

def train_and_plot(model, X, y, subplot, i, pred_type): # add `i` as parameter
    # Train the model
    history = model.fit(X, y, epochs=150, batch_size=32, validation_split=0.2, verbose=1)

    # Plot training & validation loss values in the provided subplot
    subplot.plot(history.history['loss'])
    subplot.set_title('PCA = {}, MSE = {:.3f}'.format(X.shape[1], history.history['loss'][-1]))
    subplot.set_ylabel('Loss')
    subplot.legend(['Train', 'Validation'], loc='upper right')
    
    # Save the model
    model.save(f'C:/Users/dannt/Documents/GitHub/PSDI_4_priv/IUPAC_Dataset/Scripts/ML_Models/FFNeuralNetwork/TrainedModels({pred_type})/model_pca_{i}.h5') # change the filename accordingly

def plot_predictions(predictions, y_1D):
    fig, axs = plt.subplots(5, 5, figsize=(15, 15)) 
    axs = axs.ravel()

    for i, y_hat in enumerate(predictions):
        # Calculate R-squared value
        r2 = r2_score(y_1D, y_hat)

        # Fit a linear regression line
        slope, intercept = np.polyfit(y_1D.flatten(), y_hat.flatten(), 1)  # using .flatten() to convert y_1D and y_hat to 1D arrays
        reg_line_x = np.linspace(np.min(y_1D), np.max(y_1D), 100)
        reg_line_y = slope * reg_line_x + intercept

        axs[i].scatter(y_1D, y_hat, marker='.', s=5)
        axs[i].plot(reg_line_x, reg_line_y, color='black') # Regression line

        axs[i].set_title(f'PCA = {i+1}, R^2 = {r2:.3f}')
        axs[i].set_ylabel('Predicted values')

    plt.tight_layout()
    plt.show()

def make_predictions_val(y, pred_type=None):
    predictions = []
    for i in range(1, 26):  # for PCA components from 1 to 25
        pca = PCA(n_components=i)
        X_pca = pca.fit_transform(X)
        
        # Save the pca object
        with open(f'C:/Users/dannt/Documents/GitHub/PSDI_4_priv/IUPAC_Dataset/Scripts/ML_Models/FFNeuralNetwork/PCATransformers({pred_type})/pca_{i}.pkl', 'wb') as f:
            pickle.dump(pca, f)
        
        X_pca_val = pca.transform(X_val)  # Use transform, not fit_transform
        model = DNN(i)  
        train_and_plot(model, X_pca, y, axs[i-1], i, pred_type)  # pass `i` to the function
        y_hat = model.predict(X_pca_val)
        predictions.append(y_hat)

    return predictions

def make_predictions_test(X, y, pred_type=None):
    results = pd.DataFrame()
    predictions = []  # Initialize predictions list
    y_tests = []  # Initialize test target values list

    pca = PCA(n_components=25)
    X_pca = pca.fit_transform(X)

    with open(f'pca_{25}_{pred_type}.pkl', 'wb') as f:
        pickle.dump(pca, f)

    losses = []
    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)

        model = DNN(25)
        history = model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1)
        loss = history.history['loss'][-1]

        losses.append(loss)

        # Make predictions and add to the list
        y_hat = model.predict(X_test)
        predictions.append(y_hat)
        y_tests.append(y_test)  # Add the test target values to the list

    results[f'PCA_{25}'] = losses

    results.to_csv(f'pca_losses_{pred_type}.csv', index=False)

    return predictions, y_tests
def make_predictions_test(X, y, pred_type=None):
    results = pd.DataFrame()
    predictions = []  # Initialize predictions list
    y_tests = []  # Initialize test target values list
    r2_scores = []  # Initialize list to hold R-squared scores

    pca = PCA(n_components=7)
    X_pca = pca.fit_transform(X)

    with open(f'pca_{7}_{pred_type}.pkl', 'wb') as f:
        pickle.dump(pca, f)

    losses = []
    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)

        model = DNN(7)
        history = model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1)
        loss = history.history['loss'][-1]

        losses.append(loss)

        # Make predictions and add to the list
        y_hat = model.predict(X_test)
        predictions.append(y_hat)
        y_tests.append(y_test)  # Add the test target values to the list

        # Calculate R-squared for these predictions and add to the list
        r2 = r2_score(y_test, y_hat)
        r2_scores.append(r2)

    # Calculate and print the average R-squared score
    avg_r2_score = np.mean(r2_scores)
    print(f"Average R-squared score: {avg_r2_score}")

    results[f'PCA_{7}'] = losses

    results.to_csv(f'pca_losses_{pred_type}.csv', index=False)

    return predictions, y_tests  # Return the predictions list and the test target values list

# The main block (execute this script directly)
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    current_directory = os.getcwd()
    main_directory = current_directory[:-len('Scripts\ML_Models\FFNeuralNetwork')]
    file_directory = f"{main_directory}Dataset"
    os.chdir(file_directory)

    dataset_main = pd.read_csv(f"{file_directory}\method2_dataset_(log).csv")
    dataset_val = pd.read_csv(f"{file_directory}\master_validation_set.csv")
    dataset = dataset_main # .append(dataset_val, ignore_index=True)

    # Getting X
    X = dataset.iloc[:, 7:]
    X_val = dataset.iloc[:, 7:]

    y_2D = np.array(dataset.iloc[:, 5:7]) # Both the mole fraction mean and std
    y_1D_mean = np.array(dataset.iloc[:, 5:6]) # Just the mole fraction mean
    y_1D_std = np.array(dataset.iloc[:, 6:7])

    y_2D_val = np.array(dataset.iloc[:, 5:7])
    y_1D_val_mean = np.array(dataset.iloc[:, 5:6])
    y_1D_val_std = np.array(dataset.iloc[:, 6:7])

    fig, axs = plt.subplots(5, 5, figsize=(15,15)) 
    axs = axs.ravel()  
    
    # pred_mean = make_predictions_val(y_1D_mean, pred_type='MEAN')
    # pred_std = make_predictions_val(y_1D_std, pred_type='STD')
    pred_mean = make_predictions_test(X, y_1D_mean, pred_type='MEAN')
    # pred_std = make_predictions_test(X, y_1D_std, pred_type='STD')

    # plot_predictions(pred_mean, y_1D_mean)
    # plot_predictions(pred_std, y_1D_val_std)

