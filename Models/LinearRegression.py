import pandas as pd
import random as rand
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, max_error

import warnings
warnings.filterwarnings('ignore')

### Model Functions ###
def elastic_net_model():
    MSE = []
    R2 = []
    Max_Error = []

    pca = PCA(n_components = 12)
    pcaX = pca.fit_transform(X)

    model = ElasticNet(alpha = 0.1, l1_ratio = 0.1)
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(pcaX, y, test_size = 0.2, random_state = rand.randint(0, 100))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        MSE.append(mean_squared_error(y_test, y_pred))
        R2.append(r2_score(y_test, y_pred))
        Max_Error.append(max_error(y_test, y_pred))
    
    return MSE, R2, Max_Error

def l1_model():
    MSE = []
    R2 = []
    Max_Error = []

    pca = PCA(n_components = 8)
    pcaX = pca.fit_transform(X)

    model = Lasso(alpha = 0.1)
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(pcaX, y, test_size = 0.2, random_state = rand.randint(0, 100))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        MSE.append(mean_squared_error(y_test, y_pred))
        R2.append(r2_score(y_test, y_pred))
        Max_Error.append(max_error(y_test, y_pred))
    
    return MSE, R2, Max_Error

def l2_model():
    MSE = []
    R2 = []
    Max_Error = []

    pca = PCA(n_components = 8)
    pcaX = pca.fit_transform(X)

    model = Ridge(alpha = 0.1)
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(pcaX, y, test_size = 0.2, random_state = rand.randint(0, 100))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        MSE.append(mean_squared_error(y_test, y_pred))
        R2.append(r2_score(y_test, y_pred))
        Max_Error.append(max_error(y_test, y_pred))
    
    return MSE, R2, Max_Error

def linear_model():
    model = LinearRegression()
    MSE = []
    R2 = []
    Max_Error = []

    pca = PCA(n_components = 18)
    pcaX = pca.fit_transform(X)

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(pcaX, y, test_size = 0.2, random_state = rand.randint(0, 100))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        MSE.append(mean_squared_error(y_test, y_pred))
        R2.append(r2_score(y_test, y_pred))
        Max_Error.append(max_error(y_test, y_pred))
    
    Linear_Results_Dict = {"Mean_MSE": np.mean(MSE),
                        "Std_MSE": np.std(MSE),
                        "Mean_R2": np.mean(R2),
                        "Std_R2": np.std(R2),
                        "Mean_Max_Error": np.mean(Max_Error),
                        "Std_Max_Error": np.std(Max_Error)}
    
    Linear_Results_Raw = pd.DataFrame(Linear_Results_Dict, index = [0])

    return Linear_Results_Raw

### Evaluation Function ###
def eval(MSE, R2, Max_Error):
    Results_Dict = {"Mean_MSE": np.mean(MSE),
                    "Std_MSE": np.std(MSE),
                    "Mean_R2": np.mean(R2),
                    "Std_R2": np.std(R2),
                    "Mean_Max_Error": np.mean(Max_Error),
                    "Std_Max_Error": np.std(Max_Error)}
    
    Results_Raw = pd.DataFrame(Results_Dict, index = [0])
    return Results_Raw

### Importing Dataset & Data Preprocessing ###
dataset = pd.read_csv('IUPAC_Dataset/Dataset/method3_dataset.csv')

zero_cols = list(dataset.columns[(dataset == 0).all()]) #Removing columns with all zeros
for col in zero_cols:
    del dataset[col]

y = dataset.iloc[0:, 6].values #mole fraction mean
X = pd.DataFrame(dataset.iloc[0:, 7:].values)
models = ["OLS", "Elastic Net", "L1", "L2"]

### Model Fitting ###
Linear_results = linear_model()
MSE, R2, Max_Error = elastic_net_model()
EN_results = eval(MSE, R2, Max_Error)

MSE, R2, Max_Error = l1_model()
L1_results = eval(MSE, R2, Max_Error)

MSE, R2, Max_Error = l2_model()
L2_results = eval(MSE, R2, Max_Error)

ResultsDF = pd.concat([Linear_results, EN_results, L1_results, L2_results], axis = 0)
ResultsDF.insert(0, "Models", models, True)

print(ResultsDF)