import pandas as pd
import random as rand
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, max_error

import warnings
warnings.filterwarnings('ignore')

### Model Functions ###
def elastic_net_model(comp):
    model = GridSearchCV(ElasticNet(), param_grid = EN_grid, cv = shuffle_split, verbose = 0, n_jobs = -1, scoring = scoring, refit = "MSE")
    model.fit(X, y)
    EN_bp = model.best_params_

    EN_Results_Dict = {"Components": comp}|model.best_params_|model.cv_results_
    EN_Results_Raw = pd.DataFrame(EN_Results_Dict)

    return EN_Results_Raw, EN_bp

def elastic_net_bp_model(EN_bp):
    MSE = []
    R2 = []
    Max_Error = []
    model = ElasticNet(alpha = EN_bp["alpha"], l1_ratio = EN_bp["l1_ratio"])
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(pcaX, y, test_size = 0.2, random_state = rand.randint(0, 100))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        MSE.append(mean_squared_error(y_test, y_pred))
        R2.append(r2_score(y_test, y_pred))
        Max_Error.append(max_error(y_test, y_pred))
    
    return MSE, R2, Max_Error

def l1_model(comp):
    model = GridSearchCV(Lasso(), param_grid = L1_grid, cv = shuffle_split, verbose = 0, n_jobs = -1, scoring = scoring, refit = "MSE")
    model.fit(pcaX, y)
    l1_bp = model.best_params_

    L1_Results_Dict = {"Components": comp}|model.best_params_|model.cv_results_
    L1_Results_Raw = pd.DataFrame(L1_Results_Dict)

    return L1_Results_Raw, l1_bp

def l1_bp_model(l1_bp):
    MSE = []
    R2 = []
    Max_Error = []
    model = Lasso(alpha = l1_bp["alpha"])
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(pcaX, y, test_size = 0.2, random_state = rand.randint(0, 100))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        MSE.append(mean_squared_error(y_test, y_pred))
        R2.append(r2_score(y_test, y_pred))
        Max_Error.append(max_error(y_test, y_pred))
    
    return MSE, R2, Max_Error

def l2_model(comp):
    model = GridSearchCV(Ridge(), param_grid = L2_grid, cv = shuffle_split, verbose = 0, n_jobs = -1, scoring = scoring, refit = "MSE")
    model.fit(pcaX, y)
    l2_bp = model.best_params_

    L2_Results_Dict = {"Components": comp}|model.best_params_|model.cv_results_
    L2_Results_Raw = pd.DataFrame(L2_Results_Dict)

    return L2_Results_Raw, l2_bp

def l2_bp_model(l2_bp):
    MSE = []
    R2 = []
    Max_Error = []
    model = Ridge(alpha = l2_bp["alpha"])
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(pcaX, y, test_size = 0.2, random_state = rand.randint(0, 100))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        MSE.append(mean_squared_error(y_test, y_pred))
        R2.append(r2_score(y_test, y_pred))
        Max_Error.append(max_error(y_test, y_pred))
    
    return MSE, R2, Max_Error

def linear_model(comp):
    model = LinearRegression()
    MSE = []
    R2 = []
    Max_Error = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(pcaX, y, test_size = 0.2, random_state = rand.randint(0, 100))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        MSE.append(mean_squared_error(y_test, y_pred))
        R2.append(r2_score(y_test, y_pred))
        Max_Error.append(max_error(y_test, y_pred))
    
    Linear_Results_Dict = {"Components": comp,
                        "Mean_MSE": np.mean(MSE),
                        "Std_MSE": np.std(MSE),
                        "Mean_R2": np.mean(R2),
                        "Std_R2": np.std(R2),
                        "Mean_Max_Error": np.mean(Max_Error),
                        "Std_Max_Error": np.std(Max_Error)}
    
    Linear_Results_Raw = pd.DataFrame(Linear_Results_Dict, index = [0])

    return Linear_Results_Raw

### Evaluation Function ###
def bp_eval(comp, MSE, R2, Max_Error):
    Results_Dict = {"Components": comp,
                        "Mean_MSE": np.mean(MSE),
                        "Std_MSE": np.std(MSE),
                        "Mean_R2": np.mean(R2),
                        "Std_R2": np.std(R2),
                        "Mean_Max_Error": np.mean(Max_Error),
                        "Std_Max_Error": np.std(Max_Error)}
    
    Results_Raw = pd.DataFrame(Results_Dict, index = [0])
    return Results_Raw

### Importing Dataset & Data Preprocessing ###
dataset = pd.read_csv('IUPAC_Dataset/Dataset/master_dataset_diff_prcd.csv')

zero_cols = list(dataset.columns[(dataset == 0).all()]) #Removing columns with all zeros
for col in zero_cols:
    del dataset[col]

y = dataset.iloc[0:, 5:6].values #mole fraction mean
X = pd.DataFrame(dataset.iloc[0:, 7:].values)
shuffle_split = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = rand.randint(0, 1000)) #Setting up test-train splits

scoring = {"MSE": "neg_mean_squared_error",
            "R2": "r2",
            "Max Error": "max_error"}
components = list(np.linspace(2, 20, 10, dtype = int))

### Elastic Net Variables ###
EN_grid = {'l1_ratio': list(np.linspace(0.1, 1, 10)),
         'alpha': list(np.linspace(0.1, 1, 10))}
EN_ResultsDF = pd.DataFrame()
EN_bp_ResultsDF = pd.DataFrame()

### L1 Variables ###
L1_grid = {'alpha': list(np.linspace(0.1, 1, 10))}
L1_ResultsDF = pd.DataFrame()
L1_bp_ResultsDF = pd.DataFrame()

### L2 Variables ###
L2_grid = {'alpha': list(np.linspace(0.1, 1, 10))}
L2_ResultsDF = pd.DataFrame()
L2_bp_ResultsDF = pd.DataFrame()

Linear_ResultsDF = pd.DataFrame()

### Model Fitting ###
for i in tqdm(range(0, len(components))):
    pca = PCA(n_components = components[i])
    pcaX = pca.fit_transform(X)

    EN_Results, EN_bp = elastic_net_model(components[i])
    MSE, R2, Max_Error = elastic_net_bp_model(EN_bp)
    EN_bp_results = bp_eval(components[i], MSE, R2, Max_Error)
    EN_ResultsDF = EN_ResultsDF.append(EN_Results, ignore_index = True)
    EN_bp_ResultsDF = EN_bp_ResultsDF.append(EN_bp_results, ignore_index = True)

    L1_Results, l1_bp = l1_model(components[i])
    MSE, R2, Max_Error = l1_bp_model(l1_bp)
    L1_bp_results = bp_eval(components[i], MSE, R2, Max_Error)
    L1_ResultsDF = L1_ResultsDF.append(L1_Results, ignore_index = True)
    L1_bp_ResultsDF = L1_bp_ResultsDF.append(L1_bp_results, ignore_index = True)

    L2_Results, l2_bp = l1_model(components[i])
    MSE, R2, Max_Error = l1_bp_model(l2_bp)
    L2_bp_results = bp_eval(components[i], MSE, R2, Max_Error)
    L2_ResultsDF = L2_ResultsDF.append(L2_Results, ignore_index = True)
    L2_bp_ResultsDF = L2_bp_ResultsDF.append(L2_bp_results, ignore_index = True)

    Linear_Results = linear_model(components[i])
    Linear_ResultsDF = Linear_ResultsDF.append(Linear_Results, ignore_index = True)

### Saving Results ###
print(EN_ResultsDF)
print(EN_bp_ResultsDF)

print(L1_ResultsDF)
print(L1_bp_ResultsDF)

print(L2_ResultsDF)
print(L2_bp_ResultsDF)

print(Linear_ResultsDF)

EN_ResultsDF.to_csv("IUPAC_Dataset/Scripts/ML_Models/ParamOptimisation/ENLR_opt.csv", index = False)
EN_bp_ResultsDF.to_csv("IUPAC_Dataset/Scripts/ML_Models/ParamOptimisation/ENLR_bp_opt.csv", index = False)

L1_ResultsDF.to_csv("IUPAC_Dataset/Scripts/ML_Models/ParamOptimisation/L1LR_opt.csv", index = False)
L1_bp_ResultsDF.to_csv("IUPAC_Dataset/Scripts/ML_Models/ParamOptimisation/L1LR_bp_opt.csv", index = False)

L2_ResultsDF.to_csv("IUPAC_Dataset/Scripts/ML_Models/ParamOptimisation/L2LR_opt.csv", index = False)
L2_bp_ResultsDF.to_csv("IUPAC_Dataset/Scripts/ML_Models/ParamOptimisation/L2LR_bp_opt.csv", index = False)

Linear_ResultsDF.to_csv("IUPAC_Dataset/Scripts/ML_Models/ParamOptimisation/LinearLR_opt.csv", index = False)