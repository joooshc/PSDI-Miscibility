import pandas as pd
import random as rand
from tqdm import tqdm
import numpy as np
import catboost as cb

from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, max_error

import warnings
warnings.filterwarnings('ignore')

def bp_eval(MSE, R2, Max_Error, comp):
    Results_Dict = {"Compenents": comp,
                    "Mean_MSE": np.mean(MSE),
                    "Std_MSE": np.std(MSE),
                    "Mean_R2": np.mean(R2),
                    "Std_R2": np.std(R2),
                    "Mean_Max_Error": np.mean(Max_Error),
                    "Std_Max_Error": np.std(Max_Error)}
    
    Results_Raw = pd.DataFrame(Results_Dict, index = [0])
    return Results_Raw

def meow(comp):
    cb_model = cb.CatBoostRegressor(loss_function='RMSE', logging_level='Silent')
    gscv = GridSearchCV(cb_model, cb_grid, cv = shuffle_split, verbose = False)
    gscv.fit(X, y)
    best_params = gscv.best_params_

    All_Results_Dict = {"Components": comp}|gscv.best_params_|gscv.cv_results_
    All_Results_DF = pd.DataFrame(All_Results_Dict)
    return best_params, All_Results_DF

def yowl(best_params):
    MSE = []; R2 = []; Max_Error = []
    for i in tqdm(range(10)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rand.randint(0, 1000))

        model = cb.CatBoostRegressor(iterations = best_params['iterations'],
                                    learning_rate = best_params['learning_rate'],
                                    depth = best_params['depth'],
                                    l2_leaf_reg = best_params['l2_leaf_reg'],
                                    loss_function = 'RMSE',
                                    logging_level='Silent')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        MSE.append(mean_squared_error(y_test, y_pred))
        R2.append(r2_score(y_test, y_pred))
        Max_Error.append(max_error(y_test, y_pred))

    return MSE, R2, Max_Error

dataset = pd.read_csv('IUPAC_Dataset/Dataset/master_dataset_diff_prcd.csv')

zero_cols = list(dataset.columns[(dataset == 0).all()]) #Removing columns with all zeros
for col in zero_cols:
    del dataset[col]

y = dataset.iloc[0:, 6].values #mole fraction mean

cb_grid = {"iterations": [100, 150, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "depth": [4, 6, 8],
            "l2_leaf_reg": [0.2, 0.6, 1]}

components = list(np.linspace(5, 20, 4, dtype = int))
bpresults = pd.DataFrame()
All_Results = pd.DataFrame()

for i in tqdm(range(len(components))):
    pca = PCA(n_components = components[i])
    X = pd.DataFrame(dataset.iloc[0:, 7:].values)
    X = pca.fit_transform(X)
    shuffle_split = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = rand.randint(0, 1000)) #Setting up test-train splits

    best_params, All_Results_raw = meow(components[i])
    MSE, R2, Max_Error = yowl(best_params)
    bpresults_raw = bp_eval(MSE, R2, Max_Error, components[i])

    bpresults = bpresults.append(bpresults_raw, ignore_index = True)
    All_Results = All_Results.append(All_Results_raw, ignore_index = True)

print(bpresults)
print(All_Results)
bpresults.to_csv('IUPAC_Dataset/Scripts/ML_Models/ParamOptimisation/CatBoostOptResults.csv', index = False)
All_Results.to_csv('IUPAC_Dataset/Scripts/ML_Models/ParamOptimisation/CatBoostOptAllResults.csv', index = False)