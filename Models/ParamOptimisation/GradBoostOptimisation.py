import pandas as pd
import random as rand
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, max_error

import warnings
warnings.filterwarnings('ignore')

def gradient_boost_model(comp):
    GB_model = GridSearchCV(GradientBoostingRegressor(), param_grid = GB_grid, cv = shuffle_split, verbose = 0, scoring = scoring, refit = "MSE")
    GB_model.fit(X, y)
    GB_bp_model = GradientBoostingRegressor(learning_rate = GB_model.best_params_["learning_rate"], max_depth = GB_model.best_params_["max_depth"], n_estimators = GB_model.best_params_["n_estimators"])

    for i in tqdm(range(15)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rand.randint(0, 100))
        GB_bp_model.fit(X_train, y_train)
        GB_Results_Dict = {"Components": comp,
                            "Linear_MSE": mean_squared_error(y_test, GB_bp_model.predict(X_test)),
                            "Linear_R2": r2_score(y_test, GB_bp_model.predict(X_test)),
                            "Linear_Max_Error": max_error(y_test, GB_bp_model.predict(X_test))} | GB_model.best_params_
        
        print(GB_Results_Dict)
    GB_Results_Raw = pd.DataFrame(GB_Results_Dict, index = [0])

    return GB_Results_Raw

### Importing Dataset & Data Preprocessing ###
dataset = pd.read_csv('IUPAC_Dataset/Dataset/master_dataset_diff_prcd.csv')

zero_cols = list(dataset.columns[(dataset == 0).all()]) #Removing columns with all zeros
for col in zero_cols:
    del dataset[col]

y = dataset.iloc[0:, 6].values #mole fraction mean
X = pd.DataFrame(dataset.iloc[0:, 7:].values)
shuffle_split = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = rand.randint(0, 1000)) #Setting up test-train splits

scoring = {"MSE": "neg_mean_squared_error",
            "R2": "r2",
            "Max Error": "max_error"}
GB_grid = {"n_estimators": np.linspace(325, 1000, 4, dtype = int),
           "max_depth": np.linspace(3, 7, 5, dtype = int),
           "learning_rate": np.linspace(0.0325, 0.0775, 4),
           "min_samples_split": np.linspace(0.1, 0.5, 5)}

pca = PCA(n_components = 7)
pcaX = pca.fit_transform(X)
Results = gradient_boost_model(7)
Results.to_csv("IUPAC_Dataset/Scripts/ML_Models/ParamOptimisation/Results/GB_Results.csv", index = False)