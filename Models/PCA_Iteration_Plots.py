import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
import warnings
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, r2_score
import random as rand
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import catboost as cb
from sklearn.ensemble import GradientBoostingRegressor

### Model Functions ###
def ElasticNetModel():
    mseList = []; scoreList = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size = 0.2, random_state = rand.randint(0, 100))
        model = ElasticNet(l1_ratio = 0.8, alpha = 0.2).fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        score = r2_score(y_test, y_pred)
        mseList.append(mse)
        scoreList.append(score)
        
        r2 = np.mean(scoreList)
    return r2

def L1():
    mseList = []; scoreList = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size = 0.2, random_state = rand.randint(0, 100))
        model = Lasso(alpha = 0.01).fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        score = r2_score(y_test, y_pred)
        mseList.append(mse)
        scoreList.append(score)

        r2 = np.mean(scoreList)
    return r2

def L2():
    mseList = []; scoreList = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size = 0.2, random_state = rand.randint(0, 100))
        model = Ridge(alpha = 0.02).fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        score = r2_score(y_test, y_pred)
        mseList.append(mse)
        scoreList.append(score)

        r2 = np.mean(scoreList)
    return r2

def OLS():
    mseList = []; scoreList = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size = 0.2, random_state = rand.randint(0, 100))
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        score = r2_score(y_test, y_pred)
        mseList.append(mse)
        scoreList.append(score)

        r2 = np.mean(scoreList)
    return r2

def yowl():
    MSE = []; R2 = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(pcaX, y, test_size = 0.2, random_state = rand.randint(0, 1000))

        model = cb.CatBoostRegressor(iterations = 200,
                                    learning_rate = 0.05,
                                    depth = 7,
                                    l2_leaf_reg = 0.2,
                                    loss_function = 'RMSE',
                                    logging_level='Silent')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        MSE.append(mean_squared_error(y_test, y_pred))
        R2.append(r2_score(y_test, y_pred))

        r2 = np.mean(R2)
    return r2

def GradBoost():
    mseList = []; scoreList = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(pcaX, y, test_size = 0.2, random_state = rand.randint(0, 100))
        model = GradientBoostingRegressor(n_estimators = 550, max_depth = 5, min_samples_split = 7, learning_rate = 0.0775, loss = "squared_error").fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        score = r2_score(y_test, y_pred)
        mseList.append(mse)
        scoreList.append(score)

        r2 = np.mean(scoreList)
    return r2

###

def plotter(ResultsDict, model):
    plt.clf()
    plt.figure(figsize = (16, 9))
    plt.title(model)
    plt.xlabel("Number of Components")
    plt.ylabel("R2 Score")
    plt.plot(list(ResultsDict.keys()), list(ResultsDict.values()), color = 'blue')
    plt.tight_layout()
    plt.savefig(f"IUPAC_Dataset/Results/PCA_Iteration/{model}.png")

dataset = pd.read_csv('IUPAC_Dataset/Dataset/method2_dataset_(log).csv')

zero_cols = list(dataset.columns[(dataset == 0).all()]) #Removing columns with all zeros
for col in zero_cols:
    del dataset[col]

y = dataset.iloc[0:, 6].values #mole fraction mean
X = pd.DataFrame(dataset.iloc[0:, 7:].values)
components = np.linspace(2, 30, 15, dtype = int)

ENDict = {}; L1Dict = {}; L2Dict = {}; OLSDict = {}; CatDict = {}; GradDict = {}

models = ["Elastic Net", "L1", "L2", "OLS", "CatBoost", "Gradient Boosting"]
Dicts = [ENDict, L1Dict, L2Dict, OLSDict, CatDict, GradDict]

for i in tqdm(range(len(components))):
    pca = PCA(n_components = components[i])
    pcaX = pca.fit_transform(X)
    poly = PolynomialFeatures(degree = 2, include_bias = False)
    poly_features = poly.fit_transform(pcaX)
    poly_features = poly_features.astype(float)

    ENDict.update({f"{components[i]}" : ElasticNetModel()})
    L1Dict.update({f"{components[i]}" : L1()})
    L2Dict.update({f"{components[i]}" : L2()})
    OLSDict.update({f"{components[i]}" : OLS()})
    CatDict.update({f"{components[i]}" : yowl()})
    GradDict.update({f"{components[i]}" : GradBoost()})

for i in range(len(models)):
    plotter(Dicts[i], models[i])