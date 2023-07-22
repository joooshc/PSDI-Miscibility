import catboost as cb
import numpy as np
import pandas as pd
import warnings
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, r2_score
import random as rand
from tqdm import tqdm
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('IUPAC_Dataset/Dataset/method1_dataset.csv')

zero_cols = list(dataset.columns[(dataset == 0).all()]) #Removing columns with all zeros
for col in zero_cols:
    del dataset[col]

y = dataset.iloc[0:, 6].values #mole fraction mean
X = pd.DataFrame(dataset.iloc[0:, 7:].values)
pca = PCA(n_components = 7)
pcaX = pca.fit_transform(X)
eval_dict = {}

def models():
    mseList = []; scoreList = []
    for i in tqdm(range(10)):
        X_train, X_test, y_train, y_test = train_test_split(pcaX, y, test_size = 0.2, random_state = rand.randint(0, 100))
        train = cb.Pool(X_train, y_train)
        test = cb.Pool(X_test, y_test)
        # model = cb.CatBoostRegressor(loss_function = "RMSE", logging_level='Silent')

        # grid = {"iterations": [100, 150, 200],
                # "learning_rate": [0.01, 0.05, 0.1],
                # "depth": [4, 6, 8],
                # "l2_leaf_reg": [0.2, 0.6, 1]}
        # model.grid_search(grid, train, verbose = False)

        model = cb.CatBoostRegressor(iterations = 200,
                            learning_rate = 0.05,
                            depth = 7,
                            l2_leaf_reg = 0.2,
                            loss_function = 'RMSE',
                            logging_level='Silent')
        model.fit(train)

        y_pred = model.predict(X_test)
        mseList.append(mean_squared_error(y_test, y_pred))
        scoreList.append(r2_score(y_test, y_pred))
    return mseList, scoreList

def evaluation(mseList, scoreList): #Evaluation metrics
    avScore = np.mean(scoreList)
    scoreCount = sum(i < 0 for i in scoreList)
    mseCount = sum(i > 1 for i in mseList)
    avmse = np.mean(mseList)

    eval_dict_vals = [mseCount, max(mseList), min(mseList), np.std(mseList), avmse, scoreCount,  max(scoreList), min(scoreList), np.std(scoreList), avScore]
    # eval_dict.update(eval_dict_vals)

    return eval_dict_vals

mseList, scoreList = models()
eval_dict_vals = evaluation(mseList, scoreList)
print(eval_dict_vals)

eval_cols = ["mseCount", "maxMSE", "minMSE", "stdMSE", "avMSE", "scoreCount", "maxScore", "minScore", "stdScore", "avScore"]
eval_df = pd.DataFrame([eval_dict_vals], columns = eval_cols)
print(eval_df)
eval_df.to_csv('IUPAC_Dataset/Scripts/ML_Models/CatBoost/m1.csv')