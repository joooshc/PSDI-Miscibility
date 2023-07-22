import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import warnings
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, r2_score
import random as rand
from tqdm import tqdm
import numpy as np

dataset = pd.read_csv('IUPAC_Dataset/Dataset/method1_dataset.csv')

zero_cols = list(dataset.columns[(dataset == 0).all()]) #Removing columns with all zeros
for col in zero_cols:
    del dataset[col]

y = dataset.iloc[0:, 6].values #mole fraction mean
X = pd.DataFrame(dataset.iloc[0:, 7:].values)
pca = PCA(n_components = 14)
pcaX = pca.fit_transform(X)

# def models(estimators, depth, samples, rate): 
def models():
    mseList = []
    scoreList = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(pcaX, y, test_size = 0.2, random_state = rand.randint(0, 100))

        # params = {'n_estimators': estimators, 'max_depth': depth, 'min_samples_split': samples, 'learning_rate': rate, 'loss': "squared_error"}
        params = {'n_estimators': 550, 'max_depth': 5, 'min_samples_split': 7, 'learning_rate': 0.0775, 'loss': "squared_error"}

        model = GradientBoostingRegressor(**params).fit(X_train, y_train)
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

    return eval_dict_vals

### Hyperparameter tuning ###
# estimators = list(np.linspace(100, 1000, 5, dtype = int))
# depth = list(np.linspace(1, 10, 5, dtype = int))
# samples = list(np.linspace(2, 10, 4, dtype = int))
# rate = list(np.linspace(0.01, 0.1, 5))

###
eval_dict = {}

# for e in range(0, len(estimators)):
#     for d in range(0, len(depth)):
#         for s in range(0, len(samples)):
#             for r in tqdm(range(0, len(rate)), desc='Estimators: ' + str(estimators[e]) + ', Depth: ' + str(depth[d]) + ', Samples: ' + str(samples[s])):
#                 mseList, scoreList = models(estimators[e], depth[d], samples[s], rate[r])
#                 eval_dict_vals = evaluation(mseList, scoreList)
#                 eval_dict[(estimators[e], depth[d], samples[s], rate[r])] = eval_dict_vals

mseList, scoreList = models()
eval_dict_vals = evaluation(mseList, scoreList)

eval_df = pd.DataFrame.from_dict(eval_dict, orient = 'index', columns = ['mseCount', 'maxMSE', 'minMSE', 'stdMSE', 'avMSE', 'scoreCount', 'maxScore', 'minScore', 'stdScore', 'avScore'])
eval_cols = ["mseCount", "maxMSE", "minMSE", "stdMSE", "avMSE", "scoreCount", "maxScore", "minScore", "stdScore", "avScore"]
eval_df = pd.DataFrame([eval_dict_vals], columns = eval_cols)

components = np.linspace(2, 20, 10, dtype = int) #Number of components to test
components = list(components)
eval_dict = {}

print(eval_df)
# eval_df.to_csv('IUPAC_Dataset/Scripts/ML_Models/GradientBoosting/m1.csv')