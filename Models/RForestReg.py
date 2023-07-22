import os, math, time, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def drop_zero_cols(dataset):
    zero_cols = list(dataset.columns[(dataset == 0).all()])
    for col in zero_cols:
        del dataset[col]
    return dataset

def generate_file_suffix(directory, prefix):
    files = os.listdir(directory)
    count = 0
    for file in files:
        if file.startswith(prefix):
            count += 1
    return count + 1

def plot_avg_r2(r2_scores):
    r2_means = [np.mean(scores) for scores in r2_scores]
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(r2_means)+1), r2_means, marker='o')  # adjust the range to match the length of r2_means
    plt.title('Mean R2 Score over PCA Components')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Mean R2 Score')
    plt.grid(True)
    plt.show()

def train_PCA(X, y, file_directory, grid_search_flag):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    results = pd.DataFrame(columns=["PCA_components", "Best_Params", "Max_MSE", "Min_MSE", "Max_R2", "Min_R2", "Max_RMSE", "Min_RMSE", "Max_MAE", "Min_MAE"])
    all_model_results = pd.DataFrame(columns=["PCA_components", "Model", "MSE", "R2", "RMSE", "MAE", "Best Params"])
    all_r2_scores = []; mean_r2_scores = []; y_tests = []; y_preds = []
    # for n_components in range(1, X.shape[1]-150):
    n_components = 7

    loop_model_results = pd.DataFrame(columns=["PCA_components", "Model", "MSE", "R2", "RMSE", "MAE", "Best Params"])
    best_params_df = pd.DataFrame(columns=['Model', 'n_estimators', 'max_depth', 'min_samples_split', 'max_features'])

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    mse_scores = []; r2_scores = []; rmse_scores = []; mae_scores = []; best_params_list = []; y_test_list = []; y_pred_list = []

    for j in tqdm(range(10)):
        rand_state = random.randint(1, 101)
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=rand_state)

        model = RandomForestRegressor(random_state=rand_state)
        
        best_params = {}

        if grid_search_flag:
            param_grid = {
                        'n_estimators': [10, 50, 100, 200],
                        'max_depth': [None, 10, 20, 30, 40],
                        'min_samples_split': [2, 5, 10],
                        'max_features': ['None', 'sqrt', 'log2'],
                    }

            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
            grid_search.fit(X_train, y_train.values.ravel())

            best_params = grid_search.best_params_
            best_params_list.append(best_params)

            best_params_df.loc[j, 'Model'] = j+1
            best_params_df.loc[j, 'n_estimators'] = best_params['n_estimators']
            best_params_df.loc[j, 'max_depth'] = best_params['max_depth']
            best_params_df.loc[j, 'min_samples_split'] = best_params['min_samples_split']
            best_params_df.loc[j, 'max_features'] = best_params['max_features']

            model = RandomForestRegressor(**best_params)

        model.fit(X_train, y_train.ravel())
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)

        y_test_list.append(y_test)
        y_pred_list.append(y_pred)

        loop_model_results.loc[j, ["PCA_components", "Model", "MSE", "R2", "RMSE", "MAE", "Best Params"]] = [n_components, j+1, mse, r2, rmse, mae, str(best_params)]

    all_r2_scores.append(r2_scores)
    mean_r2_scores.append([np.mean(r2_scores), n_components])
    y_tests.append(y_test_list)
    y_preds.append(y_pred_list)

    combined_results = pd.concat([best_params_df.set_index('Model'), loop_model_results.set_index('Model').drop(['PCA_components', 'Best Params'], axis=1)], axis=1).reset_index()
    print(f"PCA Components: {n_components}")
    print(combined_results.to_string(index=False))
    print("-----------------------------")

    all_model_results = all_model_results.append(loop_model_results)

    results = results.append({
        "PCA_components": n_components,
        "Best_Params": str(best_params),
        "Max_MSE": max(mse_scores),
        "Min_MSE": min(mse_scores),
        "Max_R2": max(r2_scores),
        "Min_R2": min(r2_scores),
        "Max_RMSE": max(rmse_scores),
        "Min_RMSE": min(rmse_scores),
        "Max_MAE": max(mae_scores),
        "Min_MAE": min(mae_scores)}, ignore_index=True)

    plot_avg_r2(all_r2_scores)

    suffix = generate_file_suffix(file_directory, "RForest_model_results_")
    pd.DataFrame(results).to_csv(f"{file_directory}\RForest_model_results_(summary)_{suffix}.csv", index=False)
    all_model_results.to_csv(f"{file_directory}\\RForest_model_results_(all)_{suffix}.csv", index=False)

    best_r2_score, best_n_components = max(mean_r2_scores, key=lambda item:item[0])
    idx = next(i for i, v in enumerate(mean_r2_scores) if v[1] == best_n_components)
    best_y_tests = y_tests[idx]
    best_y_preds = y_preds[idx]

    return best_r2_score, best_n_components, best_y_tests, best_y_preds

def train_RFE(X, y, file_directory, grid_search_flag):
    warnings.simplefilter(action='ignore', category=FutureWarning)

    model = RandomForestRegressor(random_state=42)
    
    # RFECV performs RFE in a cross-validation loop to find the optimal number or the best number of features.
    rfecv = RFECV(estimator=model, step=1, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    print("Starting RFECV...")
    rfecv.fit(X, y)
    print("RFECV completed.")

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.title("Performance of the RFECV")
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (neg MSE)")  
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), -rfecv.cv_results_['mean_test_score'], marker='o')
    plt.show()

    feature_rankings = pd.DataFrame({'Feature': X.columns, 'Ranking': rfecv.ranking_})
    feature_rankings.to_csv(f"{file_directory}/RFE_feature_rankings.csv", index=False)

    X = rfecv.transform(X)

    if grid_search_flag:
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'max_features': ['None', 'sqrt', 'log2'],
        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
        print("Starting GridSearchCV...")
        grid_search.fit(X, y)
        print("GridSearchCV completed.")
        model = RandomForestRegressor(**grid_search.best_params_)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")

    joblib.dump(model, f"{file_directory}/RFE_model.pkl")

def plot_scatterplots(best_n_components, y_tests, y_preds):
    fig, axs = plt.subplots(5, 2, figsize=(15, 25))  # adjust the size as needed
    axs = axs.ravel()

    for i in range(len(y_tests)):
        y_tests[i] = np.squeeze(np.asarray(y_tests[i]))
        y_preds[i] = np.squeeze(np.asarray(y_preds[i]))

        axs[i].scatter(y_tests[i], y_preds[i])

        m, b = np.polyfit(y_tests[i], y_preds[i], 1)
        axs[i].plot(y_tests[i], m*y_tests[i] + b, color='black')

        r2 = r2_score(y_tests[i], y_preds[i])
        axs[i].legend([f'R2 = {r2:.2f}'])

        axs[i].set_xlabel('y_test')
        axs[i].set_ylabel('y_pred')

    fig.suptitle(f'PCA Components: {best_n_components}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

# The main block (execute this script directly)
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    current_directory = os.getcwd()
    main_directory = current_directory[:-len('Scripts\ML_Models\RandomForestRegression')]
    file_directory = f"{main_directory}Dataset"
    os.chdir(file_directory)

    dataset = drop_zero_cols(pd.read_csv(f"{file_directory}\method1_dataset.csv"))
    features = dataset.iloc[:, 8:]
    X = features
    y = dataset.iloc[:, 6]

    best_r2_score, best_n_components, best_y_tests, best_y_preds = train_PCA(X, y, file_directory, grid_search_flag=False)
    
    print(best_n_components, best_r2_score)
    print()
    plot_scatterplots(best_n_components, best_y_tests, best_y_preds)

    # train_RFE(X, y, file_directory, grid_search_flag=False)

