import os, math, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
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
    plt.plot(range(1, len(r2_means)+1), r2_means, marker='o')
    plt.title('Mean R2 Score over PLS Components')
    plt.xlabel('Number of PLS Components')
    plt.ylabel('Mean R2 Score')
    plt.grid(True)
    plt.show()

def train(X, y, file_directory):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    results = pd.DataFrame(columns=["PLS_components", "MSE", "R2", "RMSE", "MAE"])
    all_r2_scores = []
    all_model_results = pd.DataFrame(columns=["PLS_components", "Model", "MSE", "R2", "RMSE", "MAE"]) # moved outside the loop
    
    for n_components in range(1, 51):

        mse_scores = []; r2_scores = []; rmse_scores = []; mae_scores = []

        for j in tqdm(range(10)):
            rand_state = random.randint(1, 100)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

            model = PLSRegression(n_components=n_components)
            model.fit(X_train, y_train.values.ravel())
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = math.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mse_scores.append(mse)
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            mae_scores.append(mae)

            all_model_results.loc[len(all_model_results), ["PLS_components", "Model", "MSE", "R2", "RMSE", "MAE"]] = [n_components, j+1, mse, r2, rmse, mae]

        all_r2_scores.append(r2_scores)
        print(f"PLS Components: {n_components}")
        print(all_model_results.to_string(index=False))
        print("-----------------------------")

        results = results.append({
            "PLS_components": n_components,
            "MSE": np.mean(mse_scores),
            "R2": np.mean(r2_scores),
            "RMSE": np.mean(rmse_scores),
            "MAE": np.mean(mae_scores)}, ignore_index=True)

    plot_avg_r2(all_r2_scores)

    suffix = generate_file_suffix(file_directory, "PLS_model_results_")
    results.to_csv(f"{file_directory}/PLS_model_results_{suffix}.csv", index=False)
    all_model_results.to_csv(f"{file_directory}/PLS_model_results_all_{suffix}.csv", index=False)

# The main block (execute this script directly)
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    current_directory = os.getcwd()
    main_directory = current_directory[:-len('Scripts\ML_Models\PLSRegression')]
    file_directory = f"{main_directory}Dataset"
    os.chdir(file_directory)

    dataset = drop_zero_cols(pd.read_csv(f"{file_directory}/master_dataset_diff_prcd.csv"))
    features = dataset.iloc[:, 8:]
    X = features
    y = dataset.iloc[:, 6]

    train(X, y, file_directory)
