import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, time, warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, QuantileTransformer

def drop_zero_cols(dataset):
    zero_cols = list(dataset.columns[(dataset == 0).all()])
    for col in zero_cols:
        del dataset[col]

    return dataset

def save_dropped_data(dropped_rows_dict):
    dropped_data = []
    for feature, dropped_rows in dropped_rows_dict.items():
        for row in dropped_rows:
            dropped_data.append(dataset.loc[row])  
    dropped_data_df = pd.DataFrame(dropped_data) 
    dropped_data_df.to_csv("dropped_data.csv", index=None)

def top_non_zero_features(dataset, num_features):
    non_zero_counts = dataset.iloc[:, 7:].astype(bool).sum(axis=0)
    top_features = non_zero_counts.sort_values(ascending=False).head(num_features).index.tolist()

    return top_features

def plot_hists(dataset, num_features):
    """
    For the top non zero features use:

    top_features = top_non_zero_features(dataset, dataset.shape[1])[1:num_features+1]
    """
    
    # Joining the top_features and iupac_features
    for feature_window in range(0, len(dataset.iloc[:, 7:].columns), num_features):
        features_to_plot = dataset.iloc[:, 7+feature_window:(7+feature_window)+num_features]
        n_rows = 5
        n_cols = 4

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 2*n_rows)) # Adjust figsize to (15, 2*n_rows) for better subplot size
        axes = axes.flatten()

        for idx, feature in enumerate(features_to_plot.columns):
            if idx >= len(axes):  # Stop the loop if idx goes beyond the length of axes
                break
            ax = axes[idx]
            ax.hist(features_to_plot[feature].dropna(), bins=60)
            ax.set_title(f'{feature}')

        plt.tight_layout()
        plt.show()

def check_skewness(dataset):
    skewness_index = []
    skewness = []
    for col in dataset.iloc[:, 7:].columns:
        print(f'Skewness of {col}: {dataset[col].skew()}')
        skewness_index.append((col, dataset[col].skew()))
        skewness.append(dataset[col].skew())
    print(f"\n\nAverage Skewness: {np.mean(skewness)}")
    skewness_index_df = pd.DataFrame(skewness_index)
    skewness_index_df.rename(columns={0: 'Features', 1: 'Skewness'}, inplace=True)
    # skewness_index_df.to_csv("skewness_index.csv")

def feature_removal(dataset, main_directory):
    FeatureScaling = pd.read_csv(f"{main_directory}Results\Preprocessing\FeatureScaling.csv")
    to_remove = np.array(FeatureScaling.iloc[:, :1].dropna()).T[0]

    for feature in dataset.columns:
        if feature in to_remove or feature.endswith('.1'):
            del dataset[feature]

    return dataset

def feature_scaling(dataset_selected, scaling_method, min_val_dict=None):
    to_scale = dataset_selected.iloc[:, 7:].columns

    if scaling_method != 'none':
        print(f"After {scaling_method} scaling:")

    inf_idxs = []; nan_idxs = []; dropped_rows_dict = {}

    # Initialize a dictionary to store the min values if it's not provided (assumes training data)
    if min_val_dict is None:
        min_val_dict = {}

    for feature in to_scale:
        scaler = None

        if np.isinf(dataset_selected[feature]).any():
                max_val = dataset_selected[feature][~np.isinf(dataset_selected[feature])].max()
                min_val = dataset_selected[feature][~np.isinf(dataset_selected[feature])].min()
                dataset_selected[feature] = dataset_selected[feature].replace([np.inf, -np.inf], [max_val, min_val])

        if dataset_selected[feature].isnull().any():
            mode_val = dataset_selected[feature].mode()[0]
            dataset_selected[feature].fillna(mode_val, inplace=True)

        if scaling_method == 'log':
            # If the min_val_dict has a value for this feature, use it. Otherwise, calculate it and store.
            if feature in min_val_dict:
                min_val = min_val_dict[feature]
            else:
                min_val = dataset_selected[feature].min()
                min_val_dict[feature] = min_val
            
            if min_val <= 0:
                dataset_selected[feature] = np.log(dataset_selected[feature] - min_val + 1)
            else:
                dataset_selected[feature] = np.log(dataset_selected[feature] + 1e-7)
        
        elif scaling_method == 'quantile':
            scaler = QuantileTransformer(output_distribution = 'normal')

        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()

        elif scaling_method == 'standard':
            scaler = StandardScaler()

        elif scaling_method == 'normalize':
            mean = np.mean(dataset_selected[feature])
            std = np.std(dataset_selected[feature])
            low_limit = mean - 3 * std
            high_limit = mean + 3 * std

            # Store the outliers in the dropped_rows_dict
            outlier_rows = dataset_selected.loc[(dataset_selected[feature] < low_limit) | (dataset_selected[feature] > high_limit)]
            dropped_rows_dict[feature] = outlier_rows.index.tolist()

            # Replace the outliers with the corresponding threshold values
            dataset_selected.loc[dataset_selected[feature] < low_limit, feature] = low_limit # + 0.25*std
            dataset_selected.loc[dataset_selected[feature] > high_limit, feature] = high_limit # - 0.25*std

            # Perform normalization
            dataset_selected[feature] = (dataset_selected[feature] - mean) / std

        if scaler is not None:
            dataset_selected[feature] = scaler.fit_transform(dataset_selected[feature].values.reshape(-1,1)).flatten()

        num_infs = np.count_nonzero(dataset_selected[feature] == (-np.inf))
        num_nans = dataset_selected[feature].isna().sum()
        inf_indices = np.where(dataset_selected[feature] == (-np.inf))
        nan_indices = np.where(dataset_selected[feature].isna())
        inf_idxs.append(inf_indices)
        nan_idxs.append(nan_indices)

        if scaling_method != 'none':
            print(f"{feature} --- Num of infs: {num_infs}, Num of NaNs: {num_nans}")

    return dataset_selected, dropped_rows_dict, min_val_dict

def scale_y(y):
    y_log = np.log(y)
    y_log = y_log.replace([np.inf, -np.inf], np.nan)
    
    mode_val = y_log.mode(dropna=True)[0]  # Compute the mode
    y_log.fillna(mode_val, inplace=True)  # Fill NA/NaN values using the computed mode

    '''
    # Reshape y_log to a 2D array
    y_reshaped = y_log.values.reshape(-1, 1)
    y_scaled = MinMaxScaler().fit_transform(y_reshaped)
    # to revert y back to the original scale use:
    y = MinMaxScaler().inverse_transform(y_scaled)
    '''
    return y_log

def generate_val(dataset_scaled, dropped_data_path):
    # Shuffle the dataset_scaled
    dataset_scaled_shuffled = dataset_scaled.sample(frac=1)
    dataset_scaled_part = dataset_scaled_shuffled.sample(frac=0.20) # take out 14%

    # Remove the validation data from the original dataset
    dataset_scaled = dataset_scaled.drop(dataset_scaled_part.index)

    # dropped_data = pd.read_csv(dropped_data_path)
    # dropped_data_shuffled = dropped_data.sample(frac=1)
    # dropped_data_part = dropped_data_shuffled.sample(frac=0.06) # take out 6%

    # Combine dataset_scaled_part and dropped_data_part

    # master_validation_set = pd.concat([dataset_scaled_part, dropped_data_part], ignore_index=True)
    # master_validation_set.to_csv("master_validation_set.csv", index=False)

    return dataset_scaled_part, dataset_scaled

# The main block (execute this script directly)
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 

    current_directory = os.getcwd()
    main_directory = current_directory[:-len('Scripts')]
    file_directory = f"{main_directory}Dataset"
    os.chdir(file_directory)

    dataset = pd.read_csv(f"{file_directory}\master_dataset_diff.csv")
    dataset_non_zeros = drop_zero_cols(dataset)
    dataset_selected = dataset_non_zeros
    # dataset_selected = feature_removal(dataset_non_zeros, main_directory)
    validation_set, dataset_selected = generate_val(dataset_selected, "dropped_data.csv")

    # Initialize a dictionary to store the min values for log scaling
    min_val_dict = {}

    # For the training dataset
    dataset_scaled, dropped_rows_dict, min_val_dict = feature_scaling(dataset_selected, 'standard', min_val_dict)

    # Save the dropped data into a csv file
    save_dropped_data(dropped_rows_dict)

    y1 = dataset_scaled.iloc[:, 5]
    y2 = dataset_scaled.iloc[:, 6]
    dataset_scaled.iloc[:, 5] = y1
    dataset_scaled.iloc[:, 6] = y2

    check_skewness(dataset_scaled)
    # plot_hists(dataset_scaled, 20)

    print(pd.DataFrame(dataset_scaled).shape)
    dataset_scaled.to_csv("method1_dataset.csv", index=False)


