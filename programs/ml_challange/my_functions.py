# Core libraries
import time
from datetime import datetime
import numpy as np
import pandas as pd
import io
import scipy.stats as stats

# Scikit-learn modules
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
# from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# XGBoost
import xgboost as xgb
from xgboost import XGBRegressor

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Function for Box-Cox inverse transformation
from scipy.special import inv_boxcox

# for not printing convergence warnings
import warnings

#for Step 2
from transformers import AutoTokenizer, AutoModel
import torch
from PIL import Image
# from torchvision import models, transforms
# import torch.nn as nn

# for PCA
from sklearn.decomposition import PCA 

import os



# --- Functions for each task ---

# read a sample of rows from a CSV file as a dataframe
def initialize(input_file, sample_fraction=1):
    df = pd.read_csv(input_file)
    sampled_df = df.sample(frac=sample_fraction, random_state=42)  # Ensures reproducibility
    return sampled_df

# capture df.info() output and return it as a DataFrame
def get_info_as_dataframe(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue().splitlines()
    info_data = []
    for line in info_str[5:-2]:  # Skipping header and footer lines
        parts = line.split()
        col_name = parts[1]  # Column name
        non_null_count = int(parts[2])  # Non-null count
        dtype = parts[-1]  # Data type
        info_data.append([col_name, non_null_count, dtype])
    info_df = pd.DataFrame(info_data, columns=['Column', 'Non-Null Count', 'Data Type'])
    return info_df

# extract dataframe, target column (after applying Box-Cox transformation), lambda_optimal (Box-Cox transformation), description column. It crops 'nr_' and 'sh_' from the beginning of entries of feature_1 and feature_2. It returns a fraction of dataframe
def prepare_data(df, sample_fraction=1):
    # Store the 'description' column before dropping
    df_copy = df.copy()
    deleted_column = df_copy['description'] if 'description' in df_copy.columns else None
    if 'description' in df_copy.columns:
        df_copy.drop(columns=['description'], inplace=True)
    sampled_df_copy = df_copy.sample(frac=sample_fraction, random_state=42)  # Sampling
    df_copy = sampled_df_copy
    # Modify features
    df_copy['feature_1'] = df_copy['feature_1'].str.replace('nr_', '').astype(int)
    df_copy['feature_2'] = df_copy['feature_2'].str.replace('sh_', '').astype(int)
    # Split features and target
    X = df_copy.drop(columns=['target'])
    # Apply Box-Cox transformation
    y, lambda_optimal = stats.boxcox(df_copy['target'])
    return X, y, lambda_optimal, deleted_column

#drops colums with less thna 2 non-null values, removes outliers from train set, scale X_train and X_test, imput mean in missing values. Input dataframe may contain a description column 
def scale_and_impute_data(X_train, X_test, y_train):
    # Separate the "description" column from the rest of the features
    contains_description=False
    if 'description' in X_train.columns:
        contains_description=True
        description_train = X_train['description']
        description_test = X_test['description']
        X_train = X_train.drop(columns=['description'])
        X_test = X_test.drop(columns=['description'])
    # Drop columns with less than 2 non-null values
    X_train = X_train.dropna(thresh=2, axis=1)
    X_test = X_test[X_train.columns]  # Ensure X_test has the same columns as X_train
    # Calculate Z-scores and remove outliers in y_train
    z_scores = np.abs(stats.zscore(y_train))
    X_train_no_outliers = X_train[z_scores < 4]
    y_train_no_outliers = y_train[z_scores < 4]
    if contains_description:
        description_train_no_outliers = description_train[z_scores < 4]
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_no_outliers)
    X_test_scaled = scaler.transform(X_test)
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train_scaled)
    X_test_imputed = imputer.transform(X_test_scaled)
    # Convert arrays back to DataFrames, keeping column names
    X_train_imputed_df = pd.DataFrame(X_train_imputed, columns=X_train_no_outliers.columns, index=X_train_no_outliers.index)
    X_test_imputed_df = pd.DataFrame(X_test_imputed, columns=X_train_no_outliers.columns, index=X_test.index)
    # Add the "description" column back
    if contains_description:
        X_train_imputed_df['description'] = description_train_no_outliers
        X_test_imputed_df['description'] = description_test
    return X_train_imputed_df, X_test_imputed_df, y_train_no_outliers

# check normality of the target variable
def check_normality(y, pictures_folder="."):
    # Add current hour to file name
    current_hour = datetime.now().strftime("%H:%M:%S")
    plt.figure(figsize=(10, 6))
    sns.histplot(y, bins=30, kde=True, color='blue', alpha=0.7)
    plt.title('Histogram of y after Box-Cox')
    plt.xlabel('Log(Target)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f"{pictures_folder}/histogram_after_boxcox_{current_hour}.svg", format="svg")
    #plt.show(block=False)
    # Q-Q plot
    plt.figure(figsize=(10, 6))
    stats.probplot(y, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.grid(True)
    plt.savefig(f"{pictures_folder}/qq_plot_{current_hour}.svg", format="svg")
    #plt.show(block=False)
    # Shapiro-Wilk test
    stat, p_value = stats.shapiro(y)
    print(f'Statistic: {stat}, p-value: {p_value}')
    alpha = 0.05
    if p_value > alpha:
        print("Fail to reject the null hypothesis: Data is normally distributed")
    else:
        print("Reject the null hypothesis: Data is not normally distributed")

# Function to perform train-test split and add deleted column if not None
def split_data(X, y, deleted_column=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Add deleted column back to train and test sets
    if deleted_column is not None:
        deleted_train, deleted_test = train_test_split(deleted_column, test_size=0.2, random_state=42)
        X_train['description'] = deleted_train
        X_test['description'] = deleted_test
    return X_train, X_test, y_train, y_test

# Function to handle feature selection (removing features with few non-null values)
def filter_features(X_train_scaled, info_df):
    features_names = info_df[
        (info_df['Non-Null Count'] > 1) & (info_df['Column'] != 'target')
    ]['Column'].tolist()
    columns_to_drop = info_df[info_df['Non-Null Count'] < 2]['Column'].tolist()
    all_columns = X_train_scaled.columns.tolist()
    columns_to_drop_indices = [all_columns.index(col) for col in columns_to_drop]
    indices_to_keep = [i for i in range(X_train_scaled.shape[1]) if i not in columns_to_drop_indices]
    return X_train_scaled[:, indices_to_keep], features_names

# Function to train models (random forest, xgboost, svr, elasticnet) and perform GridSearchCV over set of parameters
def train_models(X_train, y_train, X_test, y_test):
    # Ignore ConvergenceWarning
    start_time = time.time()
    print(f"Current Time: {datetime.now().strftime('%H:%M:%S')}")
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.neural_network')
    models_and_params = [
        # Random Forest
        {
            'name': 'Random Forest',
            'model': RandomForestRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5]
            }
        },
        # XGBoost
        {
            'name': 'XGBoost',
            'model': XGBRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.7, 1.0],
                }
        },
        # SVR
        {
            'name': 'SVR',
            'model': SVR(),
            'param_grid': {
                'C': [0.1, 1, 10],
                'epsilon': [0.01, 0.1, 0.5],
                'kernel': ['rbf', 'linear']
            }
        },
        # ElasticNet
        {
            'name': 'ElasticNet',
            'model': ElasticNet(random_state=42),
            'param_grid': {
                'alpha': [0.01, 0.1, 1],
                'l1_ratio': [0.1, 0.5, 0.9]
            }
        },
        # # Neural Network
        # {
        #     'name': 'Neural Network',
        #     'model': MLPRegressor(random_state=42),
        #     'param_grid': {
        #         'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        #         'activation': ['relu', 'tanh'],
        #         'learning_rate_init': [0.001, 0.01],
        #         'max_iter': [500]
        #     }
        # }
    ]
    results = []
    for model_info in models_and_params:
        name = model_info['name']
        model = model_info['model']
        param_grid = model_info['param_grid']
        print(f"Training {name}...")
        step_start_time = time.time()
        # Hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        execution_time = time.time() - step_start_time
        results.append({
            'Model': name,
            'Best Params': grid_search.best_params_,
            'RMSE': rmse,
            'Execution Time (s)': round(execution_time, 2)
        })
    overall_execution_time = time.time() - start_time
    print(f"Execution Time: {round(overall_execution_time, 2)} seconds")
    print(f"Current Time: {datetime.now().strftime('%H:%M:%S')}")
    play_sound()
    return pd.DataFrame(results).sort_values(by='RMSE')


def train_and_plot_xgboost(X_train_imputed, y_train, X_test_imputed, y_test, lambda_optimal, CPU=True, only_one_parameter=True, pictures_folder="."):
    return train_and_plot_model(X_train_imputed, y_train, X_test_imputed, y_test, lambda_optimal, model_type='xgboost', CPU=CPU, only_one_parameter=only_one_parameter, pictures_folder=pictures_folder)

def train_and_plot_model(X_train_imputed, y_train, X_test_imputed, y_test, lambda_optimal, model_type='xgboost', CPU=True, only_one_parameter=True, pictures_folder="."):
    """
    Train and evaluate a specified model ('xgboost', 'svr', or 'elasticnet').
    Perform hyperparameter tuning using GridSearchCV, plot results, and calculate metrics.
    Parameters:
    - X_train_imputed, y_train: Training features and target values.
    - X_test_imputed, y_test: Test features and target values.
    - lambda_optimal: Optimal lambda value for inverse Box-Cox transformation.
    - model_type: The type of model to train ('xgboost', 'svr', or 'elasticnet').
    - CPU: Boolean, if True, use CPU for training (specific to XGBoost).
    - only_one_parameter: Boolean, if True, use a simplified parameter grid.
    - pictures_folder: Path to save output plots.
    Returns:
    - Dictionary with best parameters, RMSE, R^2, EVS, predictions, feature importances, and feature names.
    """
    print(f"Initializing {model_type.upper()} model training...")
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(
            random_state=42,
            objective='reg:squarederror',
            tree_method='gpu_hist' if not CPU else 'hist',
            device="gpu" if not CPU else "cpu",
            n_jobs=-1,
            eval_metric='rmse',
            verbosity=1,
            max_bin=64
        )
        param_grid = {
            'n_estimators': [400],
            'max_depth': [9],
            'learning_rate': [0.05],
            'subsample': [0.7]
        }
        if not only_one_parameter:
            param_grid = {
                'n_estimators': [100, 200, 250, 300, 350, 400],
                'max_depth': [5, 8, 9, 10, 15],
                'learning_rate': [0.1, 0.05, 0.01],
                'subsample': [0.7, 0.8]
            }
    elif model_type == 'svr':
        model = SVR()
        param_grid = {
            'C': [0.1, 1],
            'epsilon': [0.01, 0.1],
            'kernel': ['rbf']
        }
    elif model_type == 'elasticnet':
        model = ElasticNet(random_state=42)
        param_grid = {
            'alpha': [0.01, 0.1, 1],
            'l1_ratio': [0.1, 0.5, 0.9]
        }
    else:
        raise ValueError("Invalid model_type. Choose 'xgboost', 'svr', or 'elasticnet'.")
    # Start time measurement
    print("Current Time:", datetime.now().strftime("%H:%M:%S"))
    step_start_time = time.time()  # Step timer
    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=3)
    grid_search.fit(X_train_imputed, y_train)
    best_model = grid_search.best_estimator_
    y_pred_boxcox = best_model.predict(X_test_imputed)
    # Apply inverse Box-Cox transformation
    y_pred_original = inv_boxcox(y_pred_boxcox, lambda_optimal)
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_boxcox))
    r2 = r2_score(y_test, y_pred_original)
    evs = explained_variance_score(y_test, y_pred_original)
    rmse_original = np.sqrt(mean_squared_error(inv_boxcox(y_test, lambda_optimal), y_pred_original))
    # Plot results
    plot_predictions_vs_target(y_pred_boxcox, y_test, pictures_folder=pictures_folder)
    inverse_boxcox_transform_and_plot(y_pred_boxcox, y_test, lambda_optimal, pictures_folder=pictures_folder)
    # Get feature importances for XGBoost
    importances = None
    feature_names = X_train_imputed.columns  # Assuming X_train_imputed is a DataFrame
    if model_type == 'xgboost':
        importances = best_model.feature_importances_
    overall_execution_time = time.time() - step_start_time
    print(f"Execution Time: {round(overall_execution_time, 2)} seconds")
    play_sound()
    result_dict = {
        'Best Params': grid_search.best_params_,
        'RMSE': rmse,
        'RMSE_original': rmse_original,
        'R2': r2,
        'EVS': evs,
        'y_pred': y_pred_boxcox,
        'Importances': importances,
        'Feature Names': feature_names
    }
    return result_dict



def train_and_plot_each_desc(X_train, y_train, X_test, y_test, lambda_optimal, model_type='xgboost', CPU=True, pictures_folder="."):
    """
    Train and evaluate a specified model ('xgboost', 'svr', or 'elasticnet') for each unique value 
    in the 'description' column, including NaN values.
    Calculate and return the RMSE for all predictions together.
    Parameters:
    - X_train, X_test: DataFrames with a 'description' column to filter by unique values.
    - y_train, y_test: Series or numpy arrays with target values corresponding to the training and test sets.
    - lambda_optimal: Optimal lambda value for regularization (used only for XGBoost).
    - model_type: The type of model to train ('xgboost', 'svr', or 'elasticnet').
    - CPU: Boolean, if True, use CPU for training (specific to XGBoost).
    - pictures_folder: Path to save output plots (if applicable).
    Returns:
    - Dictionary with results for each description, including best parameters and RMSE.
    - Overall RMSE across all predictions.
    """
    start_time = time.time()
    print(f"Current Time: {datetime.now().strftime('%H:%M:%S')}")
    unique_descriptions = X_train['description'].unique()
    model_results = {}
    all_preds = []
    all_true = []

    for desc in unique_descriptions:
        # Filter the train and test data based on the description
        if pd.isna(desc):
            X_train_filtered = X_train[X_train['description'].isna()]
            X_test_filtered = X_test[X_test['description'].isna()]
        else:
            X_train_filtered = X_train[X_train['description'] == desc]
            X_test_filtered = X_test[X_test['description'] == desc]

        # Check if there is any data after filtering
        if X_train_filtered.empty or X_test_filtered.empty:
            print(f"Skipping {desc} because there is no data for this description.")
            continue

        # Reset indices to avoid mismatches
        X_train_filtered = X_train_filtered.reset_index(drop=True)
        X_test_filtered = X_test_filtered.reset_index(drop=True)
        y_train_filtered = y_train[X_train_filtered.index]
        y_test_filtered = y_test[X_test_filtered.index]

        # Train and evaluate based on the specified model type
        if model_type == 'xgboost':
            result = train_and_plot_model(
                X_train_filtered.drop(columns=['description']),
                y_train_filtered,
                X_test_filtered.drop(columns=['description']),
                y_test_filtered,
                lambda_optimal, 
                model_type='xgboost', 
                CPU=CPU, pictures_folder=pictures_folder)
        elif model_type == 'svr':
            result = train_and_plot_model(
                X_train_filtered.drop(columns=['description']),
                y_train_filtered,
                X_test_filtered.drop(columns=['description']),
                y_test_filtered,
                lambda_optimal, 
                model_type='svr', 
                CPU=CPU, pictures_folder=pictures_folder)
        elif model_type == 'elasticnet':
            result = train_and_plot_model(
                X_train_filtered.drop(columns=['description']),
                y_train_filtered,
                X_test_filtered.drop(columns=['description']),
                y_test_filtered,
                lambda_optimal, 
                model_type='elasticnet', 
                CPU=CPU, pictures_folder=pictures_folder)
        else:
            raise ValueError("Invalid model_type. Choose 'xgboost', 'svr', or 'elasticnet'.")

        # Store the result for this subset
        model_results[desc] = {
            'Best Params': result['Best Params'],
            'RMSE': result['RMSE']
        }

        # Add the predictions and true values to the lists for overall RMSE calculation
        all_preds.extend(result['y_pred'])
        all_true.extend(y_test_filtered)

        print(f"Description: {desc}")
        print("Best Params:", result['Best Params'])
        print(f"{model_type.upper()} RMSE:", result['RMSE'])

    # Calculate overall RMSE if there were any predictions
    if all_preds and all_true:
        overall_rmse = np.sqrt(mean_squared_error(all_true, all_preds))
        print(f"Overall {model_type.upper()} RMSE:", overall_rmse)
    else:
        overall_rmse = np.nan  # If no predictions were made, set RMSE to NaN

    overall_execution_time = time.time() - start_time
    print(f"Execution Time: {round(overall_execution_time, 2)} seconds")
    print(f"Current Time: {datetime.now().strftime('%H:%M:%S')}")
    play_sound()
    return model_results, overall_rmse



#Function for Scatter Plot (Predictions vs Target)
def plot_predictions_vs_target(y_pred, y_test, title="Scatter Plot of Predictions vs Target", pictures_folder="."):
    # Add current hour to file name
    current_hour = datetime.now().strftime("%H:%M:%S") 
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=y_test, alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel('XGBoost Predictions')
    plt.ylabel('Target')
    plt.gca().set_aspect('equal', adjustable='box')
    max_value = max(max(y_pred), max(y_test))
    min_value = min(min(y_pred), min(y_test))
    plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', linewidth=2)
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)
    plt.grid(True)
    plt.savefig(f"{pictures_folder}/scatter_predictions_vs_target_{current_hour}.svg", format="svg")
    #plt.show(block=False)


#Function for Inverse Box-Cox and Original Value Plotting
def inverse_boxcox_transform_and_plot(y_pred, y_test, lambda_optimal, tick_interval=60000, pictures_folder="."):
    y_pred_original = inv_boxcox(y_pred, lambda_optimal)
    y_test_original = inv_boxcox(y_test, lambda_optimal)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred_original, y=y_test_original, alpha=0.7, color='blue')
    plt.title('Scatter Plot of Original Values (Predictions vs Target)')
    plt.xlabel('XGBoost Predictions (Original Scale)')
    plt.ylabel('Target (Original Scale)')
    plt.gca().set_aspect('equal', adjustable='box')
    max_value = max(max(y_pred_original), max(y_test_original))
    min_value = min(min(y_pred_original), min(y_test_original))
    plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', linewidth=2)
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)
    plt.xticks(np.arange(min_value, max_value + tick_interval, tick_interval), rotation=-90)
    plt.yticks(np.arange(min_value, max_value + tick_interval, tick_interval))
    plt.grid(True)
    # Add current hour to file name
    current_hour = datetime.now().strftime("%H:%M:%S") 
    plt.savefig(f"{pictures_folder}/scatter_predictions_vs_target_orginal_scale_{current_hour}.svg", format="svg")
    #plt.show(block=False)
    # RMSE for original values
    mse_original = mean_squared_error(y_test_original, y_pred_original)
    rmse_original = np.sqrt(mse_original)
    # print(f"RMSE (Original Values): {rmse_original}")
    return y_pred_original, y_test_original, rmse_original

def play_sound():
    pass
    #if you want to play sound uncomment the following lines
    # import numpy as np
    # import sounddevice as sd
    # duration = 0.5  # seconds
    # frequency = 440  # Hz
    # sample_rate = 44100
    # t = np.linspace(0, duration, int(sample_rate * duration), False)
    # wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    # sd.play(wave, sample_rate)
    # sd.wait()  # Wait until the sound finishes playing
    
#Functions for STEP 2:

# change description column to its encoding using pre-trained BERT model (columns desc_i)
def add_encoded_description(X_train, X_test):
    # Load the pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModel.from_pretrained('distilbert-base-uncased')
    # Helper function to encode descriptions
    def encode_descriptions(descriptions):
        # Clean NaN values
        descriptions = [desc if isinstance(desc, str) else '' for desc in descriptions]
        tokens = tokenizer(descriptions, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model(**tokens).last_hidden_state.mean(dim=1)
        return embeddings
    # Encode descriptions for both train and test sets
    train_embeddings = encode_descriptions(X_train['description'].tolist())
    test_embeddings = encode_descriptions(X_test['description'].tolist())
    # Convert embeddings to DataFrames with appropriate column names
    train_embeddings_df = pd.DataFrame(
        train_embeddings.cpu().numpy(),
        columns=[f'desc_{i}' for i in range(train_embeddings.size(1))],
        index=X_train.index
    )
    test_embeddings_df = pd.DataFrame(
        test_embeddings.cpu().numpy(),
        columns=[f'desc_{i}' for i in range(test_embeddings.size(1))],
        index=X_test.index
    )
    # Concatenate embeddings with the original DataFrames, excluding the "description" column
    X_train_with_desc = pd.concat([X_train.drop(columns=['description']), train_embeddings_df], axis=1)
    X_test_with_desc = pd.concat([X_test.drop(columns=['description']), test_embeddings_df], axis=1)
    return X_train_with_desc, X_test_with_desc

#Apply PCA to columns starting with 'desc_'. It only shows plots
def pca_desc_columns(dataframe, prefix='desc_', pictures_folder="."):
    # Add current hour to file name
    current_hour = datetime.now().strftime("%H:%M:%S") 
    # Select columns starting with 'desc_'
    desc_columns = dataframe.filter(like=prefix).columns
    # # Standardize the data
    scaled_data=dataframe[desc_columns]
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(dataframe[desc_columns])
    # Apply PCA
    pca = PCA()
    pca.fit(scaled_data)
    # Get eigenvalues
    eigenvalues = pca.explained_variance_
    # Plot eigenvalues
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
    plt.title('Eigenvalues of Principal Components')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.xticks(range(1, len(eigenvalues) + 1))
    plt.savefig(f"{pictures_folder}/pca_predictions_vs_target_all_{current_hour}.svg", format="svg")
    #plt.show(block=False)
    # Sort eigenvalues
    sorted_eigenvalues = np.sort(eigenvalues)
    # Get the 20 highest and 20 lowest eigenvalues
    highest_eigenvalues = sorted_eigenvalues[-21:-1]
    lowest_eigenvalues = sorted_eigenvalues[:20]
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    # Plot the highest eigenvalues
    axs[0].plot(range(1, 21), highest_eigenvalues, marker='o', color='blue')
    axs[0].set_title('20 Highest Eigenvalues')
    axs[0].set_xlabel('Principal Component Index')
    axs[0].set_ylabel('Eigenvalue')
    axs[0].grid()
    # Plot the lowest eigenvalues
    axs[1].plot(range(1, 21), lowest_eigenvalues, marker='o', color='red')
    axs[1].set_title('20 Lowest Eigenvalues')
    axs[1].set_xlabel('Principal Component Index')
    axs[1].set_ylabel('Eigenvalue')
    axs[1].grid()
    plt.tight_layout()
    plt.savefig(f"{pictures_folder}/pca_predictions_vs_target_20_20_{current_hour}.svg", format="svg")
    #plt.show(block=False)

#Apply PCA to all columns or these starting with prefix. Replace the columns with nr of most important principal components. Shows plots
def pca_transform_and_clean(train_dataframe, test_dataframe, nr=20, prefix=None, pictures_folder="."):
    start_time = time.time()
    train_df_copy = train_dataframe.copy()
    test_df_copy = test_dataframe.copy()    
    # Determine file name and title based on prefix
    if prefix:
        desc_columns_train = train_dataframe.filter(like=prefix).columns
        title_suffix = f" with Prefix '{prefix}'"
        file_suffix = f"{prefix}_"
    else:
        desc_columns_train = train_dataframe.columns
        title_suffix = " for All Columns"
        file_suffix = "all_columns_" 
    # Add current hour to file name
    current_hour = datetime.now().strftime("%H:%M:%S")   
    # Apply PCA
    pca = PCA()
    pca_components_train = pca.fit_transform(train_df_copy[desc_columns_train])
    explained_variance_ratio = pca.explained_variance_ratio_ * 100
    cumulative_explained_variance = explained_variance_ratio.cumsum()    
    # Plot and save explained variance
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
    plt.title(f"Percentage of Variance Explained by Each Component{title_suffix}")
    plt.xlabel("Principal Component")
    plt.ylabel("Percentage of Total Variance Explained (%)")
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
    plt.title(f"Cumulative Percentage of Variance Explained{title_suffix}")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance Explained (%)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{pictures_folder}/pca_all_{file_suffix}{current_hour}.svg", format="svg")
    #plt.show(block=False)
    # Plot and save variance for top nr components
    explained_variance_ratio_nr = explained_variance_ratio[:nr]
    cumulative_explained_variance_nr = cumulative_explained_variance[:nr]
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, nr + 1), explained_variance_ratio_nr, marker='o')
    plt.title(f"Percentage of Variance Explained by Top {nr} Components{title_suffix}")
    plt.xlabel("Principal Component")
    plt.ylabel("Percentage of Total Variance Explained (%)")
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, nr + 1), cumulative_explained_variance_nr, marker='o')
    plt.title(f"Cumulative Variance Explained by Top {nr} Components{title_suffix}")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance Explained (%)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{pictures_folder}/pca_{nr}_{file_suffix}{current_hour}.svg", format="svg")
    #plt.show(block=False)    
    # Create new columns for PCA components and drop original columns in train set
    for i in range(nr):
        train_df_copy[f'{prefix or "feat_"}pca_{i}'] = pca_components_train[:, i]
    train_df_copy.drop(columns=desc_columns_train, inplace=True)    
    # Transform and keep only first nr components in test set
    pca_components_test = pca.transform(test_df_copy[desc_columns_train])[:, :nr]
    for i in range(nr):
        test_df_copy[f'{prefix or "feat_"}pca_{i}'] = pca_components_test[:, i]
    test_df_copy.drop(columns=desc_columns_train, inplace=True)    
    overall_execution_time = time.time() - start_time
    print(f"Execution Time: {round(overall_execution_time, 2)} seconds")
    if prefix:
        play_sound()
    return train_df_copy, test_df_copy, cumulative_explained_variance


#Apply PCA to columns starting with 'desc_'. Replace 'desc_' columns with nr of most important principal components. Shows plots
def pca_transform_and_clean_desc(train_dataframe, test_dataframe, nr=20, prefix='desc_', pictures_folder="."):
    return pca_transform_and_clean(train_dataframe=train_dataframe, test_dataframe=test_dataframe, nr=nr, prefix=prefix, pictures_folder=pictures_folder)


# Function to process individual image - change resolution, remove transparency, flatten to 1D
def process_image(image_path, target_size=(64, 64)):
    img = Image.open(image_path).convert("RGBA")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1] range
    # Remove alpha channel if present
    if img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Keep only RGB channels
    img_array = img_array.flatten()  # Flatten to 1D array
    return img_array

# add columns img_i corresponding to pixels of pictures
def add_image_data(df, image_directory, target_size=(64, 64)):
    print(f"Current Time: {datetime.now().strftime('%H:%M:%S')}")
    step_start_time = time.time()
    # Extract unique descriptions
    unique_descriptions = df['description'].unique()
    # Dictionary to store processed image data
    image_data = {}
    # Process each unique image only once
    for desc in unique_descriptions:
        if pd.isna(desc):  # Check if the description is NaN
            image_data[desc] = np.zeros(target_size[0] * target_size[1] * 3)  # Assign zero vector
            continue  # Skip to the next iteration
        # Check for both .png and .jpg extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            filename = desc.lower().replace(' ', '_') + ext
            img_path = os.path.join(image_directory, filename)
            if os.path.exists(img_path):
                image_data[desc] = process_image(img_path, target_size)
                break  # Exit the loop if the image is found
            else:
                # print(f"Warning: {filename} not found in {image_directory}")
                image_data[desc] = np.zeros(target_size[0] * target_size[1] * 3)  # Fallback for missing images
    # Create DataFrame from processed image data
    img_df = pd.DataFrame.from_dict(image_data, orient='index')
    img_df.columns = [f'img_{i}' for i in range(img_df.shape[1])]
    img_df.index.name = 'description'
    img_df.reset_index(inplace=True)
    # Merge with original DataFrame, mapping processed data back to all rows
    df = df.merge(img_df, on='description', how='left')
    overall_execution_time = time.time() - step_start_time
    print(f"Execution Time: {round(overall_execution_time, 2)} seconds")
    print(f"Current Time: {datetime.now().strftime('%H:%M:%S')}")
    play_sound()
    return df

#saving resized images
def resize_and_convert_images(input_directory, output_directory, target_size=(224, 224)):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    # Process each image in the input directory
    for filename in os.listdir(input_directory):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Check for image file types
            img_path = os.path.join(input_directory, filename)
            img = Image.open(img_path).convert("RGB")  # Convert to RGB (removing alpha if present)
            img = img.resize(target_size)  # Resize the image to target size
            output_path = os.path.join(output_directory, filename)
            img.save(output_path)  # Save the processed image
    print(f"All images have been resized and saved to '{output_directory}'.")
    
#produce histogram of target for each description


def plot_y_train_histograms(df, y_train, description_column='description', pictures_folder='.'):
    # Add current hour to file name
    current_hour = datetime.now().strftime("%H:%M:%S")   
    # Ensure y_train is a pandas Series with the same index as df
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train, index=df.index)
    # Take the logarithm of y_train, adding a small constant to avoid log(0) issues
    if max(y_train)>50:
        log_y_train = np.log(y_train + 1e-9)
    else:
        log_y_train = y_train
    # Calculate overall min and max of log_y_train for consistent x-axis range
    y_min, y_max = log_y_train.min(), log_y_train.max()
    # Treat NaN as a unique description by replacing NaN with 'NaN' as a string
    df[description_column] = df[description_column].fillna('NaN')
    # Get unique descriptions sorted in alphabetical order
    unique_descriptions = np.sort(df[description_column].unique())
    for desc in unique_descriptions:
        # Get indices of rows with the current description
        indices = df[df[description_column] == desc].index
        # Extract log-transformed y_train values for those indices
        subset = log_y_train.loc[indices]
        # Plot histogram
        plt.figure(figsize=(8, 5))
        plt.hist(subset, bins=30, edgecolor='black', alpha=0.7)
        plt.title(f'Histogram of log(y_train) for description: {desc}')
        plt.xlabel('y_train values')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        # Set consistent x-axis range and line thickness
        plt.xlim(y_min, y_max)
        plt.gca().spines['top'].set_linewidth(1.5)
        plt.gca().spines['right'].set_linewidth(1.5)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.savefig(f"{pictures_folder}/histogram_description_{desc}_target_{current_hour}.svg", format="svg")
        # Show the plot
        #plt.show(block=False)
        

#initialise, optionally embedds description column using BERT as desc_*
def initialize_with_desc(input_file, embedding=False, sample_fraction=1, desc_column=False):
    # print("STEP 2 from the ASESSMENT")
    print(f"Current Time: {datetime.now().strftime('%H:%M:%S')}")
    step_start_time = time.time()
    # print(sample_fraction)
    sampled_df = initialize(input_file, sample_fraction=sample_fraction)
    X, y, lambda_optimal, description_column = prepare_data(sampled_df, sample_fraction=1)
    X_train, X_test, y_train, y_test = split_data(X, y, description_column)
    X_train_imputed, X_test_imputed, y_train = scale_and_impute_data(X_train, X_test, y_train)
    X_train_with_desc, X_test_with_desc = X_train_imputed, X_test_imputed
    #including the embedding of a description column using BERT:
    # Back up 'description' column
    description_train = X_train_imputed['description'].copy()
    description_test = X_test_imputed['description'].copy()
    if embedding:
        X_train_with_desc, X_test_with_desc = add_encoded_description(X_train_imputed, X_test_imputed)
    if desc_column:
        X_train_with_desc['description'] = description_train
        X_test_with_desc['description'] = description_test
    # X_test_with_desc.to_csv('X_test_with_desc.csv', sep=';', index=False)
    overall_execution_time = time.time() - step_start_time #Time: 202 seconds
    print(f"execution Time: {round(overall_execution_time, 2)} seconds")
    print(f"Current Time: {datetime.now().strftime('%H:%M:%S')}")
    play_sound()
    return X_train_with_desc, X_test_with_desc, y_train, y_test, lambda_optimal

#initialise, removing description column and including the embedding of a description column using BERT as desc_*
def initialize_with_desc_embedding(input_file, sample_fraction=1, desc_column=False):
    return initialize_with_desc(input_file=input_file, embedding=True, sample_fraction=sample_fraction, desc_column=desc_column)

# add one-hot-encoding columns instead of description
#assuming X_train and X_test contain column 'description'
def add_one_hot_encoded_description(X_train, X_test):
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None)
    # Fit on training data and transform both train and test
    description_train_encoded = one_hot_encoder.fit_transform(X_train[['description']])
    description_test_encoded = one_hot_encoder.transform(X_test[['description']])    
    # # Get the new column names with 'desc_' prefix
    # description_column_names = [f"desc_{category}" for category in one_hot_encoder.get_feature_names_out(['description'])]    
    # Create column names as 'desc_0', 'desc_1', ...
    description_column_names = [f"desc_{i}" for i in range(description_train_encoded.shape[1])]
    # Convert encoded arrays to DataFrames with new column names
    description_train_df = pd.DataFrame(description_train_encoded, columns=description_column_names, index=X_train.index)
    description_test_df = pd.DataFrame(description_test_encoded, columns=description_column_names, index=X_test.index)    
    # Concatenate the one-hot encoded description to the rest of the features
    X_train_with_desc = pd.concat([X_train.drop(columns=['description']), description_train_df], axis=1)
    X_test_with_desc = pd.concat([X_test.drop(columns=['description']), description_test_df], axis=1)    
    return X_train_with_desc, X_test_with_desc


def initialize_with_desc_one_hot(input_file, sample_fraction=1):
    print(f"Current Time: {datetime.now().strftime('%H:%M:%S')}")
    step_start_time = time.time()
    print(sample_fraction)
    sampled_df = initialize(input_file, sample_fraction=sample_fraction)
    X, y, lambda_optimal, description_column = prepare_data(sampled_df, sample_fraction=1)
    X_train, X_test, y_train, y_test = split_data(X, y, description_column)
    X_train_imputed, X_test_imputed, y_train = scale_and_impute_data(X_train, X_test, y_train)
    # Use one-hot encoding for the description column
    X_train_with_desc, X_test_with_desc = add_one_hot_encoded_description(X_train_imputed, X_test_imputed)
    overall_execution_time = time.time() - step_start_time
    print(f"Execution Time: {round(overall_execution_time, 2)} seconds")
    print(f"Current Time: {datetime.now().strftime('%H:%M:%S')}")
    play_sound()
    return X_train_with_desc, X_test_with_desc, y_train, y_test, lambda_optimal

#plots feature importance for xgboost_result
def feature_importance_plot(xgboost_result, nr=30, pictures_folder="."):
    # Prepare feature importances for visualization
    importances = xgboost_result['Importances']
    feature_names = xgboost_result['Feature Names']
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    # Display the feature importances
    print(importance_df)
    # Plotting the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'][:nr], importance_df['Importance'][:nr], color='skyblue')
    plt.xlabel('Importance Score')
    plt.title('Feature Importance from XGBoost')
    plt.gca().invert_yaxis()  # To display the highest importance at the top
    # Add current hour to file name
    current_hour = datetime.now().strftime("%H:%M:%S") 
    plt.savefig(f"{pictures_folder}/feature_importance_{current_hour}.svg", format="svg")
    #plt.show(block=False)
    return
