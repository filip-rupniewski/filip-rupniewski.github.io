# Set the working directory
import os
import time
from datetime import datetime


# Set main working directories
main_folder = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory if running as a script
os.chdir(main_folder)
# Define relative paths
input_file = os.path.join(main_folder, "candidates_data.csv")
pictures_folder = os.path.join(main_folder, "pictures")
spacecraft_images_folder = os.path.join(main_folder, "spacecraft_images")
# Create pictures folder if it doesn't exist
os.makedirs(pictures_folder, exist_ok=True)

# --- Functions for each task ---
# from my_functions import *
from my_functions import (
    add_image_data, check_normality, feature_importance_plot, initialize, 
    initialize_with_desc, initialize_with_desc_one_hot, inv_boxcox, 
    inverse_boxcox_transform_and_plot, pca_desc_columns, pca_transform_and_clean, 
    pca_transform_and_clean_desc, play_sound, plot_y_train_histograms, 
    prepare_data, scale_and_impute_data, split_data, train_and_plot_each_desc, 
    train_and_plot_xgboost, train_models
)
    
# Main program execution

# Timing utility for overall execution
overall_start_time = time.time()
print(f"Script started at: {datetime.now().strftime('%H:%M:%S')}")
step_start_time = time.time()  # Step timer

#Sample data processing (using 1% of data)
sample_fraction=0.01
sampled_df = initialize(input_file, sample_fraction=sample_fraction)

#Data preparation
# extract dataframe, target column (after applying Box-Cox transformation), lambda_optimal (Box-Cox transformation), description column. It crops 'nr_' and 'sh_' from the beginning of entries of feature_1 and feature_2. It returns a fraction of dataframe.
X, y, lambda_optimal, description_column = prepare_data(sampled_df, sample_fraction=1)

#checking normality of target data
check_normality(y, pictures_folder=pictures_folder)
#Data Splitting, scaling and imputing
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_imputed, X_test_imputed, y_train = scale_and_impute_data(X_train, X_test, y_train)
print(f"Initialization, Data Preparation, Data Splitting, Scale and Impute Data Time: {round(time.time() - overall_start_time, 2)} seconds")

#PCA for Dimensionality Reduction
#45 highest principal components describe >99% data
X_train_imputed_pca, X_test_imputed_pca, cumulative_explained_variance = pca_transform_and_clean(X_train_imputed, X_test_imputed, nr=45, pictures_folder=pictures_folder)
print(cumulative_explained_variance)

#Model Training and Evaluation (1% of Data, PCA and No PCA)
results_df_pca = train_models(X_train_imputed_pca, y_train, X_test_imputed_pca, y_test) #Time: 76 s
results_df_no_pca = train_models(X_train_imputed, y_train, X_test_imputed, y_test) #Time: 64 s
print("results for the 1% of data and 45 highest principal components")
print(results_df_pca.iloc[:,[0,2,3]])
print("results for the 1% of data and no restrictions")
print(results_df_no_pca.iloc[:,[0,2,3]])
    #   for the 1% of data but without restriction to highest P.C.
    #             Model      RMSE  Execution Time (s)
    # 1         XGBoost  0.574574               52.55
    # 0   Random Forest  0.585949               19.50
    # 2             SVR  0.607558                5.26
    # 3      ElasticNet  0.608824                0.06
    # 4  Neural Network  4.519947               17.29
    #   restricted to 45 highest principal components:
    #             Model      RMSE  Execution Time (s)
    # 2             SVR  0.606724                5.19
    # 3      ElasticNet  0.645124                0.09
    # 0   Random Forest  0.696769               21.17
    # 1         XGBoost  0.698909               59.23
    # 4  Neural Network  3.957845               13.87
    #
    #   for the 10% of data and 45 highest principal components
    #            Model      RMSE  Execution Time (s)
    # 1        XGBoost  0.524714               58.59
    # 2            SVR  0.557516                5.40
    # 0  Random Forest  0.559146               21.40
    # 3     ElasticNet  0.577160                0.09
    #   for the 10% of data and no restrictions    
    #            Model      RMSE  Execution Time (s)
    # 1        XGBoost  0.478751               38.06
    # 0  Random Forest  0.496532               18.78
    # 2            SVR  0.558672                6.57
    # 3     ElasticNet  0.573751                0.11
print(f"Train Models and Evaluate Performance Time: {round(time.time() - step_start_time, 2)} seconds") 
    #Time: 100 seconds for 1% of data 
    #Time: 889 seconds for 10% of data

#Full Data Processing
#XGBoost fine tuning on all the data (80% for trainning, 20% for test)
# Data Splitting and Scaling on Full Data
sample_fraction=1
sampled_df = initialize(input_file, sample_fraction)
X, y, lambda_optimal, description_column = prepare_data(sampled_df, sample_fraction)
check_normality(y, pictures_folder=pictures_folder)
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_imputed, X_test_imputed, y_train = scale_and_impute_data(X_train, X_test, y_train)

#PCA - 50 vectors with the highiest eigenvalues
X_train_imputed_pca, X_test_imputed_pca, cumulative_explained_variance = pca_transform_and_clean(X_train_imputed, X_test_imputed, nr=50, pictures_folder=pictures_folder)
#let us find out what are the best parameters for xgboost using GridsearchCV. 
xgboost_result_pca = train_and_plot_xgboost(X_train_imputed_pca, y_train, X_test_imputed_pca, y_test, lambda_optimal, CPU=True, pictures_folder=pictures_folder) #Execution Time (CPU): 30.68 seconds; (GPU): 19 s
xgboost_result_no_pca = train_and_plot_xgboost(X_train_imputed, y_train, X_test_imputed, y_test, lambda_optimal, CPU=True, pictures_folder=pictures_folder) #Execution Time: 24.27 seconds

#I have already found it, so it was calculated for optimal parameters only to save time.
#However, if you want to check for the best parameters, uncomment the following line (much more time consuming ~40min)
#xgboost_result = train_and_plot_xgboost(X_train_imputed, y_train, X_test_imputed, y_test, lambda_optimal, CPU=True, only_one_parameter=False) #Time (CPU): 40 min

print("Best XGBoost Params with PCA:", xgboost_result_pca['Best Params'])
print("Best XGBoost Params without PCA:", xgboost_result_no_pca['Best Params'])
print("XGBoost RMSE with PCA:", xgboost_result_pca['RMSE'])
print("XGBoost RMSE without PCA:", xgboost_result_no_pca['RMSE'])
    # Best XGBoost Params with and without PCA: {'learning_rate': 0.05, 'max_depth': 9, 'n_estimators': 400, 'subsample': 0.7}
    # XGBoost RMSE with PCA (50 highest principal components): 0.3801972985953054
    # XGBoost RMSE without PCA (for all data): 0.3156168110462055
    
# # Summary of Results of Step 1 of the Challenge
overall_execution_time = time.time() - overall_start_time #Time: 112 seconds
print(f"Overall Execution Time: {round(overall_execution_time, 2)} seconds")
# print(xgboost_result)
y_pred_box_cox = xgboost_result_no_pca['y_pred']
# If we want predicted target on original scale, then we apply inverse Box-Cox transformation
# Final Output: Inverse Box-Cox Transformation on Predictions
y_pred_original = inv_boxcox(y_pred_box_cox, lambda_optimal)
inverse_boxcox_transform_and_plot(y_pred_box_cox, y_test, lambda_optimal, tick_interval=60000, pictures_folder=pictures_folder)
play_sound() # Play sound after execution completion
print(f"Current Time: {datetime.now().strftime('%H:%M:%S')}")

#####STEP 2 from the Challenge#########
# Evaluation with BERT and One-Hot Encoded Descriptions
sample_fraction=1
#Embed description column as desc_* columns using BERT 
X_train_with_desc, X_test_with_desc, y_train, y_test, lambda_optimal = initialize_with_desc(input_file, embedding=True, sample_fraction=sample_fraction, desc_column=False)
# Time: 200 s (for whole data)
#check xgboost for all data including desc_* columns   
xgboost_result_desc_bert = train_and_plot_xgboost(X_train_with_desc, y_train, X_test_with_desc, y_test, lambda_optimal, CPU=True, pictures_folder=pictures_folder) #Time (CPU): 19s
print("XGBoost with Description Embedding - Best Params:", xgboost_result_desc_bert['Best Params'])
print("XGBoost RMSE with Description Embedding:", xgboost_result_desc_bert['RMSE'])
    #for all data:   
    # Time: 344s
    # XGBoost with Description Embedding - Best Params: {'learning_rate': 0.05, 'max_depth': 9, 'n_estimators': 400, 'subsample': 0.7}
    # XGBoost RMSE with Description Embedding: 0.3386622919928372
    #for 0.1 of data:   
    # Time: 165s
    # XGBoost with Description Embedding - Best Params: {'learning_rate': 0.05, 'max_depth': 9, 'n_estimators': 400, 'subsample': 0.7}
    # XGBoost RMSE with Description Embedding: 0.4287830325768249
    
# Prepare feature importances for visualization
feature_importance_plot(xgboost_result_desc_bert, nr=30, pictures_folder=pictures_folder)
    #For all data: Features nr 27, 52, 46 (not coming from description embedding) are the most important. Then next 15 most important columns comes from the embedding of description.
    #for 0.1 of data: columns  coming from description are the most important. Only the feature nr 27 (not coming from the description) appears among 30 most important column names

# PCA for Description Embedding
#XGBoost with 50, 10, 1 P.C. instead of all desc_columns
#plot of 20 most and 20 least important P.C.
pca_desc_columns(X_train_with_desc, pictures_folder=pictures_folder)
for nr_components in [50,10,1]:
    X_train_with_desc_pca, X_test_with_desc_pca, cumulative_explained_variance = pca_transform_and_clean_desc(X_train_with_desc, X_test_with_desc, nr=nr_components, pictures_folder=pictures_folder)
    xgboost_result = train_and_plot_xgboost(X_train_with_desc_pca, y_train, X_test_with_desc_pca, y_test, lambda_optimal, CPU=True, pictures_folder=pictures_folder)
    print("nr of PCA components: "+str(nr_components))
    print("XGBoost Best Params:", xgboost_result['Best Params'])
    print(f"XGBoost with {nr_components} PCA components for Description Embedding - RMSE:", xgboost_result['RMSE'])
    feature_importance_plot(xgboost_result, nr=30, pictures_folder=pictures_folder)
    # {'learning_rate': 0.05, 'max_depth': 9, 'n_estimators': 400, 'subsample': 0.7}
    #n  XGBoost RMSE    time (s)
    #50     0.3338475665276554  33
    #10     0.3294018897852198  25
    #1      0.3223946825063526  23
    #for 0.1 of data:
    # XGBoost RMSE: 0.4347419601167268
    # XGBoost RMSE: 0.42365132171915176
    # XGBoost RMSE: 0.46960425567422925
    

# Embed description column as desc_* columns using One-Hot Encoded Descriptions
X_train_with_desc, X_test_with_desc, y_train, y_test, lambda_optimal= initialize_with_desc_one_hot(input_file, sample_fraction=sample_fraction)
#Time: 198.17 seconds
#check xgboost for all data with desc_* columns    
xgboost_result_desc_ohe = train_and_plot_xgboost(X_train_with_desc, y_train, X_test_with_desc, y_test, lambda_optimal, CPU=True, pictures_folder=pictures_folder) #Time (GPU): 17s, (CPU): 19s
print("XGBoost with One-Hot Encoded Descriptions - Best Params:", xgboost_result_desc_ohe['Best Params'])
print("XGBoost RMSE with One-Hot Encoding:", xgboost_result_desc_ohe['RMSE'])
    # Time: 18s
    # XGBoost with One-Hot Encoded Descriptions - Best Params: {'learning_rate': 0.05, 'max_depth': 9, 'n_estimators': 400, 'subsample': 0.7}
    # XGBoost RMSE with One-Hot Encoding: 0.33894577791389  
    #for 0.1 of data:
    # Time: 9.39 seconds
    # XGBoost with One-Hot Encoded Descriptions - Best Params: {'learning_rate': 0.05, 'max_depth': 9, 'n_estimators': 400, 'subsample': 0.7}
    # XGBoost RMSE with One-Hot Encoding: 0.4179117061907738
# Prepare feature importances for visualization
feature_importance_plot(xgboost_result_desc_ohe, nr=30, pictures_folder=pictures_folder)
    #for whole data: Features nr 27, 52, 46 (not coming from description embedding) are the most important. Then there are more columns coming from description than features but they are both well distributed.
  

# Trying independent predictions for each value of 'description' column
#ANALYSIS
#produce histogram of target for each description
X_train_with_desc, X_test_with_desc, y_train, y_test, lambda_optimal= initialize_with_desc(input_file, embedding=False, sample_fraction=sample_fraction, desc_column=False)
plot_y_train_histograms(X_train_with_desc, y_train, pictures_folder=pictures_folder)
# XGBoost for all data for each unique value of 'description' column independently
xgboost_results, overall_rmse = train_and_plot_each_desc(X_train_with_desc, y_train, X_test_with_desc, y_test, lambda_optimal, model_type='xgboost', CPU=True, pictures_folder=pictures_folder)
# Overall XGBOOST RMSE: 1.093016890817071
# Execution Time: 303.84 seconds
# Time: 198 s
print(xgboost_results)
print(overall_rmse)

#Trying the same with SVR and ElasticNet, checking different parameters for each. 
#It gave worse results than XGBooost, so I commented what follows. 
# for model in ['svr', 'elasticnet']:
#     xgboost_results, overall_rmse = train_and_plot_each_desc(X_train_with_desc2, y_train2, X_test_with_desc2, y_test2, lambda_optimal2, model_type=model, CPU=True, pictures_folder=pictures_folder)
#     print('#####')
#     time.sleep(5)
#     # Overall SVR RMSE: 1.0078484142725739
#     # Execution Time: 757.16 seconds
#     # Current Time: 18:59:35
#     # Overall ELASTICNET RMSE: 1.0040451356353042
#     # Execution Time: 60 seconds

# Spacecraft images processing: ###############
sample_fraction=0.1
X_train_with_desc, X_test_with_desc, y_train, y_test, lambda_optimal = initialize_with_desc(input_file, embedding=True, sample_fraction=sample_fraction, desc_column=True)
# Add image data to DataFrame
df = add_image_data(X_train_with_desc, spacecraft_images_folder, target_size=(128, 128))
df_test = add_image_data(X_test_with_desc, spacecraft_images_folder, target_size=(128, 128))
print(df.head())
# PCA transformation on description and image data
df_pca, df_test_pca, cumulative_explained_variance = pca_transform_and_clean_desc(df, df_test, nr=46, prefix='img_', pictures_folder=pictures_folder) #Time 320s
df_pca, df_test_pca, cumulative_explained_variance = pca_transform_and_clean_desc(df_pca, df_test_pca, nr=50, prefix='desc_', pictures_folder=pictures_folder) #Time 33s
# Remove description column after PCA transformation
df_pca_no_description = df_pca.drop(columns=['description'])
df_test_pca_no_description = df_test_pca.drop(columns=['description'])
# Train and evaluate XGBoost model on PCA-transformed data
xgboost_result = train_and_plot_xgboost(df_pca_no_description, y_train, df_test_pca_no_description, y_test, lambda_optimal=lambda_optimal, CPU=True, pictures_folder=pictures_folder) 
print(xgboost_result['Best Params'])
print("XGBoost RMSE:", xgboost_result['RMSE'])
#Results with dimension reduction by PCA on img columns only (without desc columns):
# Execution Time: 21.65 seconds
# {'learning_rate': 0.05, 'max_depth': 9, 'n_estimators': 400, 'subsample': 0.7}
# XGBoost RMSE: 0.43178883079898556
#Results with dimension reduction by PCA on desc and img columns:
# Execution Time: 27.03 seconds
# {'learning_rate': 0.05, 'max_depth': 9, 'n_estimators': 400, 'subsample': 0.7}
# XGBoost RMSE: 0.43812996301600254


# Final timing output
overall_execution_time = time.time() - overall_start_time
print(f"Overall Execution Time: {round(overall_execution_time, 2)} seconds")
print("Script completed at:", datetime.now().strftime('%H:%M:%S'))
# Overall Execution Time: 877.7 seconds ~15 min


