# ----------------------------------------------------------------------------------------------------------------------
# Set parameters
# ----------------------------------------------------------------------------------------------------------------------
target = "foot"
session_name = 'Run 1 - imbalanced, street-filter'
sampling_method = 'imbalanced'  # choose 'imbalanced, 'undersampling' or 'oversampling'
ti_filter = False  # only apply if target is 'caution'
street_filter = False  # if True all rows with 'street_binary' = 1 points are dropped (and variable 'street_binary') ==> don't forget to adjust R-Script
tunnel_filter = True  # only apply if target is 'foot'
scaling = False  # can be applied to caution or foot
col_drop = ['country', 'fd', 'glacier', 'id', 'lake', 'ski', 'snowshoe']
promising_features = ['fd_risk', 'fold', 'crevasse', 'ele']
r_interpreter = './R-4.2.3/bin/Rscript.exe'
r_script = "./GAM.R"
score_path = f'../scores/{target}_scores.xlsx'

# ----------------------------------------------------------------------------------------------------------------------
# Load libraries and functions
# ----------------------------------------------------------------------------------------------------------------------
import datetime
import openpyxl
import pandas as pd
import time
import warnings
from itertools import combinations
from my_functions import print_section_title, find_threshold, calculate_confusion_matrix,\
    evaluate, relative_count, impute_missing, find_missing_ids, feature_engineering, train_val_test, execute_r_script
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------------------------------------------------
# Suppress FutureWarning and ConvergenceWarning
# ----------------------------------------------------------------------------------------------------------------------
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------------------------------------------------
# Run Main
# ----------------------------------------------------------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------------------------------------------------------
    # Print parameters
    # ------------------------------------------------------------------------------------------------------------------
    print_section_title("START")
    start_time = time.time()
    current_date = datetime.date.today()
    print("Running main function...")
    print(f"Target variable is '{target}'")
    print(f"Choosen method is {sampling_method}")
    print(f"Session name is '{session_name}'")
    print(f"Dropped columns are {col_drop}")
    print(f"ti_filter is set to {ti_filter}")
    print(f"Street filter is set to {street_filter}")
    print(f"Tunnel filter is set to {tunnel_filter}")
    print(f"Standard scaling is set to {scaling}")

    # ------------------------------------------------------------------------------------------------------------------
    # Clean data and split into train, validation and test
    # ------------------------------------------------------------------------------------------------------------------
    df = pd.read_csv("../data/Stage_1.csv")
    df_models = pd.read_excel(score_path)
    df_clean = feature_engineering(df, target, col_drop, ti_filter=ti_filter, tunnel_filter=tunnel_filter, street_filter=street_filter)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test(df_clean, target, method=sampling_method, scaling=scaling)

    # ------------------------------------------------------------------------------------------------------------------
    # Modelling: Logistic regression (LR)
    # ------------------------------------------------------------------------------------------------------------------
    print_section_title("MODELLING LOGISTIC REGRESSION")
    # LR (all features)-------------------------------------------------------------------------------------------------
    logistic_model = LogisticRegression(solver='liblinear')
    logistic_model.fit(X_train, y_train)
    df_models = evaluate(df_models, logistic_model, 'LR (all features)', X_val, y_val, session_name)
    print("LR (all features) done!")

    # LR (only promising features)--------------------------------------------------------------------------------------
    X_train_selected = X_train[promising_features]
    X_val_selected = X_val[promising_features]
    logistic_model.fit(X_train_selected, y_train)
    df_models = evaluate(df_models, logistic_model, 'LR (promising EDA features)', X_val_selected, y_val, session_name)
    print("LR (only primising features) done!")

    # LR (feature selection by RFE)-------------------------------------------------------------------------------------
    rfe = RFE(estimator=logistic_model, n_features_to_select=3)
    rfe.fit(X_train, y_train)
    selected_features = X_train.columns[rfe.support_]
    print("Selected Features (RFE):", selected_features)
    X_train_rfe = rfe.transform(X_train)
    X_val_rfe = rfe.transform(X_val)
    logistic_model.fit(X_train_rfe, y_train)
    df_models = evaluate(df_models, logistic_model, 'LR (RFE)', X_val_rfe, y_val, session_name)
    print("LR (feature selection by RFE) done!")

    # LR (all permutations)---------------------------------------------------------------------------------------------
    # Initialize an empty list to store the results
    all_models = []

    # Get all possible combinations of feature variables
    all_combinations = []
    for r in range(1, len(X_train.columns) + 1):
        all_combinations.extend(combinations(X_train.columns, r))

    # Iterate over all combinations
    for combination in all_combinations:
        # Select the features
        X_train_selected = X_train[list(combination)]
        X_val_selected = X_val[list(combination)]

        # Create logistic regression model with L2 regularization
        logistic_regression_model = LogisticRegression(solver='liblinear')

        # Train the logistic regression model on training data with selected features
        logistic_regression_model.fit(X_train_selected, y_train)

        # Make predictions on validation data with selected features
        y_pred = logistic_regression_model.predict(X_val_selected)

        # Count the number of features in the combination
        num_features = len(combination)

        # Evaluate the model and store the results
        model_name = 'LR {}D '.format(num_features) + ' + '.join(combination)
        df_models = evaluate(df_models, logistic_regression_model, model_name, X_val_selected, y_val, session_name)
        all_models.append((model_name, logistic_regression_model))
    print("LR (Permutations) done!")

    # ------------------------------------------------------------------------------------------------------------------
    # Modelling: Random forest
    # ------------------------------------------------------------------------------------------------------------------
    print_section_title("MODELLING RANDOM FOREST")
    # RF (all features)-------------------------------------------------------------------------------------------------
    random_forest_model = RandomForestClassifier(random_state=0)
    random_forest_model.fit(X_train, y_train)
    df_models = evaluate(df_models, random_forest_model, 'RF (all features)', X_val, y_val, session_name)
    print("RF (all features) done!")

    # RF (only promising features + GridSearch--------------------------------------------------------------------------
    f1_scorer = make_scorer(f1_score)
    X_train_selected = X_train[promising_features]
    X_val_selected = X_val[promising_features]
    random_forest_model = RandomForestClassifier(random_state=0)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100],  # Number of trees in the forest
        'max_depth': [None, 10, 20],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required at each leaf node
    }

    # Perform hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(random_forest_model, param_grid=param_grid, scoring=f1_scorer, cv=5)
    grid_search.fit(X_train_selected, y_train)

    # Retrieve the best estimator
    best_rf = grid_search.best_estimator_

    # Train the best estimator on the full training set
    best_rf.fit(X_train_selected, y_train)
    df_models = evaluate(df_models, best_rf, 'RF (Tuned, only promising features)', X_val_selected, y_val, session_name)
    print("Best Parameters:", grid_search.best_params_)
    print("RF (only promising features + GridSearch) done!")

    # ------------------------------------------------------------------------------------------------------------------
    # Modelling: Gradient boosting
    # ------------------------------------------------------------------------------------------------------------------
    print_section_title("MODELLING GRADIENT BOOSTING")
    # GB (all features)-------------------------------------------------------------------------------------------------
    # Define the parameter grid for Gradient Boosting
    param_grid_gb = {
        'n_estimators': [50, 100],  # Number of boosting stages to be run
        'learning_rate': [0.1, 0.01],  # Learning rate shrinks the contribution of each tree
        'max_depth': [3, 5],  # Maximum depth of the individual estimators
    }

    gb_model = GradientBoostingClassifier()

    # Perform hyperparameter tuning using GridSearchCV for Gradient Boosting
    grid_search_gb = GridSearchCV(gb_model, param_grid=param_grid_gb, scoring=f1_scorer, cv=5)
    grid_search_gb.fit(X_train, y_train)

    # Retrieve the best estimator for Gradient Boosting
    best_gb = grid_search_gb.best_estimator_

    # Evaluate the best estimator for Gradient Boosting
    df_models = evaluate(df_models, best_gb, 'GB (Tuned, all)', X_val, y_val, session_name)
    print("Best Parameters:", grid_search_gb.best_params_)
    print("GB (All features) done!")

    # GB (onliy promising features--------------------------------------------------------------------------------------
    # Select only the desired features for training and validation sets
    X_train_selected = X_train[promising_features]
    X_val_selected = X_val[promising_features]

    # Define the parameter grid for Gradient Boosting
    param_grid_gb = {
        'n_estimators': [50, 100],  # Number of boosting stages to be run
        'learning_rate': [0.1, 0.01],  # Learning rate shrinks the contribution of each tree
        'max_depth': [3, 5],  # Maximum depth of the individual estimators
    }

    # Create Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier()

    # Perform hyperparameter tuning using GridSearchCV for Gradient Boosting
    grid_search_gb = GridSearchCV(gb_model, param_grid=param_grid_gb, scoring=f1_scorer, cv=5)
    grid_search_gb.fit(X_train_selected, y_train)

    # Retrieve the best estimator for Gradient Boosting
    best_gb = grid_search_gb.best_estimator_

    # Evaluate the best estimator for Gradient Boosting
    df_models = evaluate(df_models, best_gb, 'GB (Tuned, only promising features)', X_val_selected, y_val, session_name)
    print("Best Parameters:", grid_search_gb.best_params_)
    print("GB (only promising features + GridSearch) done!")

    #------------------------------------------------------------------------------------------------------------------
    # SAVE DATAFRAME
    #------------------------------------------------------------------------------------------------------------------
    print_section_title("EXPORT LR, RF AND GB RESULTS TO XLSX")
    df_models.to_excel(score_path, index=False)
    print(f"'{target}' scores successfully saved.")


    # ------------------------------------------------------------------------------------------------------------------
    # Modelling: GAM (R-Script)
    # ------------------------------------------------------------------------------------------------------------------
    print_section_title("MODELLING GAM (via R-SCRIPT)")
    print("Execute R-Script...")
    execute_r_script(r_interpreter, r_script, target, session_name)
    print("GAM modelling done and saved to xlsx!")

    print_section_title("END")
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Runtime: {hours} hours, {minutes} minutes, {seconds} seconds")

if __name__ == "__main__":
    main()