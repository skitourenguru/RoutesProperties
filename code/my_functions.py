# ----------------------------------------------------------------------------------------------------------------------
# Load libraries
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import subprocess

from datetime import datetime
from itertools import combinations
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix, average_precision_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

# ----------------------------------------------------------------------------------------------------------------------
# Important functions
# ----------------------------------------------------------------------------------------------------------------------
def feature_engineering(df, target, col_drop, ti_filter=None, tunnel_filter=None, street_filter=None):
    """
    Performs feature engineering on DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame containing the dataset to be engineered.
    - target (str): The target variable for the modelling. It can be either 'foot' or 'caution'.
    - col_drop (list): A list of column names to be dropped from the DataFrame.
    - ti_filter (bool, optional): If True, filters the DataFrame based on the value of 'ti' (only for 'caution').
    - tunnel_filter (bool, optional): If True, excludes data points related to tunnels (only for 'foot').
    - street_filter (bool, optional): If True, excludes data points related to streets.

    Returns:
    - DataFrame: The engineered DataFrame based on the specified target and filters.
    """

    print_section_title("FEATURE ENGINEERING")

    # Drop route points abroad
    df = df.dropna(subset=['country'])
    print("Data points abroad excluded.")

    # Drop route points on lake
    df = df[df['lake'] != 1]
    print("Data points on lakes excluded.")

    # Impute crevasse and street
    df['crevasse'].fillna(0, inplace=True)
    df['street'].fillna(4, inplace=True)
    print("NAs for 'crevasse' and 'street' imputed.")

    # Impute missing values
    missing_ids = find_missing_ids(df, 'ti')

    # Impute missing ti values for each missing ID
    for missing_id in missing_ids:
        imputed_value = impute_missing(df, 'ti', missing_id)
        print(f"Imputed value for ID {missing_id}: {imputed_value}")

    # Create variable 'aspect_binary'
    conditions = [(df['aspect'] >= 0) & (df['aspect'] <= 45),
                  (df['aspect'] >= 315) & (df['aspect'] <= 360),
                  (df['aspect'] > 45) & (df['aspect'] < 315)]
    values = [1, 1, 0]

    df['aspect_binary'] = None

    for condition, value in zip(conditions, values):
        df.loc[condition, 'aspect_binary'] = value

    df['aspect_binary'] = df['aspect_binary'].astype(int)
    df.drop('aspect', axis=1, inplace=True)
    print("New variable 'aspect_binary created.")

    # Create variable 'street_binary'
    conditions = [(df['street'] < 4),
                  (df['street'] == 4)]
    values = [1, 0]

    df['street_binary'] = None

    for condition, value in zip(conditions, values):
        df.loc[condition, 'street_binary'] = value

    df['street_binary'] = df['street_binary'].astype(int)
    df.drop('street', axis=1, inplace=True)
    print("New variable 'street_binary created.")

    # Create new variable fd_risk
    df['fd_risk'] = df['fd_maxv'] * df['slope']
    print("New variable 'fd_risk created.")

    # Clip variables (limit outliers)
    df['fd_risk'] = df['fd_risk'].clip(lower=0, upper=2500)
    print("Values for 'fd_risk' clipped.")
    df['planc7'] = df['planc7'].clip(lower=-350, upper=350)
    print("Values for 'planc7' clipped.")

    # Filter evrey tenth point
    df = df.sort_values(by="id")
    df = df.reset_index(drop=True)
    df_every_10th = df.iloc[::10]
    df = df_every_10th.copy()
    print("Only every 10th point considered for analysis.")

    df.drop(columns=col_drop, inplace=True)
    print(f"Dropped the variables {col_drop}")

    if street_filter == True:
        df = df[df['street_binary'] != 1]
        df.drop('street_binary', axis=1, inplace=True)
        print("Streets are excluded.")
        print("Variable 'street_binary is dropped.")

    if target == 'foot':
        df_f = df.copy()
        df_f.drop('caution', axis=1, inplace=True)

        if tunnel_filter == True:
            df_f = df_f[(df_f['street_binary'] != 1) | (df_f['foot'] != 1)]
            print("Tunnels are excluded.")
        return df_f

    elif target == 'caution':
        df_c = df.copy()

        if ti_filter == True:
            df_c = df_c[df_c['foot'] != 1]
            df_c.drop(columns=['foot'], inplace=True)
            df_c = df_c.drop(df_c[(df_c['caution'] == 1) & (df_c['ti'] < 0.25)].index)
            df_c = df_c.drop(df_c[(df_c['caution'] == 0) & (df_c['ti'] > 0.75)].index)
            print("ti-values > 0.75 (for caution == 0) and ti-values < 0.25 (for caution == 1) removed.")

        return df_c

    else:
        return none

def train_val_test(df, target, method='imbalanced', scaling = False):
    """
    Preprocesses the input DataFrame by splitting it into training, validation, and test sets,
    potentially addressing class imbalance and performing scaling if specified, and exports
    the sets to CSV files.

    Parameters:
    - df: DataFrame, the input data containing features and target variable
    - target: str, the name of the target variable
    - method: str, optional, method for dealing with class imbalance (default is 'imbalanced')
        Possible values: 'imbalanced', 'undersampling', 'oversampling'
    - scaling: bool, optional, whether to apply feature scaling (default is False)

    Returns:
    - X_train, y_train, X_val, y_val, X_test, y_test: DataFrames and Series,
      containing the features and target variables for training, validation, and test sets
    """

    print_section_title("TRAIN VALIDATION TEST SPLIT")

    # Splitting into training, validation, and test sets
    train_set, temp_set = train_test_split(df, test_size=0.3, random_state=42)
    val_set, test_set = train_test_split(temp_set, test_size=1 / 3, random_state=42)

    if method == 'undersampling':
        # Undersampling only the training set
        rus = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = rus.fit_resample(train_set.drop(columns=[target]), train_set[target])
        train_set_resampled = pd.concat([X_train_resampled, y_train_resampled], axis=1)
        train_set = train_set_resampled

    elif method == 'oversampling':
        # Oversampling only the training set
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(train_set.drop(columns=[target]), train_set[target])
        train_set_resampled = pd.concat([X_train_resampled, y_train_resampled], axis=1)
        train_set = train_set_resampled

    if scaling == True:
        binary_vars = detect_binary_columns(train_set)
        non_binary_vars = [col for col in train_set.columns if col not in binary_vars]
        non_binary_vars_excluding_ordinal = [col for col in non_binary_vars if col != 'crevasse']

        scaler = StandardScaler()
        scaler.fit(train_set[non_binary_vars_excluding_ordinal])
        train_set[non_binary_vars_excluding_ordinal] = scaler.transform(train_set[non_binary_vars_excluding_ordinal])
        val_set[non_binary_vars_excluding_ordinal] = scaler.transform(
            val_set[non_binary_vars_excluding_ordinal])
        test_set[non_binary_vars_excluding_ordinal] = scaler.transform(test_set[non_binary_vars_excluding_ordinal])

        print("StandardScaling on non-binary and non-categorical variables successfully applied!")

    # Print set lengths
    print("Training set: {:.2f}%".format(len(train_set) / len(df) * 100))
    print("Validation set: {:.2f}%".format(len(val_set) / len(df) * 100))
    print("Test set: {:.2f}%".format(len(test_set) / len(df) * 100))

    if target is not None:
        print(f"{target} in training set: {relative_count(train_set):.2%}")
        print(f"{target} in validation set: {relative_count(val_set):.2%}")
        print(f"{target} in test set: {relative_count(test_set):.2%}")
    else:
        print("Target variable is not defined.")

    # Exporting to CSV
    train_set.to_csv(f'../data/stage_2_{target}_train.csv', index=False)
    val_set.to_csv(f'../data/stage_2_{target}_val.csv', index=False)
    test_set.to_csv(f'../data/stage_2_{target}_test.csv', index=False)

    # Extracting features and target variables
    X_train = train_set.drop(columns=[target])
    y_train = train_set[target]
    X_val = val_set.drop(columns=[target])
    y_val = val_set[target]
    X_test = test_set.drop(columns=[target])
    y_test = test_set[target]

    return X_train, y_train, X_val, y_val, X_test, y_test


def find_threshold(probabilities, y_true, steps=1000):
    """
    Perform a binary search algorithm to find the threshold where False Positives (FP)
    are equal to False Negatives (FN) in a binary classification scenario.

    Parameters:
    - probabilities (array-like): Predicted probabilities of the positive class.
    - y_true (array-like): True binary labels (0s and 1s) for the samples.
    - steps (int, optional): Number of steps for the binary search algorithm. Default is 1000.

    Returns:
    - float or None: The threshold where FP equals FN, or None if such a threshold is not found.

    The function iteratively searches for the threshold value that balances False Positives (FP)
    and False Negatives (FN) in a binary classification scenario. It uses a binary search
    algorithm to efficiently locate this threshold within the range of 0 to 1.
    """

    start = 0
    end = 1
    threshold = None

    while start <= end:
        mid = (start + end) / 2
        threshold = mid

        TN, FP, FN, TP = calculate_confusion_matrix(probabilities, y_true, threshold)

        if FP == FN:
            return threshold

        if FP < FN:
            end = mid - 0.0001

        else:
            start = mid + 0.0001

    return threshold

def calculate_confusion_matrix(probabilities, y_true, threshold):
    """
    Calculate the confusion matrix based on predicted probabilities and true labels,
    using a specified threshold for binary classification.

    Parameters:
    - probabilities (array-like): Predicted probabilities of the positive class.
    - y_true (array-like): True binary labels (0s and 1s) for the samples.
    - threshold (float): Threshold value for converting probabilities to binary predictions.

    Returns:
    - int, int, int, int: True Negative (TN), False Positive (FP),
      False Negative (FN), and True Positive (TP) counts, respectively.

    This function converts the predicted probabilities to binary predictions based on the
    specified threshold. It then computes the confusion matrix using the true labels and
    binary predictions, and extracts the counts of True Negatives (TN), False Positives (FP),
    False Negatives (FN), and True Positives (TP) from the confusion matrix.
    """

    # Convert probabilities to binary predictions based on the threshold
    predictions = (probabilities > threshold).astype(int)

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, predictions)

    # Extract TN, FP, FN, TP from the confusion matrix
    TN, FP, FN, TP = conf_matrix.ravel()

    return TN, FP, FN, TP

def evaluate(df_models, model, model_name, X_val, y_val, session_name):
    """
    Evaluates a logistic regression model on validation data and computes various classification metrics.

    Parameters:
    - df_models (DataFrame): DataFrame to store evaluation metrics of different models.
    - model: Fitted model to be evaluated.
    - model_name (str): Name of the model.
    - X_val (array-like): Validation input features.
    - y_val (array-like): True labels for the validation set.
    - session_name (str): Name of the session (run).

    Returns:
    - df_models (DataFrame): Updated DataFrame containing evaluation metrics for the model.
    """

    # Get the current date
    current_date = datetime.now()
    formatted_date = current_date.strftime('%d.%m.%Y')

    # Predict probabilities
    probabilities = model.predict_proba(X_val)[:, 1]
    
    # Call optimize_threshold function to get the optimized threshold and min_diff
    threshold = find_threshold(probabilities, y_val)

    # Convert probabilities to binary predictions based on the optimized threshold
    y_pred = (probabilities > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    auc_score = roc_auc_score(y_val, y_pred)

    f1 = f1_score(y_val, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    
    # Calculate Confusion Score
    if tp == 0:
        confusion_score = 99999
    else:
        confusion_score = (fp + fn) * 100 / tp
    
    # Create a DataFrame with new metrics
    new_metrics = pd.DataFrame({'Date' : [formatted_date],
                                'Model': [model_name],
                                'Session_Name' : [session_name],
                                'Optimized_Threshold': [threshold],
                                'Confusion_Score': [confusion_score],
                                'Accuracy': [accuracy],
                                'Precision': [precision],
                                'Recall': [recall],
                                'F1': [f1],
                                'ROC_AUC': [auc_score]
                                })
    
    # Concatenate new_metrics with df_models
    df_models = pd.concat([df_models, new_metrics], ignore_index=True)
    
    return df_models

def execute_r_script(r_interpreter, r_script, target, name):
    """
    Executes an R script using the specified R interpreter, passing variables as command-line arguments.

    Args:
        r_interpreter (str): The path to the R interpreter executable.
        r_script (str): The path to the R script file to be executed.
        target (str): The value to pass as 'target' variable to the R script.
        name (str): The value to pass as 'name' variable to the R script.

    Returns:
        None

    Prints:
        If there's an error, it prints the error message.
        If the execution is successful, it prints a success message.
    """

    process = subprocess.Popen([r_interpreter, r_script, target, name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Check if there was any error
    if process.returncode != 0:
        print("Error:", stderr.decode())
    else:
        print("R script executed successfully.")

# ----------------------------------------------------------------------------------------------------------------------
# Additional functions
# ----------------------------------------------------------------------------------------------------------------------
def plot_curves_test(y_pred_proba_train, y_pred_proba_val, y_pred_proba_test, y_train, y_val, y_test, target):
    """
    Plot Receiver Operating Characteristic (ROC) curve and Precision-Recall curve
    for a binary classification model.

    Parameters:
    y_pred_proba_train (array-like): Predicted probabilities for training data.
    y_pred_proba_val (array-like): Predicted probabilities for validation data.
    y_pred_proba_test (array-like): Predicted probabilities for test data.
    y_train (array-like): True binary labels of shape (n_samples,) for training data.
    y_val (array-like): True binary labels of shape (n_samples,) for validation data.
    y_test (array-like): True binary labels of shape (n_samples,) for test data.
    target (str): Name of the target variable for which the curves are plotted.

    Returns:
    None
    """

    # Compute ROC curve and ROC area for validation data
    fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_proba_val)
    roc_auc_val = auc(fpr_val, tpr_val)

    # Compute Precision-Recall curve and area for validation data
    precision_val, recall_val, _ = precision_recall_curve(y_val, y_pred_proba_val)
    average_precision_val = average_precision_score(y_val, y_pred_proba_val)

    # Compute ROC curve and ROC area for training data
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
    roc_auc_train = auc(fpr_train, tpr_train)

    # Compute Precision-Recall curve and area for training data
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_pred_proba_train)
    average_precision_train = average_precision_score(y_train, y_pred_proba_train)

    # Compute ROC curve and ROC area for test data
    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)
    roc_auc_test = auc(fpr_test, tpr_test)

    # Compute Precision-Recall curve and area for test data
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_proba_test)
    average_precision_test = average_precision_score(y_test, y_pred_proba_test)

    # Plot ROC curve
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr_val, tpr_val, color='#FF9999', lw=2, label='Validation ROC curve (AUC = %0.2f)' % roc_auc_val)
    plt.plot(fpr_train, tpr_train, color='#ADD8E6', lw=2, label='Training ROC curve (AUC = %0.2f)' % roc_auc_train)
    plt.plot(fpr_test, tpr_test, color='#90EE90', lw=2, label='Test ROC curve (AUC = %0.2f)' % roc_auc_test)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xticks([i / 10 for i in range(11)])
    plt.yticks([i / 10 for i in range(11)])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # Plot Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall_val, precision_val, color='#FF9999', lw=2, label='Validation Precision-Recall curve (AP = %0.2f)' % average_precision_val)
    plt.plot(recall_train, precision_train, color='#ADD8E6', lw=2, label='Training Precision-Recall curve (AP = %0.2f)' % average_precision_train)
    plt.plot(recall_test, precision_test, color='#90EE90', lw=2, label='Test Precision-Recall curve (AP = %0.2f)' % average_precision_test)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xticks([i / 10 for i in range(11)])
    plt.yticks([i / 10 for i in range(11)])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('True Positive Rate (Recall)')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(f'../plots/ROC_PR_{target}.png')
    plt.show()

def relative_count(data):
    """
    Calculate the relative count of caution occurrences within the provided dataset.
    If 'caution' is not found, it uses 'foot' instead.

    Args:
        data (DataFrame): A pandas DataFrame containing the dataset.

    Returns:
        float: The relative count of caution or foot occurrences, calculated as the sum of values divided by the total number of rows in the dataset.
    """
    try:
        return (data['caution'].sum() / len(data))
    except KeyError:
        try:
            return (data['foot'].sum() / len(data))
        except KeyError:
            print("Error: Neither 'caution' nor 'foot' found in the dataset.")
            return None

def impute_missing(df, column, missing_id):
    """
    Imputes missing values for the 'terrain indicator' (ti) variable in a DataFrame by taking the mean
    of the previous and next available values.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - ti_column (str): The name of the column containing the terrain indicator values.
    - missing_id (int): The ID of the row with the missing ti value.

    Returns:
    - float or None: The imputed ti value if both previous and next ti values are available,
                     otherwise returns None.
    """
    # Find the IDs of the previous and next points
    prev_id = missing_id - 1
    next_id = missing_id + 1

    # Find 'ti' values for prev_id and next_id
    prev_ti = None
    next_ti = None
    for index, row in df.iterrows():
        if row['id'] == prev_id:
            prev_ti = row[column]
        elif row['id'] == next_id:
            next_ti = row[column]

    # If both previous and next ti values are available, impute the missing value
    if prev_ti is not None and next_ti is not None:
        mean = (prev_ti + next_ti) / 2
        return mean
    else:
        # If either previous or next ti value is missing, return None
        return None

def find_missing_ids(df, column):
    """
    Finds the IDs of the rows with missing terrain indicator (ti) values in a DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - ti_column (str): The name of the column containing the terrain indicator values.

    Returns:
    - list: A list of IDs of the rows with missing ti values.
    """
    missing_ids = []
    for index, row in df.iterrows():
        if pd.isnull(row[column]):
            missing_ids.append(row['id'])
    return missing_ids

def print_section_title(title):
    """
    Prints a formatted section title with a specified string.

    Parameters:
        title (str): The title of the section to be printed.

    Returns:
        None
    """
    print("\n" + "=" * 100)
    print(f"{title.upper():^100}")
    print("=" * 100)

def detect_binary_columns(df):
    """
    Detect binary columns in a DataFrame.

    Parameters:
    - df: DataFrame
      Input DataFrame to detect binary columns from.

    Returns:
    - binary_cols: list
      List of column names identified as binary.
    """

    binary_cols = []
    for col in df.columns:
        if df[col].nunique() == 2:
            binary_cols.append(col)
    return binary_cols