# Master Thesis
**ARTIFICIAL INTELLIGENCE IN SKI TOURING**  
**Prediction of *foot*- and *caution*-sections for ski touring routes with a machine learning approach**  
  
**Contributors**  
[Claudio Furrer](https://www.linkedin.com/in/furrclaudio/) (Master Student, Hochschule Luzern)  
[Martin Rumo](https://www.linkedin.com/in/martinrumo/) (Supervisor, Hochschule Luzern)  
[Günter Schmudlach](https://info.skitourenguru.ch/index.php/about) (Co-Supervisor & Client, Skitourenguru.ch)  

**University**  
Hochschule Luzern  
Master of Science in Applied Information and Data Science

**Citation**  
*Furrer, C., Schmudlach, G., & Rumo, M. (2024). Artifical intelligence in ski touring: Prediction of foot- and caution-sections for ski touring routes with a machine learning approach (Master thesis). Hochschule Luzern.*

## Overview
Welcome to the repository **RoutesProperties**! This project is about the prediction of *foot*- and *caution*-sections for ski touring routes with a machine learning approach. Whether you're a scientist, a student or ski tourer, this repository is designed to solve the problem of these section classifications. Happy reading!  

## Structure
¦- code (*reproducible Python and R code*)  
¦- data (*download link for data*)  
¦- logs (*log files for all runs*)  
¦- notebooks (*non-reproducible Jupyter notebooks for data analysis*)  
¦- paper (*master thesis*)  
¦- plots (*plots used for the master thesis*)  
¦- scores (*model scores on validation data*)  
¦- summaries (*summaries of winning models*)  

## Data
The routes properties data files were too large for the GitHub repository, which is why a download link is stored in the 'data' folder. The following features are in this data set: 

![Feature Table](https://github.com/skitourenguru/RoutesProperties/blob/main/plots/feature_table.jpg)

## Requirements  
Python Version: 3.10.7  
R Version: 4.2.3  
required libraries (Python) and packages (R) see [requirements.txt](https://github.com/skitourenguru/RoutesProperties/blob/main/code/requirements.txt)

## Code
The following code was used for model training and evaluation:  
[main.py](https://github.com/skitourenguru/RoutesProperties/blob/main/code/main.py)  
[my_functions.py](https://github.com/skitourenguru/RoutesProperties/blob/main/code/my_functions.py)  
[GAM.R](https://github.com/skitourenguru/RoutesProperties/blob/main/code/GAM.R) 
[winners.R](https://github.com/skitourenguru/RoutesProperties/blob/main/code/winners.R)  

For the modelling part, the Python script [main.py](https://github.com/skitourenguru/RoutesProperties/blob/main/code/main.py) is used for modelling both *caution* and *foot*. Logistic regression, random forest and gradient boosting are modelled directly in Python with the library scikit-learn. Additionally, an R script has been created for the general additive models, which are implemented in the mgcv package. However, the [main.py](https://github.com/skitourenguru/RoutesProperties/blob/main/code/main.py) script is created in such a way that it calls the R script [GAM.R](https://github.com/skitourenguru/RoutesProperties/blob/main/code/GAM.R) directly. At the end of the [main.py](https://github.com/skitourenguru/RoutesProperties/blob/main/code/main.py) script, the scores are added to an existing Excel file containing all the evaluation scores. The script is reusable and uses several functions from the [my_functions.py](https://github.com/skitourenguru/RoutesProperties/blob/main/code/my_functions.py) script (function description see below), where for example the filters and methods can be specified. This makes it possible to approach different feature engineering techniques and observe how the evaluation scores of the models change. Thus, before each execution of the [main.py](https://github.com/skitourenguru/RoutesProperties/blob/main/code/main.py) script, the variable *session_name* must be defined, which is saved in the Excel file at the end. This allows the feature engineering methods used (e.g. ti-filter, street-filter, oversampling, undersampling, scaling) to be traced when the scores of different runs are compared later. For transparency reasons of the individual run, the console output of each run was saved in a log file in case something needs to be retraced later (see [logs](https://github.com/skitourenguru/RoutesProperties/tree/main/logs)). The log files also document certain parameters such as the selected variables with RFE or the best parameters according to GridSearch. The functions in the [my_functions.py](https://github.com/skitourenguru/RoutesProperties/blob/main/code/my_functions.py) script are imported into the [main.py](https://github.com/skitourenguru/RoutesProperties/blob/main/code/main.py) script. The most important functions for modelling are briefly described below, whereby the detailed descriptions and parameters can be viewed directly in the [my_functions.py](https://github.com/skitourenguru/RoutesProperties/blob/main/code/my_functions.py) script.

## Functions
**function** *feature_engineering*  
```python
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
``` 
  
**function** *train_val_test*  
```python
"""
Preprocesses the input DataFrame by splitting it into training, validation, and test sets,
    potentially addressing class imbalance and performing scaling if specified, and exports
    the sets to CSV files.

    Parameters:
    - df (DataFrame): the input data containing features and target variable
    - target (str): the name of the target variable
    - method (str): optional, method for dealing with class imbalance (default is 'imbalanced')
        Possible values: 'imbalanced', 'undersampling', 'oversampling'
    - scaling (bool, optional): whether to apply feature scaling (default is False)

    Returns:
    - X_train (DataFrame), y_train (Series), X_val (DataFrame), y_val (Series), X_test (DataFrame), y_test (Series):
      DataFrames and Series, containing the features and target variables for training, validation, and test sets
"""
``` 
  
**function** *find_threshold*  
```python
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
```
  
**function** *calculate_confusion_matrix*  
```python
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
```
  
**function** *evaluate*  
```python
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
```

## Contact
E-Mail: cla.furrer@gmail.com
