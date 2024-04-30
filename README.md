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

## Structure
¦- code (*reproducible Python and R code*)  
¦- logs (*log files for all runs*)  
¦- notebooks (*non-reproducible Jupyter notebooks for data analysis*)  
¦- paper (*master thesis*)  
¦- plots (*plots used for the master thesis*)  
¦- scores (*model scores on validation data*)  
¦- summaries (*summaries of winning models*)  

## Dependencies
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
Applies all the feature engineering tasks from chapter 5.1. Filters can optionally be specified in the
function call (e.g. ti_filter=True, tunnel_filter=False, street_filter=False).
Additionally, features to be excluded can also be specified (e.g. col_drop=List)  
  
**function** *train_val_test*  
Preprocesses the cleaned input data by splitting it into training, validation, and test set, and exports the
sets to CSV files (CSVs will be used for reading in with GAM.R). Scaling, over- or undersampling can
optionally be applied (e.g. scaling=True, method=’oversampled’). Additionally, the target
has to be specified in the function argument (e.g. target=’caution’).
  
**function** *find_threshold*  
The function uses a binary search algorithm to find the p-threshold value for which FP = FN results in
the predictions. Thanks to the binary search, not all thresholds have to be searched and the search time
can be greatly reduced as the algorithm iteratively narrows down the search range. Within each iteration,
it calls the function calculate_confusion_matrix for the given threshold and adjusts the search
range accordingly. If the confusion matrix yields FP = FN, the search is terminated and the p-threshold
found is returned.
  
**function** *calculate_confusion_matrix*  
Calculates the confusion matrix based on predicted probabilities and true labels at a specified threshold
for binary classification. As described in the previous function find_threshold, the function is
called for as long as it takes to find the optimum p-threshold which yields FP = FN.
  
**function** *evaluate*  
Evaluates the binary classification model on validation data and computes various classification metrics
at the optimized p-threshold, where FP = FN. The function writes the performance metrics of the model
into the data frame, which is later written into the Excel scoring file


## Usage



## Contact
E-Mail: cla.furrer@gmail.com
