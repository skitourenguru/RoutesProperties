#! /usr/bin/Rscript

# ------------------------------------------------------------------------------
# Extract variables from command-line arguments
# ------------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
target <- as.character(args[1])
session_name <- as.character(args[2])

# ------------------------------------------------------------------------------
# Load Packages
# ------------------------------------------------------------------------------
library(mgcv)
library(caret)
library(pROC)
library(ROCR)
library(openxlsx)

# ------------------------------------------------------------------------------
# Define target, set wd and read data
# ------------------------------------------------------------------------------
setwd("../data/")
train_file <- paste0("Stage_2_", target, "_train.csv")
val_file <- paste0("Stage_2_", target, "_val.csv")
df_train <- read.csv(train_file)
df_val <- read.csv(val_file)
today_date <- Sys.Date()
formatted_date <- format(today_date, "%d.%m.%Y")

# ------------------------------------------------------------------------------
# Define evaluation table headers
# ------------------------------------------------------------------------------
result_table <- data.frame(
  Date = character(),
  Model = character(),
  Data = character(),
  `Optimized Threshold` = numeric(),
  `Confusion Score` = numeric(),
  Accuracy = numeric(),
  Precision = numeric(),
  Recall = numeric(),
  F1 = numeric(),
  `ROC-AUC` = numeric(),
  stringsAsFactors = FALSE
)

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
calculate_confusion_matrix <- function(probabilities, y_true, threshold) {
  # Parameters:
  # - probabilities: A numeric vector containing predicted probabilities for 
  #   binary classification
  # - y_true: A numeric vector containing true binary labels (0 or 1)
  # - threshold: A numeric value between 0 and 1 to classify probabilities into 
  #   binary predictions
  
  # Output:
  # - A numeric vector of length 4 containing counts for True Negatives (TN), 
  #   False Positives (FP), False Negatives (FN), and True Positives (TP)
  
  if (length(probabilities) != length(y_true)) {
    stop("Length of probabilities and y_true do not match")
  }
  
  predictions <- ifelse(probabilities > threshold, 1, 0)
  
  if (length(predictions) != length(y_true)) {
    stop("Length of predictions and y_true do not match")
  }
  
  conf_matrix <- matrix(0, nrow = 2, ncol = 2, dimnames = list(Actual = c(0, 1), Predicted = c(0, 1)))
  
  for (i in 1:length(predictions)) {
    conf_matrix[y_true[i] + 1, predictions[i] + 1] <- conf_matrix[y_true[i] + 1, predictions[i] + 1] + 1
  }
  
  TN <- conf_matrix[1, 1]
  FP <- conf_matrix[1, 2]
  FN <- conf_matrix[2, 1]
  TP <- conf_matrix[2, 2]
  
  return(c(TN, FP, FN, TP))
}

find_threshold <- function(probabilities, y_true, steps = 1000) {
  # Finds the optimal threshold for binary classification based on balancing 
  # false positives and false negatives.
  #
  # Parameters:
  #   probabilities: A numeric vector of predicted probabilities for the
  #   positive class.
  #   y_true: A binary vector of true class labels (0 for negative class, 1 
  #   for positive class).
  #   steps: Number of steps for binary search algorithm (default is 1000).
  #
  # Output:
  #   A numeric value representing the optimal threshold that balances false 
  #   positives and false negatives.
  
  start <- 0
  end <- 1
  threshold <- NULL
  
  while (start <= end) {
    mid <- (start + end) / 2
    threshold <- mid
    
    conf_matrix <- calculate_confusion_matrix(probabilities, y_true, threshold)
    FP <- conf_matrix[2]
    FN <- conf_matrix[3]
    
    if (FP == FN) {
      return(threshold)
    }
    
    if (FP < FN) {
      end <- mid - 0.0000001
    } else {
      start <- mid + 0.0000001
    }
  }
  
  return(threshold)
}

evaluate_model <- function(model_name, model, dtype, df_val, df_val_true) {
  # Evaluates the performance of a machine learning model on a val dataset.
  
  # Parameters:
  #   - model_name: A character string specifying the name of the model.
  #   - model: The trained machine learning model object.
  #   - dtype: A character string indicating the type of data or session name.
  #   - df_val: The validation dataset as a dataframe.
  #   - df_val_true: The true labels of the validation dataset.
  # Output:
  #   - A dataframe containing evaluation metrics for the model on the
  #     validation dataset.
  
  predictions <- predict(model, newdata = df_val, type = "response")
  threshold <- find_threshold(probabilities = predictions, y_true = df_val_true)
  binary_predictions <- ifelse(predictions >= threshold, 1, 0)
  
  conf_matrix <- calculate_confusion_matrix(probabilities = binary_predictions, y_true = df_val_true, threshold = threshold)
  
  TN <- conf_matrix[1]
  FP <- conf_matrix[2]
  FN <- conf_matrix[3]
  TP <- conf_matrix[4]
  
  if (TP == 0) {
    confusion_score <- 99999
  } else {
    confusion_score <- (FP + FN) * 100 / TP
  }
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  f1_score <- 2 * precision * recall / (precision + recall)
  roc_auc <- roc(df_val_true, predictions)$auc
  
  new_row <- data.frame(
    Date = formatted_date,
    Model = model_name,
    Session_Name = dtype,
    Optimized_Threshold = threshold,
    Confusion_Score = confusion_score,
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1 = f1_score,
    ROC_AUC = roc_auc
  )
  
  return(new_row)
}

# ------------------------------------------------------------------------------
# Build models
# ------------------------------------------------------------------------------
# Define a list of model formulas
model_formulas <- list(
  # 1D models ------------------------------------------------------------------
  "GAM 1D s(ele)" = "s(ele)",
  "GAM 1D s(fd_maxv)" = "s(fd_maxv)",
  "GAM 1D s(fd_risk)" = "s(fd_risk)",
  "GAM 1D s(fold)" = "s(fold)",
  "GAM 1D s(slope)" = "s(slope)",
  "GAM 1D s(ti)" = "s(ti)",
  "GAM 1D s(planc7)" = "s(planc7)",
  # 2D models with smoother on 'ti' or interaction term-------------------------
  "GAM 2D s(ti) + aspect_binary" = "s(ti) + aspect_binary",
  "GAM 2D s(ti) + crevasse" = "s(ti) + crevasse",
  "GAM 2D ti : crevasse" = "ti : crevasse",
  "GAM 2D s(ti) + ele" = "s(ti) + ele",
  "GAM 2D ti : ele" = "ti : ele",
  "GAM 2D s(ti) + fd_maxv" = "s(ti) + fd_maxv",
  "GAM 2D s(ti) + fd_risk" = "s(ti) + fd_risk",
  "GAM 2D s(ti) + fold" = "s(ti) + fold",
  "GAM 2D s(ti) + forest" = "s(ti) + forest",
  "GAM 2D s(ti) + slope" = "s(ti) + slope",
  "GAM 2D s(ti) + street_binary" = "s(ti) + street_binary",
  # 2D models with smoother on 'fd_risk' or interaction term--------------------
  "GAM 2D s(fd_risk) + aspect_binary" = "s(fd_risk) + aspect_binary",
  "GAM 2D s(fd_risk) + crevasse" = "s(fd_risk) + crevasse",
  "GAM 2D fd_risk : crevasse" = "fd_risk : crevasse",
  "GAM 2D s(fd_risk) + ele" = "s(fd_risk) + ele",
  "GAM 2D fd_risk : ele" = "fd_risk : ele",
  "GAM 2D s(fd_risk) + fd_maxv" = "s(fd_risk) + fd_maxv",
  "GAM 2D s(fd_risk) + ti" = "s(fd_risk) + ti",
  "GAM 2D fd_risk : ti" = "fd_risk : ti",
  "GAM 2D s(fd_risk) + fold" = "s(fd_risk) + fold",
  "GAM 2D fd_risk : fold" = "fd_risk : fold",
  "GAM 2D s(fd_risk) + forest" = "s(fd_risk) + forest",
  "GAM 2D s(fd_risk) + slope" = "s(fd_risk) + slope",
  "GAM 2D s(fd_risk) + street_binary" = "s(fd_risk) + street_binary",
  # 3D models with most promising variables so far------------------------------
  "GAM 3D s(fd_risk) + fold + interaction" = "s(fd_risk) + fold + fd_risk : fold",
  "GAM 3D s(fd_risk) + crevasse + interaction" = "s(fd_risk) + crevasse + fd_risk : crevasse",
  "GAM 3D s(fd_risk) + ti + interaction" = "s(fd_risk) + ti + fd_risk : ti",
  "GAM 3D s(ti) + crevasse + interaction" = "s(ti) + crevasse + ti : crevasse",
  "GAM 3D s(ti) + fd_maxv + interaction" = "s(ti) + fd_maxv + ti : fd_maxv",
  "GAM 3D s(ti) + fd_risk + crevasse" = "s(ti) + fd_risk + crevasse",
  "GAM 3D ti + s(fd_risk) + crevasse" = "ti + s(fd_risk) + crevasse",
  "GAM 3D s(ti) + s(fd_risk) + crevasse" = "s(ti) + s(fd_risk) + crevasse",
  "GAM 3D s(fd_risk) + slope + interaction" = "s(fd_risk) + slope + fd_risk : slope",
  "GAM 3D s(ti) + fd_risk + ele" = "s(ti) + fd_risk + ele",
  "GAM 3D ti + s(fd_risk) + ele" = "ti + s(fd_risk) + ele",
  # 4D models with most promising variables-------------------------------------
  "GAM 4D s(ti) + fd_risk + crevasse + ele" = "s(ti) + fd_risk + crevasse + ele",
  "GAM 4D ti + s(fd_risk) + crevasse + ele" = "ti + s(fd_risk) + crevasse + ele",
  "GAM 4D s(ti) + fd_risk + crevasse + fold" = "s(ti) + fd_risk + crevasse + fold",
  "GAM 4D ti + s(fd_risk) + crevasse + fold" = "ti + s(fd_risk) + crevasse + fold",
  "GAM 4D s(ti) + fd_risk + crevasse + street_binary" = "s(ti) + fd_risk + crevasse + street_binary",
  "GAM 4D ti + s(fd_risk) + crevasse + street_binary" = "ti + s(fd_risk) + crevasse + street_binary",
  # Some extra GAMs for the foot modelling (including planc7)-------------------
  "GAM 2D s(fold) + planc7" = "s(fold) + planc7",
  "GAM 2D fold + s(planc7)" = "fold + s(planc7)",
  "GAM 2D s(fold) + crevasse" = "s(fold) + crevasse",
  "GAM 2D fold + s(crevasse)" = "fold + s(crevasse)",
  "GAM 2D s(fold) + fd_risk" = "s(fold) + fd_risk",
  "GAM 2D fold + s(fd_risk)" = "fold + s(fd_risk)",
  "GAM 2D s(fold) + slope" = "s(fold) + slope",
  "GAM 2D fold + s(slope)" = "fold + s(slope)",
  
  "GAM 3D s(fold) + fd_risk + planc7" = "s(fold) + fd_risk + planc7",
  "GAM 3D fold + s(fd_risk) + planc7" = "fold + s(fd_risk) + planc7",
  "GAM 3D fold + fd_risk + s(planc7)" = "fold + fd_risk + s(planc7)",
  "GAM 3D s(fold) + fd_risk + ele" = "s(fold) + fd_risk + ele",
  "GAM 3D fold + s(fd_risk) + ele" = "fold + s(fd_risk) + ele",
  "GAM 3D fold + fd_risk + s(ele)" = "fold + fd_risk + s(ele)",
  "GAM 3D s(fold) + fd_risk + crevasse" = "s(fold) + fd_risk + crevasse",
  "GAM 3D fold + s(fd_risk) + crevasse" = "fold + s(fd_risk) + crevasse",
  "GAM 3D fold + fd_risk + s(crevasse)" = "fold + fd_risk + s(crevasse)",
  "GAM 3D s(fold) + fd_risk + slope" = "s(fold) + fd_risk + slope",
  "GAM 3D fold + s(fd_risk) + slope" = "fold + s(fd_risk) + slope",
  "GAM 3D fold + fd_risk + s(slope)" = "fold + fd_risk + s(slope)",

  "GAM 4D s(fold) + fd_risk + planc7 + slope" = "s(fold) + fd_risk + planc7 + slope",
  "GAM 4D fold + s(fd_risk) + planc7 + slope" = "fold + s(fd_risk) + planc7 + slope",  
  "GAM 4D fold + fd_risk + s(planc7) + slope" = "fold + fd_risk + s(planc7) + slope",  
  "GAM 4D fold + fd_risk + planc7 + s(slope)" = "fold + fd_risk + planc7 + s(slope)",  
  "GAM 4D s(fold) + fd_risk + planc7 + ele" = "s(fold) + fd_risk + planc7 + ele",
  "GAM 4D fold + s(fd_risk) + planc7 + ele" = "fold + s(fd_risk) + planc7 + ele",  
  "GAM 4D fold + fd_risk + s(planc7) + ele" = "fold + fd_risk + s(planc7) + ele",  
  "GAM 4D fold + fd_risk + planc7 + s(ele)" = "fold + fd_risk + planc7 + s(ele)",
  "GAM 4D s(fold) + fd_risk + planc7 + crevasse" = "s(fold) + fd_risk + planc7 + crevasse",
  "GAM 4D fold + s(fd_risk) + planc7 + crevasse" = "fold + s(fd_risk) + planc7 + crevasse",  
  "GAM 4D fold + fd_risk + s(planc7) + crevasse" = "fold + fd_risk + s(planc7) + crevasse",  
  "GAM 4D fold + fd_risk + planc7 + s(crevasse)" = "fold + fd_risk + planc7 + s(crevasse)" 
  )

# ------------------------------------------------------------------------------
# Train and evaluate models (full data)
# ------------------------------------------------------------------------------

# Loop over each model formula
for (model_name in names(model_formulas)) {
  tryCatch({
    # Get the formula
    formula <- as.formula(paste(target, "~", model_formulas[[model_name]]))
    
    # Fit the model
    gam_model <- gam(formula, data = df_train, family = binomial)
    
    # Evaluate the model
    new_row <- evaluate_model(model_name, gam_model, session_name, df_val, df_val[[target]])
    
    # Append the results to the result table
    result_table <- rbind(result_table, new_row)
  }, error = function(e) {
    # Handle errors
    cat("Error occurred while fitting model", model_name, ": ", conditionMessage(e), "\n")
  })
}

# ------------------------------------------------------------------------------
# Output
# ------------------------------------------------------------------------------
# Update scores in the xlsx
# File path
file_path <- paste0("../scores/", target, "_scores.xlsx")

# Check if the file exists
if (file.exists(file_path)) {
  # Read existing data
  existing_data <- read.xlsx(file_path)
  
  # Append new data
  combined_data <- rbind(existing_data, result_table)
} else {
  # If the file doesn't exist, use the new data
  combined_data <- result_table
}

# Write the combined data to the file
write.xlsx(combined_data, file_path, row.names = FALSE)

# ------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------