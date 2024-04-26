# ------------------------------------------------------------------------------
# Note
# ------------------------------------------------------------------------------
# This file is only for the analysis of the winner models.
# The file is not intended for re usability!

# ------------------------------------------------------------------------------
# Define target, set wd and read data
# ------------------------------------------------------------------------------
target <- 'caution'
setwd("../data/")

train_file <- paste0("Stage_2_", target, "_train.csv")
val_file <- paste0("Stage_2_", target, "_val.csv")
test_file <- paste0("Stage_2_", target, "_test.csv")

df_train <- read.csv(train_file)
df_val <- read.csv(val_file)
df_test <- read.csv(test_file)


# ------------------------------------------------------------------------------
# Load Packages
# ------------------------------------------------------------------------------
library(mgcv)

# ------------------------------------------------------------------------------
# Load Functions
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

# ------------------------------------------------------------------------------
# Analysis logistic regression ('ti' threshold) [streets are excluded]
# ------------------------------------------------------------------------------
file_path <- "../summaries/inverse_ti.txt"
file_conn <- file(file_path, "w")
cat("# ------------------------------------------------------------------------------\n", file = file_conn)
cat("# Analysis logistic regression ('ti' threshold) [streets are excluded]\n", file = file_conn)
cat("# ------------------------------------------------------------------------------\n", file = file_conn)

# Fit logistic regression model
logistic_model <- glm(caution ~ ti, data = df_train, family = binomial)
predictions <- predict(logistic_model, newdata = df_val, type = "response")

# Find threshold where FP = FN
threshold <- find_threshold(probabilities = predictions, y_true = df_val[[target]])
cat("Threshold:", threshold, "\n", file = file_conn)
coefficients <- coef(logistic_model)
a <- coefficients[1]  # => Intercept
b <- coefficients[2]  # => Coefficient for ti
cat("Coefficients:\n", file = file_conn)
cat("Intercept (a):", a, "\n", file = file_conn)
cat("Coefficient for ti (b):", b, "\n", file = file_conn)

# Invert formula => p() = x * ti + b => inverse is (p - b) / x = ti
z <- log(threshold / (1 - threshold))
ti_threshold <- (z - a) / b
cat("ti yields 'caution = 1' at:", ti_threshold, "\n", file = file_conn)

# Close the file connection
close(file_conn)

# ------------------------------------------------------------------------------
# Analysis logistic regression ('slope' threshold) [streets are excluded]
# ------------------------------------------------------------------------------
# Open a text file for appending
file_path <- "../summaries/inverse_slope.txt"
file_conn <- file(file_path, "a")
cat("\n# ------------------------------------------------------------------------------\n", file = file_conn)
cat("# Analysis logistic regression ('slope' threshold) [streets are excluded]\n", file = file_conn)
cat("# ------------------------------------------------------------------------------\n", file = file_conn)

# Fit logistic regression model
logistic_model <- glm(caution ~ slope, data = df_train, family = binomial)
predictions <- predict(logistic_model, newdata = df_val, type = "response")

# Find threshold where FP = FN
threshold <- find_threshold(probabilities = predictions, y_true = df_val[[target]])
cat("Threshold:", threshold, "\n", file = file_conn)
coefficients <- coef(logistic_model)
a <- coefficients[1]  # => Intercept
b <- coefficients[2]  # => Coefficient for slope
cat("Coefficients:\n", file = file_conn)
cat("Intercept (a):", a, "\n", file = file_conn)
cat("Coefficient for slope (b):", b, "\n", file = file_conn)

# Invert formula => p() = x * slope + b => inverse is (p - b) / x = slope
z <- log(threshold / (1 - threshold))
slope_threshold <- (z - a) / b
cat("Slope yields 'caution = 1' at:", slope_threshold, "\n", file = file_conn)

# Close the file connection
close(file_conn)

# ------------------------------------------------------------------------------
# Analysis logistic regression ('fd_risk' threshold) [streets are excluded]
# ------------------------------------------------------------------------------
# Open a text file for appending
file_path <- "../summaries/inverse_fd_risk.txt"
file_conn <- file(file_path, "a")
cat("\n# ------------------------------------------------------------------------------\n", file = file_conn)
cat("# Analysis logistic regression ('fd_risk' threshold) [streets are excluded]\n", file = file_conn)
cat("# ------------------------------------------------------------------------------\n", file = file_conn)

# Fit logistic regression model
logistic_model <- glm(caution ~ fd_risk, data = df_train, family = binomial)
predictions <- predict(logistic_model, newdata = df_val, type = "response")

# Find threshold where FP = FN
threshold <- find_threshold(probabilities = predictions, y_true = df_val[[target]])
cat("Threshold:", threshold, "\n", file = file_conn)
coefficients <- coef(logistic_model)
a <- coefficients[1]  # => Intercept
b <- coefficients[2]  # => Coefficient for fd_risk
cat("Coefficients:\n", file = file_conn)
cat("Intercept (a):", a, "\n", file = file_conn)
cat("Coefficient for fd_risk (b):", b, "\n", file = file_conn)

# Invert formula => p() = x * fd_risk + b => inverse is (p - b) / x = fd_risk
z <- log(threshold / (1 - threshold))
fd_risk_threshold <- (z - a) / b
cat("fd_risk yields 'caution = 1' at:", fd_risk_threshold, "\n", file = file_conn)

# Close the file connection
close(file_conn)

# ------------------------------------------------------------------------------
# Analysis logistic regression ('crevasse' threshold) [non-glacier are excluded]
# ------------------------------------------------------------------------------
file_path <- "../summaries/inverse_crevasse.txt"
file_conn <- file(file_path, "a")
cat("\n# ------------------------------------------------------------------------------\n", file = file_conn)
cat("# Analysis logistic regression ('crevasse' threshold) [non-glacier are excluded]\n", file = file_conn)
cat("# ------------------------------------------------------------------------------\n", file = file_conn)

# Drop all datapoints not on a glacier
df_train_cr <- subset(df_train, crevasse != 0)
df_val_cr <- subset(df_val, crevasse != 0)

# Fit logistic regression model
logistic_model <- glm(caution ~ crevasse, data = df_train_cr, family = binomial)
predictions <- predict(logistic_model, newdata = df_val_cr, type = "response")

# Find threshold where FP = FN
threshold <- find_threshold(probabilities = predictions, y_true = df_val_cr[[target]])
cat("Threshold:", threshold, "\n", file = file_conn)
coefficients <- coef(logistic_model)
a <- coefficients[1]  # => Intercept
b <- coefficients[2]  # => Coefficient for crevasse
cat("Coefficients:\n", file = file_conn)
cat("Intercept (a):", a, "\n", file = file_conn)
cat("Coefficient for crevasse (b):", b, "\n", file = file_conn)

# Invert formula => p() = x * crevasse + b => inverse is (p - b) / x = crevasse
z <- log(threshold / (1 - threshold))
crevasse_threshold <- (z - a) / b
cat("Crevasse yields 'caution = 1' at:", crevasse_threshold, "\n", file = file_conn)

# Close the file connection
close(file_conn)

# ------------------------------------------------------------------------------
# Analysis GAM model (winner 'caution')
# ------------------------------------------------------------------------------
winner_model <- gam(caution ~ s(ti) + fd_risk + crevasse, data = df_train, family = binomial)
summary(winner_model)
print(coef(winner_model))

# Output summary
summary_output <- capture.output(summary(winner_model))
writeLines(summary_output, "../summaries/winner_caution.txt")

# Make predictions
pred_train <- predict(winner_model, newdata = df_train, type = "response")
pred_val <- predict(winner_model, newdata = df_val, type = "response")
pred_test <- predict(winner_model, newdata = df_test, type = "response")

# Create data frames for predictions
df_pred_train <- data.frame(probability = pred_train)
df_pred_val <- data.frame(probability = pred_val)
df_pred_test <- data.frame(probability = pred_test)

# Write predictions to CSV files
write.csv(df_pred_train, file = "./predictions/pred_train_caution.csv", row.names = FALSE)
write.csv(df_pred_val, file = "./predictions/pred_val_caution.csv", row.names = FALSE)
write.csv(df_pred_test, file = "./predictions/pred_test_caution.csv", row.names = FALSE)

# ------------------------------------------------------------------------------
# Define target, set wd and read data
# ------------------------------------------------------------------------------
target <- 'foot'
setwd("../data/")

train_file <- paste0("Stage_2_", target, "_train.csv")
val_file <- paste0("Stage_2_", target, "_val.csv")
test_file <- paste0("Stage_2_", target, "_test.csv")

df_train <- read.csv(train_file)
df_val <- read.csv(val_file)
df_test <- read.csv(test_file)

# ------------------------------------------------------------------------------
# Analyse GAM model (winner 'foot')
# ------------------------------------------------------------------------------
winner_model <- gam(foot ~ s(fold) + fd_risk + ele, data = df_train, family = binomial)
summary(winner_model)
print(coef(winner_model))

# Output summary
summary_output <- capture.output(summary(winner_model))
writeLines(summary_output, "../summaries/winner_foot.txt")

# Make predictions
pred_train <- predict(winner_model, newdata = df_train, type = "response")
pred_val <- predict(winner_model, newdata = df_val, type = "response")
pred_test <- predict(winner_model, newdata = df_test, type = "response")

# Create data frames for predictions
df_pred_train <- data.frame(probability = pred_train)
df_pred_val <- data.frame(probability = pred_val)
df_pred_test <- data.frame(probability = pred_test)

# Write predictions to CSV files
write.csv(df_pred_train, file = "./predictions/pred_train_foot.csv", row.names = FALSE)
write.csv(df_pred_val, file = "./predictions/pred_val_foot.csv", row.names = FALSE)
write.csv(df_pred_test, file = "./predictions/pred_test_foot.csv", row.names = FALSE)