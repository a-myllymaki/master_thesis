library(pacman)
pacman::p_load(dplyr,tidyr,iml, xgboost, counterfactuals,data.table, reshape2,tidyverse, 
               caret, cluster,factoextra, plotly, ggcorrplot, purrr,class, SHAPforxgboost,
               ggrepel,jmotif, ggplot2,combinat, corrplot, viridis, ggpubr, ggthemes,
               gridExtra, patchwork,colorblindr,psych,tibble,dichromat,tidyverse)

####################### CLASSIFICATION TASK #######################

#=================================== 
# DATA PREPARATION
#=================================== 
clf_data <- merged_data %>%
  inner_join(ASRS_scores %>% select(id, ASRS_screener), by = "id") %>%
  left_join(chronotype, by = "id") %>%
  ungroup()%>%
  select(-c(id)) %>%
  select(-ASRS_screener,  everything(), ASRS_screener ) 
  
screener_data_clf <- clf_data$ASRS_screener # screener variable

set.seed(2025)
train_index <- sample(1:nrow(reg_data), 0.7 * nrow(reg_data))
train_data_clf <- clf_data[train_index, ]
test_data_clf  <- clf_data[-train_index, ]

# Convert data to DMatrix so it's compatbile with xgboost function
dtrain_clf <- xgb.DMatrix(data = as.matrix(train_data_clf %>% select(-ASRS_screener)),
                          label = train_data_clf$ASRS_screener)
dtest_clf <- xgb.DMatrix(data = as.matrix(test_data_clf %>% select(-ASRS_screener)),
                         label = test_data_clf$ASRS_screener)
dfull_clf <- xgb.DMatrix(data = as.matrix(clf_data %>% select(-ASRS_screener)),
                         label = clf_data$ASRS_screener)

#=================================== 
# 5 FOLD CV - LEARNING CURVES 
#=================================== 

cv_results_clf <- map_dfr(max_depths, ~{
  set.seed(2025)
  params <- list(
    objective = "binary:logistic",
    eval_metric = "rmse",
    max_depth = .x,
    lambda = 0,
    gamma = 0,
    eta = 0.3
  )
  
  xgb.cv(
    params = params,
    data = dfull_clf,
    nrounds = 100, 
    nfold = 5,
    verbose = FALSE
  )$evaluation_log %>% 
    mutate(max_depth = as.factor(.x))
})

# Learning curve plots
ggplot(cv_results_clf, aes(x = iter, y = train_rmse_mean, color = max_depth)) +
  geom_line(size = 1) +
  geom_line(aes(x = iter, y=test_rmse_mean, color = max_depth))+
  labs(
    x = "Number of boosting rounds (nrounds)",
    y = "CV error (test RMSE)",
    title = "Learning curves using 5-fold CV",
    subtitle = "Thin lines = testing data, Thick lines = training data",
    color = "Max depth",
    fill = "Max depth"
  ) +
  theme_minimal() 

#=================================== 
# TRAIN FINAL CLASSIFICATION MODEL
#===================================

params_clf <- list(
  objective = "binary:logistic",
  eval_metric = "rmse",
  max_depth = 2,
  eta = 0.3,
  lambda =1,
  scale_pos_weight = 1
  
)

clf_model <- xgb.train(
  params = params_clf,
  data = dtrain_clf,
  nrounds = 6,
  watchlist = list(test = dtest_clf),
  verbose = 0
)

pred_probs <- predict(clf_model, dtest_clf)
pred_labels <- ifelse(pred_probs > 0.5, 1, 0)

# Confusion matrix
conf_matrix <- confusionMatrix(factor(pred_labels), factor(test_data_clf $ASRS_screener))
print(conf_matrix)

####################### XGBOOST VS. KNN #######################
#=================================== 
# XGBOOST LEARNING CURVES
#===================================

labels <- as.numeric(clf_data$ASRS_screener) - 1  # This will convert "X0" to 0 and "X1" to 1

cv_results_XB_clf <- map_dfr(max_depths, ~{
  set.seed(2025)
  params <- list(
    objective = "binary:logistic",
    eval_metric = "error",
    max_depth = 2,
    lambda = 1,
    gamma = 0,
    eta = 0.3
  )
  
  xgb.cv(
    params = params,
    data = dfull_clf,
    nrounds = 100, 
    nfold = 5,
    verbose = FALSE
  )$evaluation_log %>% 
    mutate(max_depth = as.factor(2))
})

# learning curve plots
ggplot(cv_results_XB_clf, aes(x = iter)) +
  geom_line(aes(y = train_error_mean, color = "Training"), size = 1) +
  geom_line(aes(y = test_error_mean, color = "Validation"), size = 1) +
  scale_color_manual(name = "Dataset",
                     values = c("Training" = "red", "Validation" = "blue")) +
  labs(
    x = "Number of boosting rounds (nrounds)",
    y = "Classification error",
    title = "XGBoost learning curves"
  ) +
  theme_minimal() +theme(
    legend.position = "bottom",          # Moves legend to bottom
    axis.text.x.top = element_text(),    # Ensures top axis labels are visible
    legend.box = "horizontal"            # Optional: Aligns legend items horizontally
  )

#=================================== 
# kNN LEARNING CURVES
#===================================

# Function to calculate classification error using 5-fold CV 
get_cv_error_rates <- function(full_data, k_values, target_col = "ASRS_screener", folds = 5) {
  # Create 5-fold CV splits
  set.seed(2025)
  cv_folds <- createFolds(full_data[[target_col]], k = folds, list = TRUE, returnTrain = TRUE)
  
  results <- data.frame(k = k_values, 
                        train_error = NA, 
                        validation_error = NA)
  
  # For each k value
  for (i in seq_along(k_values)) {
    k <- k_values[i]
    
    # Store errors across folds
    train_errors <- numeric(folds)
    val_errors <- numeric(folds)
    
    # For each fold
    for (fold in 1:folds) {
      # Get train and validation indices for this fold
      train_indices <- cv_folds[[fold]]
      val_indices <- setdiff(1:nrow(full_data), train_indices)
      
      # Split data
      fold_train <- full_data[train_indices, ]
      fold_val <- full_data[val_indices, ]
      
      # Preprocess data
      preProc <- preProcess(fold_train[, -which(names(fold_train) == target_col)], 
                            method = c("center", "scale"))
      fold_train_proc <- predict(preProc, fold_train)
      fold_val_proc <- predict(preProc, fold_val)
      
      # Extract features and targets
      fold_train_x <- fold_train_proc %>% select(-all_of(target_col)) %>% as.matrix()
      fold_train_y <- fold_train_proc[[target_col]]
      fold_val_x <- fold_val_proc %>% select(-all_of(target_col)) %>% as.matrix()
      fold_val_y <- fold_val_proc[[target_col]]
      
      # Calculate training error (predictions on training data)
      train_pred <- knn(train = fold_train_x, test = fold_train_x, cl = fold_train_y, k = k)
      train_errors[fold] <- mean(train_pred != fold_train_y)
      
      # Calculate validation error (predictions on validation data)
      val_pred <- knn(train = fold_train_x, test = fold_val_x, cl = fold_train_y, k = k)
      val_errors[fold] <- mean(val_pred != fold_val_y)
    }
    
    # mean errors across folds
    results$train_error[i] <- mean(train_errors)
    results$validation_error[i] <- mean(val_errors)
  }
  
  return(results)
}

# k values to test
k_values <- seq(1, 21, by = 2)
error_rates <- get_cv_error_rates(clf_data, k_values)

error_rates_long <- error_rates %>%
  pivot_longer(cols = c(train_error, validation_error),
               names_to = "dataset",
               values_to = "error_rate")

ggplot(error_rates_long, aes(x = k, y = error_rate, color = dataset)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  scale_color_manual(values = c("red", "blue"),
                     labels = c("Training", "Validation"),
                     name = "Dataset") +
  labs(title = "Knn classification error vs. k  (5-fold CV)",
       x = "Number of neighbors (k)",
       y = "Classification Error " )+
  theme_minimal() +
  theme(legend.position = "bottom")


# Optimal k value
optimal_k <- k_values[which.min(error_rates$validation_error)]
cat("Optimal k value:", optimal_k, "\n")
cat("Validation error at optimal k:", min(error_rates$validation_error), "\n")


#### Add N/k (degree of freedom) to  kNN results
error_rates <- error_rates %>%
  mutate(N_over_k = nrow(clf_data)/k)


ggplot(error_rates, aes(x = N_over_k)) +
  geom_line(aes(y = train_error, color = "Training"), size = 1) +
  geom_line(aes(y = validation_error, color = "Validation"), size = 1) +
  scale_color_manual(name = "Dataset",
                     values = c("Training" = "red", "Validation" = "blue")) +
  labs(title = "kNN learning curves",
       x = "N/k",
       y = "Classification Error",
       color = "Dataset") +
  theme_minimal()+ theme(
    legend.position = "bottom",          # Moves legend to bottom
    axis.text.x.top = element_text(),    # Ensures top axis labels are visible
    legend.box = "horizontal"            # Optional: Aligns legend items horizontally
  )

pred_probs_knn <- predict(clf_model, dtest_clf)
pred_labels_knn <- ifelse(pred_probs > 0.5, 1, 0)

# kNN confusion matrix
conf_matrix_knn <- confusionMatrix(factor(pred_labels_knn), factor(test_data_clf $ASRS_screener))
print(conf_matrix_knn)


## FINAL KNN MODEL


# Apply kNN with k=19
k <- 19
predicted_labels <- knn(
  train = train_data_clf %>% select(-ASRS_screener), 
  test = test_data_clf %>% select(-ASRS_screener), 
  cl = train_data_clf$ASRS_screener, 
  k = k
)

# kNN confusion matrix
confusion_matrix <- table(Predicted = predicted_labels, Actual = test_data_clf$ASRS_screener)
print("Confusion Matrix:")
print(confusion_matrix)

####################### TRY COMBINATIONS OF 5 FEATURES #######################

# Get all feature names (excluding the target variable)
all_features <- colnames(clf_data %>% select(-ASRS_screener))

# Generate all combinations of 5 features from the 9 features
feature_combinations <- combn(all_features, 5, simplify = FALSE)

combinations_df <- data.frame(   # dataframe to store results
  feature_combination = character(),
  min_test_error = numeric(),
  best_nrounds = integer(),
  stringsAsFactors = FALSE
)

# Function to evaluate a single feature combination
evaluate_feature_combination <- function(features, seed_val = 2025) {
  set.seed(seed_val)
  
  # Create DMatrix with only the selected features
  dfull_subset <- xgb.DMatrix(
    data = as.matrix(clf_data %>% select(all_of(features))),
    label = clf_data$ASRS_screener
  )
  
  params <- list(
    objective = "binary:logistic",
    eval_metric = "rmse",
    max_depth = 2,  
    lambda = 1,
    gamma = 0,
    eta = 0.3,
    scale_pos_weight = 1
  )
  
  # Run cross-validation
  cv_result <- xgb.cv(
    params = params,
    data = dfull_subset,
    nrounds = 60,  
    nfold = 5,
    verbose = FALSE
  )
  
  # Get the evaluation log
  eval_log <- cv_result$evaluation_log
  
  # Find the best performance
  best_index <- which.min(eval_log$test_rmse_mean)
  min_test_error <- eval_log$test_rmse_mean[best_index]
  best_nrounds <- best_index
  
  # Return results
  return(list(
    feature_combination = paste(features, collapse = ", "),
    min_test_error = min_test_error,
    best_nrounds = best_nrounds
  ))
}

# Process all feature combinations
for (i in 1:length(feature_combinations)) {
  cat(sprintf("Processing combination %d/%d\n", i, length(feature_combinations)))
  
  combination_result <- evaluate_feature_combination(feature_combinations[[i]])
  
  # Add to results dataframe
  combinations_df <- rbind(combinations_df, data.frame(
    feature_combination = combination_result$feature_combination,
    min_test_error = combination_result$min_test_error,
    best_nrounds = combination_result$best_nrounds,
    stringsAsFactors = FALSE
  ))
}

# Sort the results by test error
combinations_df <- combinations_df %>% arrange(min_test_error)

# Display the top 10 best combinations
head(combinations_df, 10)

# Find the best combination
best_combination <- combinations_df %>% 
  dplyr::slice(1) %>% 
  pull(feature_combination) %>%
  strsplit(", ") %>% 
  unlist()

best_comb_params <- list(
  objective = "binary:logistic",
  eval_metric = "rmse",
  max_depth = 2,
  lambda = 1,
  gamma = 0,
  eta = 0.3,
  scale_pos_weight = 1
)

# best n_rounds from CV
best_nrounds <- combinations_df %>% dplyr::slice(1) %>% pull(best_nrounds)

#=================================== 
# TRAIN MODEL WITH BEST FEATURES
#===================================

best_clf_data <- clf_data %>% select((c(all_of(best_combination), ASRS_screener)))

train_data_clf_best  <- best_clf_data[train_index_chr, ]
test_data_clf_best   <- best_clf_data[-train_index_chr, ]

dtrain_clf_best <- xgb.DMatrix(data = as.matrix(train_data_clf_best  %>% select(-ASRS_screener)),
                               label = train_data_clf_best $ASRS_screener)
dtest_clf_best <- xgb.DMatrix(data = as.matrix(test_data_clf_best  %>% select(-ASRS_screener)),
                              label = test_data_clf_best $ASRS_screener)
dfull_clf_best <- xgb.DMatrix(data = as.matrix(best_clf_data  %>% select(-ASRS_screener)),
                              label = best_clf_data$ASRS_screener)


best_comb_clf_model <- xgb.train(
  params = best_comb_params,
  data = dtrain_clf_best,
  nrounds = best_nrounds,
  watchlist = list(test = dtest_clf_best),
  verbose = 0
)

pred_probs <- predict(best_comb_clf_model, dtest_clf_best)
pred_labels <- ifelse(pred_probs > 0.5, 1, 0)

conf_matrix <- confusionMatrix(factor(pred_labels), factor(test_data_clf_best$ASRS_screener))
print(conf_matrix)


