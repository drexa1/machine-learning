# House Prices: Advanced Regression Techniques
#
# 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, 
# this competition challenges you to predict the final price of each home.
#
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques

library(readr)
library(ggplot2)
library(scales)
library(dplyr)
library(VIM)
library(pROC)
library(caret)
library(mice)
library(glmnet)
library(assertthat)

set.seed(123)
start_time <- Sys.time()

# Import data
setwd("C:/Users/drexa/git/machine-learning/")
train <- read.csv("./datasets/houseprices_train.csv", stringsAsFactors = FALSE)
test <- read.csv("./datasets/houseprices_test.csv", stringsAsFactors = FALSE)
test_Id <- test$Id # save for submission file

# Plot target distribution from the data provided
print(summary(train$SalePrice))
dev.new()
p_SalePrice <- ggplot(train, aes(SalePrice)) +
               geom_histogram(color = "white", fill = "lightblue") +
               scale_x_continuous(labels = comma) +
               ggtitle("SalePrice from historical data")
print(p_SalePrice)

# Combine datasets to apply wrangling homogeneously
combined <- rbind(dplyr::select(train, MSSubClass:SaleCondition), 
                  dplyr::select(test, MSSubClass:SaleCondition))

# COMBINED DATASET WRANGLING ###################################################################

feature_classes <- sapply(names(combined), function(x) { class(combined[[x]]) })
categorical_features <- names(feature_classes[feature_classes == "character"])
numerical_features <-names(feature_classes[feature_classes != "character"])

# 1-hot encoding for categorical features
dummies <- caret::dummyVars( ~ ., combined[categorical_features])
categorical_1_hot <- predict(dummies, combined[categorical_features])
categorical_1_hot[is.na(categorical_1_hot)] <- -1

# For NA in numeric features impute zero
for (x in numerical_features) {
    combined[[x]] [is.na(combined[[x]])] <- -1
}

# Reconstruct
combined <- cbind(combined[numerical_features], categorical_1_hot)

# Split again into train and test
original_sample <- 1:nrow(train)
train_and_validate <- combined[original_sample, ]
train_and_validate <- cbind(train_and_validate, train$SalePrice)
setnames(train_and_validate, "train$SalePrice", "SalePrice")
test <- combined[-original_sample, ]

# Sample a validate set from training
validate_sample <- caret::createDataPartition(train_and_validate$SalePrice, p = .75, list = FALSE)
train <- train_and_validate[validate_sample, ]
validate  <- train_and_validate[-validate_sample, ]

# BASELINE MODEL ###################################################################

target <- match("SalePrice", names(train))

x_train <- as.matrix(train[ ,-target]) # exclude the target
y_train <- as.numeric(train$SalePrice) # only the target
x_validate <- as.matrix(validate[ ,-target])
y_validate <- as.numeric(validate$SalePrice) 

# Linear model with ridge penalty
cv.ridge <- cv.glmnet(x_train, y_train, alpha = 0, standardize = TRUE, type.measure = "mse")
pred.train.ridge <- predict(cv.ridge, x_train, s = cv.ridge$lambda.1se, type="response")
pred.validate.ridge <- predict(cv.ridge, newx = x_validate, s = cv.ridge$lambda.1se, type="response")

# Linear model with lasso penalty
cv.lasso <- cv.glmnet(x_train, y_train, alpha = 1, standardize = TRUE, type.measure = "mse")
pred.train.lasso <- predict(cv.lasso, x_train, s = cv.lasso$lambda.1se, type="response")
pred.validate.lasso <- predict(cv.lasso, newx = x_validate, s = cv.lasso$lambda.1se, type="response")

# RMSE. Also doable by RMSE(obs, pred)
rmse.train.ridge <- sqrt(mean((log(pred.train.ridge) - log(train$SalePrice))^2))
print(rmse.train.ridge)
rmse.validate.ridge <- sqrt(mean((log(pred.validate.ridge) - log(validate$SalePrice))^2))
print(rmse.validate.ridge)
rmse.train.lasso <- sqrt(mean((log(pred.train.lasso) - log(train$SalePrice))^2))
print(rmse.train.lasso)
rmse.validate.lasso <- sqrt(mean((log(pred.validate.lasso) - log(validate$SalePrice))^2))
print(rmse.validate.lasso)

# AUC (cpu intensive) Not of much use for regression
# roc.ridge <- pROC::multiclass.roc(validate$SalePrice, pred.validate.ridge)
# auc.ridge <- pROC::auc(roc.ridge)
# print(auc.ridge)
# roc.lasso <- pROC::multiclass.roc(validate$SalePrice, pred.validate.lasso)
# auc.lasso <- pROC::auc(roc.lasso)
# print(auc.lasso)

# Retrain best model with the whole original set for unlabeled data
x_retrain <- as.matrix(train_and_validate[ ,-target])
y_retrain <- as.numeric(train_and_validate$SalePrice)
x_test <- as.matrix(test[ ,-target])
y_test <- as.numeric(test$SalePrice)

cv.submit <- cv.glmnet(x_retrain, y_retrain, alpha = 0, standardize = TRUE, type.measure = "mse")
pred.test <- predict(cv.submit, newx = x_test, s = cv.lasso$lambda.1se, type="response")

# Plot submission distribution
dev.new()
p_submit <- ggplot(test, aes(pred.test)) +
    geom_histogram(color = "white", fill = "lightblue") +
    scale_x_continuous(labels = comma) +
    ggtitle("SalePrice predicted")
print(p_submit)

# Export for submission
submit <- data.frame(Id = test_Id, SalePrice = as.numeric(pred.test))
filename <- paste("lm_submission", format(Sys.time(),"%Y%m%d_%H%M%S"), "csv", sep=".")
write.csv(submit, filename, quote = FALSE, row.names = FALSE)

# MODEL OPTIMIZATION ###################################################################

# Feature engineering

# Variable importance

# Predictive imputation

# Preprocessing

# Parameter tuning search

# XGBoost 
# randomForest
# SVM

# LOOC CV

print("Done")
