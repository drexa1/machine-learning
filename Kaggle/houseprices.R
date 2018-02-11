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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

system("ls ../input")
train <- read.csv("../input/houseprices_train.csv", stringsAsFactors = FALSE)
test <- read.csv("../input/houseprices_test.csv", stringsAsFactors = FALSE)
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

feature_classes <- sapply(names(combined), function(x) class(combined[[x]]) )
categorical_features <- names(feature_classes[feature_classes == "character"])
numerical_features <-names(feature_classes[feature_classes != "character"])

# Convert all string like features to factor
combined[ ,categorical_features] <- lapply(combined[ ,categorical_features], factor)

# Some of the numerical features in this particular case are more like discrete categories
combined$MSSubClass <- factor(combined$MSSubClass)
combined$OverallQual <- factor(combined$OverallQual)
combined$OverallCond <- factor(combined$OverallCond)
combined$BsmtFullBath <- factor(combined$BsmtFullBath)
combined$BsmtHalfBath <- factor(combined$BsmtHalfBath)
combined$FullBath <- factor(combined$FullBath)
combined$HalfBath <- factor(combined$HalfBath)
combined$BedroomAbvGr <- factor(combined$BedroomAbvGr)
combined$KitchenAbvGr <- factor(combined$KitchenAbvGr)
combined$TotRmsAbvGrd <- factor(combined$TotRmsAbvGrd)
combined$Fireplaces <- factor(combined$Fireplaces)
combined$GarageCars <- factor(combined$GarageCars)
combined$MoSold <- factor(combined$MoSold)
combined$YrSold <- factor(combined$YrSold)

# Check how many values are NA
NA_summary <- sapply(combined, function(x) sum(is.na(x)))
dev.new()
p_NA_1 <- VIM::aggr(combined, col = mdc(1:2), axes = FALSE, border = NA, sortVars = FALSE)
print(p_NA_1)

# [OPTIMIZATION] Features too empty, sorry but drop
too_empty <- which(NA_summary > 1000)
combined <- combined[ , -too_empty]

# update the classes list after the casts and deletions
feature_classes <- sapply(names(combined), function(x) class(combined[[x]]) )
categorical_features <- names(feature_classes[feature_classes == "factor"])
numerical_features <-names(feature_classes[feature_classes != "factor"])
assert_that(length(categorical_features) + length(numerical_features) == length(names(combined)))

NA_summary <- sapply(combined, function(x) sum(is.na(x)))
dev.new()
p_NA_2 <- VIM::aggr(combined, col = mdc(1:2), axes = FALSE, border = NA, sortVars = FALSE)
print(p_NA_2)
assert_that(length(which(NA_summary > 1000)) == 0)

# [OPTIMIZATION] Predictive imputation (CPU intensive)
delta_opt <- Sys.time()
partially_empty <- which(NA_summary > 0) # returns col indexes, not counts
print(Sys.time() - delta_opt)
imputed <- mice(combined[ ,partially_empty], method = "rf", printFlag = TRUE) # based on randomForest
completed_features <- mice::complete(imputed)
assert_that(!anyNA(completed_features))
combined[ ,partially_empty] <- completed_features
assert_that(!anyNA(combined))

# [OPTIMIZATION] Feature engineering
combined$HouseArea <- combined$X1stFlrSF + combined$X2ndFlrSF + combined$TotalBsmtSF + combined$GarageArea
numerical_features <- c(numerical_features, "HouseArea")

# 1-hot encoding for categorical features
dummies <- caret::dummyVars( ~ ., dplyr::select(combined, categorical_features))
categorical_1_hot <- predict(dummies, dplyr::select(combined, categorical_features))

# For all NA impute out of range value -optimized above
# categorical_1_hot[is.na(categorical_1_hot)] <- OOR
# for (x in numerical_features) {
#     combined[[x]] [is.na(combined[[x]])] <- OOR
# }

# Reconstruct
combined <- cbind(combined[numerical_features], categorical_1_hot)

NA_summary <- sapply(combined, function(x) sum(is.na(x)))
dev.new()
p_NA_3 <- VIM::aggr(combined, col = mdc(1:2), axes = FALSE, border = NA, sortVars = FALSE)
print(p_NA_3)
assert_that(!anyNA(combined))

# [OPTIMIZATION] Normalization
preObj <- caret::preProcess(combined, method = c("center", "scale"))
print(preObj)
combined <- predict(preObj, combined)

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

# AUC (CPU intensive) -Not of much use for regression
# roc.ridge <- pROC::multiclass.roc(validate$SalePrice, pred.validate.ridge)
# auc.ridge <- pROC::auc(roc.ridge)
# print(auc.ridge)
# roc.lasso <- pROC::multiclass.roc(validate$SalePrice, pred.validate.lasso)
# auc.lasso <- pROC::auc(roc.lasso)
# print(auc.lasso)

# Retrain best model with the whole original set for unlabeled data
x_fullset <- as.matrix(train_and_validate[ ,-target])
y_fullset <- as.numeric(train_and_validate$SalePrice)

cv.submit <- cv.glmnet(x_fullset, y_fullset, alpha = 0, standardize = TRUE, type.measure = "mse")
pred.test <- predict(cv.submit, newx = as.matrix(test), s = cv.lasso$lambda.1se, type="response")

# Export for submission
submit.lm <- data.frame(Id = test_Id, SalePrice = as.numeric(pred.test))
filename <- paste("lm_submission", format(Sys.time(),"%Y%m%d_%H%M%S"), "csv", sep=".")
write.csv(submit.lm, filename, quote = FALSE, row.names = FALSE)
submission_file.lm <- read.csv(filename, stringsAsFactors = FALSE)

# MODEL OPTIMIZATION ###################################################################

# [OPTIMIZATION] feature selection (CPU intensive)
saCtrl <- safsControl(functions = caretSA, improve = 25, allowParallel = TRUE, verbose = TRUE) # via simulated annealing

# Linear model
delta_sa.lm <- Sys.time()
sa_fs.lm <- caret::safs(x = x_fullset, y = y_fullset, iters = 10, safsControl = saCtrl, method = "glmnet")
print(Sys.time() - delta_sa.lm)
x_sa_set.lm <- dplyr::select(train_and_validate, sa_fs.lm$optVariables)

# XGBoost           
sa_fs.xgb <- caret::safs(x = x_fullset, y = y_fullset, iters = 3, safsControl = saCtrl, method = "xgbTree")
print(Sys.time() - start_time)
x_sa_set.xgb <- dplyr::select(train_and_validate, sa_fs.xgb$optVariables)

# SVM with radial kernel
sa_fs.svm <- caret::safs(x = x_fullset, y = y_fullset, iters = 3, safsControl = saCtrl, method = "svmRadial")
print(Sys.time() - start_time)
x_sa_set.svm <- dplyr::select(train_and_validate, sa_fs.svm$optVariables)

# [OPTIMIZATION] Hyperparameter search
trainCtrl <- trainControl(method="cv", number = 10, verbose = TRUE)

# Linear model
hyper.fit.lm <- caret::train(x_sa_set.lm, y_fullset, method = "glmnet", trControl = trainCtrl)
print(hyper.fit.lm)
pred.hyper.lm <- predict(hyper.fit.lm, test)

submit.lm <- data.frame(Id = test_Id, SalePrice = as.numeric(pred.hyper.lm))
filename <- paste("hyper_submission.lm", format(Sys.time(),"%Y%m%d_%H%M%S"), "csv", sep=".")
write.csv(submit.lm, filename, quote = FALSE, row.names = FALSE)

# XGBoost           
hyper.fit.xgb <- caret::train(x = x_sa_set.xgb, y = y_fullset, method = 'xgbTree', metric = 'RMSE', trControl = trainCtrl)
print(hyper.fit.xgb)
pred.hyper.xgb <- predict(hyper.fit.xgb, test)

submit.xgb <- data.frame(Id = test_Id, SalePrice = as.numeric(pred.hyper.xgb))
filename <- paste("hyper_submission.xgb", format(Sys.time(),"%Y%m%d_%H%M%S"), "csv", sep=".")
write.csv(submit.xgb, filename, quote = FALSE, row.names = FALSE)

# SVM with radial kernel
hyper.fit.svm <- caret::train(x = x_sa_set.nn, y = y_fullset, method = 'svmRadial', metric = 'RMSE', trControl = trainCtrl)
print(hyper.fit.svm)
pred.sa.svm <- predict(hyper.fit.svm, test)


print("Done")