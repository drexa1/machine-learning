library("C50")

library("caret")
library("irr")

options(warn = -1)
set.seed(123)

cat("\n")
cat("------------------------------------------------------------- \n")
cat(" Classification using Decision Trees (Performance evaluation) \n")
cat("------------------------------------------------------------- \n")

setwd("C:/Users/drexa/git/machine-learning/Decision Trees")

# Import data (majority of nominal features)
credits <- read.csv("../datasets/credit.csv", stringsAsFactors = TRUE)
cat("*** Hamburg credit agency loans dataset imported \n\n")

# Recode $default as a factor
credits$default <- factor(credits$default, c("1", "2"), c("no", "yes"))

# k-fold CV 
folds <- createFolds(credits$default, k = 10)

results <- lapply(folds, function(x) {
    train <- credits[ - x,]
    test <- credits[x,]
    model <- C5.0(default ~ ., data = train)
    credit_pred <- predict(model, test)
    credit_actual <- test$default
    kappa <- kappa2(data.frame(credit_actual, credit_pred))
    return(kappa$value)
})
cat("*** C5.0 - Kappa statistic mean for 10 folds: ")
print(mean(unlist(results)))

results <- lapply(folds, function(x) {
    train <- credits[ - x,]
    test <- credits[x,]
    model <- C5.0(default ~ ., data = train, trials = 10)
    credit_pred <- predict(model, test)
    credit_actual <- test$default
    kappa <- kappa2(data.frame(credit_actual, credit_pred))
    return(kappa$value)
})
cat("*** C5.0 boosted - Kappa statistic mean for 10 folds: ")
print(mean(unlist(results)))


