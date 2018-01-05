library("C50")

library("caret")

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

# Caret automated data munging and parameter tunning
cat("*** Model lookup... ")
model <- train(default ~ ., data = credits, method = "C5.0")
print(model)

# Full set, not indicative for unseen data
# pred <- predict(model, credits)
# pred_prob <- predict(model, credits, type = "prob")
# caret_conf_matrix <- confusionMatrix(pred, credits$default, positive = "yes")
#print(caret_conf_matrix)

folds <- createFolds(credits$default, k = 10)
results <- lapply(folds, function(x) {
    train <- credits[-x,]
    test <- credits[x,]

    model <- C5.0(default ~ ., data = train, trials = 20, model = tree, winnow = FALSE)
    
    pred <- predict(model, test)
    pred_prob <- predict(model, credits, type = "prob")

    kappa <- kappa2(data.frame(test$default, pred))
    return(kappa$value)
})
cat("\n*** C5.0 - Kappa statistic mean for 10 folds: ")
print(mean(unlist(results)))
