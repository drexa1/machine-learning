library("C50")

library("caret")

options(warn = -1)
set.seed(123)

cat("\n")
cat("------------------------------------------------------------- \n")
cat(" Classification using Decision Trees (Performance evaluation) \n")
cat("------------------------------------------------------------- \n")

# setwd("C:/Users/drexa/git/R/MachineLearning/Decision Trees")
setwd("C:/Users/dr186049/git/MachineLearning/Decision Trees")

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
pred <- predict(model, credits)
pred_prob <- predict(model, credits, type = "prob")

caret_conf_matrix <- confusionMatrix(pred, credits$default, positive = "yes")
print(caret_conf_matrix)
results <- data.frame(pred, pred_prob$yes)
irr.kappa <- kappa2(results[1:2])
cat("\n*** Kappa: ")
print(unlist(irr.kappa$value))

# ROC visualizing
roc_pred <- prediction(results$pred_prob.yes, results$pred)
roc_perf <- performance(roc_pred, measure = "tpr", x.measure = "fpr")
plot(roc_perf, main = "ROC curve for current classifier", col = "lavender", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
cat("*** AUC: ")
roc_perf.uac <- performance(roc_pred, measure = "auc")
print(unlist(roc_perf.uac@y.values))