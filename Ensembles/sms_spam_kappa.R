library("tm")
library("wordcloud")
library("SnowballC")
library("e1071")

library("caret")
library("irr")
library("ROCR")

options(warn = -1)
set.seed(123)

cat("\n")
cat("---------------------------------------------------------- \n")
cat(" Classification using Naive Bayes (Performance evaluation) \n")
cat("---------------------------------------------------------- \n")

setwd("C:/Users/drexa/git/machine-learning/Naive Bayes")

# Import data
results <- read.csv("../datasets/sms_results.csv", stringsAsFactors = FALSE)
caret_conf_matrix <- confusionMatrix(results$predict_type, 
                                     results$actual_type, 
                                     positive = "spam")
# print(caret_conf_matrix)
cat("\n*** Severe class imbalance, real possibility of a correct prediction by chance alone: \n")
print(caret_conf_matrix$overall[2])
cat("\n*** Alternative Kappa: ")
irr.kappa <- kappa2(results[1:2])
print(unlist(irr.kappa$value))

# ROC visualizing
roc_pred <- prediction(results$prob_spam, results$actual_type)
roc_perf <- performance(roc_pred, measure = "tpr", x.measure = "fpr")
plot(roc_perf, main = "ROC curve for current classifier", col = "lavender", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
cat("*** AUC: ")
roc_perf.uac <- performance(roc_pred, measure = "auc")
print(unlist(roc_perf.uac@y.values))
